"""
HTTP training server.

Runs on Jarvis (RTX 4090). Accepts self-play samples via HTTP,
trains models on GPU, serves weights to clients.

Usage:
    julia --threads 4 --project scripts/training_server.jl \\
        --port 9090 --data-dir ./sessions/alphazero-server
"""

using HTTP
using JSON
using MsgPack
using Dates
using SHA

include(joinpath(@__DIR__, "eval_manager.jl"))
using .EvalManager

"""Cumulative, lock-free counters for the sample-ingest and self-play path."""
mutable struct ServerMetrics
    upload_requests::Threads.Atomic{Int64}
    accepted_batches::Threads.Atomic{Int64}
    duplicate_batches::Threads.Atomic{Int64}
    rejected_batches::Threads.Atomic{Int64}
    dedupe_evictions::Threads.Atomic{Int64}
    request_bytes::Threads.Atomic{Int64}
    upload_ns::Threads.Atomic{Int64}
    mcts_simulations::Threads.Atomic{Int64}
    tree_hits::Threads.Atomic{Int64}
    tree_misses::Threads.Atomic{Int64}
    nn_evaluations::Threads.Atomic{Int64}
    oracle_calls::Threads.Atomic{Int64}
    bearoff_hits::Threads.Atomic{Int64}
    bearoff_misses::Threads.Atomic{Int64}
    search_ns::Threads.Atomic{Int64}
    max_depth::Threads.Atomic{Int64}
end

ServerMetrics() = ServerMetrics((Threads.Atomic{Int64}(0) for _ in 1:16)...)

function _atomic_max!(a::Threads.Atomic{Int64}, value::Int64)
    old = a[]
    while value > old
        observed = Threads.atomic_cas!(a, old, value)
        observed == old && return value
        old = observed
    end
    return old
end

"""Read a consistent-enough cumulative snapshot for monitoring and rate deltas."""
function server_metrics_snapshot(state)
    m = state.metrics
    return (
        upload_requests=m.upload_requests[], accepted_batches=m.accepted_batches[],
        duplicate_batches=m.duplicate_batches[], rejected_batches=m.rejected_batches[],
        dedupe_evictions=m.dedupe_evictions[], request_bytes=m.request_bytes[],
        upload_ns=m.upload_ns[], mcts_simulations=m.mcts_simulations[],
        tree_hits=m.tree_hits[], tree_misses=m.tree_misses[],
        nn_evaluations=m.nn_evaluations[], oracle_calls=m.oracle_calls[],
        bearoff_hits=m.bearoff_hits[], bearoff_misses=m.bearoff_misses[],
        search_ns=m.search_ns[], max_depth=m.max_depth[])
end

# Client tracking
mutable struct ClientStats
    client_id::String
    client_type::String    # "julia" or "web"
    name::String
    git_commit::String
    eval_capable::Bool
    has_wildbg::Bool
    protocol_version::Int
    assigned_seed::Int
    games_contributed::Int
    samples_contributed::Int
    first_seen::DateTime
    last_seen::DateTime
end

"""Compute samples/sec for a client from its contribution history."""
function client_samples_per_sec(c::ClientStats)
    elapsed = Dates.value(c.last_seen - c.first_seen) / 1000.0  # seconds
    elapsed < 1.0 ? 0.0 : c.samples_contributed / elapsed
end

mutable struct ServerState
    # Models and training
    iteration::Threads.Atomic{Int}
    contact_version::Threads.Atomic{Int}
    race_version::Threads.Atomic{Int}
    total_games::Threads.Atomic{Int}
    total_samples::Threads.Atomic{Int}

    # Cached weight blobs (updated after each training iteration)
    contact_weight_bytes::Vector{UInt8}
    race_weight_bytes::Vector{UInt8}
    contact_onnx_bytes::Vector{UInt8}
    race_onnx_bytes::Vector{UInt8}
    weight_lock::ReentrantLock

    # Weight history: version → (contact_bytes, race_bytes)
    # Kept alive while an eval job references that version
    weight_history::Dict{Int, Tuple{Vector{UInt8}, Vector{UInt8}}}

    # Loss metrics
    contact_loss::Float64
    race_loss::Float64

    # Client tracking
    clients::Dict{String, ClientStats}
    clients_lock::ReentrantLock

    # Bounded idempotency window for retrying uploads after lost responses.
    accepted_batches::Dict{String,Int}
    accepted_batch_order::Vector{String}
    accepted_batches_lock::ReentrantLock

    # Aggregate observability (kept separate from training state and buffer locks)
    metrics::ServerMetrics

    # Config served to clients
    config::Dict{String, Any}

    # API key
    api_key::String

    # Server start time
    start_time::DateTime

    # Client restart flag (set via /api/restart-clients, cleared after all clients disconnect)
    restart_clients::Threads.Atomic{Bool}

    # Operational state: draining stops new uploads and asks the training loop to
    # finish its current safe boundary before writing a final checkpoint.
    accepting_samples::Threads.Atomic{Bool}
    shutdown_requested::Threads.Atomic{Bool}
end

function ServerState(; api_key::String, config::Dict{String, Any})
    ServerState(
        Threads.Atomic{Int}(0), Threads.Atomic{Int}(0), Threads.Atomic{Int}(0),
        Threads.Atomic{Int}(0), Threads.Atomic{Int}(0),
        UInt8[], UInt8[], UInt8[], UInt8[],
        ReentrantLock(),
        Dict{Int, Tuple{Vector{UInt8}, Vector{UInt8}}}(),
        0.0, 0.0,
        Dict{String, ClientStats}(),
        ReentrantLock(),
        Dict{String,Int}(),
        String[],
        ReentrantLock(),
        ServerMetrics(),
        config,
        api_key,
        now(),
        Threads.Atomic{Bool}(false),
        Threads.Atomic{Bool}(true),
        Threads.Atomic{Bool}(false),
    )
end

# --- Authentication ---

function check_auth(req::HTTP.Request, state::ServerState)::Bool
    auth = HTTP.header(req, "Authorization", "")
    expected = "Bearer $(state.api_key)"
    return auth == expected
end

function stable_client_seed(base_seed::Integer, client_id::AbstractString)
    base_seed >= 0 || throw(ArgumentError("base seed must be non-negative"))
    digest = sha256(codeunits(client_id))
    offset = zero(UInt64)
    @inbounds for i in 1:8
        offset = (offset << 8) | UInt64(digest[i])
    end
    return Int(mod(UInt64(base_seed) + offset, UInt64(typemax(Int) - 1))) + 1
end

# --- HTTP Handlers ---

function handle_register(req::HTTP.Request, state::ServerState)
    check_auth(req, state) || return HTTP.Response(401, "Unauthorized")
    try
        body = JSON.parse(String(req.body))
        client_id = get(body, "client_id", "unknown")
        client_type = get(body, "client_type", "julia")
        name = get(body, "name", client_id)
        git_commit = get(body, "git_commit", "unknown")
        eval_capable = get(body, "eval_capable", false)
        has_wildbg = get(body, "has_wildbg", false)
        protocol_version = Int(get(body, "protocol_version", 0))
        protocol_version == DISTRIBUTED_PROTOCOL_VERSION || return HTTP.Response(
            409, "Distributed protocol mismatch: server=$DISTRIBUTED_PROTOCOL_VERSION " *
                 "client=$protocol_version")

        # Assign unique seed based on client count (deterministic, non-overlapping)
        assigned_seed = lock(state.clients_lock) do
            existing = get(state.clients, client_id, nothing)
            if existing !== nothing
                existing.name = name
                existing.git_commit = git_commit
                existing.eval_capable = eval_capable
                existing.has_wildbg = has_wildbg
                existing.protocol_version = protocol_version
                existing.last_seen = now()
                return existing.assigned_seed
            end
            # Stable across server restarts and independent of registration order.
            seed = stable_client_seed(Int(state.config["seed"]), client_id)
            state.clients[client_id] = ClientStats(
                client_id, client_type, name, git_commit,
                eval_capable, has_wildbg, protocol_version, seed,
                0, 0, now(), now()
            )
            eval_str = eval_capable ? " [eval-capable$(has_wildbg ? "+wildbg" : "")]" : ""
            println("[Server] Client registered: $name ($client_id) commit=$git_commit$eval_str")
            seed
        end

        return HTTP.Response(200, JSON.json(Dict(
            "session_id" => client_id,
            "assigned_seed" => assigned_seed,
            "protocol_version" => DISTRIBUTED_PROTOCOL_VERSION,
            "contract_fingerprint" => get(state.config, "contract_fingerprint", ""),
        )))
    catch e
        return HTTP.Response(400, "Bad request: $e")
    end
end

function handle_config(req::HTTP.Request, state::ServerState)
    check_auth(req, state) || return HTTP.Response(401, "Unauthorized")
    return HTTP.Response(200, ["Content-Type" => "application/json"],
                         JSON.json(state.config))
end

function handle_samples(req::HTTP.Request, state::ServerState, buffer::PERBuffer)
    started_ns = time_ns()
    Threads.atomic_add!(state.metrics.upload_requests, Int64(1))
    Threads.atomic_add!(state.metrics.request_bytes, Int64(length(req.body)))
    try
        if !check_auth(req, state)
            Threads.atomic_add!(state.metrics.rejected_batches, Int64(1))
            return HTTP.Response(401, "Unauthorized")
        end
        state.accepting_samples[] || return HTTP.Response(
            503, ["Retry-After" => "5"], "Server is draining; retry after restart")
        content_type = HTTP.header(req, "Content-Type", "application/msgpack")
        client_id = HTTP.header(req, "X-Client-Id", "unknown")

        # Deserialize and validate OUTSIDE the buffer lock.
        envelope = if occursin("json", content_type)
            unpack_samples_json_envelope(String(req.body))
        else
            unpack_samples_envelope(req.body)
        end
        validate_sample_envelope!(
            envelope, String(state.config["contract_fingerprint"]))
        batch = envelope.batch

        duplicate = lock(state.accepted_batches_lock) do
            if haskey(state.accepted_batches, envelope.batch_id)
                true
            else
                # Keep append + batch-ID recording atomic with respect to retries.
                per_add_batch!(buffer,
                    batch.states, batch.policies, batch.values,
                    batch.equities, batch.has_equity, batch.is_chance,
                    batch.is_contact, batch.is_bearoff;
                    source_iteration=envelope.source_iteration)
                state.accepted_batches[envelope.batch_id] = Int(batch.n)
                push!(state.accepted_batch_order, envelope.batch_id)
                if length(state.accepted_batch_order) > 10_000
                    expired = popfirst!(state.accepted_batch_order)
                    delete!(state.accepted_batches, expired)
                    Threads.atomic_add!(state.metrics.dedupe_evictions, Int64(1))
                end
                false
            end
        end

        # Update stats
        if !duplicate
            Threads.atomic_add!(state.total_samples, Int(batch.n))
            Threads.atomic_add!(state.total_games, Int(envelope.metrics.games))
            Threads.atomic_add!(state.metrics.accepted_batches, Int64(1))
            Threads.atomic_add!(state.metrics.mcts_simulations, envelope.metrics.mcts_simulations)
            Threads.atomic_add!(state.metrics.tree_hits, envelope.metrics.tree_hits)
            Threads.atomic_add!(state.metrics.tree_misses, envelope.metrics.tree_misses)
            Threads.atomic_add!(state.metrics.nn_evaluations, envelope.metrics.nn_evaluations)
            Threads.atomic_add!(state.metrics.oracle_calls, envelope.metrics.oracle_calls)
            Threads.atomic_add!(state.metrics.bearoff_hits, envelope.metrics.bearoff_hits)
            Threads.atomic_add!(state.metrics.bearoff_misses, envelope.metrics.bearoff_misses)
            Threads.atomic_add!(state.metrics.search_ns, envelope.metrics.search_ns)
            _atomic_max!(state.metrics.max_depth, envelope.metrics.max_depth)
            lock(state.clients_lock) do
                if haskey(state.clients, client_id)
                    state.clients[client_id].games_contributed += Int(envelope.metrics.games)
                    state.clients[client_id].samples_contributed += Int(batch.n)
                    state.clients[client_id].last_seen = now()
                end
            end
        else
            Threads.atomic_add!(state.metrics.duplicate_batches, Int64(1))
        end

        resp = Dict{String,Any}("accepted" => Int(batch.n), "buffer_size" => buf_length(buffer),
                    "restart" => state.restart_clients[], "duplicate" => duplicate,
                    "batch_id" => envelope.batch_id)

        # Assign eval work to eval-capable clients (one chunk at a time per client)
        lock(EVAL_LOCK) do
            job = EVAL_JOB[]
            job === nothing && return
            is_eval_client = lock(state.clients_lock) do
                c = get(state.clients, client_id, nothing)
                c !== nothing && c.eval_capable && c.has_wildbg
            end
            is_eval_client || return
            # Don't assign if this client already has a checked-out chunk
            cid_str = String(client_id)
            has_checkout = any(c -> c.checked_out_by == cid_str && !c.completed, job.chunks)
            has_checkout && return
            chunk = EvalManager.checkout_chunk!(job, cid_str)
            chunk === nothing && return
            println("[Eval] Assigned chunk $(chunk.chunk_id) to $client_id (iter=$(job.iter))")
            resp["eval_chunk"] = Dict(
                "chunk_id" => chunk.chunk_id,
                "position_range_start" => first(chunk.position_range),
                "position_range_end" => last(chunk.position_range),
                "az_is_white" => chunk.az_is_white,
                "weights_version" => job.weights_version,
                "eval_iter" => job.iter,
            )
        end

        return HTTP.Response(200, ["Content-Type" => "application/json"], JSON.json(resp))
    catch e
        Threads.atomic_add!(state.metrics.rejected_batches, Int64(1))
        @warn "Error handling samples" exception=(e, catch_backtrace())
        return HTTP.Response(400, "Bad request: $e")
    finally
        Threads.atomic_add!(state.metrics.upload_ns, Int64(time_ns() - started_ns))
    end
end

function handle_health(state::ServerState)
    return HTTP.Response(200, ["Content-Type" => "application/json"], JSON.json(Dict(
        "status" => "alive", "iteration" => state.iteration[],
        "accepting_samples" => state.accepting_samples[])))
end

function handle_ready(state::ServerState)
    weights_ready = lock(state.weight_lock) do
        !isempty(state.contact_weight_bytes) && !isempty(state.race_weight_bytes)
    end
    contract_ready = !isempty(String(get(state.config, "contract_fingerprint", "")))
    ready = weights_ready && contract_ready && state.accepting_samples[]
    body = JSON.json(Dict(
        "status" => ready ? "ready" : "not_ready",
        "weights_ready" => weights_ready,
        "contract_ready" => contract_ready,
        "accepting_samples" => state.accepting_samples[]))
    return HTTP.Response(ready ? 200 : 503, ["Content-Type" => "application/json"], body)
end

function handle_drain(req::HTTP.Request, state::ServerState)
    check_auth(req, state) || return HTTP.Response(401, "Unauthorized")
    state.accepting_samples[] = false
    state.shutdown_requested[] = true
    return HTTP.Response(202, ["Content-Type" => "application/json"],
                         JSON.json(Dict("draining" => true,
                                        "iteration" => state.iteration[])))
end

function handle_weights(req::HTTP.Request, state::ServerState, model::Symbol)
    check_auth(req, state) || return HTTP.Response(401, "Unauthorized")
    # Parse ?version=N query param for pinned weight downloads (eval)
    requested_version = nothing
    query_str = split(req.target, "?"; limit=2)
    if length(query_str) == 2
        for param in split(query_str[2], "&")
            kv = split(param, "="; limit=2)
            if length(kv) == 2 && kv[1] == "version"
                requested_version = tryparse(Int, kv[2])
            end
        end
    end

    lock(state.weight_lock) do
        if requested_version !== nothing
            # Serve pinned version from history (for eval)
            entry = get(state.weight_history, requested_version, nothing)
            entry === nothing && return HTTP.Response(404, "Weight version $requested_version not available")
            bytes = model == :contact ? entry[1] : entry[2]
            return HTTP.Response(200,
                ["Content-Type" => "application/octet-stream",
                 "X-Iteration" => string(state.iteration[]),
                 "X-Version" => string(requested_version)],
                bytes)
        else
            # Serve latest (normal self-play sync)
            bytes = model == :contact ? state.contact_weight_bytes : state.race_weight_bytes
            isempty(bytes) && return HTTP.Response(404, "No weights available yet")
            version = model == :contact ? state.contact_version[] : state.race_version[]
            return HTTP.Response(200,
                ["Content-Type" => "application/octet-stream",
                 "X-Iteration" => string(state.iteration[]),
                 "X-Version" => string(version)],
                bytes)
        end
    end
end

function handle_weights_onnx(req::HTTP.Request, state::ServerState, model::Symbol)
    check_auth(req, state) || return HTTP.Response(401, "Unauthorized")
    lock(state.weight_lock) do
        bytes = model == :contact ? state.contact_onnx_bytes : state.race_onnx_bytes
        isempty(bytes) && return HTTP.Response(404, "No ONNX weights available yet")
        return HTTP.Response(200,
            ["Content-Type" => "application/octet-stream",
             "X-Iteration" => string(state.iteration[])],
            bytes)
    end
end

function handle_weights_version(req::HTTP.Request, state::ServerState)
    check_auth(req, state) || return HTTP.Response(401, "Unauthorized")
    resp = Dict(
        "contact_version" => state.contact_version[],
        "race_version" => state.race_version[],
        "iteration" => state.iteration[],
    )
    return HTTP.Response(200, ["Content-Type" => "application/json"], JSON.json(resp))
end

function handle_status(req::HTTP.Request, state::ServerState, buffer::PERBuffer)
    # Status endpoint doesn't require auth (public info)
    metrics = server_metrics_snapshot(state)
    upload_latency_ms = metrics.upload_requests == 0 ? 0.0 :
        metrics.upload_ns / metrics.upload_requests / 1e6
    resp = Dict(
        "iteration" => state.iteration[],
        "buffer_size" => buf_length(buffer),
        "contact_loss" => state.contact_loss,
        "race_loss" => state.race_loss,
        "total_games" => state.total_games[],
        "total_samples" => state.total_samples[],
        "total_clients" => length(state.clients),
        "accepting_samples" => state.accepting_samples[],
        "shutdown_requested" => state.shutdown_requested[],
        "uptime_seconds" => round(Dates.value(now() - state.start_time) / 1000, digits=1),
        "observability" => Dict(
            "upload_requests" => metrics.upload_requests,
            "accepted_batches" => metrics.accepted_batches,
            "duplicate_batches" => metrics.duplicate_batches,
            "rejected_batches" => metrics.rejected_batches,
            "dedupe_evictions" => metrics.dedupe_evictions,
            "avg_upload_latency_ms" => round(upload_latency_ms, digits=2),
            "request_bytes" => metrics.request_bytes,
            "mcts_simulations" => metrics.mcts_simulations,
            "tree_hits" => metrics.tree_hits,
            "tree_misses" => metrics.tree_misses,
            "nn_evaluations" => metrics.nn_evaluations,
            "oracle_calls" => metrics.oracle_calls,
            "bearoff_hits" => metrics.bearoff_hits,
            "bearoff_misses" => metrics.bearoff_misses,
            "search_seconds" => round(metrics.search_ns / 1e9, digits=3),
            "max_depth" => metrics.max_depth,
        ),
    )
    return HTTP.Response(200, ["Content-Type" => "application/json"], JSON.json(resp))
end

function handle_clients(req::HTTP.Request, state::ServerState)
    check_auth(req, state) || return HTTP.Response(401, "Unauthorized")
    clients = lock(state.clients_lock) do
        [Dict(
            "client_id" => c.client_id,
            "type" => c.client_type,
            "name" => c.name,
            "git_commit" => c.git_commit,
            "eval_capable" => c.eval_capable,
            "has_wildbg" => c.has_wildbg,
            "protocol_version" => c.protocol_version,
            "assigned_seed" => c.assigned_seed,
            "games_contributed" => c.games_contributed,
            "samples_contributed" => c.samples_contributed,
            "first_seen" => Dates.format(c.first_seen, "yyyy-mm-dd HH:MM:SS"),
            "last_seen" => Dates.format(c.last_seen, "yyyy-mm-dd HH:MM:SS"),
            "samples_per_sec" => round(client_samples_per_sec(c), digits=1),
        ) for c in values(state.clients)]
    end
    return HTTP.Response(200, ["Content-Type" => "application/json"], JSON.json(clients))
end

function handle_restart_clients(req::HTTP.Request, state::ServerState)
    check_auth(req, state) || return HTTP.Response(401, "Unauthorized")
    Threads.atomic_xchg!(state.restart_clients, true)
    n = length(state.clients)
    println("[Server] Restart signal sent — $n client(s) will restart after next upload")
    return HTTP.Response(200, JSON.json(Dict("ok" => true, "clients" => n)))
end

function handle_cancel_restart(req::HTTP.Request, state::ServerState)
    check_auth(req, state) || return HTTP.Response(401, "Unauthorized")
    Threads.atomic_xchg!(state.restart_clients, false)
    println("[Server] Restart signal cancelled")
    return HTTP.Response(200, JSON.json(Dict("ok" => true)))
end

"""Get aggregate cluster performance stats."""
function get_cluster_stats(state::ServerState)
    lock(state.clients_lock) do
        cutoff = now() - Dates.Minute(5)
        active_clients = filter(c -> c.last_seen >= cutoff, collect(values(state.clients)))
        total_sps = sum(client_samples_per_sec(c) for c in active_clients; init=0.0)
        n_active = length(active_clients)
        per_client = Dict{String, Dict{String, Float64}}()
        for c in active_clients
            per_client[c.client_id] = Dict(
                "samples_per_sec" => round(client_samples_per_sec(c), digits=1),
            )
        end
        return (total_samples_per_sec=total_sps, total_clients=n_active, per_client=per_client)
    end
end

# --- Distributed Eval State ---

const EVAL_JOB = Ref{Union{Nothing, EvalManager.EvalJob}}(nothing)
const EVAL_LOCK = ReentrantLock()
const EVAL_CHUNK_SIZE = 100         # 100 games per chunk → 20 chunks for 2000 games
const EVAL_CHECKOUT_LEASE = 300.0   # 5 minutes — heartbeats extend; expires if client dies
const EVAL_JOB_TIMEOUT = 7200.0    # 2 hours total — eval completes naturally, never replaced mid-run

# --- Distributed Eval Handlers ---

function handle_eval_status(req::HTTP.Request, state::ServerState)
    check_auth(req, state) || return HTTP.Response(401, "Unauthorized")
    resp = lock(EVAL_LOCK) do
        job = EVAL_JOB[]
        if job === nothing
            Dict("eval_iter" => 0)
        else
            s = EvalManager.status(job)
            Dict("eval_iter" => s.eval_iter,
                 "total_chunks" => s.total_chunks,
                 "completed" => s.completed,
                 "checked_out" => s.checked_out,
                 "available" => s.available,
                 "weights_version" => job.weights_version)
        end
    end
    return HTTP.Response(200, ["Content-Type" => "application/json"], JSON.json(resp))
end

function handle_eval_checkout(req::HTTP.Request, state::ServerState)
    check_auth(req, state) || return HTTP.Response(401, "Unauthorized")
    try
        body = MsgPack.unpack(req.body)
        client_name = get(body, "client_name", "unknown")

        resp = lock(EVAL_LOCK) do
            job = EVAL_JOB[]
            if job === nothing
                return Dict("chunk_id" => 0)
            end
            chunk = EvalManager.checkout_chunk!(job, client_name)
            if chunk === nothing
                return Dict("chunk_id" => 0)
            end
            Dict("chunk_id" => chunk.chunk_id,
                 "position_range_start" => first(chunk.position_range),
                 "position_range_end" => last(chunk.position_range),
                 "az_is_white" => chunk.az_is_white,
                 "weights_version" => job.weights_version,
                 "eval_iter" => job.iter)
        end
        return HTTP.Response(200, ["Content-Type" => "application/json"], JSON.json(resp))
    catch e
        @warn "Error handling eval checkout" exception=(e, catch_backtrace())
        return HTTP.Response(400, "Bad request: $e")
    end
end

function handle_eval_submit(req::HTTP.Request, state::ServerState)
    check_auth(req, state) || return HTTP.Response(401, "Unauthorized")
    try
        body = MsgPack.unpack(req.body)
        chunk_id = Int(body["chunk_id"])
        client_name = get(body, "client_name", "unknown")
        rewards = Float64.(body["rewards"])
        value_nn = Float64.(get(body, "value_nn", Float64[]))
        value_opp = Float64.(get(body, "value_opp", Float64[]))
        value_is_contact = Bool.(get(body, "value_is_contact", Bool[]))
        reward_is_contact = Bool.(get(body, "reward_is_contact", fill(false, length(rewards))))

        # Reject length-skewed value arrays here, at the boundary. finalize_eval
        # does `value_nn .- value_opp` and cor(...) over the aggregated arrays; a
        # single mismatched submission would throw DimensionMismatch inside the
        # training loop's finalize and could take training down.
        if length(value_nn) != length(value_opp)
            return HTTP.Response(400, "value_nn ($(length(value_nn))) and value_opp ($(length(value_opp))) length mismatch")
        end
        if length(value_is_contact) != length(value_nn)
            return HTTP.Response(400, "value_is_contact ($(length(value_is_contact))) and value_nn ($(length(value_nn))) length mismatch")
        end
        if length(reward_is_contact) != length(rewards)
            return HTTP.Response(400, "reward_is_contact ($(length(reward_is_contact))) and rewards ($(length(rewards))) length mismatch")
        end

        eval_complete = false
        accepted = false
        error_status = 409
        error_message = "Eval submit rejected"
        lock(EVAL_LOCK) do
            job = EVAL_JOB[]
            if job === nothing
                error_message = "No active eval job"
                return
            end
            chunk_idx = findfirst(c -> c.chunk_id == chunk_id, job.chunks)
            if chunk_idx === nothing
                error_status = 404
                error_message = "Unknown eval chunk $chunk_id"
                return
            end
            chunk = job.chunks[chunk_idx]
            if chunk.completed
                error_message = "Eval chunk $chunk_id is already complete"
                return
            end
            if chunk.checked_out_by != client_name
                owner = something(chunk.checked_out_by, "nobody")
                error_message = "Eval chunk $chunk_id is owned by $owner, not $client_name"
                @warn "Eval submit rejected" chunk_id client_name owner
                return
            end
            expected = length(chunk.position_range)
            if length(rewards) != expected
                error_status = 400
                error_message = "Eval chunk $chunk_id: expected $expected rewards, got $(length(rewards))"
                @warn error_message chunk_id client_name
                return
            end
            az_is_white = chunk.az_is_white
            result = EvalManager.EvalChunkResult(chunk_id, az_is_white,
                                                  rewards, value_nn, value_opp,
                                                  value_is_contact, reward_is_contact)
            EvalManager.submit_chunk!(job, result)
            accepted = true
            n_done = count(c -> c.completed, job.chunks)
            n_total = length(job.chunks)
            if n_done % 10 == 0 || n_done == n_total
                println("Eval iter $(job.iter): $n_done/$n_total chunks submitted")
            end
            if EvalManager.is_complete(job)
                eval_complete = true
                # Finalize: aggregate results and log to TensorBoard
                stats = EvalManager.finalize_eval(job)
                iter = job.iter
                println("Eval iter $iter complete: equity=$(round(stats.equity, digits=4)), win%=$(round(stats.win_pct * 100, digits=1)), $(stats.num_games) games")
                try
                    with_logger(TB_LOGGER) do
                        @info "05_eval_strength/equity" value=stats.equity log_step_increment=0
                        @info "05_eval_strength/win_pct" value=stats.win_pct * 100 log_step_increment=0
                        @info "05_eval_strength/white_equity" value=stats.white_equity log_step_increment=0
                        @info "05_eval_strength/black_equity" value=stats.black_equity log_step_increment=0
                        @info "05_eval_strength/contact_value_mse" value=stats.contact_value_mse log_step_increment=0
                        @info "05_eval_strength/contact_value_corr" value=stats.contact_value_corr log_step_increment=0
                        @info "05_eval_strength/race_value_mse" value=stats.race_value_mse log_step_increment=0
                        @info "05_eval_strength/race_value_corr" value=stats.race_value_corr log_step_increment=0
                    end
                catch e
                    @warn "Failed to log eval to TensorBoard" exception=e
                end
                EVAL_JOB[] = nothing
            end
        end

        if !accepted
            resp = Dict("accepted" => false, "error" => error_message, "eval_complete" => false)
            return HTTP.Response(error_status, ["Content-Type" => "application/json"], JSON.json(resp))
        end

        resp = Dict("accepted" => true, "eval_complete" => eval_complete)
        return HTTP.Response(200, ["Content-Type" => "application/json"], JSON.json(resp))
    catch e
        @warn "Error handling eval submit" exception=(e, catch_backtrace())
        return HTTP.Response(400, "Bad request: $e")
    end
end

function handle_eval_heartbeat(req::HTTP.Request, state::ServerState)
    check_auth(req, state) || return HTTP.Response(401, "Unauthorized")
    try
        body = MsgPack.unpack(req.body)
        chunk_id = Int(body["chunk_id"])
        client_name = get(body, "client_name", "unknown")

        extended = lock(EVAL_LOCK) do
            job = EVAL_JOB[]
            job === nothing && return false
            EvalManager.extend_lease!(job, chunk_id, client_name)
        end

        return HTTP.Response(200, ["Content-Type" => "application/json"],
                             JSON.json(Dict("lease_extended" => extended)))
    catch e
        return HTTP.Response(400, "Bad request: $e")
    end
end

# --- Router ---

function create_router(state::ServerState, buffer::PERBuffer)
    router = HTTP.Router()

    HTTP.register!(router, "POST", "/api/register", req -> handle_register(req, state))
    HTTP.register!(router, "GET", "/api/health", _ -> handle_health(state))
    HTTP.register!(router, "GET", "/api/ready", _ -> handle_ready(state))
    HTTP.register!(router, "POST", "/api/drain", req -> handle_drain(req, state))
    HTTP.register!(router, "GET", "/api/config", req -> handle_config(req, state))
    HTTP.register!(router, "POST", "/api/samples", req -> handle_samples(req, state, buffer))
    HTTP.register!(router, "GET", "/api/weights/contact", req -> handle_weights(req, state, :contact))
    HTTP.register!(router, "GET", "/api/weights/race", req -> handle_weights(req, state, :race))
    HTTP.register!(router, "GET", "/api/weights/onnx/contact", req -> handle_weights_onnx(req, state, :contact))
    HTTP.register!(router, "GET", "/api/weights/onnx/race", req -> handle_weights_onnx(req, state, :race))
    HTTP.register!(router, "GET", "/api/weights/version", req -> handle_weights_version(req, state))
    HTTP.register!(router, "GET", "/api/status", req -> handle_status(req, state, buffer))
    HTTP.register!(router, "GET", "/api/clients", req -> handle_clients(req, state))
    # /api/client_stats removed — samples_per_sec computed server-side
    HTTP.register!(router, "POST", "/api/restart-clients", req -> handle_restart_clients(req, state))
    HTTP.register!(router, "POST", "/api/cancel-restart", req -> handle_cancel_restart(req, state))

    # Distributed eval endpoints
    HTTP.register!(router, "GET", "/api/eval/status", req -> handle_eval_status(req, state))
    HTTP.register!(router, "POST", "/api/eval/checkout", req -> handle_eval_checkout(req, state))
    HTTP.register!(router, "POST", "/api/eval/submit", req -> handle_eval_submit(req, state))
    HTTP.register!(router, "POST", "/api/eval/heartbeat", req -> handle_eval_heartbeat(req, state))

    # File serving endpoint — clients download start positions, eval positions, etc.
    HTTP.register!(router, "GET", "/api/file/*", function(req)
        filename = split(req.target, "/api/file/")[end]
        # Security: only serve from known data directories, no path traversal
        if contains(filename, "..") || contains(filename, "/")
            return HTTP.Response(400, "Invalid filename")
        end
        # Search known data directories
        search_paths = [
            get(ENV, "BACKGAMMONNET_EVAL_DATA_DIR", ""),
            get(state.config, "eval_data_dir", ""),
            get(state.config, "data_dir", ""),
        ]
        for dir in search_paths
            isempty(dir) && continue
            path = joinpath(dir, filename)
            if isfile(path)
                data = read(path)
                return HTTP.Response(200, ["Content-Type" => "application/octet-stream",
                    "Content-Length" => string(length(data))], data)
            end
        end
        return HTTP.Response(404, "File not found: $filename")
    end)

    return router
end

"""Update cached weight blobs after training iteration."""
function update_weight_cache!(state::ServerState, contact_network, race_network;
                              contact_width::Int, contact_blocks::Int,
                              race_width::Int, race_blocks::Int)
    contact_header = WeightHeader(0x01, Int32(state.iteration[]),
                                   Int32(contact_width), Int32(contact_blocks), UInt64(0))
    race_header = WeightHeader(0x02, Int32(state.iteration[]),
                                Int32(race_width), Int32(race_blocks), UInt64(0))

    lock(state.weight_lock) do
        state.contact_weight_bytes = serialize_weights_with_header(contact_network, contact_header)
        state.race_weight_bytes = serialize_weights_with_header(race_network, race_header)
        Threads.atomic_add!(state.contact_version, 1)
        Threads.atomic_add!(state.race_version, 1)

        # Stash current weights in history (version is post-increment)
        cv = state.contact_version[]
        state.weight_history[cv] = (copy(state.contact_weight_bytes), copy(state.race_weight_bytes))

        # Prune history: keep only versions referenced by active eval job
        prune_weight_history!(state)
    end
end

"""Remove weight history entries not needed by any active eval job. Must hold weight_lock."""
function prune_weight_history!(state::ServerState)
    needed = Set{Int}()
    # Always keep the latest version
    needed_latest = max(state.contact_version[], state.race_version[])
    push!(needed, needed_latest)
    # Keep version referenced by active eval job
    job = EVAL_JOB[]
    if job !== nothing
        push!(needed, job.weights_version)
    end
    for v in keys(state.weight_history)
        if v ∉ needed
            delete!(state.weight_history, v)
        end
    end
end

"""Save client stats to disk."""
function save_client_stats(state::ServerState, path::String)
    lock(state.clients_lock) do
        clients = Dict(id => Dict(
            "client_type" => c.client_type,
            "name" => c.name,
            "protocol_version" => c.protocol_version,
            "assigned_seed" => c.assigned_seed,
            "games_contributed" => c.games_contributed,
            "samples_contributed" => c.samples_contributed,
            "first_seen" => Dates.format(c.first_seen, "yyyy-mm-dd HH:MM:SS"),
            "last_seen" => Dates.format(c.last_seen, "yyyy-mm-dd HH:MM:SS"),
            "samples_per_sec" => round(client_samples_per_sec(c), digits=1),
        ) for (id, c) in state.clients)
        open(path, "w") do f
            JSON.print(f, clients, 2)
        end
    end
end

"""Start the HTTP server (non-blocking)."""
function start_server!(state::ServerState, buffer::PERBuffer; port::Int=9090)
    router = create_router(state, buffer)
    @info "Starting training server on port $port"
    server = HTTP.serve!(router, "0.0.0.0", port)
    return server
end
