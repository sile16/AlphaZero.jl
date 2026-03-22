"""
HTTP training server.

Runs on Jarvis (RTX 4090). Accepts self-play samples via HTTP,
trains models on GPU, serves weights to clients.

Usage:
    julia --threads 4 --project scripts/training_server.jl \\
        --port 9090 --data-dir /home/sile/alphazero-server
"""

using HTTP
using JSON
using MsgPack
using Dates

include(joinpath(@__DIR__, "eval_manager.jl"))
using .EvalManager

# Client tracking
mutable struct ClientStats
    client_id::String
    client_type::String    # "julia" or "web"
    name::String
    games_contributed::Int
    samples_contributed::Int
    first_seen::DateTime
    last_seen::DateTime
    # Performance metrics (updated via POST /api/client_stats)
    cpu_percent::Float64
    memory_used_gb::Float64
    memory_total_gb::Float64
    games_per_sec::Float64
    samples_per_sec::Float64
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

    # Loss metrics
    contact_loss::Float64
    race_loss::Float64

    # Client tracking
    clients::Dict{String, ClientStats}
    clients_lock::ReentrantLock

    # Config served to clients
    config::Dict{String, Any}

    # API key
    api_key::String

    # Server start time
    start_time::DateTime
end

function ServerState(; api_key::String, config::Dict{String, Any})
    ServerState(
        Threads.Atomic{Int}(0), Threads.Atomic{Int}(0), Threads.Atomic{Int}(0),
        Threads.Atomic{Int}(0), Threads.Atomic{Int}(0),
        UInt8[], UInt8[], UInt8[], UInt8[],
        ReentrantLock(),
        0.0, 0.0,
        Dict{String, ClientStats}(),
        ReentrantLock(),
        config,
        api_key,
        now(),
    )
end

# --- Authentication ---

function check_auth(req::HTTP.Request, state::ServerState)::Bool
    auth = HTTP.header(req, "Authorization", "")
    expected = "Bearer $(state.api_key)"
    return auth == expected
end

# --- HTTP Handlers ---

function handle_register(req::HTTP.Request, state::ServerState)
    check_auth(req, state) || return HTTP.Response(401, "Unauthorized")
    try
        body = JSON.parse(String(req.body))
        client_id = get(body, "client_id", "unknown")
        client_type = get(body, "client_type", "julia")
        name = get(body, "name", client_id)

        # Assign unique seed based on client count (deterministic, non-overlapping)
        assigned_seed = lock(state.clients_lock) do
            n = length(state.clients)
            seed = state.config["seed"] + (n + 1) * 104729  # large prime stride
            state.clients[client_id] = ClientStats(
                client_id, client_type, name,
                0, 0, now(), now(),
                0.0, 0.0, 0.0, 0.0, 0.0
            )
            seed
        end

        return HTTP.Response(200, JSON.json(Dict(
            "session_id" => client_id,
            "assigned_seed" => assigned_seed,
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
    check_auth(req, state) || return HTTP.Response(401, "Unauthorized")
    try
        content_type = HTTP.header(req, "Content-Type", "application/msgpack")
        client_id = HTTP.header(req, "X-Client-Id", "unknown")

        # Deserialize OUTSIDE the buffer lock (reduces lock contention)
        local batch::SampleBatch
        if occursin("json", content_type)
            batch = unpack_samples_json(String(req.body))
        else
            batch = unpack_samples(req.body)
        end

        # Add columnar data directly to buffer (no NamedTuple allocation)
        per_add_batch!(buffer,
            batch.states, batch.policies, batch.values,
            batch.equities, batch.has_equity, batch.is_contact, batch.is_bearoff)

        # Update stats
        Threads.atomic_add!(state.total_samples, Int(batch.n))
        lock(state.clients_lock) do
            if haskey(state.clients, client_id)
                state.clients[client_id].samples_contributed += Int(batch.n)
                state.clients[client_id].last_seen = now()
            end
        end

        resp = Dict("accepted" => Int(batch.n), "buffer_size" => buf_length(buffer))
        return HTTP.Response(200, ["Content-Type" => "application/json"], JSON.json(resp))
    catch e
        @warn "Error handling samples" exception=(e, catch_backtrace())
        return HTTP.Response(400, "Bad request: $e")
    end
end

function handle_weights(req::HTTP.Request, state::ServerState, model::Symbol)
    check_auth(req, state) || return HTTP.Response(401, "Unauthorized")
    lock(state.weight_lock) do
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
    resp = Dict(
        "iteration" => state.iteration[],
        "buffer_size" => buf_length(buffer),
        "contact_loss" => state.contact_loss,
        "race_loss" => state.race_loss,
        "total_games" => state.total_games[],
        "total_samples" => state.total_samples[],
        "total_clients" => length(state.clients),
        "uptime_seconds" => round(Dates.value(now() - state.start_time) / 1000, digits=1),
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
            "games_contributed" => c.games_contributed,
            "samples_contributed" => c.samples_contributed,
            "first_seen" => Dates.format(c.first_seen, "yyyy-mm-dd HH:MM:SS"),
            "last_seen" => Dates.format(c.last_seen, "yyyy-mm-dd HH:MM:SS"),
            "cpu_percent" => c.cpu_percent,
            "memory_used_gb" => c.memory_used_gb,
            "memory_total_gb" => c.memory_total_gb,
            "games_per_sec" => c.games_per_sec,
            "samples_per_sec" => c.samples_per_sec,
        ) for c in values(state.clients)]
    end
    return HTTP.Response(200, ["Content-Type" => "application/json"], JSON.json(clients))
end

function handle_client_stats(req::HTTP.Request, state::ServerState)
    check_auth(req, state) || return HTTP.Response(401, "Unauthorized")
    try
        body = JSON.parse(String(req.body))
        client_id = get(body, "client_id", "unknown")

        lock(state.clients_lock) do
            if haskey(state.clients, client_id)
                c = state.clients[client_id]
                c.cpu_percent = Float64(get(body, "cpu_percent", 0.0))
                c.memory_used_gb = Float64(get(body, "memory_used_gb", 0.0))
                c.memory_total_gb = Float64(get(body, "memory_total_gb", 0.0))
                c.games_per_sec = Float64(get(body, "games_per_sec", 0.0))
                c.samples_per_sec = Float64(get(body, "samples_per_sec", 0.0))
                c.last_seen = now()
            end
        end

        return HTTP.Response(200, JSON.json(Dict("ok" => true)))
    catch e
        return HTTP.Response(400, "Bad request: $e")
    end
end

"""Get aggregate cluster performance stats."""
function get_cluster_stats(state::ServerState)
    lock(state.clients_lock) do
        cutoff = now() - Dates.Minute(5)
        active_clients = filter(c -> c.last_seen >= cutoff, collect(values(state.clients)))
        total_gps = sum(c.games_per_sec for c in active_clients; init=0.0)
        total_sps = sum(c.samples_per_sec for c in active_clients; init=0.0)
        n_active = length(active_clients)
        per_client = Dict{String, Dict{String, Float64}}()
        for c in active_clients
            per_client[c.client_id] = Dict(
                "games_per_sec" => c.games_per_sec,
                "cpu_percent" => c.cpu_percent,
            )
        end
        return (total_games_per_sec=total_gps, total_samples_per_sec=total_sps,
                total_clients=n_active, per_client=per_client)
    end
end

# --- Distributed Eval State ---

const EVAL_JOB = Ref{Union{Nothing, EvalManager.EvalJob}}(nothing)
const EVAL_LOCK = ReentrantLock()
const EVAL_CHUNK_SIZE = 50
const EVAL_CHECKOUT_LEASE = 300.0  # 5 minutes
const EVAL_JOB_TIMEOUT = 1800.0   # 30 minutes

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

        eval_complete = false
        lock(EVAL_LOCK) do
            job = EVAL_JOB[]
            job === nothing && return
            chunk_idx = findfirst(c -> c.chunk_id == chunk_id, job.chunks)
            chunk_idx === nothing && return
            chunk = job.chunks[chunk_idx]
            # Validate ownership: only the client that checked out can submit
            if chunk.checked_out_by !== nothing && chunk.checked_out_by != client_name
                @warn "Eval submit rejected: chunk $chunk_id owned by $(chunk.checked_out_by), not $client_name"
                return
            end
            az_is_white = chunk.az_is_white
            result = EvalManager.EvalChunkResult(chunk_id, az_is_white,
                                                  rewards, value_nn, value_opp, value_is_contact)
            EvalManager.submit_chunk!(job, result)
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
                println("Eval iter $iter complete: equity=$(round(stats.equity, digits=4)), win%=$(round(stats.win_pct, digits=1)), $(stats.n_games) games")
                try
                    with_logger(TB_LOGGER) do
                        @info "eval/equity" value=stats.equity log_step_increment=0
                        @info "eval/win_pct" value=stats.win_pct log_step_increment=0
                        @info "eval/white_equity" value=stats.white_equity log_step_increment=0
                        @info "eval/black_equity" value=stats.black_equity log_step_increment=0
                        @info "eval/value_mse" value=stats.value_mse log_step_increment=0
                        @info "eval/value_corr" value=stats.value_corr log_step_increment=0
                        @info "eval/games" value=stats.n_games log_step_increment=0
                    end
                catch e
                    @warn "Failed to log eval to TensorBoard" exception=e
                end
                EVAL_JOB[] = nothing
            end
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
    HTTP.register!(router, "GET", "/api/config", req -> handle_config(req, state))
    HTTP.register!(router, "POST", "/api/samples", req -> handle_samples(req, state, buffer))
    HTTP.register!(router, "GET", "/api/weights/contact", req -> handle_weights(req, state, :contact))
    HTTP.register!(router, "GET", "/api/weights/race", req -> handle_weights(req, state, :race))
    HTTP.register!(router, "GET", "/api/weights/onnx/contact", req -> handle_weights_onnx(req, state, :contact))
    HTTP.register!(router, "GET", "/api/weights/onnx/race", req -> handle_weights_onnx(req, state, :race))
    HTTP.register!(router, "GET", "/api/weights/version", req -> handle_weights_version(req, state))
    HTTP.register!(router, "GET", "/api/status", req -> handle_status(req, state, buffer))
    HTTP.register!(router, "GET", "/api/clients", req -> handle_clients(req, state))
    HTTP.register!(router, "POST", "/api/client_stats", req -> handle_client_stats(req, state))

    # Distributed eval endpoints
    HTTP.register!(router, "GET", "/api/eval/status", req -> handle_eval_status(req, state))
    HTTP.register!(router, "POST", "/api/eval/checkout", req -> handle_eval_checkout(req, state))
    HTTP.register!(router, "POST", "/api/eval/submit", req -> handle_eval_submit(req, state))
    HTTP.register!(router, "POST", "/api/eval/heartbeat", req -> handle_eval_heartbeat(req, state))

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
    end
end

"""Save client stats to disk."""
function save_client_stats(state::ServerState, path::String)
    lock(state.clients_lock) do
        clients = Dict(id => Dict(
            "client_type" => c.client_type,
            "name" => c.name,
            "games_contributed" => c.games_contributed,
            "samples_contributed" => c.samples_contributed,
            "first_seen" => Dates.format(c.first_seen, "yyyy-mm-dd HH:MM:SS"),
            "last_seen" => Dates.format(c.last_seen, "yyyy-mm-dd HH:MM:SS"),
            "cpu_percent" => c.cpu_percent,
            "memory_used_gb" => c.memory_used_gb,
            "memory_total_gb" => c.memory_total_gb,
            "games_per_sec" => c.games_per_sec,
            "samples_per_sec" => c.samples_per_sec,
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
