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

# Client tracking
mutable struct ClientStats
    client_id::String
    client_type::String    # "julia" or "web"
    name::String
    games_contributed::Int
    samples_contributed::Int
    first_seen::DateTime
    last_seen::DateTime
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

        lock(state.clients_lock) do
            state.clients[client_id] = ClientStats(
                client_id, client_type, name,
                0, 0, now(), now()
            )
        end

        return HTTP.Response(200, JSON.json(Dict("session_id" => client_id)))
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

        local batch::SampleBatch
        if occursin("json", content_type)
            batch = unpack_samples_json(String(req.body))
        else
            batch = unpack_samples(req.body)
        end

        # Convert to NamedTuples and add to buffer
        samples = batch_to_samples(batch)
        per_add!(buffer, samples)

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
        ) for c in values(state.clients)]
    end
    return HTTP.Response(200, ["Content-Type" => "application/json"], JSON.json(clients))
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
