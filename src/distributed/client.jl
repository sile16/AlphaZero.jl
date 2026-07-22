"""
HTTP client for self-play workers.

Connects to the training server to:
- Download config (self-play params)
- Download model weights
- Upload self-play samples
- Poll for weight updates
"""

using HTTP
using JSON
using MsgPack
using Random

mutable struct SelfPlayClient
    server_url::String
    api_key::String
    client_id::String
    client_type::String

    # Weight versioning
    contact_version::Int
    race_version::Int
    contact_iteration::Int
    race_iteration::Int

    # Upload batching
    upload_buffer::Vector{Any}
    upload_threshold::Int
    process_nonce::String
    batch_sequence::Int
    contract_fingerprint::String
    pending_batch_id::Union{Nothing,String}
    pending_upload_bytes::Union{Nothing,Vector{UInt8}}
    pending_upload_count::Int

    # Config from server
    config::Dict{String, Any}
end

function SelfPlayClient(server_url::String, api_key::String;
                        client_id::String="julia-$(gethostname())-$(Base.Libc.getpid())",
                        client_type::String="julia",
                        upload_threshold::Int=5000)
    SelfPlayClient(
        rstrip(server_url, '/'),
        api_key,
        client_id,
        client_type,
        0, 0,
        0, 0,
        Any[],
        upload_threshold,
        string(rand(UInt128); base=16),
        0,
        "",
        nothing,
        nothing,
        0,
        Dict{String, Any}(),
    )
end

function auth_headers(client::SelfPlayClient)
    ["Authorization" => "Bearer $(client.api_key)",
     "X-Client-Id" => client.client_id]
end

"""Register and return (success, assigned_seed). Server assigns unique seed per client."""
function register!(client::SelfPlayClient; name::String=client.client_id,
                   eval_capable::Bool=false, has_wildbg::Bool=false)
    git_commit = try
        strip(read(`git -C $(dirname(@__DIR__)) rev-parse --short HEAD`, String))
    catch
        "unknown"
    end
    body = JSON.json(Dict(
        "client_id" => client.client_id,
        "client_type" => client.client_type,
        "name" => name,
        "git_commit" => git_commit,
        "eval_capable" => eval_capable,
        "has_wildbg" => has_wildbg,
        "protocol_version" => DISTRIBUTED_PROTOCOL_VERSION,
    ))
    resp = HTTP.post("$(client.server_url)/api/register",
                     auth_headers(client),
                     body;
                     status_exception=false)
    if resp.status != 200
        @warn "Registration failed" status=resp.status body=String(resp.body)
        return (success=false, assigned_seed=nothing,
                contract_fingerprint=nothing)
    end
    result = JSON.parse(String(resp.body))
    seed = get(result, "assigned_seed", nothing)
    fingerprint = get(result, "contract_fingerprint", nothing)
    return (success=true, assigned_seed=seed,
            contract_fingerprint=fingerprint)
end

function next_batch_id!(client::SelfPlayClient)
    client.batch_sequence += 1
    return "$(client.client_id)-$(client.process_nonce)-$(client.batch_sequence)"
end

"""Fetch self-play config from server."""
function fetch_config!(client::SelfPlayClient)
    resp = HTTP.get("$(client.server_url)/api/config",
                    auth_headers(client);
                    status_exception=false)
    if resp.status != 200
        error("Failed to fetch config: $(resp.status)")
    end
    client.config = JSON.parse(String(resp.body))
    return client.config
end

"""Check if server has newer weights."""
function check_weight_version(client::SelfPlayClient)
    resp = HTTP.get("$(client.server_url)/api/weights/version",
                    auth_headers(client);
                    status_exception=false,
                    connect_timeout=10, readtimeout=30)
    if resp.status != 200
        @warn "Weight version check failed" status=resp.status
        return nothing
    end
    return JSON.parse(String(resp.body))
end

"""Download weights for a model. Returns (header, weight_arrays) or nothing."""
function download_weights(client::SelfPlayClient, model::Symbol;
                          expected_version::Union{Nothing, Int}=nothing,
                          pinned_version::Union{Nothing, Int}=nothing)
    endpoint = model == :contact ? "contact" : "race"
    url = "$(client.server_url)/api/weights/$endpoint"
    if pinned_version !== nothing
        url *= "?version=$pinned_version"
    end
    resp = HTTP.get(url, auth_headers(client);
                    status_exception=false,
                    connect_timeout=10, readtimeout=120)
    if resp.status != 200
        @warn "Weight download failed" model status=resp.status
        return nothing
    end
    downloaded_version = try
        parse(Int, HTTP.header(resp, "X-Version", "0"))
    catch
        0
    end
    if !isnothing(expected_version) && downloaded_version != expected_version
        error("Weight version mismatch for $model: expected $expected_version, got $downloaded_version")
    end
    return deserialize_weights_with_header(resp.body)
end

"""Sync weights if server has newer version. Returns true if weights were updated."""
function sync_weights!(client::SelfPlayClient, contact_network, race_network)
    version = check_weight_version(client)
    version === nothing && return false

    updated = false

    if version["contact_version"] > client.contact_version
        result = download_weights(client, :contact; expected_version=version["contact_version"])
        if result !== nothing
            header, weights = result
            FluxLib.load_weights!(contact_network, weights)
            client.contact_version = version["contact_version"]
            client.contact_iteration = Int(header.iteration)
            updated = true
            @info "Updated contact weights" version=client.contact_version iteration=header.iteration
        end
    end

    if version["race_version"] > client.race_version
        result = download_weights(client, :race; expected_version=version["race_version"])
        if result !== nothing
            header, weights = result
            FluxLib.load_weights!(race_network, weights)
            client.race_version = version["race_version"]
            client.race_iteration = Int(header.iteration)
            updated = true
            @info "Updated race weights" version=client.race_version iteration=header.iteration
        end
    end

    return updated
end

"""Add samples to upload buffer. Flushes when threshold is reached."""
function buffer_samples!(client::SelfPlayClient, samples::Vector)
    append!(client.upload_buffer, samples)
    if length(client.upload_buffer) >= client.upload_threshold
        flush_samples!(client)
    end
end

"""Upload all buffered samples to server."""
function prepare_pending_upload!(client::SelfPlayClient)
    isempty(client.upload_buffer) && return nothing
    if isnothing(client.pending_upload_bytes)
        client.pending_upload_count = length(client.upload_buffer)
        batch = samples_to_batch(client.upload_buffer[1:client.pending_upload_count])
        client.pending_batch_id = next_batch_id!(client)
        client.pending_upload_bytes = pack_samples(
            batch; contract_fingerprint=client.contract_fingerprint,
            batch_id=client.pending_batch_id,
            source_iteration=min(client.contact_iteration, client.race_iteration))
    end
    return (batch_id=client.pending_batch_id::String,
            bytes=client.pending_upload_bytes::Vector{UInt8},
            n=client.pending_upload_count)
end

function acknowledge_pending_upload!(client::SelfPlayClient, batch_id::AbstractString,
                                     accepted::Integer)
    client.pending_batch_id == batch_id || error(
        "Acknowledged batch $batch_id does not match pending $(client.pending_batch_id)")
    accepted == client.pending_upload_count || error(
        "Server acknowledged $accepted of $(client.pending_upload_count) samples " *
        "for batch $batch_id")
    deleteat!(client.upload_buffer, 1:Int(accepted))
    client.pending_batch_id = nothing
    client.pending_upload_bytes = nothing
    client.pending_upload_count = 0
    return Int(accepted)
end

function flush_samples!(client::SelfPlayClient)
    pending = prepare_pending_upload!(client)
    pending === nothing && return 0
    batch_id, bytes, n_uploaded = pending

    headers = vcat(auth_headers(client),
                   ["Content-Type" => "application/msgpack"])
    resp = HTTP.post("$(client.server_url)/api/samples",
                     headers,
                     bytes;
                     status_exception=false)

    if resp.status != 200
        @warn "Sample upload failed" status=resp.status n_samples=n_uploaded
        return 0
    end

    result = JSON.parse(String(resp.body))
    accepted = Int(result["accepted"])
    acknowledge_pending_upload!(client, batch_id, accepted)
    @debug "Uploaded samples" accepted=result["accepted"] buffer_size=result["buffer_size"]
    return accepted
end

"""Get server status."""
function server_status(client::SelfPlayClient)
    resp = HTTP.get("$(client.server_url)/api/status";
                    status_exception=false,
                    connect_timeout=10, readtimeout=30)
    if resp.status != 200
        return nothing
    end
    return JSON.parse(String(resp.body))
end
