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

mutable struct SelfPlayClient
    server_url::String
    api_key::String
    client_id::String
    client_type::String

    # Weight versioning
    contact_version::Int
    race_version::Int

    # Upload batching
    upload_buffer::Vector{Any}
    upload_threshold::Int

    # Config from server
    config::Dict{String, Any}
end

function SelfPlayClient(server_url::String, api_key::String;
                        client_id::String="julia-$(gethostname())",
                        client_type::String="julia",
                        upload_threshold::Int=5000)
    SelfPlayClient(
        rstrip(server_url, '/'),
        api_key,
        client_id,
        client_type,
        0, 0,
        Any[],
        upload_threshold,
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
    ))
    resp = HTTP.post("$(client.server_url)/api/register",
                     auth_headers(client),
                     body;
                     status_exception=false)
    if resp.status != 200
        @warn "Registration failed" status=resp.status body=String(resp.body)
        return (success=false, assigned_seed=nothing)
    end
    result = JSON.parse(String(resp.body))
    seed = get(result, "assigned_seed", nothing)
    return (success=true, assigned_seed=seed)
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
function download_weights(client::SelfPlayClient, model::Symbol; expected_version::Union{Nothing, Int}=nothing)
    endpoint = model == :contact ? "contact" : "race"
    resp = HTTP.get("$(client.server_url)/api/weights/$endpoint",
                    auth_headers(client);
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
function flush_samples!(client::SelfPlayClient)
    isempty(client.upload_buffer) && return 0

    batch = samples_to_batch(client.upload_buffer)
    bytes = pack_samples(batch)

    headers = vcat(auth_headers(client),
                   ["Content-Type" => "application/msgpack"])
    resp = HTTP.post("$(client.server_url)/api/samples",
                     headers,
                     bytes;
                     status_exception=false)

    n_uploaded = length(client.upload_buffer)
    empty!(client.upload_buffer)

    if resp.status != 200
        @warn "Sample upload failed" status=resp.status n_samples=n_uploaded
        return 0
    end

    result = JSON.parse(String(resp.body))
    @debug "Uploaded samples" accepted=result["accepted"] buffer_size=result["buffer_size"]
    return result["accepted"]
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
