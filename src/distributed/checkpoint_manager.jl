"""Crash-safe, checksummed checkpoint bundles."""
module CheckpointManager

using Dates
using JSON
using SHA
using UUIDs

export CHECKPOINT_SCHEMA_VERSION, write_checkpoint_bundle!, validate_checkpoint_bundle,
       latest_valid_checkpoint, checkpoint_manifest

const CHECKPOINT_SCHEMA_VERSION = 1

_bundle_name(iteration::Integer) = "bundle_iter_$(lpad(iteration, 8, '0'))"

function _file_sha256(path::AbstractString)
    open(path, "r") do io
        return bytes2hex(sha256(io))
    end
end

function _write_json_atomic(path::AbstractString, value)
    mkpath(dirname(path))
    temporary = path * ".tmp-" * string(uuid4())
    try
        open(temporary, "w") do io
            JSON.print(io, value)
            flush(io)
        end
        mv(temporary, path; force=true)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
    return path
end

"""Read a checkpoint manifest without validating its payload files."""
checkpoint_manifest(path::AbstractString) =
    JSON.parsefile(joinpath(path, "manifest.json"))

"""Validate schema, required files, sizes, and SHA-256 hashes."""
function validate_checkpoint_bundle(path::AbstractString;
                                    required_files::AbstractVector{<:AbstractString}=String[])
    isdir(path) || throw(ArgumentError("checkpoint bundle does not exist: $path"))
    manifest_path = joinpath(path, "manifest.json")
    isfile(manifest_path) || throw(ArgumentError("checkpoint manifest missing: $manifest_path"))
    manifest = JSON.parsefile(manifest_path)
    Int(get(manifest, "schema_version", 0)) == CHECKPOINT_SCHEMA_VERSION ||
        throw(ArgumentError("unsupported checkpoint schema $(get(manifest, "schema_version", 0))"))
    files = get(manifest, "files", Dict{String,Any}())
    for required in required_files
        haskey(files, required) || throw(ArgumentError(
            "checkpoint is missing required file record: $required"))
    end
    for (name, record) in files
        basename(name) == name || throw(ArgumentError("unsafe checkpoint filename: $name"))
        file_path = joinpath(path, name)
        isfile(file_path) || throw(ArgumentError("checkpoint payload missing: $name"))
        expected_size = Int(record["bytes"])
        filesize(file_path) == expected_size || throw(ArgumentError(
            "checkpoint payload size mismatch for $name"))
        actual_hash = _file_sha256(file_path)
        actual_hash == String(record["sha256"]) || throw(ArgumentError(
            "checkpoint payload checksum mismatch for $name"))
    end
    return manifest
end

"""
    write_checkpoint_bundle!(root, iteration, writers; metadata=Dict())

Each value in `writers` is called with a destination path inside a temporary
directory. The directory is checksum-validated and then renamed into place;
`latest.json` is updated only after the complete bundle is visible.
"""
function write_checkpoint_bundle!(root::AbstractString, iteration::Integer,
                                  writers::AbstractDict{<:AbstractString};
                                  metadata::AbstractDict=Dict{String,Any}())
    iteration >= 0 || throw(ArgumentError("checkpoint iteration must be non-negative"))
    mkpath(root)
    final_path = joinpath(root, _bundle_name(iteration))
    if ispath(final_path)
        validate_checkpoint_bundle(final_path)
        return final_path
    end
    temporary = joinpath(root, ".tmp-$(_bundle_name(iteration))-$(uuid4())")
    mkdir(temporary)
    try
        records = Dict{String,Any}()
        for name in sort!(String.(collect(keys(writers))))
            basename(name) == name || throw(ArgumentError("checkpoint writer name must be a basename: $name"))
            destination = joinpath(temporary, name)
            writers[name](destination)
            isfile(destination) || error("checkpoint writer did not create $name")
            records[name] = Dict(
                "bytes" => filesize(destination),
                "sha256" => _file_sha256(destination),
            )
        end
        manifest = Dict{String,Any}(
            "schema_version" => CHECKPOINT_SCHEMA_VERSION,
            "iteration" => Int(iteration),
            "created_at_utc" => string(Dates.now(Dates.UTC)),
            "metadata" => Dict{String,Any}(String(k) => v for (k, v) in metadata),
            "files" => records,
        )
        open(joinpath(temporary, "manifest.json"), "w") do io
            JSON.print(io, manifest)
            flush(io)
        end
        validate_checkpoint_bundle(temporary)
        mv(temporary, final_path)
        _write_json_atomic(joinpath(root, "latest.json"), Dict(
            "bundle" => basename(final_path), "iteration" => Int(iteration)))
        return final_path
    catch
        isdir(temporary) && rm(temporary; recursive=true, force=true)
        rethrow()
    end
end

"""Return the newest valid bundle, falling back past incomplete/corrupt ones."""
function latest_valid_checkpoint(root_or_bundle::AbstractString;
                                 required_files::AbstractVector{<:AbstractString}=String[])
    if isfile(joinpath(root_or_bundle, "manifest.json"))
        validate_checkpoint_bundle(root_or_bundle; required_files)
        return abspath(root_or_bundle)
    end
    isdir(root_or_bundle) || return nothing
    candidates = Tuple{Int,String}[]
    for name in readdir(root_or_bundle)
        match_result = match(r"^bundle_iter_(\d+)$", name)
        match_result === nothing && continue
        push!(candidates, (parse(Int, match_result.captures[1]),
                           joinpath(root_or_bundle, name)))
    end
    sort!(candidates; by=first, rev=true)
    for (_, candidate) in candidates
        try
            validate_checkpoint_bundle(candidate; required_files)
            return abspath(candidate)
        catch
            # A prior process may have died mid-write or storage may be corrupt.
            # Continue to the newest earlier complete bundle.
        end
    end
    return nothing
end

end # module
