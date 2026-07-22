"""Immutable evaluation-suite manifests and grouped result summaries."""
module EvalManifest

using Dates
using JSON
using SHA

export build_eval_manifest, write_eval_manifest, validate_eval_manifest,
       evaluation_leakage, grouped_eval_summary

_sha256_file(path) = open(io -> bytes2hex(sha256(io)), path, "r")

function build_eval_manifest(artifact_path::AbstractString, positions;
                             suite::AbstractString,
                             contract_fingerprint::AbstractString,
                             classify::Function,
                             fingerprint::Function)
    strata = Dict{String,Int}()
    fingerprints = String[]
    for position in positions
        stratum = String(classify(position))
        strata[stratum] = get(strata, stratum, 0) + 1
        push!(fingerprints, String(fingerprint(position)))
    end
    return Dict{String,Any}(
        "schema" => "alphazero_eval_manifest_v1",
        "suite" => String(suite),
        "created_at_utc" => string(Dates.now(Dates.UTC)),
        "artifact" => basename(artifact_path),
        "artifact_sha256" => _sha256_file(artifact_path),
        "contract_fingerprint" => String(contract_fingerprint),
        "num_positions" => length(positions),
        "strata" => strata,
        "position_set_sha256" => bytes2hex(sha256(codeunits(join(sort!(fingerprints), "\n")))),
        "unique_positions" => length(unique(fingerprints)),
    )
end

function write_eval_manifest(path::AbstractString, manifest::AbstractDict)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, manifest, 2)
    end
    return path
end

function validate_eval_manifest(manifest_path::AbstractString,
                                artifact_path::AbstractString, positions;
                                contract_fingerprint::AbstractString,
                                fingerprint::Function)
    manifest = JSON.parsefile(manifest_path)
    get(manifest, "schema", "") == "alphazero_eval_manifest_v1" ||
        throw(ArgumentError("unsupported evaluation manifest schema"))
    String(manifest["contract_fingerprint"]) == contract_fingerprint ||
        throw(ArgumentError("evaluation manifest contract fingerprint mismatch"))
    String(manifest["artifact_sha256"]) == _sha256_file(artifact_path) ||
        throw(ArgumentError("evaluation artifact checksum mismatch"))
    Int(manifest["num_positions"]) == length(positions) ||
        throw(ArgumentError("evaluation position count mismatch"))
    fingerprints = sort!(String[String(fingerprint(position)) for position in positions])
    set_hash = bytes2hex(sha256(codeunits(join(fingerprints, "\n"))))
    String(manifest["position_set_sha256"]) == set_hash ||
        throw(ArgumentError("evaluation position-set fingerprint mismatch"))
    return manifest
end

evaluation_leakage(training_fingerprints, evaluation_fingerprints) =
    intersect(Set(String.(training_fingerprints)), Set(String.(evaluation_fingerprints)))

function grouped_eval_summary(rewards::AbstractVector{<:Real}, strata::AbstractVector)
    length(rewards) == length(strata) || throw(ArgumentError("reward/stratum length mismatch"))
    grouped = Dict{String,Vector{Float64}}()
    for (reward, stratum) in zip(rewards, strata)
        push!(get!(grouped, String(stratum), Float64[]), Float64(reward))
    end
    return Dict(name => Dict(
        "n" => length(values),
        "equity" => sum(values) / length(values),
        "win_rate" => count(>(0), values) / length(values),
    ) for (name, values) in grouped)
end

end # module
