"""Corpus-level quality aggregation layered on authoritative artifact validators."""
module DataQuality

using Dates
using JSON

export QualityAccumulator, add_artifact!, quality_report, write_quality_report

mutable struct QualityAccumulator
    artifacts::Vector{Dict{String,Any}}
    training_fingerprints::Set{String}
    evaluation_fingerprints::Set{String}
    duplicate_training::Int
    duplicate_evaluation::Int
end

QualityAccumulator() = QualityAccumulator(Dict{String,Any}[], Set{String}(),
                                          Set{String}(), 0, 0)

function add_artifact!(acc::QualityAccumulator;
                       path::AbstractString, role::AbstractString,
                       fingerprints, strata::AbstractDict,
                       illegal_policy_entries::Integer=0,
                       nonfinite_entries::Integer=0,
                       metadata::AbstractDict=Dict{String,Any}())
    role in ("train", "eval") || throw(ArgumentError("role must be train or eval"))
    destination = role == "train" ? acc.training_fingerprints : acc.evaluation_fingerprints
    duplicates = 0
    for fingerprint in fingerprints
        value = String(fingerprint)
        value in destination && (duplicates += 1)
        push!(destination, value)
    end
    role == "train" ? (acc.duplicate_training += duplicates) :
                      (acc.duplicate_evaluation += duplicates)
    push!(acc.artifacts, Dict{String,Any}(
        "path" => String(path), "role" => String(role),
        "samples" => length(fingerprints), "duplicates" => duplicates,
        "strata" => Dict{String,Any}(String(k) => v for (k, v) in strata),
        "illegal_policy_entries" => Int(illegal_policy_entries),
        "nonfinite_entries" => Int(nonfinite_entries),
        "metadata" => Dict{String,Any}(String(k) => v for (k, v) in metadata),
    ))
    return acc
end

function quality_report(acc::QualityAccumulator)
    leakage = intersect(acc.training_fingerprints, acc.evaluation_fingerprints)
    illegal = sum(Int(a["illegal_policy_entries"]) for a in acc.artifacts; init=0)
    nonfinite = sum(Int(a["nonfinite_entries"]) for a in acc.artifacts; init=0)
    return Dict{String,Any}(
        "schema" => "alphazero_data_quality_v1",
        "created_at_utc" => string(Dates.now(Dates.UTC)),
        "ok" => isempty(leakage) && illegal == 0 && nonfinite == 0,
        "artifacts" => acc.artifacts,
        "unique_training_positions" => length(acc.training_fingerprints),
        "unique_evaluation_positions" => length(acc.evaluation_fingerprints),
        "duplicate_training_positions" => acc.duplicate_training,
        "duplicate_evaluation_positions" => acc.duplicate_evaluation,
        "train_eval_leakage" => length(leakage),
        "train_eval_leakage_examples" => first(sort!(collect(leakage)), min(20, length(leakage))),
        "illegal_policy_entries" => illegal,
        "nonfinite_entries" => nonfinite,
    )
end

function write_quality_report(base_path::AbstractString, report::AbstractDict)
    json_path = endswith(base_path, ".json") ? base_path : base_path * ".json"
    markdown_path = replace(json_path, r"\.json$" => ".md")
    mkpath(dirname(json_path))
    open(json_path, "w") do io
        JSON.print(io, report, 2)
    end
    open(markdown_path, "w") do io
        println(io, "# AlphaZero training-data quality")
        println(io)
        println(io, "- Status: ", report["ok"] ? "PASS" : "FAIL")
        println(io, "- Artifacts: ", length(report["artifacts"]))
        println(io, "- Unique training positions: ", report["unique_training_positions"])
        println(io, "- Unique evaluation positions: ", report["unique_evaluation_positions"])
        println(io, "- Train/eval leakage: ", report["train_eval_leakage"])
        println(io, "- Illegal policy entries: ", report["illegal_policy_entries"])
        println(io, "- Non-finite entries: ", report["nonfinite_entries"])
    end
    return (json=json_path, markdown=markdown_path)
end

end # module
