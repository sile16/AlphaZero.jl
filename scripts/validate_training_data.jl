#!/usr/bin/env julia

using ArgParse
using JSON
using SHA
import BackgammonNet

include(joinpath(@__DIR__, "..", "src", "distributed", "data_quality.jl"))
using .DataQuality

settings = ArgParseSettings()
@add_arg_table! settings begin
    "--train"
        nargs = '+'
        default = String[]
    "--eval"
        nargs = '+'
        default = String[]
    "--output"
        default = joinpath("sessions", "validation", "data_quality_report.json")
end
args = parse_args(settings)
isempty(args["train"]) && isempty(args["eval"]) && error("provide --train and/or --eval artifacts")

fingerprint(game) = bytes2hex(sha256(codeunits(repr(BackgammonNet.game_state_fingerprint(game)))))
accumulator = QualityAccumulator()

git_revision(path) = try
    strip(read(`git -C $path rev-parse HEAD`, String))
catch
    "unknown"
end

git_dirty(path) = try
    !isempty(strip(read(`git -C $path status --porcelain`, String)))
catch
    nothing
end

function inspect_artifact!(path::String, role::String)
    artifact = BackgammonNet.load_training_artifact(path) # authoritative BGN validation
    strata = Dict("contact" => 0, "race" => 0, "chance" => 0, "cube" => length(artifact.cube_states))
    fingerprints = String[]
    illegal = 0
    for (index, game) in enumerate(artifact.states)
        push!(fingerprints, fingerprint(game))
        contact = BackgammonNet.is_contact_position(game)
        strata[contact ? "contact" : "race"] += 1
        BackgammonNet.is_chance_node(game) && (strata["chance"] += 1)
        legal = Set(Int.(BackgammonNet.legal_actions(game)))
        for action in artifact.data.checker_action_ids[index]
            Int(action) in legal || (illegal += 1)
        end
    end
    metadata = artifact.data.metadata isa AbstractDict ? artifact.data.metadata : Dict{String,Any}()
    add_artifact!(accumulator; path=abspath(path), role,
        fingerprints, strata, illegal_policy_entries=illegal,
        metadata=Dict(
            "artifact_kind" => get(metadata, "artifact_kind", "unknown"),
            "block_id" => get(metadata, "block_id", "unknown"),
            "observation_encoding" => artifact.obs_encoding,
        ))
    println("Validated $(length(fingerprints)) checker samples: $path")
end

for path in args["train"]
    inspect_artifact!(path, "train")
end
for path in args["eval"]
    inspect_artifact!(path, "eval")
end

report = quality_report(accumulator)
backgammonnet_root = dirname(dirname(pathof(BackgammonNet)))
report["runtime"] = Dict(
    "julia_version" => string(VERSION),
    "backgammonnet_version" => string(Base.pkgversion(BackgammonNet)),
    "backgammonnet_commit" => git_revision(backgammonnet_root),
    "backgammonnet_dirty" => git_dirty(backgammonnet_root),
    "value_head_contract" => BackgammonNet.VALUE_HEAD_CONTRACT,
    "value_head_order" => String.(BackgammonNet.VALUE_HEAD_ORDER),
)
paths = write_quality_report(abspath(args["output"]), report)
println("Quality report: $(paths.json) and $(paths.markdown)")
report["ok"] || exit(2)
