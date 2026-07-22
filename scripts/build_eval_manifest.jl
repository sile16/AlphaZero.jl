#!/usr/bin/env julia

using ArgParse
using Serialization
using SHA
using StaticArrays
using AlphaZero
import BackgammonNet

include(joinpath(@__DIR__, "..", "src", "distributed", "protocol.jl"))
include(joinpath(@__DIR__, "..", "src", "distributed", "eval_manifest.jl"))
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
using .EvalManifest

settings = ArgParseSettings()
@add_arg_table! settings begin
    "--input"
        required = true
    "--output"
        default = ""
    "--suite"
        default = "fixed-eval-v1"
end
args = parse_args(settings)
input = abspath(args["input"])
output = isempty(args["output"]) ? input * ".manifest.json" : abspath(args["output"])
positions = Serialization.deserialize(input)
gspec = GameSpec()
contract = contract_fingerprint(backgammon_ml_contract(gspec))

function as_game(position)
    position isa BackgammonNet.BackgammonGame && return position
    p0, p1, cp = position
    return backgammon_game(p0, p1, SVector{2,Int8}(0, 0), Int8(0), cp,
                           false, 0.0f0; observation_type=:minimal_flat)
end
classify(position) = BackgammonNet.is_contact_position(as_game(position)) ? "contact" : "race"
fingerprint(position) = bytes2hex(sha256(codeunits(repr(BackgammonNet.game_state_fingerprint(as_game(position))))))

manifest = build_eval_manifest(input, positions; suite=args["suite"],
    contract_fingerprint=contract, classify, fingerprint)
write_eval_manifest(output, manifest)
println("Wrote $(manifest["num_positions"])-position manifest: $output")
