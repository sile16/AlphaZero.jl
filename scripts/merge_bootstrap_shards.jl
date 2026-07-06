#!/usr/bin/env julia
#=
Merge sharded contact-bootstrap files (produced by generate_contact_bootstrap.jl
run in parallel with distinct seeds) into a single NamedTuple file in the exact
format training_server.jl --bootstrap-file expects: (states, policies, values, equity).

Usage:
  julia --project scripts/merge_bootstrap_shards.jl \
      --glob='/path/shard_*.jls' --out=/path/contact_bootstrap_wildbg_300k.jls --max=300000
=#
using ArgParse, Serialization
using AlphaZero
import BackgammonNet

function parse_args_merge()
    s = ArgParseSettings(autofix_names=true)
    @add_arg_table! s begin
        "--dir";    arg_type = String; required = true   # directory of shard_*.jls files
        "--prefix"; arg_type = String; default = "shard_"
        "--out";    arg_type = String; required = true
        "--max";    arg_type = Int;    default = 300_000
    end
    return ArgParse.parse_args(s)
end

function main()
    a = parse_args_merge()
    files = sort([joinpath(a["dir"], f) for f in readdir(a["dir"])
                  if startswith(f, a["prefix"]) && endswith(f, ".jls")])
    isempty(files) && error("no shard files ($(a["prefix"])*.jls) in $(a["dir"])")
    states = BackgammonNet.BackgammonGame[]
    policies = Vector{Float32}[]
    values = Float32[]
    equities = Vector{Float32}[]
    for f in files
        d = deserialize(f)
        append!(states, d.states); append!(policies, d.policies)
        append!(values, d.values); append!(equities, d.equity)
        println("  + $(basename(f)): $(length(d.states)) samples  (running total $(length(states)))")
    end
    n = min(a["max"], length(states))
    out = (states=states[1:n], policies=policies[1:n], values=values[1:n], equity=equities[1:n])
    serialize(a["out"], out)
    println("Merged $(length(files)) shards → $n samples → $(a["out"])  (had $(length(states)) total)")
end
main()
