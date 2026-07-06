#!/usr/bin/env julia
"""
Generate a fixed set of CONTACT (non-race) positions for the ground-truth eval.

Plays wildbg-vs-wildbg from the SHORT_GAME opening and samples pre-dice
positions where contact is still possible (`is_contact_position`, which implies
NOT bearoff). Stored as (p0, p1, current_player) tuples at CHANCE nodes — the
same pre-dice format scripts/race_ground_truth.jl expects (it rolls dice to a
decision node before scoring). Deduped.

Usage:
    julia --project scripts/generate_contact_positions.jl \\
        --num-positions=2000 --out=eval_data/contact_eval_2000.jls \\
        --wildbg-lib=/path/to/libwildbg.so
"""

using ArgParse

function parse_args_gen()
    s = ArgParseSettings(autofix_names=true)
    @add_arg_table! s begin
        "--num-positions"; arg_type = Int; default = 2000
        "--sample-prob";   arg_type = Float64; default = 0.20
        "--max-games";     arg_type = Int; default = 20000
        "--seed";          arg_type = Int; default = 7
        "--wildbg-lib";    arg_type = String; default = ""
        "--out";           arg_type = String;
            default = joinpath(dirname(@__DIR__), "eval_data", "contact_eval_2000.jls")
    end
    return ArgParse.parse_args(s)
end

const ARGS = parse_args_gen()

using AlphaZero
using AlphaZero: GI
using Random
using Serialization
using BackgammonNet

ENV["BACKGAMMON_OBS_TYPE"] = "minimal_flat"
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
const gspec = GameSpec()

function find_wildbg_lib()
    isempty(ARGS["wildbg_lib"]) || return ARGS["wildbg_lib"]
    for c in (joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg.so"),
              joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg.dylib"))
        isfile(c) && return c
    end
    error("libwildbg not found. Pass --wildbg-lib")
end

function main()
    rng = Xoshiro(ARGS["seed"])
    lib = find_wildbg_lib()
    variant = filesize(lib) > 10_000_000 ? :large : :small
    variant == :large ? BackgammonNet.wildbg_set_lib_path!(large=lib) :
                        BackgammonNet.wildbg_set_lib_path!(small=lib)
    wb = BackgammonNet.WildbgBackend(nets=variant); BackgammonNet.open!(wb)
    agent = BackgammonNet.BackendAgent(wb)

    target = ARGS["num_positions"]
    sp = ARGS["sample_prob"]
    seen = Set{Tuple{UInt128, UInt128, Int8}}()
    positions = Tuple{UInt128, UInt128, Int8}[]
    println("Generating $target contact positions (wildbg self-play from SHORT_GAME opening)...")

    gi = 0
    while length(positions) < target && gi < ARGS["max_games"]
        gi += 1
        g = BackgammonNet.clone(GI.init(gspec).game)   # fresh SHORT_GAME opening
        while !BackgammonNet.game_terminated(g) && length(positions) < target
            if BackgammonNet.is_chance_node(g)
                if BackgammonNet.is_contact_position(g) && rand(rng) < sp
                    key = (g.p0, g.p1, Int8(g.current_player))
                    if !(key in seen)
                        push!(seen, key); push!(positions, key)
                    end
                end
                BackgammonNet.sample_chance!(g, rng)
            else
                a = BackgammonNet.agent_move(agent, g)
                BackgammonNet.apply_action!(g, a)
            end
        end
        if gi % 500 == 0
            println("  games=$gi  positions=$(length(positions))"); flush(stdout)
        end
    end

    positions = positions[1:min(target, length(positions))]
    serialize(ARGS["out"], positions)
    println("Wrote $(length(positions)) unique contact positions → $(ARGS["out"])  (from $gi games)")
end

main()
