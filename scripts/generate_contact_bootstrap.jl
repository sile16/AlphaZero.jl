#!/usr/bin/env julia
#=
Generate a wildbg-vs-wildbg CONTACT bootstrap: full AlphaZero training samples
(state, one-hot wildbg-move policy, game-outcome value + 5-head equity) in the
raw-columnar format that `training_server.jl --bootstrap-file` loads. Imitating
wildbg brings the contact model to ~parity FAST, so self-play + the exact-race-
frontier curriculum + deep MCTS can then EXCEED it (the actual goal — cold-start
is too slow to reach the band where the curriculum's effect is testable).

Convention MATCHES convert_trace_to_samples exactly (a mismatch trains garbage):
  value  = wp ? final_reward : -final_reward           (side-to-move, RAW points ±1/±2/±3)
  equity = equity_vector_from_outcome(outcome, wp)     (5-head joint cumulative)
  policy = one-hot at the wildbg action code           (action code IS the 1-based policy index)
Only CONTACT decision nodes are recorded (race/bearoff are covered by the table +
race net). Output NamedTuple: (states, policies, values, equity).

Example:
  julia --threads 8 --project scripts/generate_contact_bootstrap.jl \\
     --target-samples=300000 --sample-prob=0.3 \\
     --out=/homeshare/projects/AlphaZero.jl/eval_data/contact_bootstrap_wildbg.jls
=#
using ArgParse, Serialization, Random, StaticArrays

function parse_args_gen()
    s = ArgParseSettings(autofix_names=true)
    @add_arg_table! s begin
        "--out";            arg_type = String;  default = "/homeshare/projects/AlphaZero.jl/eval_data/contact_bootstrap_wildbg.jls"
        "--target-samples"; arg_type = Int;     default = 300_000
        "--max-games";      arg_type = Int;     default = 2_000_000
        "--sample-prob";    arg_type = Float64; default = 0.30
        "--policy-temp";    arg_type = Float64; default = 0.10  # softmax temp (equity units) over wildbg per-move values
        "--wildbg-lib";     arg_type = String;  default = ""
        "--seed";           arg_type = Int;     default = 1
    end
    return ArgParse.parse_args(s)
end
const ARGS_G = parse_args_gen()

using AlphaZero
using AlphaZero: GI, equity_vector_from_outcome
import BackgammonNet
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
const gspec = GameSpec()
const NUM_ACTIONS = GI.num_actions(gspec)

# Soft policy target from wildbg's PER-MOVE evaluation (not one-hot). Mirrors
# _best_action_by_evaluation: evaluate each successor, normalize to the mover's
# perspective (negate when the turn completes), softmax over the move values. Returns
# (full_policy_vector, wildbg's chosen move = argmax). This gives the policy head the
# relative value of near-tied moves instead of an overconfident one-hot.
function soft_policy_and_move(wb, g, acts, T::Float64)
    pol = zeros(Float32, NUM_ACTIONS)
    if length(acts) == 1
        (1 <= acts[1] <= NUM_ACTIONS) && (pol[acts[1]] = 1.0f0)
        return pol, acts[1]
    end
    posts = Vector{BackgammonNet.BackgammonGame}(undef, length(acts))
    g2 = BackgammonNet.clone(g)
    for (j, a) in enumerate(acts)
        BackgammonNet.copy_state!(g2, g); BackgammonNet.apply_action!(g2, a)
        posts[j] = BackgammonNet.clone(g2)
    end
    eqs = BackgammonNet.evaluate_batch(wb, posts)   # opponent/own equity of each successor
    vals = Float64[posts[j].current_player != g.current_player ? -eqs[j] : eqs[j] for j in eachindex(acts)]
    m = maximum(vals)
    p = exp.((vals .- m) ./ T); p ./= sum(p)
    @inbounds for (j, a) in enumerate(acts)
        (1 <= a <= NUM_ACTIONS) && (pol[a] = Float32(p[j]))
    end
    return pol, acts[argmax(vals)]
end

function find_wildbg_lib()
    isempty(ARGS_G["wildbg_lib"]) || return ARGS_G["wildbg_lib"]
    for c in (joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg.so"),
              joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg.dylib"))
        isfile(c) && return c
    end
    error("libwildbg not found; pass --wildbg-lib")
end

function main()
    lib = find_wildbg_lib()
    variant = filesize(lib) > 16_000_000 ? :large : :small
    variant == :large ? BackgammonNet.wildbg_set_lib_path!(large=lib) :
                        BackgammonNet.wildbg_set_lib_path!(small=lib)
    wb = BackgammonNet.WildbgBackend(nets=variant); BackgammonNet.open!(wb)
    rng = MersenneTwister(ARGS_G["seed"])
    sp = ARGS_G["sample_prob"]; target = ARGS_G["target_samples"]; T = ARGS_G["policy_temp"]

    states   = BackgammonNet.BackgammonGame[]
    policies = Vector{Float32}[]
    values   = Float32[]
    equities = Vector{Float32}[]

    println("Generating CONTACT bootstrap (wildbg $variant self-play, target $target samples)...")
    flush(stdout)
    gi = 0
    while length(states) < target && gi < ARGS_G["max_games"]
        gi += 1
        g = BackgammonNet.clone(GI.init(gspec).game)   # fresh SHORT_GAME opening
        gstates = BackgammonNet.BackgammonGame[]; gpolicies = Vector{Float32}[]
        while !BackgammonNet.game_terminated(g)
            if BackgammonNet.is_chance_node(g)
                BackgammonNet.sample_chance!(g, rng)
            else
                acts = BackgammonNet.legal_actions(g)
                isempty(acts) && break   # no legal move (shouldn't happen at a decision node)
                # Choose the move by wildbg's per-move evaluation (evaluate_batch — no
                # native-move matching, which fails on rare positions). This keeps PLAY
                # consistent with the recorded soft policy target.
                pol, a = soft_policy_and_move(wb, g, acts, T)
                if BackgammonNet.is_contact_position(g) && rand(rng) < sp
                    push!(gstates, BackgammonNet.clone(g)); push!(gpolicies, pol)
                end
                BackgammonNet.apply_action!(g, a)
            end
        end
        isempty(gstates) && continue
        final_reward = Float32(g.reward)              # white-relative ±1/±2/±3
        outcome = GI.game_outcome(GameEnv(g, rng))    # GI.GameOutcome (winner + win type)
        for (st, pol) in zip(gstates, gpolicies)
            wp = GI.white_playing(gspec, st)
            val = wp ? final_reward : -final_reward
            eq = equity_vector_from_outcome(outcome, wp)
            push!(states, st); push!(policies, pol); push!(values, Float32(val)); push!(equities, eq)
        end
        if gi % 500 == 0
            println("  games=$gi  samples=$(length(states))"); flush(stdout)
        end
    end

    n = min(target, length(states))
    out = (states=states[1:n], policies=policies[1:n], values=values[1:n], equity=equities[1:n])
    serialize(ARGS_G["out"], out)
    println("Wrote $n CONTACT bootstrap samples → $(ARGS_G["out"])  (from $gi games)")
end
main()
