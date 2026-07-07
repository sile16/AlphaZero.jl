#!/usr/bin/env julia
"""
verify_race_supervised.jl — verify a supervised race net against EXACT-table targets.

Loads a trained race checkpoint and a held-out exact-table-labeled test set
(from generate_race_table_supervised.jl), forwards every position through the net,
and reports value accuracy (correlation / MSE / MAE, raw points) and policy top-1
agreement vs the exact best move. This is the trust check for the whole pipeline.

Usage:
  julia --project scripts/verify_race_supervised.jl <checkpoint> <test.jls> \
        [--width=128] [--blocks=3] [--obs-type=min_plus_flat] [--batch=4096]
"""

using AlphaZero
using AlphaZero: GI, Network, FluxLib
import BackgammonNet
using BackgammonNet: CombinedBearoff, load_combined_bearoff, bearoff_turn_value_equity
using Serialization, Statistics, Printf
import Flux

parse_arg(args, key; default=nothing) = begin
    for a in args
        startswith(a, "--$key=") && return split(a, "=", limit=2)[2]
    end
    default
end

const CKPT = ARGS[1]
const TEST = ARGS[2]
const WIDTH = parse(Int, something(parse_arg(ARGS, "width"), "128"))
const BLOCKS = parse(Int, something(parse_arg(ARGS, "blocks"), "3"))
const BATCH = parse(Int, something(parse_arg(ARGS, "batch"), "4096"))

const ONESIDED_DIR = something(parse_arg(ARGS, "table-dir"),
    joinpath(homedir(), "github", "BackgammonNet.jl", "data", "bearoff", "bearoff_n18"))
const K7_DIR = joinpath(homedir(), "github", "BackgammonNet.jl", "data", "bearoff", "bearoff_k7_twosided")

ENV["BACKGAMMON_OBS_TYPE"] = something(parse_arg(ARGS, "obs-type"), "min_plus_flat")
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
const gspec = GameSpec()
const NUM_ACTIONS = GI.num_actions(gspec)
const _state_dim = GI.state_dim(gspec)[1]
const ORACLE_CFG = AlphaZero.BackgammonInference.OracleConfig(
    _state_dim, NUM_ACTIONS, gspec; vectorize_state! = vectorize_state_into!)

# Exact mover-relative value of playing first-action `a` from decision node `g`.
# Uses bearoff_turn_value_equity, which recurses through the rest of the turn (doubles-correct)
# and only flips at true turn boundaries — the naive one-action + sign flip is WRONG for doubles
# and forced-pass positions.
function move_equity(t::CombinedBearoff, g, a, scratch)
    BackgammonNet.copy_state!(scratch, g)
    BackgammonNet.apply_action!(scratch, a)
    v, _ = bearoff_turn_value_equity(t, scratch, Int(g.current_player))
    return v
end

function main()
    println("Loading test set: $TEST")
    d = deserialize(TEST)
    n = length(d.states)
    println("  $n positions, obs=$(d.states[1].obs_type), state_dim=$_state_dim")

    table = (isdir(ONESIDED_DIR) && isdir(K7_DIR)) ? (print("Loading combined table for move-regret... ");
        t = load_combined_bearoff(; k7_dir=K7_DIR, onesided_dir=ONESIDED_DIR); println("done."); t) : nothing
    table === nothing && println("  (tables not found — skipping move-regret)")
    scratch = BackgammonNet.clone(d.states[1])
    regret = Float64[]

    println("Loading net: $CKPT  ($(WIDTH)w×$(BLOCKS)b)")
    net = FluxLib.FCResNetMultiHead(gspec, FluxLib.FCResNetMultiHeadHP(width=WIDTH, num_blocks=BLOCKS))
    FluxLib.load_weights(CKPT, net)
    net = Flux.cpu(net)
    _, value_batch_oracle = AlphaZero.BackgammonInference.make_cpu_oracles(
        :flux, net, ORACLE_CFG; batch_size=BATCH)

    rs = Float64(GI.reward_scale(gspec))   # raw points ↔ NN scale
    pred = Vector{Float64}(undef, n)
    pol_top1 = 0
    pol_total = 0

    t0 = time()
    for lo in 1:BATCH:n
        hi = min(lo + BATCH - 1, n)
        games = d.states[lo:hi]
        out = value_batch_oracle(games)
        for (k, i) in enumerate(lo:hi)
            pol_i, val_i = out[k][1], out[k][2]
            pred[i] = Float64(val_i) * rs            # NN scale → raw points
            # policy top-1 vs exact best move. Oracle returns policy over LEGAL
            # actions (ascending id order); align to sorted legal list.
            exact_best = argmax(d.policies[i])
            legal = sort(collect(BackgammonNet.legal_actions(games[k])))
            nn_best = nothing
            if length(legal) == 1
                nn_best = legal[1]
            elseif length(pol_i) == length(legal)
                nn_best = legal[argmax(pol_i)]
            end
            if nn_best !== nothing
                pol_top1 += (nn_best == exact_best)
                pol_total += 1
                if table !== nothing
                    # regret = exact best value − exact value of the move the net picked (≥0)
                    push!(regret, d.values[i] - move_equity(table, games[k], nn_best, scratch))
                end
            end
        end
    end
    dt = time() - t0

    exact = Float64.(d.values)
    err = pred .- exact
    mse = mean(err .^ 2)
    mae = mean(abs.(err))
    corr = cor(pred, exact)
    bias = mean(err)

    # Split by doubles vs non-doubles (targets for doubles positions may be affected
    # by multi-part-turn handling — measure them separately).
    isdbl = [Bool(d.states[i].dice[1] == d.states[i].dice[2]) for i in 1:n]
    nd = .!isdbl
    @printf("\n=== VALUE by dice type ===\n")
    @printf("  non-doubles (%d): corr=%.5f  MAE=%.5f\n", count(nd), cor(pred[nd], exact[nd]), mean(abs.(err[nd])))
    @printf("  doubles     (%d): corr=%.5f  MAE=%.5f\n", count(isdbl), cor(pred[isdbl], exact[isdbl]), mean(abs.(err[isdbl])))

    println("\n=== VALUE (raw points, exact ∈ [$(round(minimum(exact),digits=2)),$(round(maximum(exact),digits=2))]) ===")
    @printf("  correlation : %.5f\n", corr)
    @printf("  MSE         : %.5f\n", mse)
    @printf("  MAE         : %.5f\n", mae)
    @printf("  bias (mean) : %+.5f\n", bias)
    println("\n=== POLICY (raw net argmax) ===")
    @printf("  top-1 agreement vs exact best move : %.2f%% (%d/%d)\n",
            100 * pol_top1 / pol_total, pol_top1, pol_total)
    if !isempty(regret)
        clamp0 = max.(regret, 0.0)   # numerical: regret is ≥0 by construction
        @printf("  move-regret (exact pts lost): mean=%.5f  median=%.5f  p95=%.4f  max=%.3f\n",
                mean(clamp0), median(clamp0), quantile(clamp0, 0.95), maximum(clamp0))
        @printf("  optimal-move rate (regret<0.01 pts): %.2f%%\n",
                100 * count(<(0.01), clamp0) / length(clamp0))
    end
    @printf("\nForwarded %d positions in %.1fs (%.0f/s)\n", n, dt, n / dt)
end

main()
