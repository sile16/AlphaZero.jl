#!/usr/bin/env julia
"""
verify_race_mcts.jl — does MCTS on top of the race net LOWER move-regret vs the raw policy?

For a subsample of exact-table-labeled race positions, compares two move choices against
the EXACT combined k7→n15 race table:
  - raw  : argmax of the net's policy head
  - mcts : the move selected by the production BatchedMCTS search (NN evaluator only —
           no exact-table leaf, so this isolates the SEARCH contribution)

Reports mean/median move-regret (exact points lost vs the table's best move) for each,
so we can see whether search improves the already-near-optimal raw policy.

Usage:
  julia --project scripts/verify_race_mcts.jl <checkpoint> <test.jls> \
        [--width=128] [--blocks=3] [--n=2000] [--iters=100] [--table-dir=...]
"""

using AlphaZero
using AlphaZero: GI, MCTS, Network, FluxLib, MctsParams, ConstSchedule, BatchedMCTS, GameLoop
using AlphaZero.BackgammonInference
import BackgammonNet
using BackgammonNet: CombinedBearoff, load_combined_bearoff, bearoff_turn_value_equity
using Serialization, Statistics, Printf, Random
import Flux

parse_arg(args, key; default=nothing) = begin
    for a in args
        startswith(a, "--$key=") && return split(a, "=", limit=2)[2]
    end
    default
end

const CKPT   = ARGS[1]
const TEST   = ARGS[2]
const WIDTH  = parse(Int, something(parse_arg(ARGS, "width"), "128"))
const BLOCKS = parse(Int, something(parse_arg(ARGS, "blocks"), "3"))
const NSUB   = parse(Int, something(parse_arg(ARGS, "n"), "2000"))
const ITERS  = parse(Int, something(parse_arg(ARGS, "iters"), "100"))
const N15_DIR = something(parse_arg(ARGS, "table-dir"),
    BackgammonNet.default_bearoff_n15_dir())
const K7_DIR = BackgammonNet.default_bearoff_k7_dir()

ENV["BACKGAMMON_OBS_TYPE"] = something(parse_arg(ARGS, "obs-type"), "min_plus_flat")
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
const gspec = GameSpec()
const NUM_ACTIONS = GI.num_actions(gspec)
const _state_dim = GI.state_dim(gspec)[1]
const ORACLE_CFG = BackgammonInference.OracleConfig(
    _state_dim, NUM_ACTIONS, gspec; vectorize_state! = vectorize_state_into!)

# Exact mover-relative value of playing first-action `a` from decision node `g`.
# Doubles-correct: recurses through the rest of the turn via bearoff_turn_value_equity.
function move_equity(t::CombinedBearoff, g, a, scratch)
    BackgammonNet.copy_state!(scratch, g)
    BackgammonNet.apply_action!(scratch, a)
    v, _ = bearoff_turn_value_equity(t, scratch, Int(g.current_player))
    return v
end

function main()
    d = deserialize(TEST)
    n = min(NSUB, length(d.states))
    println("Test positions: $n (of $(length(d.states)))   MCTS iters: $ITERS")
    table = load_combined_bearoff(; k7_dir=K7_DIR, n15_dir=N15_DIR)
    scratch = BackgammonNet.clone(d.states[1])

    net = FluxLib.FCResNetMultiHead(gspec, FluxLib.FCResNetMultiHeadHP(width=WIDTH, num_blocks=BLOCKS))
    FluxLib.load_weights(CKPT, net); net = Flux.cpu(net)
    single_oracle, batch_oracle = BackgammonInference.make_cpu_oracles(:flux, net, ORACLE_CFG; batch_size=64)

    mcts_params = MctsParams(num_iters_per_turn=ITERS, cpuct=1.5,
        temperature=ConstSchedule(0.0), dirichlet_noise_ϵ=0.0, dirichlet_noise_α=1.0)
    agent = GameLoop.MctsAgent(single_oracle, batch_oracle, mcts_params, 64, gspec)

    reg_raw = Float64[]; reg_mcts = Float64[]
    rng = Random.MersenneTwister(1234)
    t0 = time()
    for i in 1:n
        g = d.states[i]
        legal = sort(collect(BackgammonNet.legal_actions(g)))
        length(legal) == 0 && continue
        best = d.values[i]                       # exact best move value

        # raw net argmax
        P, _ = single_oracle(g)
        raw_move = length(legal) == 1 ? legal[1] :
            (length(P) == length(legal) ? legal[argmax(P)] : legal[1])
        push!(reg_raw, best - move_equity(table, g, raw_move, scratch))

        # production MCTS (NN evaluator only)
        env = GameEnv(BackgammonNet.clone(g), rng)
        player = GameLoop.create_player(agent; rng=rng)
        mcts_actions, mcts_policy, _ = GameLoop.select_action(agent, player, env)
        mcts_move = mcts_actions[argmax(mcts_policy)]   # greedy (temp 0)
        push!(reg_mcts, best - move_equity(table, g, mcts_move, scratch))

        if i % 500 == 0
            @printf("  %d/%d  (%.1fs)\n", i, n, time() - t0); flush(stdout)
        end
    end

    summ(v) = (mean(max.(v,0.0)), median(max.(v,0.0)),
               100*count(<(0.01), max.(v,0.0))/length(v))
    mr, md, opr = summ(reg_raw)
    mm, dm, opm = summ(reg_mcts)
    println("\n=== MOVE-REGRET (exact points lost vs table best) ===")
    @printf("  raw  net  : mean=%.5f  median=%.5f  optimal%%=%.2f\n", mr, md, opr)
    @printf("  MCTS(%3d) : mean=%.5f  median=%.5f  optimal%%=%.2f\n", ITERS, mm, dm, opm)
    @printf("  improvement (raw→mcts mean regret): %+.5f pts (%.1f%% lower)\n",
            mr - mm, mr > 0 ? 100*(mr-mm)/mr : 0.0)
    # Hard subset: positions where the raw net was suboptimal — where search should matter.
    hard = findall(>=(0.01), max.(reg_raw, 0.0))
    if !isempty(hard)
        hr = mean(max.(reg_raw[hard], 0.0)); hm = mean(max.(reg_mcts[hard], 0.0))
        @printf("\n  hard subset (raw regret ≥0.01): %d positions (%.1f%%)\n",
                length(hard), 100*length(hard)/length(reg_raw))
        @printf("    raw mean regret  = %.4f\n    mcts mean regret = %.4f  (%.1f%% lower)\n",
                hr, hm, hr > 0 ? 100*(hr-hm)/hr : 0.0)
    end
    @printf("\n%d positions in %.1fs\n", n, time() - t0)
end

main()
