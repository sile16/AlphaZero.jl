#!/usr/bin/env julia
"""
Pre-bearoff race value-head ground-truth eval (two-sided wildbg-vs-wildbg MC).

Measures NN value-head accuracy in the pre-bearoff race band — the band where
v12 plateaued invisibly at wildbg parity because nothing measured ground truth.

Design (correct baseline; a one-sided analytic variance-reduction is the
documented follow-on):
- For each fixed pre-bearoff race position, roll dice to a POST-DICE decision
  node — the NN value head is trained on decision nodes and is OUT OF
  DISTRIBUTION at pre-dice chance nodes (see the "never use NN as pre-dice
  frontier evaluator" law). Evaluate the NN there (in-distribution).
- Ground truth = wildbg-vs-wildbg Monte-Carlo rollout FROM that decision node
  (mover plays wildbg's move, then both sides play wildbg to completion),
  averaged over R rollouts → the decision node's true money equity under a
  strong near-optimal reference policy. NEVER the NN as rollout policy (v11 drift).
- Compare NN value (mover-relative, ×reward_scale → points) vs rollout equity
  (mover-relative points): MSE / MAE / bias / corr, plus differential (within-
  position, across dice) signal.

Usage:
    julia --threads 28 --project scripts/race_ground_truth.jl <checkpoint> \\
        --width=128 --blocks=3 --num-workers=24 --rollouts=200 --num-positions=500 \\
        --wildbg-lib=/path/to/libwildbg.so
"""

using ArgParse

function parse_args_gt()
    s = ArgParseSettings(description="Pre-bearoff race value ground-truth eval", autofix_names=true)
    @add_arg_table! s begin
        "checkpoint"
            help = "Race model checkpoint file"
            arg_type = String
            required = true
        "--obs-type"
            arg_type = String
            default = "minimal_flat"
        "--num-positions"
            help = "How many pre-bearoff race positions to score (0=all available)"
            arg_type = Int
            default = 500
        "--rollouts"
            help = "wildbg-vs-wildbg rollouts per position (variance of the target)"
            arg_type = Int
            default = 200
        "--dice-per-position"
            help = "Distinct dice outcomes sampled per position (each = one decision node)"
            arg_type = Int
            default = 1
        "--width"
            arg_type = Int
            default = 128
        "--blocks"
            arg_type = Int
            default = 3
        "--num-workers"
            arg_type = Int
            default = 22
        "--wildbg-lib"
            arg_type = String
            default = ""
        "--positions-file"
            arg_type = String
            default = joinpath(dirname(@__DIR__), "eval_data", "race_eval_2000.jls")
        "--pre-bearoff-only"
            help = "Keep only positions NOT yet in the k=7 bearoff range (the pre-bearoff band)"
            action = :store_true
        "--seed"
            arg_type = Int
            default = 42
        "--out"
            help = "Optional .jls path to dump per-position (nn_val, rollout_eq) samples"
            arg_type = String
            default = ""
    end
    return ArgParse.parse_args(s)
end

const ARGS = parse_args_gt()

using AlphaZero
using AlphaZero: GI, Network, FluxLib
using AlphaZero.NetLib
import Flux
using Random
using Statistics
using Printf
using Serialization
using StaticArrays
using BackgammonNet

ENV["BACKGAMMON_OBS_TYPE"] = ARGS["obs_type"]
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
const gspec = GameSpec()
const NUM_ACTIONS = GI.num_actions(gspec)
const _state_dim = GI.state_dim(gspec)[1]
const ORACLE_CFG = AlphaZero.BackgammonInference.OracleConfig(
    _state_dim, NUM_ACTIONS, gspec; vectorize_state! = vectorize_state_into!)
const REWARD_SCALE = Float64(GI.reward_scale(gspec))

_is_bearoff(p0, p1) = BackgammonNet.BearoffK7.is_bearoff_position(p0, p1)

function find_wildbg_lib()
    isempty(ARGS["wildbg_lib"]) || return ARGS["wildbg_lib"]
    for c in (joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg.so"),
              joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg.dylib"))
        isfile(c) && return c
    end
    error("libwildbg not found. Pass --wildbg-lib=/path/to/libwildbg")
end

"""Roll out one game to completion with wildbg on BOTH sides; return white reward."""
@inline function rollout_to_end!(wb_agent, g::BackgammonGame, rng)
    while !BackgammonNet.game_terminated(g)
        if BackgammonNet.is_chance_node(g)
            BackgammonNet.sample_chance!(g, rng)
        else
            a = BackgammonNet.agent_move(wb_agent, g)
            BackgammonNet.apply_action!(g, a)
        end
    end
    return Float64(g.reward)   # white-relative
end

"""Score one (position, dice) decision node. Computes BOTH:
- VALUE: NN value (mover-relative RAW points) vs wildbg-rollout equity.
- POLICY: does the NN policy argmax match wildbg's move, and how much policy mass
  the NN puts on wildbg's move (a cheap policy-quality proxy — the value finding
  showed value isn't the pre-bearoff bottleneck, so policy is the next suspect).
Returns a NamedTuple, or nothing if the rolled state has no legal move.
Doubles caveat: at a doubles node this compares the FIRST action of the turn."""
function eval_decision_node(oracle, wb_agent, p0::UInt128, p1::UInt128, cp::Int8,
                            rollouts::Int, rng)
    g0 = BackgammonGame(p0, p1, SVector{2,Int8}(0, 0), Int8(0), cp, false, 0.0f0;
                        obs_type=:minimal_flat)
    BackgammonNet.sample_chance!(g0, rng)          # → post-dice decision node for cp
    BackgammonNet.game_terminated(g0) && return nothing
    BackgammonNet.is_chance_node(g0) && return nothing
    mover = Int(g0.current_player)
    # Use GI.available_actions (explicitly sorted to match the oracle's policy
    # index order) rather than raw legal_actions — guarantees pol[i] ↔ acts[i]
    # alignment for the policy metrics instead of relying on legal_actions
    # happening to be sorted (verified identical, but this makes it robust).
    acts = GI.available_actions(gspec, g0)
    isempty(acts) && return nothing

    # NN policy + value at the in-distribution decision node (single_oracle:
    # policy over legal actions, value mover-relative /reward_scale).
    pol, val = oracle(g0)
    nn_val = Float64(val) * REWARD_SCALE

    # Policy vs wildbg's move (reference).
    wb_move = BackgammonNet.agent_move(wb_agent, g0)
    wb_idx = findfirst(==(wb_move), acts)
    nn_move = acts[argmax(pol)]
    agree = (nn_move == wb_move)
    wb_mass = wb_idx === nothing ? NaN : Float64(pol[wb_idx])
    top1 = Float64(maximum(pol))

    # VALUE ground truth: wildbg-vs-wildbg rollout FROM g0 (wildbg plays its move
    # first) — so this ALSO equals the equity after wildbg's chosen move.
    total = 0.0
    for _ in 1:rollouts
        g = BackgammonNet.clone(g0)
        total += rollout_to_end!(wb_agent, g, rng)
    end
    we = total / rollouts
    mover_eq = mover == 0 ? we : -we

    # POLICY move-regret: equity the NN's policy-argmax move loses vs wildbg's.
    # gt already = equity after wildbg's move, so regret = mover_eq − eq(nn_move).
    # Non-doubles only (1 action = full turn → clean); 0 when the moves agree.
    is_doubles = g0.dice[1] == g0.dice[2]
    regret = 0.0
    regret_valid = !is_doubles
    if regret_valid && !agree
        tn = 0.0
        for _ in 1:rollouts
            g = BackgammonNet.clone(g0)
            BackgammonNet.apply_action!(g, nn_move)
            tn += rollout_to_end!(wb_agent, g, rng)
        end
        en = tn / rollouts
        eq_nn = mover == 0 ? en : -en
        regret = mover_eq - eq_nn
    end
    return (nn_val=nn_val, rollout_eq=mover_eq, agree=agree, wb_mass=wb_mass, top1=top1,
            regret=regret, regret_valid=regret_valid)
end

function main()
    Random.seed!(ARGS["seed"])
    positions_file = ARGS["positions_file"]
    isfile(positions_file) || error("Positions file not found: $positions_file")
    all_positions = deserialize(positions_file)
    println("Loaded $(length(all_positions)) race positions from $positions_file")

    if ARGS["pre_bearoff_only"]
        pre = [pd for pd in all_positions if !_is_bearoff(pd[1], pd[2])]
        println("Pre-bearoff filter: $(length(pre))/$(length(all_positions)) positions NOT yet in k=7 range")
        all_positions = pre
    end

    n_req = ARGS["num_positions"]
    positions = (n_req > 0 && n_req < length(all_positions)) ? all_positions[1:n_req] : all_positions
    n_pos = length(positions)
    rollouts = ARGS["rollouts"]
    dice_per = ARGS["dice_per_position"]
    num_workers = ARGS["num_workers"]

    println("Checkpoint: $(ARGS["checkpoint"])")
    println("Architecture: $(ARGS["width"])w×$(ARGS["blocks"])b | reward_scale=$REWARD_SCALE")
    println("Positions: $n_pos | dice/pos: $dice_per | rollouts/node: $rollouts | workers: $num_workers")

    # Network + value oracle (flux single-eval path — value only, no MCTS).
    network = FluxLib.FCResNetMultiHead(
        gspec, FluxLib.FCResNetMultiHeadHP(width=ARGS["width"], num_blocks=ARGS["blocks"]))
    FluxLib.load_weights(ARGS["checkpoint"], network)
    network = Flux.cpu(network)
    single_oracle, _ = AlphaZero.BackgammonInference.make_cpu_oracles(
        :flux, network, ORACLE_CFG; batch_size=1)

    # Wildbg per worker.
    wildbg_lib = find_wildbg_lib()
    lib_size = filesize(wildbg_lib)
    nets_variant = lib_size > 10_000_000 ? :large : :small
    if nets_variant == :large
        BackgammonNet.wildbg_set_lib_path!(large=wildbg_lib)
    else
        BackgammonNet.wildbg_set_lib_path!(small=wildbg_lib)
    end
    println("wildbg: $nets_variant ($(round(lib_size/1e6, digits=1))MB)")
    wb_agents = [begin
        wb = BackgammonNet.WildbgBackend(nets=nets_variant); BackgammonNet.open!(wb)
        BackgammonNet.BackendAgent(wb)
    end for _ in 1:num_workers]
    println("=" ^ 70); flush(stdout)

    n_jobs = n_pos * dice_per
    nn_vals = Vector{Float64}(undef, n_jobs)
    gt_vals = Vector{Float64}(undef, n_jobs)
    agree_v = fill(false, n_jobs)
    wbmass_v = fill(NaN, n_jobs)
    top1_v = fill(NaN, n_jobs)
    regret_v = fill(NaN, n_jobs)   # NaN = doubles node (excluded from regret)
    valid = fill(false, n_jobs)
    claimed = Threads.Atomic{Int}(0)
    done = Threads.Atomic{Int}(0)
    t0 = time()

    Threads.@threads :static for w in 1:num_workers
        wb = wb_agents[w]
        while true
            job = Threads.atomic_add!(claimed, 1) + 1
            job > n_jobs && break
            pos_idx = (job - 1) ÷ dice_per + 1
            p0, p1, cp = positions[pos_idx]
            # Deterministic per-job RNG (reproducible; distinct per dice sample).
            rng = Xoshiro(ARGS["seed"] * 1_000_003 + job)
            r = eval_decision_node(single_oracle, wb, UInt128(p0), UInt128(p1), Int8(cp), rollouts, rng)
            if r !== nothing
                nn_vals[job] = r.nn_val; gt_vals[job] = r.rollout_eq
                agree_v[job] = r.agree; wbmass_v[job] = r.wb_mass; top1_v[job] = r.top1
                regret_v[job] = r.regret_valid ? r.regret : NaN
                valid[job] = true
            end
            d = Threads.atomic_add!(done, 1) + 1
            if d % 200 == 0
                @printf("  %d/%d nodes  (%.1fs)\n", d, n_jobs, time() - t0); flush(stdout)
            end
        end
    end

    nn = nn_vals[valid]; gt = gt_vals[valid]
    n = length(nn)
    n >= 2 || error("Too few valid nodes ($n)")
    mse  = mean((nn .- gt) .^ 2)
    mae  = mean(abs.(nn .- gt))
    bias = mean(nn) - mean(gt)
    corr = n >= 3 ? cor(nn, gt) : NaN

    # Policy metrics (mover-move agreement with wildbg + policy mass on wildbg's move).
    agree = agree_v[valid]
    wbmass = wbmass_v[valid]; wbmass = wbmass[.!isnan.(wbmass)]
    top1 = top1_v[valid]
    agree_rate = mean(agree)
    mean_wbmass = isempty(wbmass) ? NaN : mean(wbmass)
    mean_top1 = mean(top1)

    println("=" ^ 70)
    println("Pre-bearoff race: NN vs wildbg-rollout ground truth")
    @printf("  nodes scored:   %d  (%.1fs, %.0f rollouts total)\n", n, time() - t0, n * rollouts)
    println("  -- VALUE HEAD --")
    @printf("  NN value  mean=%.3f std=%.3f | rollout mean=%.3f std=%.3f\n", mean(nn), std(nn), mean(gt), std(gt))
    @printf("  MSE=%.4f  MAE=%.4f  bias=%.4f  corr=%.4f\n", mse, mae, bias, corr)
    reg = regret_v[valid]; reg = reg[.!isnan.(reg)]   # non-doubles nodes only
    mean_regret = isempty(reg) ? NaN : mean(reg)
    # regret conditioned on disagreement (the equity cost when the NN differs).
    reg_dis = reg[reg .!= 0.0]
    mean_regret_dis = isempty(reg_dis) ? NaN : mean(reg_dis)
    println("  -- POLICY HEAD (argmax vs wildbg move; doubles = first action) --")
    @printf("  agree=%.1f%%  policy-mass-on-wildbg-move=%.3f  top1-prob=%.3f\n",
            agree_rate * 100, mean_wbmass, mean_top1)
    @printf("  move-regret (non-doubles, pts): mean=%.4f  mean|disagree=%.4f  (n=%d)\n",
            mean_regret, mean_regret_dis, length(reg))
    println("=" ^ 70)

    if !isempty(ARGS["out"])
        serialize(ARGS["out"], (nn=nn, gt=gt, rollouts=rollouts, checkpoint=ARGS["checkpoint"]))
        println("Saved per-node samples → $(ARGS["out"])")
    end
end

main()
