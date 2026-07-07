#!/usr/bin/env julia
"""
analyze_pr.jl — Measure our AlphaZero net's playing strength with the
INDUSTRY-STANDARD backgammon metrics: Error Rate (ER / mEMG) and PR
(Performance Rating), using GNU Backgammon (gnubg) as the reference evaluator.

WHY (vs eval_vs_wildbg / eval_vs_gnubg): win% and equity are *match-outcome*
metrics. The community-standard *strength* metric is per-decision quality: how
much equity a player throws away per move relative to a strong reference. PR is
what XG and GNUbg report and what players quote (world-class bots ~2-3 PR,
strong humans ~4-6 PR). This script reports PR/ER alongside the usual numbers.

METHOD (standard checker-play error-rate analysis):
  1. Our dual-model AZ agent plays SELF-PLAY games at MCTS-N (batched MCTS,
     contact + race nets + bearoff routing — same machinery as eval_vs_wildbg).
     Every checker decision (both sides) is a decision by OUR net, so all are
     rated. We record, at each decision, the pre-move position + the action id
     our net chose. Chance nodes, forced single-move positions, and cube
     decisions are skipped (only UNFORCED checker decisions count).
  2. For each recorded decision we evaluate EVERY legal action with the gnubg
     reference and convert to a scalar cubeless money equity, then:
         best_eq = max equity over all legal moves
         our_eq  = equity of our chosen action
         error   = max(best_eq - our_eq, 0)      (equity/points lost; clamp noise)

     HOW each move's equity is computed (IMPORTANT — see WHY below): we use the
     algorithm of BackgammonNet's DEFAULT `evaluate_actions` — for each legal
     action, clone+apply it, `evaluate_probs(gnubg, resulting_position)`, flip
     with `flip_equity_perspective` if the move handed the roll to the opponent,
     then `compute_cubeless_equity(position, probs)` (the exact conversion
     `_rule_aware_equity_from_probs` uses). This is a gnubg ply-N evaluation of
     each resulting position (= standard move-equity analysis).

     WHY NOT `evaluate_actions(gnubg, position)` directly: gnubg's backend
     OVERRIDES `evaluate_actions` with a native move-list path whose
     (action_id, probs) pairing is only reliable for current_player==1; for
     player 0 the pairing/ordering is scrambled by a board-reconstruction bug in
     the gnubg C bridge (verified: it ranks gnubg's own best move last). The
     per-resulting-position method above avoids that override, is symmetric
     between players, and for player 1 reproduces the native list EXACTLY. Since
     we iterate the position's own legal actions, our chosen action is always
     present — no unmatched skips, and doubles half-moves are handled naturally
     (intermediate-position eval; both sides graded the same way).
  3. Aggregate over N = number of unforced checker decisions:
         mean_error = sum(errors) / N            [cubeless money-equity points]

FORMULAS + UNITS (documented, print in output):
  ER (mEMG)  = 1000 * mean_error
               "milli-EMG": milli-equity (thousandths of a point) lost per
               unforced decision. Equivalent-to-money-game normalized error rate.
  PR         = 500  * mean_error   ==  ER / 2
               XG/GNUbg Performance Rating convention: PR is the mEMG error rate
               halved (equivalently 500 x mean equity lost per unforced move).
               Sanity anchors: world-class bots ~2-3, strong humans ~4-6,
               intermediate ~8-12, beginner >15.
  (These are the widely-cited XG conventions. Our equity is *cubeless money*
   equity in points; XG's reference uses rollouts/2-ply + cubeful/match EMG, so
   absolute PR here is APPROXIMATE, but it is internally consistent for
   comparing our own nets against each other under a fixed gnubg reference ply.)

CALIBRATION FLOOR: because our reference is gnubg ply-N applied per resulting
  position (not the full gnubg engine's move filter), an Option-A-optimal player
  scores PR 0, but the actual gnubg engine (its own best_move) scores a small
  nonzero PR under this same reference — the reference's noise floor. We measure
  it by running a few gnubg self-play games through the identical pipeline and
  report it as "reference floor". Interpret our net's PR relative to that floor.

REFERENCE:
  BackgammonNet.GnubgCLibBackend(ply=N). ply=1 is the reference (ply=0 is
  faster but a weaker reference). ply>=2 DEADLOCKS in gnubg's internal 2-ply
  thread pool (see eval_vs_gnubg.jl) — do NOT use ply>=2 here.

  NOTE: the gnubg C bridge is a single shared runtime guarded by an internal
  global lock; per-worker backend instances all resolve to it and gnubg calls
  serialize. Parallel workers still help the AZ self-play (MCTS) side.

SCOPE: checker-play ER/PR only. Cube (doubling / take-pass) error rate is a
  clearly-marked stub below — it lands with the cubeful architecture.

Usage:
    julia --threads 16 --project scripts/analyze_pr.jl <contact_checkpoint> [options]

    # Smoke test (few games, confirm end-to-end):
    julia --threads 16 --project scripts/analyze_pr.jl \\
        sessions/contact-flywheel/checkpoints/contact_iter_140.data \\
        --obs-type min_plus_flat --width 256 --blocks 5 --race-width 128 --race-blocks 3 \\
        --num-games 3 --mcts-iters 800 --num-workers 6 --gnubg-ply 1

    # Real run (~1000+ unforced decisions):
    julia --threads 16 --project scripts/analyze_pr.jl \\
        sessions/contact-flywheel/checkpoints/contact_iter_140.data \\
        --obs-type min_plus_flat --width 256 --blocks 5 --race-width 128 --race-blocks 3 \\
        --num-games 40 --mcts-iters 800 --num-workers 12 --gnubg-ply 1

Options:
    --obs-type=min_plus_flat  Observation type
    --num-games=40            Self-play games to run (each yields many decisions)
    --width=256               Contact network width
    --blocks=5                Contact network blocks
    --race-width=128          Race network width
    --race-blocks=3           Race network blocks
    --num-workers=12          CPU worker threads (self-play + reference eval)
    --mcts-iters=800          MCTS iterations per move
    --gnubg-ply=1             gnubg reference ply (1 = reference; ply>=2 deadlocks)
    --max-decisions=0         Cap analyzed decisions (0 = no cap)
    --seed=1                  Base RNG seed
    --race-checkpoint=PATH    Explicit race checkpoint (else auto-detect race_latest.data)
"""

using ArgParse

function parse_pr_args()
    s = ArgParseSettings(description="Measure AZ checker-play ER/PR vs gnubg", autofix_names=true)
    @add_arg_table! s begin
        "checkpoint"
            help = "Contact/main checkpoint file"
            arg_type = String
            required = true
        "--obs-type"
            help = "Observation type"
            arg_type = String
            default = "min_plus_flat"
        "--num-games"
            help = "Self-play games to run"
            arg_type = Int
            default = 40
        "--width"
            help = "Contact network width"
            arg_type = Int
            default = 256
        "--blocks"
            help = "Contact network blocks"
            arg_type = Int
            default = 5
        "--race-width"
            help = "Race network width"
            arg_type = Int
            default = 128
        "--race-blocks"
            help = "Race network blocks"
            arg_type = Int
            default = 3
        "--num-workers"
            help = "CPU worker threads"
            arg_type = Int
            default = 12
        "--mcts-iters"
            help = "MCTS iterations per move"
            arg_type = Int
            default = 800
        "--gnubg-ply"
            help = "gnubg reference ply (1 = reference; ply>=2 deadlocks — refused)"
            arg_type = Int
            default = 1
        "--max-decisions"
            help = "Cap on analyzed unforced decisions (0 = no cap)"
            arg_type = Int
            default = 0
        "--calibrate-games"
            help = "gnubg self-play games to measure the reference floor (0 = skip)"
            arg_type = Int
            default = 3
        "--seed"
            help = "Base RNG seed"
            arg_type = Int
            default = 1
        "--race-checkpoint"
            help = "Explicit race checkpoint path (else auto-detect)"
            arg_type = String
            default = ""
        "--inference-batch-size"
            help = "Inference batch size for MCTS"
            arg_type = Int
            default = 50
        "--inference-backend"
            help = "CPU inference backend: auto, fast, or flux"
            arg_type = String
            default = "auto"
        "--chance-mode"
            help = "Chance-node handling: passthrough (default) or exact_expectation"
            arg_type = String
            default = "passthrough"
    end
    return ArgParse.parse_args(s)
end

const ARGS = parse_pr_args()

# Load packages
using AlphaZero
using AlphaZero: GI, MCTS, Network, FluxLib, MctsParams, ConstSchedule, BatchedMCTS, GameLoop
using AlphaZero.NetLib
import Flux
using Random
using Statistics
using Dates
using Printf

# BackgammonNet provides game + gnubg reference backend + equity conversion
using BackgammonNet

# Set up game
ENV["BACKGAMMON_OBS_TYPE"] = ARGS["obs_type"]
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
const gspec = GameSpec()
const NUM_ACTIONS = GI.num_actions(gspec)
const _state_dim = GI.state_dim(gspec)[1]
const ORACLE_CFG = AlphaZero.BackgammonInference.OracleConfig(
    _state_dim, NUM_ACTIONS, gspec;
    vectorize_state! = vectorize_state_into!,
    route_state = s -> (s isa BackgammonNet.BackgammonGame && !BackgammonNet.is_contact_position(s) ? 2 : 1))

# ── Recorded decision ─────────────────────────────────────────────────────
"""A single unforced checker decision made by our net (pre-move position + chosen action)."""
struct CheckerDecision
    state::BackgammonNet.BackgammonGame   # owning pre-move clone
    action::Int                           # action id our net chose (same id-space as evaluate_actions)
    is_contact::Bool
end

# ── Reference move evaluation ──────────────────────────────────────────────
# We reimplement BackgammonNet's DEFAULT `evaluate_actions` algorithm here
# (clone+apply, evaluate_probs of the resulting position, flip perspective if the
# roll passed to the opponent, compute_cubeless_equity) rather than calling
# `evaluate_actions(gnubg, ...)` directly, because gnubg OVERRIDES that method
# with a native move-list path whose (action_id, probs) pairing is corrupted for
# current_player==0 (verified: it ranks gnubg's own best move last). This
# per-resulting-position method is player-symmetric and, for player 1, reproduces
# gnubg's native list exactly. All building blocks are BackgammonNet's own
# (evaluate_probs, flip_equity_perspective, compute_cubeless_equity), so the
# equity scale/signs match the rest of the codebase.

"""
    reference_error(gnubg, decision; scratch) -> (:ok, err, nmoves) | (:forced, ...)

Equity lost by our chosen action vs the reference-best legal action, in cubeless
money-equity points (clamped at 0). `:forced` if <=1 legal move. `scratch` is a
reusable BackgammonGame to avoid per-call allocation.
"""
function reference_error(gnubg, d::CheckerDecision, scratch::BackgammonNet.BackgammonGame)
    g = d.state
    (g.phase == BackgammonNet.PHASE_CHECKER_PLAY) || return (:forced, 0.0, 0)
    actions = BackgammonNet.legal_actions(g)
    length(actions) <= 1 && return (:forced, 0.0, length(actions))

    best_eq = -Inf
    our_eq = nothing
    @inbounds for a in actions
        BackgammonNet.copy_state!(scratch, g)
        BackgammonNet.apply_action!(scratch, a)
        probs = BackgammonNet.evaluate_probs(gnubg, scratch)
        if scratch.current_player != g.current_player
            probs = BackgammonNet.flip_equity_perspective(probs)
        end
        eq = Float64(BackgammonNet.compute_cubeless_equity(g, probs))
        eq > best_eq && (best_eq = eq)
        a == d.action && (our_eq = eq)
    end
    # our chosen action is one of `g`'s legal actions, so our_eq is always set.
    our_eq === nothing && return (:forced, 0.0, length(actions))
    err = max(best_eq - our_eq, 0.0)   # clamp tiny negatives from eval noise
    return (:ok, err, length(actions))
end

# ── Self-play trace collection ─────────────────────────────────────────────
"""Play one self-play game (our net vs our net), returning the unforced checker decisions."""
function selfplay_decisions(single_oracle, batch_oracle, mcts_params, batch_size; seed::Int)
    rng = MersenneTwister(seed)
    env = GI.init(gspec)
    az = GameLoop.MctsAgent(single_oracle, batch_oracle, mcts_params, batch_size, gspec)

    # Self-play: our net on both sides. Greedy (temperature 0) for strongest play.
    result = GameLoop.play_game(az, az, env;
        record_trace=true,
        rng=rng,
        temperature_fn=_ -> 0.0)

    return decisions_from_trace(result.trace)
end

"""Extract unforced checker decisions from a recorded game trace."""
function decisions_from_trace(trace)
    decisions = CheckerDecision[]
    for e in trace
        # Skip chance (never in trace), forced single-move, and cube decisions.
        e.is_chance && continue
        length(e.legal_actions) > 1 || continue
        state = e.state
        (state isa BackgammonNet.BackgammonGame) || continue
        state.phase == BackgammonNet.PHASE_CHECKER_PLAY || continue
        push!(decisions, CheckerDecision(state, e.action, e.is_contact))
    end
    return decisions
end

"""Play one gnubg-vs-gnubg game (for the reference floor calibration)."""
function gnubg_selfplay_decisions(gnubg; seed::Int)
    rng = MersenneTwister(seed)
    env = GI.init(gspec)
    agent = GameLoop.ExternalAgent(gnubg)
    result = GameLoop.play_game(agent, agent, env;
        record_trace=true, rng=rng, temperature_fn=_ -> 0.0)
    return decisions_from_trace(result.trace)
end

"""
    score_decisions(gnubg_backends, decisions, num_workers) -> (errors, contact_flag, status)

Reference-evaluate every decision (parallel across per-worker backends; the gnubg
runtime serializes its C calls internally). `errors[i]` is NaN for skipped ones.
"""
function score_decisions(gnubg_backends, decisions, num_workers)
    n = length(decisions)
    errors = fill(NaN, n)
    contact_flag = Vector{Bool}(undef, n)
    status = Vector{Symbol}(undef, n)
    idx = Threads.Atomic{Int}(0)
    Threads.@threads for w in 1:num_workers
        gb = gnubg_backends[w]
        scratch = BackgammonNet.clone(decisions[1].state)
        while true
            i = Threads.atomic_add!(idx, 1) + 1
            i > n && break
            d = decisions[i]
            contact_flag[i] = d.is_contact
            local res
            try
                res = reference_error(gb, d, scratch)
            catch e
                @warn "gnubg reference_error failed on decision $i; skipping" exception=(e, catch_backtrace()) maxlog=5
                status[i] = :error
                continue
            end
            tag, err, _ = res
            status[i] = tag
            tag == :ok && (errors[i] = err)
        end
    end
    return errors, contact_flag, status
end

"""Aggregate scored errors into ER/PR summary."""
function summarize(errors, contact_flag, status)
    ok = status .== :ok
    n_ok = count(ok)
    n_ok == 0 && return nothing
    e_ok = errors[ok]
    mean_err = sum(e_ok) / n_ok
    split(sel) = begin
        e = errors[ok .& sel]
        isempty(e) ? (n=0, ER=NaN, PR=NaN) :
            (n=length(e), ER=1000.0*sum(e)/length(e), PR=500.0*sum(e)/length(e))
    end
    return (n_ok=n_ok,
            n_forced=count(==(:forced), status),
            n_error=count(==(:error), status),
            sum_err=sum(e_ok), mean_err=mean_err,
            ER=1000.0*mean_err, PR=500.0*mean_err,
            contact=split(contact_flag), race=split(.!contact_flag))
end

# ── Cube error-rate stub (checker-play only for now) ───────────────────────
# TODO(cube): When the cubeful architecture lands, add doubling ER and take/pass
# ER here. Structure: collect PHASE_CUBE_DECISION / PHASE_CUBE_RESPONSE decisions
# during self-play (they are currently skipped above), then compare our cube
# action's cubeful equity to gnubg's `cube_action` best cubeful equity via
# BackgammonNet.compute_cubeful_equity_janowski / gnubg's native cube analysis.
# Report doubling-ER and take/pass-ER as separate mEMG figures. Left as a stub:
# cube phase is disabled by default (BACKGAMMON_CUBE_ENABLED=false) and this
# script analyzes checker play only.
# function analyze_cube_decisions(...) end

# ── Main ───────────────────────────────────────────────────────────────────
function main()
    ckpt_path = ARGS["checkpoint"]
    gnubg_ply = ARGS["gnubg_ply"]
    num_workers = ARGS["num_workers"]
    batch_size = ARGS["inference_batch_size"]
    mcts_iters = ARGS["mcts_iters"]

    if gnubg_ply >= 2
        error("gnubg ply=$gnubg_ply refused: gnubg's 2-ply evaluator deadlocks its internal " *
              "thread pool. Use --gnubg-ply 1 (reference) or 0.")
    end

    # ── Resolve race checkpoint (auto-detect race_latest.data in checkpoint dir) ──
    ckpt_dir = dirname(ckpt_path)
    ckpt_name = basename(ckpt_path)
    race_ckpt_path = nothing
    if !isempty(ARGS["race_checkpoint"])
        race_ckpt_path = ARGS["race_checkpoint"]
    elseif startswith(ckpt_name, "contact_")
        cand = joinpath(ckpt_dir, replace(ckpt_name, "contact_" => "race_"))
        if isfile(cand)
            race_ckpt_path = cand
        elseif isfile(joinpath(ckpt_dir, "race_latest.data"))
            race_ckpt_path = joinpath(ckpt_dir, "race_latest.data")
        end
    elseif isfile(joinpath(ckpt_dir, "race_latest.data"))
        race_ckpt_path = joinpath(ckpt_dir, "race_latest.data")
    end

    println("=" ^ 72)
    println("AlphaZero checker-play strength: ER (mEMG) + PR vs gnubg")
    println("=" ^ 72)
    println("Contact checkpoint: $ckpt_path")
    println("Race checkpoint:    $(race_ckpt_path === nothing ? "(none — single model)" : race_ckpt_path)")
    println("Architecture:       contact=$(ARGS["width"])w×$(ARGS["blocks"])b" *
            (race_ckpt_path === nothing ? "" : " + race=$(ARGS["race_width"])w×$(ARGS["race_blocks"])b"))
    println("Obs type:           $(ARGS["obs_type"])")
    println("Self-play games:    $(ARGS["num_games"])  (greedy, MCTS-$mcts_iters, chance=$(ARGS["chance_mode"]))")
    println("Reference:          gnubg ply-$gnubg_ply")
    println("Workers:            $num_workers CPU")
    println("=" ^ 72)
    flush(stdout)

    # ── Load networks (CPU) ──
    contact_network = FluxLib.FCResNetMultiHead(
        gspec, FluxLib.FCResNetMultiHeadHP(width=ARGS["width"], num_blocks=ARGS["blocks"]))
    FluxLib.load_weights(ckpt_path, contact_network)
    contact_network = Flux.cpu(contact_network)

    race_network = nothing
    if race_ckpt_path !== nothing
        race_network = FluxLib.FCResNetMultiHead(
            gspec, FluxLib.FCResNetMultiHeadHP(width=ARGS["race_width"], num_blocks=ARGS["race_blocks"]))
        FluxLib.load_weights(race_ckpt_path, race_network)
        race_network = Flux.cpu(race_network)
    end

    mcts_params = MctsParams(
        num_iters_per_turn=mcts_iters,
        cpuct=1.5,
        temperature=ConstSchedule(0.0),
        dirichlet_noise_ϵ=0.0,
        dirichlet_noise_α=1.0,
        chance_mode=Symbol(ARGS["chance_mode"]))

    backend = AlphaZero.BackgammonInference.resolve_cpu_backend(ARGS["inference_backend"])
    println("CPU inference: $(AlphaZero.BackgammonInference.cpu_backend_summary(backend))")
    cpu_single, cpu_batch = AlphaZero.BackgammonInference.make_cpu_oracles(
        backend, contact_network, ORACLE_CFG;
        secondary_net=race_network, batch_size=batch_size)

    # ── Phase 1: parallel self-play, collect unforced checker decisions ──
    num_games = ARGS["num_games"]
    println("\nPhase 1: self-play ($num_games games, $num_workers workers)...")
    flush(stdout)
    t0 = time()
    per_game = Vector{Vector{CheckerDecision}}(undef, num_games)
    claimed = Threads.Atomic{Int}(0)
    Threads.@threads for _ in 1:num_workers
        while true
            i = Threads.atomic_add!(claimed, 1) + 1
            i > num_games && break
            per_game[i] = selfplay_decisions(cpu_single, cpu_batch, mcts_params, batch_size;
                                             seed=ARGS["seed"] + i)
        end
    end
    decisions = CheckerDecision[]
    for g in per_game; append!(decisions, g); end
    selfplay_time = time() - t0

    max_dec = ARGS["max_decisions"]
    if max_dec > 0 && length(decisions) > max_dec
        decisions = decisions[1:max_dec]
    end
    n_total = length(decisions)
    println("  Collected $n_total unforced checker decisions from $num_games games " *
            "($(round(selfplay_time, digits=1))s, $(round(n_total/max(num_games,1), digits=1)) dec/game)")
    if n_total == 0
        println("No decisions collected — nothing to analyze."); return
    end
    flush(stdout)

    # ── Phase 2: reference-evaluate every decision with gnubg ──
    # gnubg's C runtime serializes internally, but we still parallelize: the
    # clone/apply/compute work runs outside gnubg's lock. Per-worker backend
    # instances all resolve to the one shared, lock-guarded runtime.
    println("\nPhase 2: gnubg ply-$gnubg_ply reference evaluation of $n_total decisions...")
    flush(stdout)
    t1 = time()
    gnubg_backends = [begin
        gb = BackgammonNet.GnubgCLibBackend(ply=gnubg_ply, threads=1)
        BackgammonNet.open!(gb)
        gb
    end for _ in 1:num_workers]

    errors, contact_flag, status = score_decisions(gnubg_backends, decisions, num_workers)
    ref_time = time() - t1

    s = summarize(errors, contact_flag, status)
    if s === nothing
        for gb in gnubg_backends; BackgammonNet.close(gb); end
        println("No decisions could be scored against gnubg."); return
    end
    println("  Reference eval done ($(round(ref_time, digits=1))s)")

    # ── Optional: reference floor via gnubg self-play (same pipeline) ──
    floor = nothing
    calib_games = ARGS["calibrate_games"]
    if calib_games > 0
        println("\nCalibration: $calib_games gnubg self-play game(s) (reference floor)...")
        flush(stdout)
        calib_dec = CheckerDecision[]
        for gi in 1:calib_games
            append!(calib_dec, gnubg_selfplay_decisions(gnubg_backends[1]; seed=ARGS["seed"] + 10_000 + gi))
        end
        if !isempty(calib_dec)
            cerr, cflag, cstat = score_decisions(gnubg_backends, calib_dec, num_workers)
            floor = summarize(cerr, cflag, cstat)
        end
    end
    for gb in gnubg_backends; BackgammonNet.close(gb); end
    total_time = time() - t0

    # ── Report ──
    println("\n" * "=" ^ 72)
    println("RESULTS — checker-play strength (net vs gnubg ply-$gnubg_ply)")
    println("=" ^ 72)
    println("Decisions:")
    println("  scored (unforced):     $(s.n_ok)")
    println("  skipped — forced:      $(s.n_forced)")
    s.n_error > 0 && println("  skipped — eval error:  $(s.n_error)")
    println()
    @printf("  Total equity lost:  %.4f  points over %d decisions\n", s.sum_err, s.n_ok)
    @printf("  Mean equity lost:   %.6f  points / unforced decision\n", s.mean_err)
    println()
    println("Metrics (formulas printed for the record):")
    @printf("  ER (mEMG) = 1000 * mean_equity_lost = %7.2f   [milli-equity / unforced decision]\n", s.ER)
    @printf("  PR        =  500 * mean_equity_lost = %7.2f   [ = ER/2 ; XG/GNUbg convention]\n", s.PR)
    println()
    println("  Sanity anchors: world-class bot ~2-3 PR, strong human ~4-6, intermediate ~8-12.")
    println()
    println("Breakdown by phase:")
    @printf("  Contact: n=%5d  ER=%7.2f  PR=%6.2f\n", s.contact.n, s.contact.ER, s.contact.PR)
    @printf("  Race:    n=%5d  ER=%7.2f  PR=%6.2f\n", s.race.n, s.race.ER, s.race.PR)
    if floor !== nothing
        println()
        @printf("Reference floor (gnubg self-play, same pipeline): PR=%.2f  ER=%.2f  (n=%d)\n",
                floor.PR, floor.ER, floor.n_ok)
        @printf("  => net PR above reference floor: %+.2f\n", s.PR - floor.PR)
    end
    println()
    println("Reference: gnubg ply-$gnubg_ply, evaluated per resulting position (BackgammonNet")
    println("  evaluate_probs + compute_cubeless_equity; native move-list override bypassed).")
    println("Caveats: (1) ply-1 cubeless-money reference is WEAKER than XG's 2-ply/rollout")
    println("  cubeful/match reference, so ABSOLUTE PR is approximate. (2) Interpret the net's")
    println("  PR relative to the reference floor above; the DELTA is the internally-consistent")
    println("  signal for comparing our own nets under a fixed reference ply.")
    @printf("Timing: self-play %.1fs + reference %.1fs = %.1fs total.\n",
            selfplay_time, ref_time, total_time)
    println("=" ^ 72)
    flush(stdout)
end

main()
