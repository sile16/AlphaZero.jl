#!/usr/bin/env julia
"""
analyze_pr_native.jl — TRUE-scale checker-play Error Rate (ER / mEMG) and PR
(Performance Rating) vs GNU Backgammon (gnubg), using gnubg's OWN native
move-list equities so that the reference floor → ~0 (making absolute PR
comparable to published bot/human ratings).

WHY A NEW SCRIPT (vs scripts/analyze_pr.jl):
  analyze_pr.jl grades each move by cloning the position, applying the move, and
  calling `evaluate_probs` on the RESULTING position *independently*, per move.
  That per-resulting-position ply-1 eval is NOT the same number gnubg uses when
  it internally ranks the moves, so even gnubg-vs-itself scores a ~19 PR "floor"
  through that pipeline. The floor is a pure artifact of re-evaluating each
  resulting position in isolation.

  This script instead reads gnubg's ONE native move-list evaluation for the
  position (the `fn_list_moves` C bridge call behind `ranked_moves` /
  `evaluate_actions`), which returns, from that single call, every legal move's
  resulting board together with gnubg's own per-move probability vector. Grading
  every move off that single native evaluation makes gnubg's own best move the
  argmax by construction, so the gnubg-self-play floor collapses to ~0.

THE gnubg NATIVE MOVE-LIST API (what this script reaches into — READ-ONLY):
  BackgammonNet._gnubg_clib_move_data(backend, g) issues
      ccall(runtime.fn_list_moves, Cint,
            (Ptr{Cuint}, Ref{_BGNetCubeInfo}, Cint, Cint, Cint,
             Ptr{_BGNetMoveRecord}, Cuint, Ref{Cuint}),
            board, cubeinfo, dice1, dice2, ply, moves_buf, requested, count_ref)
  where each `_BGNetMoveRecord` is
      struct _BGNetMoveRecord
          anMove::NTuple{8,Cint}       # up to 4 (from,to) sub-moves, gnubg coords
          arEvalMove::NTuple{7,Cfloat} # gnubg's eval of that move's result
      end
  `_gnubg_clib_move_data` reconstructs each move's RESULTING board (applying
  anMove to `to_gnubg_simple(g)`) and returns a Vector of
      (resulting_board::Vector{Int}(26, gnubg-simple, mover perspective),
       probs::NTuple{5,Float64})     # (P(win),P(w∧g),P(w∧bg),P(l∧g),P(l∧bg))
  from gnubg's SINGLE evaluation of the position. This is exactly the
  {resulting_board → equity} set option-3 of the task calls for.

PERSPECTIVE OF THE NATIVE PROBS (determined empirically, see report):
  arEvalMove / `probs` is already in the MOVER's (g.current_player) perspective —
  higher = better for the player on roll. So the scalar equity is
      eq(move) = compute_cubeless_equity(g, probs)      # NO perspective flip
  Verified: for BOTH players, the entry equal to gnubg's own best_move board is
  ALWAYS the argmax of this equity (gap 0.0000), and applying the flip instead
  makes the best move look terrible — confirming NO flip is correct.

MATCHING OUR MOVE BY BOARD (sidesteps the known player-0 action-id bug):
  gnubg's native list has a player-0 board-RECONSTRUCTION quirk that scrambles
  the (action_id ↔ equity) pairing produced by `ranked_moves` (it can rank
  gnubg's own best move last for player 0). We never use that pairing. Instead:
    * best_eq = max over the native list of eq(move).
    * our_eq  = eq of the native-list entry whose RESULTING BOARD equals the
                board our full move produces. We convert each native resulting
                board to (p0,p1) bitboards via `from_gnubg_simple(board,
                g.current_player)` and compare against the bitboards our own
                `apply_action!` produces. Match-by-board is player-symmetric
                (verified: gnubg's best-move board is found for BOTH players).
    * regret  = max(best_eq - our_eq, 0)
  Because best_eq and our_eq come from the SAME native evaluation, feeding
  gnubg's best move as "played" yields regret ≡ 0 → floor ~0.

PER-MOVE (per-TURN) GRADING, incl. doubles:
  gnubg's native list contains COMPLETE moves (all checkers, up to 4 for
  doubles). Our engine plays doubles as TWO half-move actions. So we grade at
  the TURN level (the standard XG/gnubg ER unit): we combine a turn's half-move
  actions into one full-move resulting board and match that against the native
  full-move list. (analyze_pr.jl instead graded each half-move against an
  intermediate-position eval; per-turn here is both the standard unit and the
  only thing gnubg's native list exposes.)

METRIC (identical convention to analyze_pr.jl):
      ER (mEMG) = 1000 * mean(regret)      [milli-equity lost / unforced move]
      PR        =  500 * mean(regret)      [ = ER/2 ; XG/GNUbg convention]
  split into contact / race. Reference = gnubg ply-1 (ply>=2 deadlocks gnubg's
  internal 2-ply thread pool — refused, as in analyze_pr.jl).

VALIDATION built in:
  * FLOOR: gnubg self-play graded by this method → PR should be ~0 (vs
    analyze_pr.jl's ~19), reported per-player to prove no player-0 asymmetry.
  * NET:   the supplied checkpoint's true-scale PR (contact/race split).
  * RANDOM: random-legal self-play → PR must be MUCH higher than the floor.

Usage:
    julia --threads 16 --project scripts/analyze_pr_native.jl <contact_checkpoint> [options]

    julia --threads 16 --project scripts/analyze_pr_native.jl \\
        sessions/contact-flywheel/checkpoints/contact_iter_140.data \\
        --obs-type min_plus_flat --width 256 --blocks 5 --race-width 128 --race-blocks 3 \\
        --num-games 40 --mcts-iters 800 --num-workers 12 --gnubg-ply 1

Options (same as analyze_pr.jl, plus --random-games):
    --obs-type=min_plus_flat  --num-games=40   --width=256   --blocks=5
    --race-width=128  --race-blocks=3  --num-workers=12  --mcts-iters=800
    --gnubg-ply=1  --max-decisions=0  --calibrate-games=3  --random-games=3
    --seed=1  --race-checkpoint=PATH  --inference-batch-size=50
    --inference-backend=auto  --chance-mode=passthrough
"""

using ArgParse

function parse_pr_args()
    s = ArgParseSettings(description="Measure AZ checker-play ER/PR vs gnubg (native move-list floor≈0)", autofix_names=true)
    @add_arg_table! s begin
        "checkpoint"
            help = "Contact/main checkpoint file"; arg_type = String; required = true
        "--obs-type";        help = "Observation type"; arg_type = String; default = "min_plus_flat"
        "--num-games";       help = "Self-play games to run"; arg_type = Int; default = 40
        "--width";           help = "Contact network width"; arg_type = Int; default = 256
        "--blocks";          help = "Contact network blocks"; arg_type = Int; default = 5
        "--race-width";      help = "Race network width"; arg_type = Int; default = 128
        "--race-blocks";     help = "Race network blocks"; arg_type = Int; default = 3
        "--num-workers";     help = "CPU worker threads"; arg_type = Int; default = 12
        "--mcts-iters";      help = "MCTS iterations per move"; arg_type = Int; default = 800
        "--gnubg-ply";       help = "gnubg reference ply (1=reference; ply>=2 deadlocks — refused)"; arg_type = Int; default = 1
        "--max-decisions";   help = "Cap on analyzed unforced moves (0 = no cap)"; arg_type = Int; default = 0
        "--calibrate-games"; help = "gnubg self-play games to measure the reference floor (0 = skip)"; arg_type = Int; default = 3
        "--random-games";    help = "random-legal self-play games for the sanity contrast (0 = skip)"; arg_type = Int; default = 3
        "--seed";            help = "Base RNG seed"; arg_type = Int; default = 1
        "--race-checkpoint"; help = "Explicit race checkpoint path (else auto-detect)"; arg_type = String; default = ""
        "--inference-batch-size"; help = "Inference batch size for MCTS"; arg_type = Int; default = 50
        "--inference-backend";    help = "CPU inference backend: auto, fast, or flux"; arg_type = String; default = "auto"
        "--chance-mode";          help = "Chance-node handling: passthrough (default) or exact_expectation"; arg_type = String; default = "passthrough"
    end
    return ArgParse.parse_args(s)
end

const ARGS = parse_pr_args()

using AlphaZero
using AlphaZero: GI, MCTS, Network, FluxLib, MctsParams, ConstSchedule, BatchedMCTS, GameLoop
using AlphaZero.NetLib
import Flux
using Random
using Statistics
using Dates
using Printf
using BackgammonNet

ENV["BACKGAMMON_OBS_TYPE"] = ARGS["obs_type"]
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
const gspec = GameSpec()
const NUM_ACTIONS = GI.num_actions(gspec)
const _state_dim = GI.state_dim(gspec)[1]
const ORACLE_CFG = AlphaZero.BackgammonInference.OracleConfig(
    _state_dim, NUM_ACTIONS, gspec;
    vectorize_state! = vectorize_state_into!,
    route_state = s -> (s isa BackgammonNet.BackgammonGame && !BackgammonNet.is_contact_position(s) ? 2 : 1))

# ── A graded UNIT is a full TURN (one player's complete checker move) ─────────
"""
Pre-move (full-roll) position + the (p0,p1) bitboards our FULL move produced.
Grading matches `res_p0/res_p1` against gnubg's native full-move list.
"""
struct MoveDecision
    state::BackgammonNet.BackgammonGame   # pre-move clone (remaining_actions as rolled)
    res_p0::UInt128                       # resulting board after our full move
    res_p1::UInt128
    is_contact::Bool
    player::Int
end

# ── Native gnubg move-list grading ────────────────────────────────────────────
"""
    native_regret(gnubg, d) -> (:ok|:forced|:unmatched, regret, nmoves)

Equity our full move throws away vs gnubg's best, using gnubg's ONE native
move-list evaluation of the pre-move position. `:forced` when <=1 legal full
move; `:unmatched` if our resulting board is not in gnubg's list (should not
happen for a legal move — reported if it ever does).
"""
function native_regret(gnubg::BackgammonNet.GnubgCLibBackend, d::MoveDecision)
    g = d.state
    (g.phase == BackgammonNet.PHASE_CHECKER_PLAY) || return (:forced, 0.0, 0)
    BackgammonNet.open!(gnubg)
    # ONE native evaluation of the whole move list (under the gnubg C lock).
    md = lock(BackgammonNet._GNUBG_CLIB_LOCK) do
        BackgammonNet._gnubg_clib_move_data(gnubg, g)
    end
    n = length(md)
    n <= 1 && return (:forced, 0.0, n)

    player = Int(g.current_player)
    best_eq = -Inf
    our_eq = nothing
    @inbounds for (tsimple, probs) in md
        # Native probs are already in the mover's perspective → NO flip.
        eq = Float64(BackgammonNet.compute_cubeless_equity(g, probs))
        eq > best_eq && (best_eq = eq)
        tp0, tp1 = BackgammonNet.from_gnubg_simple(tsimple, player)
        if tp0 == d.res_p0 && tp1 == d.res_p1
            our_eq = eq
        end
    end
    our_eq === nothing && return (:unmatched, 0.0, n)
    return (:ok, max(best_eq - our_eq, 0.0), n)
end

# ── Turn extraction ───────────────────────────────────────────────────────────
"""Group an AZ self-play trace into full-move (per-turn) MoveDecisions.

Consecutive same-player checker entries that are board-continuous (the doubles
first + second half) are combined into one full move. Non-doubles turns are a
single entry."""
function turns_from_trace(trace)
    res = MoveDecision[]
    n = length(trace)
    i = 1
    while i <= n
        e = trace[i]
        if e.is_chance || !(e.state isa BackgammonNet.BackgammonGame) ||
           e.state.phase != BackgammonNet.PHASE_CHECKER_PLAY || length(e.legal_actions) < 1
            i += 1; continue
        end
        g = e.state
        player = e.player
        s = BackgammonNet.clone(g)
        BackgammonNet.apply_action!(s, e.action)
        j = i + 1
        # doubles second half: same player, board-continuous with s after half 1
        if g.remaining_actions == 2
            while j <= n
                ej = trace[j]
                (ej.state isa BackgammonNet.BackgammonGame) || break
                (!ej.is_chance && ej.player == player &&
                 ej.state.phase == BackgammonNet.PHASE_CHECKER_PLAY &&
                 ej.state.p0 == s.p0 && ej.state.p1 == s.p1) || break
                BackgammonNet.apply_action!(s, ej.action)
                j += 1
                break   # doubles has at most 2 half-moves
            end
        end
        push!(res, MoveDecision(g, s.p0, s.p1, e.is_contact, player))
        i = j
    end
    return res
end

# ── AZ net self-play (batched MCTS) ───────────────────────────────────────────
function net_selfplay_turns(single_oracle, batch_oracle, mcts_params, batch_size; seed::Int)
    rng = MersenneTwister(seed)
    env = GI.init(gspec)
    az = GameLoop.MctsAgent(single_oracle, batch_oracle, mcts_params, batch_size, gspec)
    result = GameLoop.play_game(az, az, env; record_trace=true, rng=rng, temperature_fn=_ -> 0.0)
    return turns_from_trace(result.trace)
end

# ── Pure-engine self-play with a pluggable checker policy (gnubg / random) ─────
function _sample_dice(rng)
    r = rand(rng, Float32); c = 0.0f0
    @inbounds for i in 1:length(BackgammonNet.DICE_PROBS)
        c += BackgammonNet.DICE_PROBS[i]
        r <= c && return i
    end
    return length(BackgammonNet.DICE_PROBS)
end

"""Play one game where checker moves are chosen by `policy` (:gnubg | :random),
returning per-turn MoveDecisions. gnubg policy uses `best_move` (its cache links
the two doubles half-moves), so it must run serially on `gnubg`."""
function pure_selfplay_turns(gnubg, policy::Symbol, rng; step_cap::Int=20_000)
    g = BackgammonNet.initial_state()
    out = MoveDecision[]
    steps = 0
    while true
        steps += 1
        steps > step_cap && error("pure_selfplay_turns exceeded step cap")
        at = BackgammonNet.action_type(g)
        at == BackgammonNet.ACTION_TYPE_TERMINAL && return out
        if at == BackgammonNet.ACTION_TYPE_CHANCE
            BackgammonNet.apply_chance!(g, _sample_dice(rng)); continue
        end
        # start of a checker turn
        pre = BackgammonNet.clone(g)
        is_contact = BackgammonNet.is_contact_position(g)
        player = Int(g.current_player)
        start_player = g.current_player
        # execute the full turn (1 or 2 half-moves)
        while true
            att = BackgammonNet.action_type(g)
            (att == BackgammonNet.ACTION_TYPE_TERMINAL || att == BackgammonNet.ACTION_TYPE_CHANCE) && break
            g.current_player != start_player && break
            acts = BackgammonNet.legal_actions(g)
            isempty(acts) && break
            a = if policy === :gnubg
                # best_move can error on forced-pass / no-legal-move doubles halves;
                # fall back to a legal action (such turns are forced → graded :forced).
                length(acts) == 1 ? acts[1] :
                    try BackgammonNet.best_move(gnubg, g) catch; acts[1] end
            else
                acts[rand(rng, 1:length(acts))]
            end
            BackgammonNet.apply_action!(g, a)
            g.terminated && break
        end
        push!(out, MoveDecision(pre, g.p0, g.p1, is_contact, player))
        g.terminated && return out
    end
end

# ── Parallel scoring over a set of decisions ──────────────────────────────────
function score_moves(gnubg_backends, decisions, num_workers)
    n = length(decisions)
    errors = fill(NaN, n)
    contact_flag = Vector{Bool}(undef, n)
    player_of = Vector{Int}(undef, n)
    status = Vector{Symbol}(undef, n)
    idx = Threads.Atomic{Int}(0)
    Threads.@threads for w in 1:num_workers
        gb = gnubg_backends[w]
        while true
            i = Threads.atomic_add!(idx, 1) + 1
            i > n && break
            d = decisions[i]
            contact_flag[i] = d.is_contact
            player_of[i] = d.player
            local res
            try
                res = native_regret(gb, d)
            catch e
                @warn "native_regret failed on decision $i; skipping" exception=(e, catch_backtrace()) maxlog=5
                status[i] = :error; continue
            end
            tag, err, _ = res
            status[i] = tag
            tag == :ok && (errors[i] = err)
        end
    end
    return errors, contact_flag, player_of, status
end

# ── Aggregation ───────────────────────────────────────────────────────────────
function summarize(errors, contact_flag, player_of, status)
    ok = status .== :ok
    n_ok = count(ok)
    n_ok == 0 && return nothing
    e_ok = errors[ok]
    mean_err = sum(e_ok) / n_ok
    subset(sel) = begin
        e = errors[ok .& sel]
        isempty(e) ? (n=0, ER=NaN, PR=NaN) :
            (n=length(e), ER=1000.0*sum(e)/length(e), PR=500.0*sum(e)/length(e))
    end
    return (n_ok=n_ok,
            n_forced=count(==(:forced), status),
            n_unmatched=count(==(:unmatched), status),
            n_error=count(==(:error), status),
            sum_err=sum(e_ok), mean_err=mean_err,
            ER=1000.0*mean_err, PR=500.0*mean_err,
            contact=subset(contact_flag), race=subset(.!contact_flag),
            p0=subset(player_of .== 0), p1=subset(player_of .== 1))
end

# ── Main ──────────────────────────────────────────────────────────────────────
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

    # resolve race checkpoint (auto-detect race_latest.data in checkpoint dir)
    ckpt_dir = dirname(ckpt_path); ckpt_name = basename(ckpt_path)
    race_ckpt_path = nothing
    if !isempty(ARGS["race_checkpoint"])
        race_ckpt_path = ARGS["race_checkpoint"]
    elseif startswith(ckpt_name, "contact_")
        cand = joinpath(ckpt_dir, replace(ckpt_name, "contact_" => "race_"))
        if isfile(cand); race_ckpt_path = cand
        elseif isfile(joinpath(ckpt_dir, "race_latest.data")); race_ckpt_path = joinpath(ckpt_dir, "race_latest.data"); end
    elseif isfile(joinpath(ckpt_dir, "race_latest.data"))
        race_ckpt_path = joinpath(ckpt_dir, "race_latest.data")
    end

    println("=" ^ 72)
    println("AlphaZero checker-play strength: TRUE-scale ER (mEMG) + PR (native gnubg floor≈0)")
    println("=" ^ 72)
    println("Contact checkpoint: $ckpt_path")
    println("Race checkpoint:    $(race_ckpt_path === nothing ? "(none — single model)" : race_ckpt_path)")
    println("Architecture:       contact=$(ARGS["width"])w×$(ARGS["blocks"])b" *
            (race_ckpt_path === nothing ? "" : " + race=$(ARGS["race_width"])w×$(ARGS["race_blocks"])b"))
    println("Obs type:           $(ARGS["obs_type"])")
    println("Self-play games:    $(ARGS["num_games"])  (greedy, MCTS-$mcts_iters, chance=$(ARGS["chance_mode"]))")
    println("Reference:          gnubg ply-$gnubg_ply, NATIVE move-list equities (per TURN)")
    println("Workers:            $num_workers CPU")
    println("=" ^ 72); flush(stdout)

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
        num_iters_per_turn=mcts_iters, cpuct=1.5, temperature=ConstSchedule(0.0),
        dirichlet_noise_ϵ=0.0, dirichlet_noise_α=1.0, chance_mode=Symbol(ARGS["chance_mode"]))

    backend = AlphaZero.BackgammonInference.resolve_cpu_backend(ARGS["inference_backend"])
    println("CPU inference: $(AlphaZero.BackgammonInference.cpu_backend_summary(backend))")
    cpu_single, cpu_batch = AlphaZero.BackgammonInference.make_cpu_oracles(
        backend, contact_network, ORACLE_CFG; secondary_net=race_network, batch_size=batch_size)

    # ── Phase 1: net self-play → per-turn decisions ──
    num_games = ARGS["num_games"]
    println("\nPhase 1: net self-play ($num_games games, $num_workers workers)...")
    flush(stdout)
    t0 = time()
    per_game = Vector{Vector{MoveDecision}}(undef, num_games)
    claimed = Threads.Atomic{Int}(0)
    Threads.@threads for _ in 1:num_workers
        while true
            i = Threads.atomic_add!(claimed, 1) + 1
            i > num_games && break
            per_game[i] = net_selfplay_turns(cpu_single, cpu_batch, mcts_params, batch_size; seed=ARGS["seed"] + i)
        end
    end
    decisions = MoveDecision[]
    for g in per_game; append!(decisions, g); end
    selfplay_time = time() - t0
    max_dec = ARGS["max_decisions"]
    if max_dec > 0 && length(decisions) > max_dec; decisions = decisions[1:max_dec]; end
    n_total = length(decisions)
    println("  Collected $n_total full-move decisions from $num_games games " *
            "($(round(selfplay_time, digits=1))s, $(round(n_total/max(num_games,1), digits=1)) moves/game)")
    if n_total == 0; println("No decisions collected — nothing to analyze."); return; end
    flush(stdout)

    # ── Build gnubg backends (shared, lock-serialized C runtime) ──
    gnubg_backends = [begin
        gb = BackgammonNet.GnubgCLibBackend(ply=gnubg_ply, threads=1); BackgammonNet.open!(gb); gb
    end for _ in 1:num_workers]

    # ── Phase 2: native gnubg grading of the net's moves ──
    println("\nPhase 2: gnubg ply-$gnubg_ply NATIVE move-list grading of $n_total moves...")
    flush(stdout)
    t1 = time()
    errors, cflag, pof, status = score_moves(gnubg_backends, decisions, num_workers)
    ref_time = time() - t1
    s = summarize(errors, cflag, pof, status)
    if s === nothing
        for gb in gnubg_backends; BackgammonNet.close(gb); end
        println("No decisions could be scored against gnubg."); return
    end
    println("  Grading done ($(round(ref_time, digits=1))s)")

    # ── Floor: gnubg self-play through the same native grading ──
    floor = nothing
    calib_games = ARGS["calibrate_games"]
    if calib_games > 0
        println("\nFLOOR: $calib_games gnubg self-play game(s), same native grading...")
        flush(stdout)
        calib = MoveDecision[]
        for gi in 1:calib_games
            append!(calib, pure_selfplay_turns(gnubg_backends[1], :gnubg,
                                               MersenneTwister(ARGS["seed"] + 10_000 + gi)))
        end
        if !isempty(calib)
            cerr, ccf, cpo, cst = score_moves(gnubg_backends, calib, num_workers)
            floor = summarize(cerr, ccf, cpo, cst)
        end
    end

    # ── Random-agent sanity contrast ──
    randsum = nothing
    rnd_games = ARGS["random_games"]
    if rnd_games > 0
        println("\nSANITY: $rnd_games random-legal self-play game(s)...")
        flush(stdout)
        rnd = MoveDecision[]
        for gi in 1:rnd_games
            append!(rnd, pure_selfplay_turns(gnubg_backends[1], :random,
                                             MersenneTwister(ARGS["seed"] + 20_000 + gi)))
        end
        if !isempty(rnd)
            rerr, rcf, rpo, rst = score_moves(gnubg_backends, rnd, num_workers)
            randsum = summarize(rerr, rcf, rpo, rst)
        end
    end

    for gb in gnubg_backends; BackgammonNet.close(gb); end
    total_time = time() - t0

    # ── Report ──
    println("\n" * "=" ^ 72)
    println("RESULTS — TRUE-scale checker-play strength (net vs gnubg ply-$gnubg_ply, native)")
    println("=" ^ 72)
    println("Decisions (full moves / turns):")
    println("  scored (unforced):     $(s.n_ok)")
    println("  skipped — forced:      $(s.n_forced)")
    s.n_unmatched > 0 && println("  skipped — UNMATCHED:   $(s.n_unmatched)  (our board not in gnubg list — investigate)")
    s.n_error > 0 && println("  skipped — eval error:  $(s.n_error)")
    println()
    @printf("  Total equity lost:  %.4f  points over %d moves\n", s.sum_err, s.n_ok)
    @printf("  Mean equity lost:   %.6f  points / unforced move\n", s.mean_err)
    println()
    println("Metrics:")
    @printf("  ER (mEMG) = 1000 * mean_equity_lost = %7.2f\n", s.ER)
    @printf("  PR        =  500 * mean_equity_lost = %7.2f   ( = ER/2 ; XG/GNUbg convention)\n", s.PR)
    println()
    println("  Sanity anchors: world-class bot ~2-3 PR, strong human ~4-6, intermediate ~8-12.")
    println()
    println("Breakdown by phase:")
    @printf("  Contact: n=%5d  ER=%7.2f  PR=%6.2f\n", s.contact.n, s.contact.ER, s.contact.PR)
    @printf("  Race:    n=%5d  ER=%7.2f  PR=%6.2f\n", s.race.n, s.race.ER, s.race.PR)
    @printf("  (player split — asymmetry check: P0 PR=%.2f n=%d | P1 PR=%.2f n=%d)\n",
            s.p0.PR, s.p0.n, s.p1.PR, s.p1.n)

    if floor !== nothing
        println()
        @printf("FLOOR (gnubg self-play, native grading): PR=%.3f  ER=%.3f  (n=%d)\n",
                floor.PR, floor.ER, floor.n_ok)
        @printf("  per-player floor:  P0 PR=%.3f (n=%d)   P1 PR=%.3f (n=%d)   [both ≈0 ⇒ no player-0 bug]\n",
                floor.p0.PR, floor.p0.n, floor.p1.PR, floor.p1.n)
        floor.n_unmatched > 0 && @printf("  floor UNMATCHED: %d\n", floor.n_unmatched)
        @printf("  => net PR above native floor: %+.2f\n", s.PR - floor.PR)
        println(abs(floor.PR) < 1.0 ?
            "  => FLOOR ≈ 0 : native move-list equity extraction VALIDATED (vs analyze_pr.jl's ~19)." :
            "  => FLOOR NOT ≈0 : investigate perspective / board matching above.")
    end
    if randsum !== nothing
        println()
        @printf("RANDOM (random-legal self-play): PR=%.2f  ER=%.2f  (n=%d)  [contact PR=%.2f race PR=%.2f]\n",
                randsum.PR, randsum.ER, randsum.n_ok, randsum.contact.PR, randsum.race.PR)
        floor !== nothing && println(randsum.PR > (floor.PR + 5) ?
            "  => contrast OK: random PR ≫ floor." :
            "  => WARNING: random PR not clearly above floor — unexpected.")
    end
    println()
    println("Reference: gnubg ply-$gnubg_ply NATIVE move list (fn_list_moves via")
    println("  BackgammonNet._gnubg_clib_move_data); per-move equity = compute_cubeless_equity of")
    println("  gnubg's own per-move probs (mover perspective, no flip); our move matched BY BOARD")
    println("  (from_gnubg_simple bitboards) → floor≈0, player-symmetric.")
    @printf("Timing: self-play %.1fs + grading %.1fs = %.1fs total.\n", selfplay_time, ref_time, total_time)
    println("=" ^ 72); flush(stdout)
end

main()
