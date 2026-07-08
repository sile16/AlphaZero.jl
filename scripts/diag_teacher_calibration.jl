#!/usr/bin/env julia
"""
diag_teacher_calibration.jl — TEACHER vs DRIFT diagnostic.

QUESTION
  Our contact net's value head was trained to imitate wildbg per-position equity
  (0.99 corr vs wildbg). But measured vs GNUbg equity the NET is only ~0.79 corr
  with a large +0.72 over-optimistic bias. Two hypotheses:
    (A) TEACHER problem — wildbg's OWN equity is miscalibrated vs true equity, so
        our net faithfully learned a bad target. Fix = use gnubg/rollouts as the
        value teacher, not wildbg.
    (B) DRIFT problem — wildbg's equity is fine; the net's +0.72 over-optimism was
        added during SELF-PLAY drift. Fix = anchor self-play value to a reference.

  This script distinguishes them WITHOUT touching the net at all: it computes
  wildbg-equity and gnubg-equity on the SAME contact positions and correlates
  them. gnubg (ply-1) is the strong reference / "truth".

    * If wildbg-vs-gnubg corr ≈ 0.79 and bias ≈ +0.72 (matching the NET's own
      miscalibration) → the TEACHER is the problem (hypothesis A).
    * If wildbg-vs-gnubg corr is HIGH (~0.9+) and bias SMALL (~0) → wildbg is a
      good teacher and the net's over-optimism came from DRIFT (hypothesis B).
    * Intermediate → both contribute; the split is quantified.

METHOD (reuses machinery from scripts/benchmark_pr.jl & calibrate_pr_ladder.jl)
  1. Generate ~N CONTACT decision positions by playing cubeless money games in
     which gnubg-0ply moves both sides with seeded dice, snapshotting the PRE-MOVE
     position at the start of every checker turn (verbatim generator). Both
     players-to-move are represented (play alternates). Race/bearoff positions are
     skipped — race is table-exact and not the concern here.
  2. For each position g:  wb = evaluate(wildbg, g),  gb = evaluate(gnubg, g).
     Both BackgammonNet.evaluate(...) return rule-aware cubeless money equity in
     the MOVER's perspective (probs are for g.current_player, run through the same
     compute_cubeless_equity) — verified identical perspective, so NO flip and the
     two numbers are directly comparable on every position.
  3. Report corr, mean bias mean(wb-gb), MSE, MAE, and each mean. Split by player
     to expose any P0/P1 asymmetry. Bonus: sign-agreement + corr on the clearly-
     decided subset (|gb| large) as an orientation sanity check.

Usage:
    julia --threads 8 --project scripts/diag_teacher_calibration.jl [--n 2000 --seed 1]

Options:
    --n=2000             target CONTACT positions to evaluate
    --seed=1             base RNG seed (dice) — deterministic position set
    --gnubg-ply=1        gnubg reference ply (the strong "truth"; 1 = validated)
    --num-workers=8      CPU worker threads for the two evaluations
    --wildbg-lib=PATH    libwildbg.so (default ~/github/wildbg/target/release/libwildbg.so)
    --max-games=100000   safety cap on generator games
    --decided-thresh=0.7 |gnubg equity| above which a position is "clearly decided"
"""

using ArgParse

function parse_diag_args()
    s = ArgParseSettings(description="wildbg-vs-gnubg equity calibration (teacher vs drift)", autofix_names=true)
    @add_arg_table! s begin
        "--n";             help = "Target CONTACT positions to evaluate"; arg_type = Int; default = 2000
        "--seed";          help = "Base RNG seed (dice) — deterministic position set"; arg_type = Int; default = 1
        "--gnubg-ply";     help = "gnubg reference ply (strong truth; 1 = validated)"; arg_type = Int; default = 1
        "--num-workers";   help = "CPU worker threads for evaluation"; arg_type = Int; default = 8
        "--wildbg-lib";    help = "libwildbg.so path"; arg_type = String; default = ""
        "--max-games";     help = "Safety cap on generator games"; arg_type = Int; default = 100_000
        "--decided-thresh"; help = "|gnubg equity| above which a position is clearly decided"; arg_type = Float64; default = 0.7
    end
    return ArgParse.parse_args(s)
end

const ARGS_D = parse_diag_args()

using BackgammonNet
using Random
using Statistics
using Printf

# ── Fixed contact position: pre-move checker-play state + player to move ────────
struct DiagPosition
    state::BackgammonNet.BackgammonGame
    player::Int
end

# ── Dice sampling (verbatim from calibrate_pr_ladder.jl) ────────────────────────
function _sample_dice(rng)
    r = rand(rng, Float32); c = 0.0f0
    @inbounds for i in 1:length(BackgammonNet.DICE_PROBS)
        c += BackgammonNet.DICE_PROBS[i]
        r <= c && return i
    end
    return length(BackgammonNet.DICE_PROBS)
end

# ── Full-TURN execution by an engine's best_move (from benchmark_pr.jl) ──────────
function play_turn_with_engine!(g, engine, fail_counter::Threads.Atomic{Int})
    start_player = g.current_player
    while true
        att = BackgammonNet.action_type(g)
        (att == BackgammonNet.ACTION_TYPE_TERMINAL || att == BackgammonNet.ACTION_TYPE_CHANCE) && break
        g.current_player != start_player && break
        acts = BackgammonNet.legal_actions(g)
        isempty(acts) && break
        a = length(acts) == 1 ? acts[1] :
            try BackgammonNet.best_move(engine, g) catch
                Threads.atomic_add!(fail_counter, 1); acts[1]
            end
        BackgammonNet.apply_action!(g, a)
        g.terminated && break
    end
    return g
end

# ── Generate CONTACT positions: gnubg-0ply moves both sides, seeded dice ────────
"""Play cubeless games in which `gen_engine` (gnubg-0ply) moves both sides with
seeded dice, snapshotting the PRE-MOVE state at the start of every CONTACT checker
turn (race/bearoff skipped). Deterministic in `base_seed`."""
function generate_contact_positions(gen_engine, n_target::Int, base_seed::Int, max_games::Int)
    positions = DiagPosition[]
    fail = Threads.Atomic{Int}(0)
    gi = 0
    while length(positions) < n_target && gi < max_games
        gi += 1
        rng = MersenneTwister(base_seed + gi)
        g = BackgammonNet.initial_state()
        while true
            at = BackgammonNet.action_type(g)
            at == BackgammonNet.ACTION_TYPE_TERMINAL && break
            if at == BackgammonNet.ACTION_TYPE_CHANCE
                BackgammonNet.apply_chance!(g, _sample_dice(rng)); continue
            end
            if BackgammonNet.is_contact_position(g)
                push!(positions, DiagPosition(BackgammonNet.clone(g), Int(g.current_player)))
            end
            play_turn_with_engine!(g, gen_engine, fail)
            g.terminated && break
            length(positions) >= n_target && break
        end
    end
    return positions, gi, fail[]
end

# ── Engine factory (from calibrate_pr_ladder.jl) ────────────────────────────────
function make_wildbg(wildbg_lib::String)
    lib_size = filesize(wildbg_lib)
    nets = lib_size > 10_000_000 ? :large : :small
    nets == :large ? BackgammonNet.wildbg_set_lib_path!(large=wildbg_lib) :
                     BackgammonNet.wildbg_set_lib_path!(small=wildbg_lib)
    e = BackgammonNet.WildbgBackend(nets=nets); BackgammonNet.open!(e); return e
end

make_gnubg(ply::Int) = (e = BackgammonNet.GnubgCLibBackend(ply=ply, threads=1); BackgammonNet.open!(e); e)

# ── Paired evaluation on the common positions (parallel across workers) ─────────
"""For each position compute wb=evaluate(wildbg,g) and gb=evaluate(gnubg,g).
One wildbg + one gnubg backend PER worker (both hold serial C/cache state, so each
worker owns its own handle). Both are mover-perspective → directly comparable.
Returns (wb, gb, player) aligned; positions that fail either eval are dropped."""
function paired_eval(positions::Vector{DiagPosition}, num_workers::Int, wildbg_lib::String, gnubg_ply::Int)
    n = length(positions)
    wb = fill(NaN, n); gb = fill(NaN, n)
    okflag = fill(false, n)
    player = Vector{Int}(undef, n)
    idx = Threads.Atomic{Int}(0)
    n_fail = Threads.Atomic{Int}(0)
    Threads.@threads for w in 1:num_workers
        wengine = make_wildbg(wildbg_lib)
        gengine = make_gnubg(gnubg_ply)
        try
            while true
                i = Threads.atomic_add!(idx, 1) + 1
                i > n && break
                g = positions[i].state
                player[i] = positions[i].player
                try
                    wb[i] = BackgammonNet.evaluate(wengine, g)
                    gb[i] = BackgammonNet.evaluate(gengine, g)
                    okflag[i] = isfinite(wb[i]) && isfinite(gb[i])
                catch e
                    Threads.atomic_add!(n_fail, 1)
                    @warn "eval failed on position $i; skipping" exception=(e,) maxlog=5
                end
            end
        finally
            try BackgammonNet.close(wengine) catch end
            try BackgammonNet.close(gengine) catch end
        end
    end
    return wb, gb, player, okflag, n_fail[]
end

# ── Calibration statistics ──────────────────────────────────────────────────────
function calib_stats(wb::Vector{Float64}, gb::Vector{Float64})
    n = length(wb)
    n == 0 && return nothing
    d = wb .- gb
    corr = n >= 2 ? Statistics.cor(wb, gb) : NaN
    return (n=n,
            corr=corr,
            bias=Statistics.mean(d),                 # mean(wb - gb): + => wildbg over-optimistic
            mse=Statistics.mean(d .^ 2),
            mae=Statistics.mean(abs.(d)),
            rmse=sqrt(Statistics.mean(d .^ 2)),
            mean_wb=Statistics.mean(wb),
            mean_gb=Statistics.mean(gb),
            std_wb=n >= 2 ? Statistics.std(wb) : NaN,
            std_gb=n >= 2 ? Statistics.std(gb) : NaN)
end

function print_stats(label, s)
    s === nothing && (println("  $label: (no data)"); return)
    @printf("  %-22s n=%5d  corr=%6.3f  bias(wb-gb)=%+6.3f  MAE=%5.3f  RMSE=%5.3f  MSE=%5.3f\n",
            label, s.n, s.corr, s.bias, s.mae, s.rmse, s.mse)
    @printf("  %-22s mean_wb=%+6.3f  mean_gb=%+6.3f   std_wb=%5.3f  std_gb=%5.3f\n",
            "", s.mean_wb, s.mean_gb, s.std_wb, s.std_gb)
end

# ── Main ────────────────────────────────────────────────────────────────────────
function main()
    n_target = ARGS_D["n"]
    base_seed = ARGS_D["seed"]
    gnubg_ply = ARGS_D["gnubg_ply"]
    num_workers = ARGS_D["num_workers"]
    max_games = ARGS_D["max_games"]
    decided_thresh = ARGS_D["decided_thresh"]

    gnubg_ply >= 2 && error("gnubg reference ply=$gnubg_ply refused (2-ply deadlocks). Use --gnubg-ply 1 or 0.")

    wildbg_lib = ARGS_D["wildbg_lib"]
    isempty(wildbg_lib) && (wildbg_lib = joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg.so"))
    isfile(wildbg_lib) || error("libwildbg not found at $wildbg_lib — pass --wildbg-lib=/path/to/libwildbg.so")

    println("=" ^ 78)
    println("TEACHER-vs-DRIFT DIAGNOSTIC — wildbg equity vs gnubg equity on common positions")
    println("=" ^ 78)
    println("Positions:   $n_target CONTACT pre-move decisions (gnubg-0ply generator, both players)")
    println("wildbg:      WildbgBackend  (evaluate = rule-aware cubeless money equity, mover persp.)")
    println("gnubg:       GnubgCLibBackend(ply=$gnubg_ply)  [strong reference / 'truth']")
    println("Comparison:  both mover-perspective → directly comparable, no flip.  bias=mean(wb-gb)")
    println("Seed:        $base_seed  (deterministic)     Workers: $num_workers")
    println("wildbg lib:  $wildbg_lib")
    println("=" ^ 78); flush(stdout)

    # ── Generate the common contact-position set ──
    println("\nGENERATING contact positions (gnubg-0ply moves both sides, seeded dice)...")
    flush(stdout)
    t0 = time()
    gen_engine = make_gnubg(0)
    positions, n_games, gen_fail = generate_contact_positions(gen_engine, n_target, base_seed, max_games)
    try BackgammonNet.close(gen_engine) catch end
    np0 = count(p -> p.player == 0, positions); np1 = count(p -> p.player == 1, positions)
    println("  $(length(positions)) contact positions from $n_games games ($(round(time()-t0,digits=1))s)")
    println("  player-to-move split:  P0=$np0   P1=$np1")
    gen_fail > 0 && println("  (generator best_move fallbacks: $gen_fail)")
    isempty(positions) && (println("No positions — nothing to do."); return)
    flush(stdout)

    # ── Paired evaluation ──
    println("\nEVALUATING wildbg-equity and gnubg-equity on every position...")
    flush(stdout)
    t1 = time()
    wb, gb, player, okflag, eval_fail = paired_eval(positions, num_workers, wildbg_lib, gnubg_ply)
    println("  done ($(round(time()-t1,digits=1))s)" * (eval_fail > 0 ? "  [eval failures skipped: $eval_fail]" : ""))
    flush(stdout)

    keep = okflag
    wbk = wb[keep]; gbk = gb[keep]; pk = player[keep]

    # ── Report ──
    println("\n" * "=" ^ 78)
    println("CALIBRATION — wildbg equity vs gnubg-ply-$gnubg_ply equity (SAME positions)")
    println("=" ^ 78)
    overall = calib_stats(wbk, gbk)
    print_stats("ALL", overall)
    println()
    print_stats("player-to-move P0", calib_stats(wbk[pk .== 0], gbk[pk .== 0]))
    print_stats("player-to-move P1", calib_stats(wbk[pk .== 1], gbk[pk .== 1]))

    # ── Orientation sanity check: clearly-decided subset ──
    println("\n" * "-" ^ 78)
    println("ORIENTATION SANITY CHECK — clearly-decided positions (|gnubg equity| > $decided_thresh)")
    println("-" ^ 78)
    dec = abs.(gbk) .> decided_thresh
    ndec = count(dec)
    if ndec > 0
        sign_agree = count(sign.(wbk[dec]) .== sign.(gbk[dec])) / ndec
        ds = calib_stats(wbk[dec], gbk[dec])
        @printf("  decided positions: n=%d   SIGN-agreement(wb vs gb)=%.1f%%   corr=%.3f\n",
                ndec, 100*sign_agree, ds.corr)
        # A few worked examples spanning the equity range
        order = sortperm(gbk[dec])
        decidx = findall(dec)
        picks = unique(round.(Int, range(1, length(order); length=min(6, length(order)))))
        println("  sample (sorted by gnubg equity):    gnubg     wildbg    (wb-gb)")
        for k in picks
            j = decidx[order[k]]
            @printf("       gnubg=%+6.3f   wildbg=%+6.3f   diff=%+6.3f\n", gbk[j], wbk[j], wbk[j]-gbk[j])
        end
        println(sign_agree > 0.9 ?
            "  => Signs agree on clearly-decided positions: comparison is correctly ORIENTED." :
            "  => WARNING: sign disagreement is high — check perspective alignment before trusting the verdict.")
    else
        println("  (no positions exceeded the decided threshold)")
    end

    # ── Verdict ──
    println("\n" * "=" ^ 78)
    println("VERDICT — is the net's miscalibration inherited from the TEACHER or added by DRIFT?")
    println("=" ^ 78)
    println("Reference (our NET vs gnubg, measured previously):  corr≈0.79   bias≈+0.72")
    @printf("This run (wildbg TEACHER vs gnubg):                 corr=%.3f   bias=%+.3f\n",
            overall.corr, overall.bias)
    println()
    c = overall.corr; b = overall.bias
    if c <= 0.85 && b >= 0.4
        println(">>> TEACHER PROBLEM (hypothesis A). wildbg's OWN equity is poorly calibrated vs")
        println("    gnubg: low correlation AND a large positive (over-optimistic) bias that closely")
        println("    mirrors the net's own corr≈0.79 / bias≈+0.72. The net did NOT drift — it")
        println("    faithfully learned a miscalibrated target. FIX for the cubeful value head: use")
        println("    gnubg (or rollouts) as the value teacher, NOT wildbg per-position equity.")
    elseif c >= 0.9 && abs(b) <= 0.2
        println(">>> DRIFT PROBLEM (hypothesis B). wildbg's equity tracks gnubg well (high corr, near-")
        println("    zero bias) — it is a GOOD teacher. So the net's corr≈0.79 / +0.72 over-optimism")
        println("    was NOT inherited; it was ADDED during self-play. FIX: anchor the self-play value")
        println("    target to a strong reference (teacher/rollouts) instead of pure bootstrap.")
    else
        frac = overall.bias / 0.72
        println(">>> MIXED (both contribute). wildbg is an imperfect teacher but does not fully explain")
        @printf("    the net's miscalibration. The teacher accounts for ~%.0f%% of the +0.72 bias\n", 100*clamp(frac,0,1))
        @printf("    (wildbg carries %+.3f of it) and corr=%.3f (net 0.79); the remainder is self-play\n", b, c)
        println("    drift. FIX: BOTH — switch to a better value teacher AND anchor self-play value.")
    end
    @printf("\nNumbers:  corr=%.3f  bias=%+.3f  MAE=%.3f  RMSE=%.3f  mean_wb=%+.3f  mean_gb=%+.3f  (n=%d)\n",
            overall.corr, overall.bias, overall.mae, overall.rmse, overall.mean_wb, overall.mean_gb, overall.n)
    println("=" ^ 78); flush(stdout)
end

main()
