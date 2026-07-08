#!/usr/bin/env julia
"""
diag_value_drift.jl — CLEAN self-play VALUE-head over-optimism drift.

QUESTION
  Our contact net's VALUE head is over-optimistic vs true (GNUbg) equity. A prior
  measurement gave:
     * i140 (post-self-play):     corr 0.79 / bias +0.72   — on i140's OWN game positions
     * imitation (pre-self-play): corr 0.58 / bias +0.24   — on ITS OWN game positions
  Those two numbers are NOT comparable: each net was measured on a DIFFERENT
  position distribution (its own self-play games), so the difference conflates
  "value drift" with "which positions each net happens to visit". That is a
  position-distribution CONFOUND.

  The teacher (wildbg) is well-calibrated vs gnubg (0.957 corr / +0.04 bias on
  common positions, per scripts/diag_teacher_calibration.jl), so any large
  over-optimism is NOT inherited from the teacher — it was added by self-play.

  This script resolves the confound the METHODOLOGICALLY-CORRECT way: it evaluates
  BOTH nets' raw VALUE HEAD on the SAME fixed set of common contact positions, and
  compares each to gnubg equity on those identical positions. The clean drift is
     DRIFT = bias(i140) − bias(imitation)      [measured on identical positions]
  i.e. how much over-optimistic bias self-play ADDED, with the position
  distribution held constant.

METHOD (reuses scripts/diag_teacher_calibration.jl machinery)
  1. Generate ~N COMMON contact decision positions: gnubg-0ply moves BOTH sides
     with seeded dice, snapshotting the PRE-MOVE state at the start of every
     CONTACT checker turn (race/bearoff skipped — race is table-exact, not the
     concern). Deterministic in --seed. Both players-to-move are represented.
  2. For each position compute THREE mover-relative point-equities:
       (a) imitation net value scalar  (raw value head, NO search)
       (b) i140 net value scalar        (raw value head, NO search)
       (c) gnubg ply-1 equity           (evaluate(GnubgCLibBackend(ply=1), g))
     The NN value scalar is the joint-cumulative 5-head equity
         (2*p_win-1) + (p_wg-p_lg) + (p_wbg-p_lbg)
     via BackgammonNet.compute_cubeless_equity (rule-aware, mover perspective) —
     the SAME convention gnubg.evaluate returns and eval_vs_gnubg.jl uses (cube
     disabled → :auto search_value reduces to this cubeless equity). NO search:
     the value head is read directly off ONE forward pass, not MCTS-backed.
  3. Report, on the identical positions:
       * imitation vs gnubg:  corr, bias=mean(nn-gb), MAE
       * i140      vs gnubg:  corr, bias,               MAE
       * CLEAN DRIFT:  bias(i140) − bias(imitation),  and the corr change.
       * player-to-move split (asymmetry check).
     Plus an ORIENTATION sanity check (clearly-won positions must be strongly
     positive for all three) to confirm perspective alignment before trusting it.

Usage:
    julia --threads 8 --project scripts/diag_value_drift.jl [--n 2000 --seed 1]

Options:
    --n=2000                 target COMMON contact positions
    --seed=1                 base RNG seed (dice) — deterministic position set
    --gnubg-ply=1            gnubg reference ply (strong truth; 1 = validated)
    --num-workers=8          CPU worker threads for gnubg evaluation
    --obs-type=min_plus_flat network observation type (must match training)
    --width=256 --blocks=5   contact net architecture (both checkpoints)
    --race-width=128 --race-blocks=3   race net architecture (loaded but unused: contact only)
    --imit-ckpt=PATH         imitation (pre-self-play) contact checkpoint
    --i140-ckpt=PATH         i140 (post-self-play) contact checkpoint
    --max-games=100000       safety cap on generator games
    --decided-thresh=0.7     |gnubg equity| above which a position is clearly decided
"""

using ArgParse

function parse_drift_args()
    s = ArgParseSettings(description="Clean self-play value-head drift: both nets on identical positions vs gnubg", autofix_names=true)
    @add_arg_table! s begin
        "--n";               help = "Target COMMON contact positions"; arg_type = Int; default = 2000
        "--seed";            help = "Base RNG seed (dice) — deterministic position set"; arg_type = Int; default = 1
        "--gnubg-ply";       help = "gnubg reference ply (strong truth; 1 = validated)"; arg_type = Int; default = 1
        "--num-workers";     help = "CPU worker threads for gnubg evaluation"; arg_type = Int; default = 8
        "--obs-type";        help = "Network observation type"; arg_type = String; default = "min_plus_flat"
        "--width";           help = "Contact net width"; arg_type = Int; default = 256
        "--blocks";          help = "Contact net blocks"; arg_type = Int; default = 5
        "--race-width";      help = "Race net width"; arg_type = Int; default = 128
        "--race-blocks";     help = "Race net blocks"; arg_type = Int; default = 3
        "--imit-ckpt";       help = "Imitation (pre-self-play) contact checkpoint";
                             arg_type = String; default = "sessions/contact-imit-wbeq/checkpoints/contact_latest.data"
        "--i140-ckpt";       help = "i140 (post-self-play) contact checkpoint";
                             arg_type = String; default = "sessions/contact-flywheel/checkpoints/contact_iter_140.data"
        "--max-games";       help = "Safety cap on generator games"; arg_type = Int; default = 100_000
        "--decided-thresh";  help = "|gnubg equity| above which a position is clearly decided"; arg_type = Float64; default = 0.7
    end
    return ArgParse.parse_args(s)
end

const ARGS_D = parse_drift_args()

# Observation type must be set BEFORE including the game definition (it is baked in
# at include time via a const read from ENV).
ENV["BACKGAMMON_OBS_TYPE"] = ARGS_D["obs_type"]

using AlphaZero
using AlphaZero: GI, Network, FluxLib
import Flux
using BackgammonNet
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
const gspec = GameSpec()
const NUM_ACTIONS = GI.num_actions(gspec)
const STATE_DIM = GI.state_dim(gspec)[1]
const OBS_SYM = Symbol(ARGS_D["obs_type"])

# ── Fixed contact position: pre-move checker-play state + player to move ────────
struct DiagPosition
    state::BackgammonNet.BackgammonGame
    player::Int
end

# ── Dice sampling (verbatim from diag_teacher_calibration.jl) ────────────────────
function _sample_dice(rng)
    r = rand(rng, Float32); c = 0.0f0
    @inbounds for i in 1:length(BackgammonNet.DICE_PROBS)
        c += BackgammonNet.DICE_PROBS[i]
        r <= c && return i
    end
    return length(BackgammonNet.DICE_PROBS)
end

# ── Full-TURN execution by an engine's best_move ────────────────────────────────
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

# ── Generate COMMON contact positions: gnubg-0ply moves both sides, seeded dice ──
"""Deterministic in `base_seed`. All snapshots carry obs_type=OBS_SYM so the network
observation matches training. Only CONTACT pre-move states are kept."""
function generate_contact_positions(gen_engine, n_target::Int, base_seed::Int, max_games::Int)
    positions = DiagPosition[]
    fail = Threads.Atomic{Int}(0)
    gi = 0
    while length(positions) < n_target && gi < max_games
        gi += 1
        rng = MersenneTwister(base_seed + gi)
        g = BackgammonNet.initial_state(obs_type=OBS_SYM)
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

make_gnubg(ply::Int) = (e = BackgammonNet.GnubgCLibBackend(ply=ply, threads=1); BackgammonNet.open!(e); e)

# ── Load a contact network (CPU) ────────────────────────────────────────────────
function load_contact_net(path::String, width::Int, blocks::Int)
    isfile(path) || error("checkpoint not found: $path")
    net = FluxLib.FCResNetMultiHead(gspec, FluxLib.FCResNetMultiHeadHP(width=width, num_blocks=blocks))
    FluxLib.load_weights(path, net)
    return Flux.cpu(net)
end

@inline _sigmoid(x) = 1.0f0 / (1.0f0 + exp(-Float32(x)))

# ── Raw VALUE-HEAD scalar for every position (NO search) ────────────────────────
"""One forward pass per chunk through the contact net; convert the 5 joint-
cumulative value-head logits to a mover-perspective point equity via
compute_cubeless_equity (rule-aware; == (2pw-1)+(wg-lg)+(wbg-lbg) for these
cube-disabled money positions). Value heads depend only on the state trunk, so the
action mask A is irrelevant here (ones)."""
function net_value_scalars(net, positions::Vector{DiagPosition}; chunk::Int=512)
    n = length(positions)
    out = Vector{Float64}(undef, n)
    i = 1
    while i <= n
        j = min(i + chunk - 1, n)
        m = j - i + 1
        X = Matrix{Float32}(undef, STATE_DIM, m)
        A = ones(Float32, NUM_ACTIONS, m)
        @inbounds for k in 1:m
            vectorize_state_into!(view(X, :, k), gspec, positions[i + k - 1].state)
        end
        _, Lw, Lgw, Lbgw, Lgl, Lbgl, _ = FluxLib.forward_normalized_multihead(net, X, A)
        @inbounds for k in 1:m
            g = positions[i + k - 1].state
            heads = (_sigmoid(Lw[1, k]), _sigmoid(Lgw[1, k]), _sigmoid(Lbgw[1, k]),
                     _sigmoid(Lgl[1, k]), _sigmoid(Lbgl[1, k]))
            out[i + k - 1] = Float64(BackgammonNet.compute_cubeless_equity(g, heads))
        end
        i = j + 1
    end
    return out
end

# ── gnubg equity on every position (parallel across workers) ────────────────────
function gnubg_eval(positions::Vector{DiagPosition}, num_workers::Int, gnubg_ply::Int)
    n = length(positions)
    gb = fill(NaN, n)
    okflag = fill(false, n)
    idx = Threads.Atomic{Int}(0)
    n_fail = Threads.Atomic{Int}(0)
    Threads.@threads for _ in 1:num_workers
        gengine = make_gnubg(gnubg_ply)
        try
            while true
                i = Threads.atomic_add!(idx, 1) + 1
                i > n && break
                try
                    v = BackgammonNet.evaluate(gengine, positions[i].state)
                    gb[i] = v
                    okflag[i] = isfinite(v)
                catch e
                    Threads.atomic_add!(n_fail, 1)
                    @warn "gnubg eval failed on position $i; skipping" exception=(e,) maxlog=5
                end
            end
        finally
            try BackgammonNet.close(gengine) catch end
        end
    end
    return gb, okflag, n_fail[]
end

# ── Calibration statistics (nn vs gb) ───────────────────────────────────────────
function calib_stats(nn::Vector{Float64}, gb::Vector{Float64})
    n = length(nn)
    n == 0 && return nothing
    d = nn .- gb
    corr = n >= 2 ? Statistics.cor(nn, gb) : NaN
    return (n=n, corr=corr,
            bias=Statistics.mean(d),              # mean(nn - gb): + => net over-optimistic
            mae=Statistics.mean(abs.(d)),
            rmse=sqrt(Statistics.mean(d .^ 2)),
            mean_nn=Statistics.mean(nn),
            mean_gb=Statistics.mean(gb),
            std_nn=n >= 2 ? Statistics.std(nn) : NaN,
            std_gb=n >= 2 ? Statistics.std(gb) : NaN)
end

function print_stats(label, s)
    s === nothing && (println("  $label: (no data)"); return)
    @printf("  %-24s n=%5d  corr=%6.3f  bias(nn-gb)=%+6.3f  MAE=%5.3f  RMSE=%5.3f\n",
            label, s.n, s.corr, s.bias, s.mae, s.rmse)
    @printf("  %-24s mean_nn=%+6.3f  mean_gb=%+6.3f   std_nn=%5.3f  std_gb=%5.3f\n",
            "", s.mean_nn, s.mean_gb, s.std_nn, s.std_gb)
end

# ── Main ────────────────────────────────────────────────────────────────────────
function main()
    n_target    = ARGS_D["n"]
    base_seed   = ARGS_D["seed"]
    gnubg_ply   = ARGS_D["gnubg_ply"]
    num_workers = ARGS_D["num_workers"]
    max_games   = ARGS_D["max_games"]
    decided_thr = ARGS_D["decided_thresh"]
    imit_ckpt   = ARGS_D["imit_ckpt"]
    i140_ckpt   = ARGS_D["i140_ckpt"]

    gnubg_ply >= 2 && error("gnubg reference ply=$gnubg_ply refused (2-ply deadlocks). Use --gnubg-ply 1 or 0.")

    println("=" ^ 82)
    println("CLEAN VALUE-HEAD DRIFT — both nets' raw value head on IDENTICAL positions vs gnubg")
    println("=" ^ 82)
    println("Positions:   $n_target COMMON contact pre-move decisions (gnubg-0ply generator, both players)")
    println("Obs type:    $(ARGS_D["obs_type"])   state_dim=$STATE_DIM   num_actions=$NUM_ACTIONS")
    println("Net value:   raw joint-cumulative 5-head equity (compute_cubeless_equity), NO search")
    println("gnubg:       GnubgCLibBackend(ply=$gnubg_ply)   [strong reference / 'truth']")
    println("imitation:   $imit_ckpt")
    println("i140:        $i140_ckpt")
    println("Comparison:  all mover-perspective points → directly comparable.  bias=mean(nn-gb)")
    println("Seed:        $base_seed  (deterministic)     Workers: $num_workers")
    println("=" ^ 82); flush(stdout)

    # ── Generate the common contact-position set ──
    println("\nGENERATING common contact positions (gnubg-0ply moves both sides, seeded dice)...")
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

    # ── Net value heads on the identical positions ──
    println("\nEVALUATING imitation-net value head (raw, no search)...")
    flush(stdout); t1 = time()
    imit_net = load_contact_net(imit_ckpt, ARGS_D["width"], ARGS_D["blocks"])
    imit_v = net_value_scalars(imit_net, positions)
    println("  done ($(round(time()-t1,digits=1))s)")

    println("EVALUATING i140-net value head (raw, no search)...")
    flush(stdout); t2 = time()
    i140_net = load_contact_net(i140_ckpt, ARGS_D["width"], ARGS_D["blocks"])
    i140_v = net_value_scalars(i140_net, positions)
    println("  done ($(round(time()-t2,digits=1))s)")

    # ── gnubg equity on the identical positions ──
    println("EVALUATING gnubg-ply-$gnubg_ply equity...")
    flush(stdout); t3 = time()
    gb, okflag, eval_fail = gnubg_eval(positions, num_workers, gnubg_ply)
    println("  done ($(round(time()-t3,digits=1))s)" * (eval_fail > 0 ? "  [gnubg failures skipped: $eval_fail]" : ""))
    flush(stdout)

    keep = okflag .& isfinite.(imit_v) .& isfinite.(i140_v)
    imk = imit_v[keep]; i1k = i140_v[keep]; gbk = gb[keep]
    pk = [positions[i].player for i in 1:length(positions)][keep]

    imit_s = calib_stats(imk, gbk)
    i140_s = calib_stats(i1k, gbk)

    # ── Report ──
    println("\n" * "=" ^ 82)
    println("VALUE-HEAD vs gnubg-ply-$gnubg_ply — SAME $(imit_s.n) positions for BOTH nets")
    println("=" ^ 82)
    print_stats("imitation (pre-self-play)", imit_s)
    println()
    print_stats("i140 (post-self-play)", i140_s)

    println("\nPlayer-to-move split (asymmetry check):")
    println("  imitation:")
    print_stats("  P0", calib_stats(imk[pk .== 0], gbk[pk .== 0]))
    print_stats("  P1", calib_stats(imk[pk .== 1], gbk[pk .== 1]))
    println("  i140:")
    print_stats("  P0", calib_stats(i1k[pk .== 0], gbk[pk .== 0]))
    print_stats("  P1", calib_stats(i1k[pk .== 1], gbk[pk .== 1]))

    # ── Orientation sanity check ──
    println("\n" * "-" ^ 82)
    println("ORIENTATION SANITY CHECK — clearly-decided positions (|gnubg equity| > $decided_thr)")
    println("-" ^ 82)
    dec = abs.(gbk) .> decided_thr
    ndec = count(dec)
    if ndec > 0
        agree_im = count(sign.(imk[dec]) .== sign.(gbk[dec])) / ndec
        agree_i1 = count(sign.(i1k[dec]) .== sign.(gbk[dec])) / ndec
        @printf("  decided n=%d   SIGN-agreement: imitation=%.1f%%  i140=%.1f%%\n",
                ndec, 100*agree_im, 100*agree_i1)
        order = sortperm(gbk[dec]); decidx = findall(dec)
        picks = unique(round.(Int, range(1, length(order); length=min(6, length(order)))))
        println("  sample (sorted by gnubg):     gnubg     imit      i140")
        for k in picks
            jj = decidx[order[k]]
            @printf("       gnubg=%+6.3f   imit=%+6.3f   i140=%+6.3f\n", gbk[jj], imk[jj], i1k[jj])
        end
        println((agree_im > 0.9 && agree_i1 > 0.9) ?
            "  => Signs agree on decided positions: comparison correctly ORIENTED (mover-relative)." :
            "  => WARNING: sign disagreement high — check perspective before trusting the verdict.")
    else
        println("  (no positions exceeded the decided threshold)")
    end

    # ── The clean drift ──
    drift_bias = i140_s.bias - imit_s.bias
    drift_corr = i140_s.corr - imit_s.corr
    drift_mae  = i140_s.mae  - imit_s.mae
    println("\n" * "=" ^ 82)
    println("CLEAN DRIFT — over-optimism ADDED by self-play (position distribution held constant)")
    println("=" ^ 82)
    @printf("  imitation bias(nn-gb) = %+.3f     i140 bias(nn-gb) = %+.3f\n", imit_s.bias, i140_s.bias)
    @printf("  >>> CLEAN DRIFT (i140_bias - imitation_bias) = %+.3f  points  <<<\n", drift_bias)
    @printf("      corr change  (i140 - imitation) = %+.3f   (imit %.3f -> i140 %.3f)\n",
            drift_corr, imit_s.corr, i140_s.corr)
    @printf("      MAE  change  (i140 - imitation) = %+.3f   (imit %.3f -> i140 %.3f)\n",
            drift_mae, imit_s.mae, i140_s.mae)
    println()
    println("INTERPRETATION:")
    if drift_bias >= 0.15
        @printf("  Self-play CLEANLY ADDED over-optimistic value bias: +%.3f points on identical\n", drift_bias)
        println("  positions. This is the confound-free drift — it is NOT a position-distribution")
        println("  artifact, because both nets were scored on the very same positions.")
        if drift_corr < -0.02
            println("  Correlation also DEGRADED, so self-play made the value head both more biased")
            println("  AND worse-ranked vs true equity.")
        elseif drift_corr > 0.02
            println("  Correlation IMPROVED even as bias grew: self-play sharpened the ranking but")
            println("  pushed the whole value scale up (systematic over-optimism / mis-calibration).")
        else
            println("  Correlation essentially unchanged: the drift is almost purely a bias/offset")
            println("  shift, not a ranking-quality change.")
        end
        println("  FIX implication: anchor the self-play value target to a strong reference")
        println("  (teacher/rollouts/gnubg) rather than pure bootstrap, or recalibrate the offset.")
    elseif drift_bias <= -0.15
        @printf("  Self-play REDUCED bias by %.3f on identical positions — over-optimism did NOT\n", -drift_bias)
        println("  grow during self-play. The prior 'i140 more biased' reading was a position-")
        println("  distribution artifact (i140 simply visits more optimistic-looking positions).")
    else
        @printf("  Bias barely moved (%+.3f) on identical positions. Self-play did NOT cleanly add\n", drift_bias)
        println("  over-optimism; the prior gap between i140 (+0.72) and imitation (+0.24) was")
        println("  largely a POSITION-DISTRIBUTION confound, not true value drift.")
    end
    @printf("\nNumbers:  imit(corr=%.3f bias=%+.3f MAE=%.3f)  i140(corr=%.3f bias=%+.3f MAE=%.3f)  n=%d\n",
            imit_s.corr, imit_s.bias, imit_s.mae, i140_s.corr, i140_s.bias, i140_s.mae, imit_s.n)
    println("=" ^ 82); flush(stdout)
end

main()
