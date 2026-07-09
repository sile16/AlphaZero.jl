#!/usr/bin/env julia
"""
build_gnubg_afterstate_labels.jl — EXP: relabel pre-roll afterstates with a STRONG
gnubg teacher (replacing the weak wildbg per-move sibling labels) + oversample the
catastrophic tail classes, to push the afterstate-value net BELOW wildbg toward gnubg.

WHY gnubg move_data (not evaluate_probs per afterstate):
  Calibration: evaluate_probs = 10 ms/afterstate (serial, C-locked) → 108 min for 637k.
  gnubg _gnubg_clib_move_data(parent) evaluates a WHOLE sibling set in ONE C call
  = 23 ms/decision → ~11 min for 30k decisions (~10x faster).
  Verified: flip(move_data mover-probs) == evaluate_probs(afterstate) to mean 1e-2,
  and evaluate_probs-argmax == gnubg move-list grader argmax on 98% of decisions
  (corr 0.987) → gnubg-ply1 labels are IN THE GRADING CURRENCY.

LABEL (per candidate move m of a decision, parent P, mover = P.current_player):
  gnubg move_data(P) → (resulting_board, mover_probs) at chosen ply. The afterstate A_m
  has the OPPONENT to move; the net's label contract is opponent-perspective 5-head.
  Perspective flip (mover→opponent), exact & verified:
      opp = (1-pw, pgl, pbgl, pgw, pbgw)
  Matched to my enumerated full-turn afterstates BY RESULTING BOARD (sidesteps the
  gnubg player-0 action-id bug — no action-id pairing is used).

POSITIONS: fresh gnubg-0ply self-play (seed distinct from the eval benchmark seed=1;
  diagnostics show distribution-shift ≈ 0, so any reasonable contact distribution is
  fine). Each decision keeps its full candidate-move (sibling) set.

OVERSAMPLE (diagnostic: top-10% of decisions carry 60% of equity loss; worst classes
  BEAR_IN_VS_CONTACT PR≈40, then DEEP_ANCHOR, then back-checker middlegame):
  physically duplicate tail-class decisions (fresh decision ids) so the SAME validated
  finetune script trains on an up-weighted tail. Multipliers below.

OUTPUT: a NamedTuple .jls in the EXACT format finetune_preroll_afterstate.jl consumes
  (sib_states, sib_labels, sib_dec, ret_states, ret_labels, meta) — drop-in.

Retention (post-roll anti-forgetting): reuse the existing wildbg .jls ret_states but
  RELABEL them with gnubg evaluate_probs → fully-gnubg dataset.

Checkpoints every --checkpoint-every decisions to <out>.ckpt (crash-safe; --resume).

Usage:
  julia --project scripts/build_gnubg_afterstate_labels.jl \\
     --n-decisions 40000 --seed 2 --gnubg-ply 1 \\
     --out sessions/preroll-afterstate/gnubg_afterstates_40k.jls
"""

using ArgParse
function parse_args_b()
    s = ArgParseSettings(autofix_names=true)
    @add_arg_table! s begin
        "--n-decisions";     arg_type=Int; default=40000   # base (pre-oversample) contact decisions w/ >=2 siblings
        "--seed";            arg_type=Int; default=2
        "--gnubg-ply";       arg_type=Int; default=1
        "--gen-ply";         arg_type=Int; default=0        # self-play generator strength
        "--out";             arg_type=String; default="sessions/preroll-afterstate/gnubg_afterstates_40k.jls"
        "--ret-source";      arg_type=String; default="sessions/preroll-afterstate/sibling_afterstates_30k.jls"
        "--n-retention";     arg_type=Int; default=30000
        "--checkpoint-every";arg_type=Int; default=5000
        "--max-games";       arg_type=Int; default=2_000_000
        "--obs-type";        arg_type=String; default="min_plus_flat"
        "--no-oversample";   action=:store_true
        "--resume";          action=:store_true
    end
    return ArgParse.parse_args(s)
end
const A = parse_args_b()
ENV["BACKGAMMON_OBS_TYPE"] = A["obs_type"]

using AlphaZero
using AlphaZero: GI
using BackgammonNet
using Serialization, Random, Statistics, Printf

include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
const OBS_SYM = Symbol(A["obs_type"])

# ── perspective flip: mover-probs (move_data) → opponent-to-move afterstate label ──
@inline flip5(p) = (Float32(1.0-p[1]), Float32(p[4]), Float32(p[5]), Float32(p[2]), Float32(p[3]))
@inline eqjoint5(p) = (2f0*p[1]-1f0)+(p[2]-p[4])+(p[3]-p[5])

# ── enumerate full-turn afterstates (verbatim from afterstate_value_pr.jl) ──
function enumerate_afters(P::BackgammonNet.BackgammonGame)
    start_player = P.current_player
    seen = Dict{Tuple{UInt128,UInt128}, BackgammonNet.BackgammonGame}()
    stack = BackgammonNet.BackgammonGame[BackgammonNet.clone(P)]
    while !isempty(stack)
        s = pop!(stack)
        att = BackgammonNet.action_type(s)
        done = (att == BackgammonNet.ACTION_TYPE_TERMINAL || s.current_player != start_player ||
                att == BackgammonNet.ACTION_TYPE_CHANCE)
        if done; key=(s.p0,s.p1); haskey(seen,key)||(seen[key]=s); continue; end
        acts = BackgammonNet.legal_actions(s)
        if isempty(acts); key=(s.p0,s.p1); haskey(seen,key)||(seen[key]=s); continue; end
        for a in acts
            s2 = BackgammonNet.clone(s); BackgammonNet.apply_action!(s2, a); push!(stack, s2)
        end
    end
    return collect(values(seen))
end

function _sample_dice(rng)
    r = rand(rng, Float32); c = 0.0f0
    @inbounds for i in 1:length(BackgammonNet.DICE_PROBS)
        c += BackgammonNet.DICE_PROBS[i]; r <= c && return i
    end
    return length(BackgammonNet.DICE_PROBS)
end
function play_turn!(g, engine, fail::Base.RefValue{Int})
    sp = g.current_player
    while true
        att = BackgammonNet.action_type(g)
        (att==BackgammonNet.ACTION_TYPE_TERMINAL||att==BackgammonNet.ACTION_TYPE_CHANCE)&&break
        g.current_player != sp && break
        acts = BackgammonNet.legal_actions(g); isempty(acts)&&break
        a = length(acts)==1 ? acts[1] : (try BackgammonNet.best_move(engine,g) catch; fail[]+=1; acts[1] end)
        BackgammonNet.apply_action!(g, a); g.terminated&&break
    end
    g
end

# ── position-class heuristics (mover = P.current_player perspective) ──────────
# gnubg simple board m (perspective=mover): m[1]=mover bar, m[2:25]=points 1-24
# (positive=mover, negative=opp), m[26]=opp bar. Mover bears off from points 1-6,
# back checkers live on points 19-24, opponent's home = points 19-24.
#   BEAR_IN_VS_CONTACT : mover mostly home/off (>=11 on pts 1-6 + borne off) yet still
#                        CONTACT (opp has back presence) — worst class, PR≈40.
#   DEEP_ANCHOR        : mover holds a deep anchor (>=2 on pts 22-24) OR opp holds a
#                        deep anchor vs mover (>=2 on mover pts 1-4).
#   BACK_CHECKER_MID   : mover still has >=2 back checkers (pts 19-24), holding/prime.
#   CONTACT_GENERIC    : everything else.
function classify(P::BackgammonNet.BackgammonGame)
    cp = Int(P.current_player)
    m = BackgammonNet.to_gnubg_simple(P; perspective=cp)
    mover_home = 0; mover_on = max(m[1],0); mover_backdeep = 0; mover_back = 0
    opp_anchor = false
    @inbounds for pt in 1:24
        v = m[pt+1]
        if v > 0
            mover_on += v
            (1 <= pt <= 6)  && (mover_home += v)
            (22 <= pt <= 24)&& (mover_backdeep += v)
            (19 <= pt <= 24)&& (mover_back += v)
        elseif v <= -2 && (1 <= pt <= 4)
            opp_anchor = true
        end
    end
    mover_off = 15 - mover_on
    if mover_home + mover_off >= 11
        return :BEAR_IN_VS_CONTACT
    elseif mover_backdeep >= 2 || opp_anchor
        return :DEEP_ANCHOR
    elseif mover_back >= 2
        return :BACK_CHECKER_MID
    else
        return :CONTACT_GENERIC
    end
end

# oversample multipliers (physical decision duplication)
const OVERSAMPLE = Dict(:BEAR_IN_VS_CONTACT=>3, :DEEP_ANCHOR=>2, :BACK_CHECKER_MID=>2,
                        :CONTACT_GENERIC=>1)

struct DecRec
    afters::Vector{BackgammonNet.BackgammonGame}
    labels::Vector{NTuple{5,Float32}}
    class::Symbol
end

function main()
    Random.seed!(A["seed"])
    ckpt = A["out"] * ".ckpt"
    println("="^80); println("BUILD GNUBG AFTERSTATE LABELS"); println("="^80)
    @printf("target base decisions=%d  gnubg-ply=%d  gen-ply=%d  seed=%d  obs=%s\n",
            A["n_decisions"], A["gnubg_ply"], A["gen_ply"], A["seed"], A["obs_type"])
    @printf("out=%s\n", A["out"]); flush(stdout)

    recs = DecRec[]
    gi_start = 0
    if A["resume"] && isfile(ckpt)
        C = deserialize(ckpt)
        recs = C.recs; gi_start = C.gi
        @printf("RESUME: loaded %d decisions from ckpt (gi=%d)\n", length(recs), gi_start); flush(stdout)
    end

    labeler = BackgammonNet.GnubgCLibBackend(ply=A["gnubg_ply"], threads=1); BackgammonNet.open!(labeler)
    gen     = BackgammonNet.GnubgCLibBackend(ply=A["gen_ply"], threads=1);   BackgammonNet.open!(gen)

    fail = Ref(0); t0 = time(); gi = gi_start
    n_forced_skip = 0; n_lowsib_skip = 0; last_ck = length(recs)
    while length(recs) < A["n_decisions"] && gi < A["max_games"]
        gi += 1
        r = MersenneTwister(A["seed"]*1_000_003 + gi)
        g = BackgammonNet.initial_state(obs_type=OBS_SYM)
        while true
            at = BackgammonNet.action_type(g)
            at == BackgammonNet.ACTION_TYPE_TERMINAL && break
            if at == BackgammonNet.ACTION_TYPE_CHANCE
                BackgammonNet.apply_chance!(g, _sample_dice(r)); continue
            end
            if BackgammonNet.is_contact_position(g) && g.phase == BackgammonNet.PHASE_CHECKER_PLAY &&
               length(recs) < A["n_decisions"]
                P = BackgammonNet.clone(g)
                # gnubg move-list (one C call → whole sibling set)
                md = lock(BackgammonNet._GNUBG_CLIB_LOCK) do
                    BackgammonNet._gnubg_clib_move_data(labeler, P)
                end
                if length(md) >= 2
                    player = Int(P.current_player)
                    mdprobs = Dict{Tuple{UInt128,UInt128},NTuple{5,Float64}}()
                    for (tsimple, probs) in md
                        tp0,tp1 = BackgammonNet.from_gnubg_simple(tsimple, player)
                        mdprobs[(tp0,tp1)] = probs
                    end
                    afters = enumerate_afters(P)
                    ast = BackgammonNet.BackgammonGame[]; lbl = NTuple{5,Float32}[]
                    for Afs in afters
                        k = (Afs.p0, Afs.p1)
                        haskey(mdprobs, k) || continue
                        push!(ast, Afs); push!(lbl, flip5(mdprobs[k]))
                    end
                    if length(ast) >= 2
                        push!(recs, DecRec(ast, lbl, classify(P)))
                    else
                        n_lowsib_skip += 1
                    end
                else
                    n_forced_skip += 1
                end
            end
            play_turn!(g, gen, fail); g.terminated && break
            length(recs) >= A["n_decisions"] && break
        end
        if length(recs) - last_ck >= A["checkpoint_every"]
            serialize(ckpt, (recs=recs, gi=gi)); last_ck = length(recs)
            el = time()-t0
            @printf("  [ckpt] decisions=%d  games=%d  %.1f min  %.1f dec/s  (forced_skip=%d lowsib=%d)\n",
                    length(recs), gi, el/60, length(recs)/max(el,1e-9), n_forced_skip, n_lowsib_skip); flush(stdout)
        end
    end
    serialize(ckpt, (recs=recs, gi=gi))
    @printf("\nLABELING DONE: %d decisions, %d games, %.1f min (gen fallbacks=%d)\n",
            length(recs), gi, (time()-t0)/60, fail[]); flush(stdout)

    # ── class distribution ──
    cls = [r.class for r in recs]
    classes = [:BEAR_IN_VS_CONTACT,:DEEP_ANCHOR,:BACK_CHECKER_MID,:CONTACT_GENERIC]
    println("\nCLASS DISTRIBUTION (base):")
    for c in classes
        n = count(==(c), cls); @printf("  %-20s %6d  (%.1f%%)\n", c, n, 100n/length(recs))
    end

    # ── oversample by physical decision duplication ──
    do_os = !A["no_oversample"]
    order = collect(1:length(recs))
    if do_os
        dup = Int[]
        for (i,r) in enumerate(recs)
            for _ in 1:(OVERSAMPLE[r.class]-1); push!(dup, i); end
        end
        append!(order, dup)
        shuffle!(order)
        println("\nOVERSAMPLE: base=$(length(recs)) -> effective=$(length(order)) decisions")
        eff = [recs[i].class for i in order]
        for c in classes
            n = count(==(c), eff); @printf("  %-20s %6d  (%.1f%%)\n", c, n, 100n/length(order))
        end
    else
        println("\nOVERSAMPLE disabled.")
    end

    # ── flatten to finetune NamedTuple format ──
    sib_states = BackgammonNet.BackgammonGame[]
    sib_labels = NTuple{5,Float32}[]
    sib_dec    = Int32[]
    did = Int32(0)
    for i in order
        did += Int32(1); r = recs[i]
        for k in 1:length(r.afters)
            push!(sib_states, r.afters[k]); push!(sib_labels, r.labels[k]); push!(sib_dec, did)
        end
    end
    @printf("\nFLATTENED: %d siblings across %d (effective) decisions\n", length(sib_states), did)

    # ── retention: reuse existing ret_states, RELABEL with gnubg evaluate_probs ──
    ret_states = BackgammonNet.BackgammonGame[]; ret_labels = NTuple{5,Float32}[]
    if isfile(A["ret_source"]) && A["n_retention"] > 0
        print("retention: relabeling from $(A["ret_source"]) ... "); flush(stdout); tr=time()
        RS = deserialize(A["ret_source"])
        src = RS.ret_states
        nR = min(A["n_retention"], length(src))
        for j in 1:nR
            s = src[j]
            p = BackgammonNet.evaluate_probs(labeler, s)
            push!(ret_states, s)
            push!(ret_labels, (Float32(p[1]),Float32(p[2]),Float32(p[3]),Float32(p[4]),Float32(p[5])))
        end
        @printf("%d states relabeled (%.1f min)\n", nR, (time()-tr)/60); flush(stdout)
    else
        println("retention: none (ret_source missing or n_retention=0)")
    end

    # opening pre-roll gnubg equity (sanity anchor, teacher perspective)
    open_state = BackgammonNet.initial_state(obs_type=OBS_SYM)
    op = BackgammonNet.evaluate_probs(labeler, open_state)
    @printf("opening pre-roll gnubg equity = %+.4f (5-head=%.3f,%.3f,%.3f,%.3f,%.3f)\n", eqjoint5(op), op...)

    meta = (source="gnubg-move_data selfplay", teacher=:gnubg, gnubg_ply=A["gnubg_ply"],
            gen_ply=A["gen_ply"], n_base_decisions=length(recs), n_eff_decisions=Int(did),
            n_sibling=length(sib_states), n_retention=length(ret_states),
            oversample=do_os, seed=A["seed"], obs_type=A["obs_type"],
            open_preroll_equity=Float64(eqjoint5(op)),
            class_base=Dict(c=>count(==(c),cls) for c in classes))

    out = A["out"]; mkpath(dirname(out))
    serialize(out, (sib_states=sib_states, sib_labels=sib_labels, sib_dec=sib_dec,
                    ret_states=ret_states, ret_labels=ret_labels, meta=meta))
    @printf("\nSaved → %s\n", out)
    println("meta = ", meta)
    println("="^80)
    try BackgammonNet.close(labeler) catch end; try BackgammonNet.close(gen) catch end
end

main()
