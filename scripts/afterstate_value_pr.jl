#!/usr/bin/env julia
"""
afterstate_value_pr.jl — DECISIVE experiment: does AFTERSTATE-VALUE move selection
with the EXISTING i140 value head beat the full policy+MCTS-800 pipeline?

HYPOTHESIS (expert consult): our weak play is caused by the POLICY HEAD (it ranks
676 action-ids without ever seeing the RESULTING position, so it cannot generalize).
Every strong backgammon bot instead scores each legal move by evaluating its
RESULTING position (afterstate) with the VALUE net and ignores the policy head.

TEST: on the SAME fixed benchmark positions used for the baselines
  - i140 raw policy (mcts-1)           = 51.7 PR   (contact, gnubg-ply1 native ref)
  - i140 mcts-800 (full pipeline)      = 32.6 PR
  - wildbg-large ~14, gnubg-0ply ~2, floor ~1
does greedy 0-ply afterstate-value selection with the i140 VALUE head do better?

METHOD (per benchmark decision: pre-move position P, player to move, dice rolled)
  1. Enumerate every legal FULL-TURN move → its resulting board / afterstate A_m
     (opponent now to move, PRE-ROLL chance node). DFS over the 676-half-move tree,
     deduped by resulting (p0,p1) — identical to what gnubg's native list enumerates.
  2. Score(m) = MY equity after m = -(opponent's equity at A_m).  Two honest ways:
     (A) 0-ply-A  : evaluate A_m DIRECTLY with the value net (pre-roll, dice unset).
                    OOD (net trained post-roll) — validated by opening-eq≈0 sanity.
     (B) 0-ply-B  : E over the opponent's 21 rolls r of V(A_m + r), the net's value
                    of the opponent-to-move POST-ROLL decision position. Matches the
                    net's training distribution → the honest 0-ply afterstate score.
     For both: V(state) = compute_cubeless_equity(state, net(state)) is the MOVER's
     (=opponent's) equity, so my score is -that. Routed contact/race by board.
  3. Greedy move = argmax_m Score(m). Grade THAT move's resulting board with the
     gnubg-ply1 NATIVE move-list reference, matched BY BOARD — the SAME grader as the
     baselines (native_regret verbatim from analyze_pr_native.jl / benchmark_pr.jl).

SIGN / PERSPECTIVE sanity built in:
  - opening pre-roll direct eval (0-ply-A) must be ≈ 0.
  - WORST-scored move PR (argmin) is graded too: if the sign were flipped, the
    "greedy" move would be the worst → its PR would be tiny and the worst huge. A
    correct sign ⇒ greedy PR ≪ worst PR. Reported.
  - a random-legal move PR is graded as an additional contrast.

Positions: regenerated with benchmark_pr.jl's EXACT seed-1 gnubg-0ply generator
(deterministic → identical to the fixed benchmark set behind the baselines).

Usage:
    julia --threads 16 --project scripts/afterstate_value_pr.jl \\
        --n-positions 1000 --seed 1 --num-workers 16
Options:
    --n-positions=1000  --seed=1  --num-workers=16  --gnubg-ply=1
    --net-ckpt=sessions/contact-flywheel/checkpoints/contact_iter_140.data
    --race-ckpt=sessions/race-supervised-v2/checkpoints/race_train_latest.data
    --obs-type=min_plus_flat --width=256 --blocks=5 --race-width=128 --race-blocks=3
    --two-ply / --no-two-ply   --two-ply-topk=8   --two-ply-positions=0 (0=all)
"""

using ArgParse

function parse_args_av()
    s = ArgParseSettings(description="Afterstate-value move-selection PR vs policy+MCTS baselines", autofix_names=true)
    @add_arg_table! s begin
        "--n-positions";  arg_type=Int; default=1000
        "--seed";         arg_type=Int; default=1
        "--num-workers";  arg_type=Int; default=12
        "--gnubg-ply";    arg_type=Int; default=1
        "--net-ckpt";     arg_type=String; default="sessions/contact-flywheel/checkpoints/contact_iter_140.data"
        "--race-ckpt";    arg_type=String; default="sessions/race-supervised-v2/checkpoints/race_train_latest.data"
        "--obs-type";     arg_type=String; default="min_plus_flat"
        "--width";        arg_type=Int; default=256
        "--blocks";       arg_type=Int; default=5
        "--race-width";   arg_type=Int; default=128
        "--race-blocks";  arg_type=Int; default=3
        "--max-games";    arg_type=Int; default=100_000
        "--two-ply";      action=:store_true
        "--two-ply-topk"; arg_type=Int; default=8
        "--two-ply-positions"; arg_type=Int; default=0   # 0 = all
    end
    return ArgParse.parse_args(s)
end
const A = parse_args_av()

ENV["BACKGAMMON_OBS_TYPE"] = A["obs_type"]

using AlphaZero
using AlphaZero: GI, FluxLib
import Flux
using BackgammonNet
using Random
using Statistics
using Printf
import LinearAlgebra

include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
const gspec = GameSpec()
const NUM_ACTIONS = GI.num_actions(gspec)
const STATE_DIM = GI.state_dim(gspec)[1]
const OBS_SYM = Symbol(A["obs_type"])

@inline _sigmoid(x) = 1.0f0 / (1.0f0 + exp(-Float32(x)))

# ── graded unit: a full TURN (verbatim contract) ─────────────────────────────
struct MoveDecision
    state::BackgammonNet.BackgammonGame
    res_p0::UInt128
    res_p1::UInt128
    is_contact::Bool
    player::Int
end
struct BenchPosition
    state::BackgammonNet.BackgammonGame
    is_contact::Bool
    player::Int
end

# ── native gnubg move-list grading (verbatim from analyze_pr_native.jl) ───────
function native_regret(gnubg::BackgammonNet.GnubgCLibBackend, d::MoveDecision)
    g = d.state
    (g.phase == BackgammonNet.PHASE_CHECKER_PLAY) || return (:forced, 0.0, 0)
    BackgammonNet.open!(gnubg)
    md = lock(BackgammonNet._GNUBG_CLIB_LOCK) do
        BackgammonNet._gnubg_clib_move_data(gnubg, g)
    end
    n = length(md)
    n <= 1 && return (:forced, 0.0, n)
    player = Int(g.current_player)
    best_eq = -Inf
    our_eq = nothing
    @inbounds for (tsimple, probs) in md
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
                @warn "native_regret failed on decision $i" exception=(e, catch_backtrace()) maxlog=5
                status[i] = :error; continue
            end
            tag, err, _ = res
            status[i] = tag
            tag == :ok && (errors[i] = err)
        end
    end
    return errors, contact_flag, player_of, status
end

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
    return (n_ok=n_ok, n_forced=count(==(:forced), status),
            n_unmatched=count(==(:unmatched), status), n_error=count(==(:error), status),
            sum_err=sum(e_ok), mean_err=mean_err, ER=1000.0*mean_err, PR=500.0*mean_err,
            contact=subset(contact_flag), race=subset(.!contact_flag),
            p0=subset(player_of .== 0), p1=subset(player_of .== 1))
end

# ── dice sampling / turn execution (verbatim) ────────────────────────────────
function _sample_dice(rng)
    r = rand(rng, Float32); c = 0.0f0
    @inbounds for i in 1:length(BackgammonNet.DICE_PROBS)
        c += BackgammonNet.DICE_PROBS[i]
        r <= c && return i
    end
    return length(BackgammonNet.DICE_PROBS)
end
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
make_gnubg(ply::Int) = (e = BackgammonNet.GnubgCLibBackend(ply=ply, threads=1); BackgammonNet.open!(e); e)

# ── common benchmark positions (verbatim benchmark_pr.jl generator) ──────────
function generate_common_positions(gen_engine, n_contact::Int, base_seed::Int, max_games::Int)
    contact = BenchPosition[]
    fail = Threads.Atomic{Int}(0)
    gi = 0
    while length(contact) < n_contact && gi < max_games
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
                length(contact) < n_contact &&
                    push!(contact, BenchPosition(BackgammonNet.clone(g), true, Int(g.current_player)))
            end
            play_turn_with_engine!(g, gen_engine, fail)
            g.terminated && break
            length(contact) >= n_contact && break
        end
    end
    return contact, gi, fail[]
end

# ── enumerate every legal full-turn afterstate of P (deduped by board) ───────
"""DFS over half-moves; a full turn ends when current_player changes / terminal /
chance. Returns Vector{BackgammonGame} afterstates (opponent to move, pre-roll)
unique by resulting (p0,p1)."""
function enumerate_full_turn_afterstates(P::BackgammonNet.BackgammonGame)
    start_player = P.current_player
    seen = Dict{Tuple{UInt128,UInt128}, BackgammonNet.BackgammonGame}()
    stack = BackgammonNet.BackgammonGame[BackgammonNet.clone(P)]
    while !isempty(stack)
        s = pop!(stack)
        att = BackgammonNet.action_type(s)
        done = (att == BackgammonNet.ACTION_TYPE_TERMINAL || s.current_player != start_player ||
                att == BackgammonNet.ACTION_TYPE_CHANCE)
        if done
            key = (s.p0, s.p1); haskey(seen, key) || (seen[key] = s); continue
        end
        acts = BackgammonNet.legal_actions(s)
        if isempty(acts)
            key = (s.p0, s.p1); haskey(seen, key) || (seen[key] = s); continue
        end
        for a in acts
            s2 = BackgammonNet.clone(s)
            BackgammonNet.apply_action!(s2, a)
            push!(stack, s2)
        end
    end
    return collect(values(seen))
end

# ── batched value-net equity (mover perspective), routed contact/race ────────
"""V(state) = mover-perspective cubeless equity from the value head. Terminal
states get the exact terminal equity (mover just moved → previous mover won)."""
function value_equities!(out::Vector{Float64}, states::Vector{BackgammonNet.BackgammonGame},
                         idxs::Vector{Int}, contact_net, race_net; chunk::Int=1024)
    # split indices by contact/race, terminal handled by caller
    cidx = Int[]; ridx = Int[]
    for i in idxs
        BackgammonNet.is_contact_position(states[i]) ? push!(cidx, i) : push!(ridx, i)
    end
    _eval_group!(out, states, cidx, contact_net, chunk)
    _eval_group!(out, states, ridx, race_net, chunk)
    return out
end
function _eval_group!(out, states, group::Vector{Int}, net, chunk::Int)
    net === nothing && return
    n = length(group)
    p = 1
    while p <= n
        q = min(p + chunk - 1, n)
        m = q - p + 1
        X = Matrix{Float32}(undef, STATE_DIM, m)
        Amask = ones(Float32, NUM_ACTIONS, m)
        @inbounds for k in 1:m
            vectorize_state_into!(view(X, :, k), gspec, states[group[p + k - 1]])
        end
        _, Lw, Lgw, Lbgw, Lgl, Lbgl, _ = FluxLib.forward_normalized_multihead(net, X, Amask)
        @inbounds for k in 1:m
            gi = group[p + k - 1]
            heads = (_sigmoid(Lw[1,k]), _sigmoid(Lgw[1,k]), _sigmoid(Lbgw[1,k]),
                     _sigmoid(Lgl[1,k]), _sigmoid(Lbgl[1,k]))
            out[gi] = Float64(BackgammonNet.compute_cubeless_equity(states[gi], heads))
        end
        p = q + 1
    end
end

# terminal afterstate: the player who just moved (start_player) has borne off / won.
# my (start_player) equity is + points; sentinel large enough to always be chosen.
const TERMINAL_WIN = 12.0

"""Given afterstates of ONE benchmark position, return per-afterstate scores for
methods :A (direct pre-roll eval) and :B (roll-expectation). Score = MY equity."""
function score_afterstates(afters::Vector{BackgammonNet.BackgammonGame}, contact_net, race_net)
    na = length(afters)
    scoreA = fill(-Inf, na)
    scoreB = fill(-Inf, na)
    # --- method A: direct eval of each afterstate (opponent to move, pre-roll) ---
    liveA = Int[]
    for i in 1:na
        if afters[i].terminated
            scoreA[i] = TERMINAL_WIN; scoreB[i] = TERMINAL_WIN
        else
            push!(liveA, i)
        end
    end
    if !isempty(liveA)
        vA = fill(NaN, na)
        value_equities!(vA, afters, liveA, contact_net, race_net)
        for i in liveA
            scoreA[i] = -vA[i]                      # my equity = -(opponent equity)
        end
    end
    # --- method B: expectation over opponent's 21 rolls ---
    # build all (afterstate + roll) decision positions, batch, aggregate
    liveB = [i for i in liveA]                      # non-terminal only
    if !isempty(liveB)
        rollstates = BackgammonNet.BackgammonGame[]
        owner = Int[]; probw = Float64[]
        for i in liveB
            base = afters[i]
            for r in 1:21
                pr = BackgammonNet.DICE_PROBS[r]
                pr == 0f0 && continue
                gr = BackgammonNet.clone(base)
                BackgammonNet.apply_chance!(gr, r)
                push!(rollstates, gr); push!(owner, i); push!(probw, Float64(pr))
            end
        end
        vR = fill(NaN, length(rollstates))
        value_equities!(vR, rollstates, collect(1:length(rollstates)), contact_net, race_net)
        acc = Dict{Int,Float64}()
        for k in 1:length(rollstates)
            acc[owner[k]] = get(acc, owner[k], 0.0) + probw[k] * vR[k]   # E[opp equity]
        end
        for i in liveB
            scoreB[i] = -acc[i]                      # my equity = -E[opp equity]
        end
    end
    return scoreA, scoreB
end

# ── main afterstate pass over all benchmark positions (parallel) ─────────────
"""Returns greedy-A, greedy-B, worst-B, random decisions (Vector{MoveDecision})."""
function afterstate_pass(positions::Vector{BenchPosition}, contact_net, race_net,
                         num_workers::Int, seed::Int)
    n = length(positions)
    dA = Vector{MoveDecision}(undef, n)
    dB = Vector{MoveDecision}(undef, n)
    dWorstB = Vector{MoveDecision}(undef, n)
    dRand = Vector{MoveDecision}(undef, n)
    idx = Threads.Atomic{Int}(0)
    Threads.@threads for w in 1:num_workers
        rng = MersenneTwister(seed + 7919*w)
        while true
            i = Threads.atomic_add!(idx, 1) + 1
            i > n && break
            bp = positions[i]
            afters = enumerate_full_turn_afterstates(bp.state)
            sA, sB = score_afterstates(afters, contact_net, race_net)
            gA = afters[argmax(sA)]
            gB = afters[argmax(sB)]
            wB = afters[argmin(sB)]
            rr = afters[rand(rng, 1:length(afters))]
            dA[i]     = MoveDecision(bp.state, gA.p0, gA.p1, bp.is_contact, bp.player)
            dB[i]     = MoveDecision(bp.state, gB.p0, gB.p1, bp.is_contact, bp.player)
            dWorstB[i]= MoveDecision(bp.state, wB.p0, wB.p1, bp.is_contact, bp.player)
            dRand[i]  = MoveDecision(bp.state, rr.p0, rr.p1, bp.is_contact, bp.player)
        end
    end
    return dA, dB, dWorstB, dRand
end

# ── 2-ply (optional), SOUND leaves: score(m) = -E_r[ max_{m'} yeq(m') ] ──────
# A_m: opponent (Y) to move, pre-roll. For each roll r, Y picks best reply m'.
# Y's equity after m' = E_{r'}[ V(B_{m'} + r') ]  (B_{m'} = ME to move, pre-roll;
# leaf is the POST-ROLL, in-distribution position B_{m'}+r'). All leaves batched
# per candidate. My equity(m) = -E_r[ max_{m'} yeq(m') ].
function score_twoply(afters::Vector{BackgammonNet.BackgammonGame}, base_scoreB::Vector{Float64},
                      topk::Int, contact_net, race_net)
    na = length(afters)
    out = fill(-Inf, na)
    order = sortperm(base_scoreB; rev=true)
    cand = order[1:min(topk, na)]
    for i in cand
        A_m = afters[i]
        if A_m.terminated; out[i] = TERMINAL_WIN; continue; end
        # collect all leaf states for this candidate, with (r, rep) bookkeeping
        leaf = BackgammonNet.BackgammonGame[]
        leaf_w = Float64[]                 # DICE_PROBS[r'] weight of each leaf
        rmeta = Int[]; repmeta = Int[]     # which roll r, which reply index
        term_win = Dict{Tuple{Int,Int},Bool}()   # (r,rep) that is terminal Y-win
        reps_count = Dict{Int,Int}()       # r -> number of replies
        for r in 1:21
            prr = BackgammonNet.DICE_PROBS[r]; prr == 0f0 && continue
            gr = BackgammonNet.clone(A_m); BackgammonNet.apply_chance!(gr, r)   # Y post-roll
            reps = enumerate_full_turn_afterstates(gr)   # B_{m'}: ME to move, pre-roll (or terminal)
            reps_count[r] = length(reps)
            for (ri, rp) in enumerate(reps)
                if rp.terminated
                    term_win[(r, ri)] = true             # Y bore off → Y wins
                else
                    for rp2 in 1:21
                        prr2 = BackgammonNet.DICE_PROBS[rp2]; prr2 == 0f0 && continue
                        b = BackgammonNet.clone(rp); BackgammonNet.apply_chance!(b, rp2)
                        push!(leaf, b); push!(leaf_w, Float64(prr2))
                        push!(rmeta, r); push!(repmeta, ri)
                    end
                end
            end
        end
        # one batched net eval of all leaves (post-roll, in-distribution)
        vL = fill(NaN, length(leaf))
        isempty(leaf) || value_equities!(vL, leaf, collect(1:length(leaf)), contact_net, race_net)
        # myeq(r,rep) = E_{r'}[V(B_{m'}+r')]  (leaf B_{m'} is ME to move → V is MY equity).
        # Y's equity after playing m' = -(MY equity at B_{m'}) = -myeq.  Y maximizes.
        myacc = Dict{Tuple{Int,Int},Float64}()
        for k in 1:length(leaf)
            key = (rmeta[k], repmeta[k])
            myacc[key] = get(myacc, key, 0.0) + leaf_w[k] * vL[k]
        end
        total = 0.0
        for r in 1:21
            prr = BackgammonNet.DICE_PROBS[r]; prr == 0f0 && continue
            best_opp = -Inf                                   # Y's best (max) equity at A_m+r
            for ri in 1:get(reps_count, r, 0)
                yq = get(term_win, (r, ri), false) ? TERMINAL_WIN : -get(myacc, (r, ri), Inf)
                yq > best_opp && (best_opp = yq)
            end
            total += Float64(prr) * best_opp                  # E_r[ Y's best equity ]
        end
        out[i] = -total                                       # my equity(m) = -E_r[Y best]
    end
    return out
end
function twoply_pass(positions::Vector{BenchPosition}, contact_net, race_net,
                     num_workers::Int, topk::Int, limit::Int)
    n = limit > 0 ? min(limit, length(positions)) : length(positions)
    dT = Vector{MoveDecision}(undef, n)
    idx = Threads.Atomic{Int}(0)
    Threads.@threads for w in 1:num_workers
        while true
            i = Threads.atomic_add!(idx, 1) + 1
            i > n && break
            bp = positions[i]
            afters = enumerate_full_turn_afterstates(bp.state)
            _, sB = score_afterstates(afters, contact_net, race_net)
            sT = score_twoply(afters, sB, topk, contact_net, race_net)
            gT = afters[argmax(sT)]
            dT[i] = MoveDecision(bp.state, gT.p0, gT.p1, bp.is_contact, bp.player)
        end
    end
    return dT, n
end

function load_net(path, width, blocks)
    isfile(path) || return nothing
    net = FluxLib.FCResNetMultiHead(gspec, FluxLib.FCResNetMultiHeadHP(width=width, num_blocks=blocks))
    FluxLib.load_weights(path, net)
    return Flux.cpu(net)
end

function report(label, s)
    s === nothing && (println("  $label: (no data)"); return)
    @printf("%-32s PR=%7.2f  ER=%7.2f  (contact n=%d PR=%.2f | race n=%d PR=%.2f | scored=%d forced=%d unmatched=%d)\n",
            label, s.PR, s.ER, s.contact.n, s.contact.PR, s.race.n, s.race.PR, s.n_ok, s.n_forced, s.n_unmatched)
end

function main()
    nw = A["num_workers"]; seed = A["seed"]; ncontact = A["n_positions"]; gnubg_ply = A["gnubg_ply"]
    BLAS_note = ""
    try; LinearAlgebra.BLAS.set_num_threads(1); BLAS_note=" (BLAS=1)"; catch; end

    println("="^90)
    println("AFTERSTATE-VALUE PR — does i140's VALUE head, as an afterstate evaluator, beat MCTS-800?")
    println("="^90)
    println("Positions:  $ncontact contact pre-move decisions, benchmark_pr seed-$seed gnubg-0ply generator")
    println("Grader:     gnubg ply-$gnubg_ply NATIVE move-list, match-by-board (SAME as baselines)")
    println("Net:        $(A["net_ckpt"])  (contact $(A["width"])x$(A["blocks"]))")
    println("            race $(A["race_ckpt"])  ($(A["race_width"])x$(A["race_blocks"]))")
    println("Obs:        $(A["obs_type"])  state_dim=$STATE_DIM  num_actions=$NUM_ACTIONS  workers=$nw$BLAS_note")
    println("Baselines:  i140 raw-policy 51.7 | i140 mcts-800 32.6 | wildbg ~14 | gnubg0 ~2 | floor ~1")
    println("="^90); flush(stdout)

    contact_net = load_net(A["net_ckpt"], A["width"], A["blocks"])
    contact_net === nothing && error("contact net not found: $(A["net_ckpt"])")
    race_net = load_net(A["race_ckpt"], A["race_width"], A["race_blocks"])
    println(race_net === nothing ? "WARNING: race net missing — race afterstates use contact net" :
            "Loaded contact + race nets.")

    # ── SANITY 1: opening pre-roll direct eval (0-ply-A) should be ≈ 0 ──
    open_state = BackgammonNet.initial_state(obs_type=OBS_SYM)   # PHASE_CHANCE, dice unset
    ov = fill(NaN, 1)
    value_equities!(ov, [open_state], [1], contact_net, race_net)
    @printf("\nSANITY(A) opening pre-roll DIRECT value-net equity = %+.4f  (expect ≈ 0 if pre-roll in-distribution)\n", ov[1])
    flush(stdout)

    # ── generate common positions ──
    println("\nGenerating common benchmark positions (gnubg-0ply both sides, seeded dice)...")
    flush(stdout); t0 = time()
    gen = make_gnubg(0)
    positions, ngames, genfail = generate_common_positions(gen, ncontact, seed, A["max_games"])
    try BackgammonNet.close(gen) catch end
    np0 = count(p->p.player==0, positions); np1 = count(p->p.player==1, positions)
    println("  $(length(positions)) contact positions from $ngames games ($(round(time()-t0,digits=1))s)  P0=$np0 P1=$np1")
    genfail>0 && println("  (generator best_move fallbacks: $genfail)")
    isempty(positions) && (println("no positions"); return)
    flush(stdout)

    # ── SANITY 2: one position, show best vs worst afterstate score + native regret ──
    gb0 = make_gnubg(gnubg_ply)
    let bp = positions[1]
        afters = enumerate_full_turn_afterstates(bp.state)
        sA, sB = score_afterstates(afters, contact_net, race_net)
        gi = argmax(sB); wi = argmin(sB)
        dg = MoveDecision(bp.state, afters[gi].p0, afters[gi].p1, bp.is_contact, bp.player)
        dw = MoveDecision(bp.state, afters[wi].p0, afters[wi].p1, bp.is_contact, bp.player)
        tg = native_regret(gb0, dg); tw = native_regret(gb0, dw)
        @printf("SANITY(sign) pos#1 (%d legal moves): best-score move scoreB=%+.3f native_regret=%.4f | worst-score scoreB=%+.3f native_regret=%.4f\n",
                length(afters), sB[gi], tg[2], sB[wi], tw[2])
        println("             (correct sign ⇒ best-score regret ≪ worst-score regret)")
    end
    try BackgammonNet.close(gb0) catch end
    flush(stdout)

    # ── afterstate pass ──
    println("\nAfterstate move selection (0-ply-A, 0-ply-B, worst-B, random) over all positions...")
    flush(stdout); t1 = time()
    dA, dB, dWorstB, dRand = afterstate_pass(positions, contact_net, race_net, nw, seed)
    println("  done ($(round(time()-t1,digits=1))s)"); flush(stdout)

    dT = nothing; ntwo = 0
    if A["two_ply"]
        println("\n2-ply afterstate (top-$(A["two_ply_topk"]) candidates) ...")
        flush(stdout); t2 = time()
        dT, ntwo = twoply_pass(positions, contact_net, race_net, nw, A["two_ply_topk"], A["two_ply_positions"])
        println("  done over $ntwo positions ($(round(time()-t2,digits=1))s)"); flush(stdout)
    end

    # ── grade everything with gnubg native ──
    gnubg_backends = [make_gnubg(gnubg_ply) for _ in 1:nw]
    println("\nGrading (gnubg ply-$gnubg_ply native)...\n"); flush(stdout)
    grade(d) = summarize(score_moves(gnubg_backends, d, nw)...)
    sA  = grade(dA); sB = grade(dB); sW = grade(dWorstB); sR = grade(dRand)
    sT  = dT === nothing ? nothing : grade(dT[1:ntwo])
    for gb in gnubg_backends; try BackgammonNet.close(gb) catch end; end

    println("="^90)
    println("RESULTS — afterstate-value PR vs baselines (contact, gnubg-ply$gnubg_ply native, lower=better)")
    println("="^90)
    report("0-ply-A (direct pre-roll)", sA)
    report("0-ply-B (roll-expectation)", sB)
    sT !== nothing && report("2-ply (top-$(A["two_ply_topk"]))", sT)
    println("-"^90)
    report("[sign] WORST-B (argmin)", sW)
    report("[ctrl] RANDOM legal move", sR)
    println("-"^90)
    println("Baselines on this fixed set:  i140 raw-policy=51.7   i140 MCTS-800=32.6   wildbg≈14   gnubg0≈2   floor≈1")
    println("="^90)

    # ── verdict ──
    bpr = sB === nothing ? NaN : sB.PR
    println("\nVERDICT")
    if sB !== nothing
        @printf("  0-ply-B afterstate-value PR = %.1f   vs   MCTS-800 = 32.6   (raw policy 51.7)\n", bpr)
        if bpr < 32.6
            println("  => 0-ply afterstate-value BEATS the full policy+MCTS-800 pipeline.")
            println("     CONFIRMS the policy head was the bottleneck. Path: afterstate-value + expectimax.")
        elseif bpr < 51.7
            println("  => 0-ply afterstate-value is BETWEEN raw policy and MCTS-800 — partial confirmation.")
        else
            println("  => 0-ply afterstate-value is WORSE than raw policy — the VALUE head cannot discriminate")
            println("     sibling afterstates. Bottleneck is value sibling-discrimination, not the policy head.")
        end
    end
    if sT !== nothing
        @printf("  2-ply afterstate-value PR = %.1f (n=%d)\n", sT.PR, ntwo)
    end
    println("="^90); flush(stdout)
end

main()
