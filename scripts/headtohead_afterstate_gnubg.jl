#!/usr/bin/env julia
"""
headtohead_afterstate_gnubg.jl — HEAD-TO-HEAD game-play verification of the
pre-roll AFTERSTATE-VALUE contact evaluator against gnubg (and random).

Does the near-gnubg per-decision PR (2-ply 2.9 vs gnubg-2ply on the fixed
benchmark) translate into real playing strength? This script plays ACTUAL
cubeless-money games: our afterstate-value agent vs gnubg (GnubgCLibBackend +
GameLoop.ExternalAgent), tracking win% AND points-per-game (equity, counting
gammons/backgammons), with 95% confidence intervals.

OUR AGENT (afterstate-value move selection, policy head ignored):
  At each decision, enumerate every legal full-turn afterstate (opponent-to-move,
  pre-roll) and score each by our value net; play the move toward the best board.
  - mode :a      0-ply-A DIRECT — evaluate each pre-roll afterstate DIRECTLY with
                 the value net (1 eval/move, FAST; valid because this net is a
                 PRE-ROLL evaluator, verified by the opening-eq sanity ≈ +0.07).
  - mode :twoply 2-ply expectimax over the top-k afterstates (stronger, slower).
  Contact vs race routed by is_contact_position (race net for race positions).

IMPORTANT: matches the PR benchmark setup — STANDARD opening (short_game=false),
which is what the net was trained/benchmarked on (game.jl's GI.init uses the
short-game position, so we build the env directly here instead).

Opponent gnubg via ExternalAgent(GnubgCLibBackend(ply=P)); gnubg is globally
serialized but thread-safe; ply-0/1 run fine under many workers (ply>=2 can
deadlock — not used here). Random opponent used first as a harness/sign sanity.

VARIANCE REDUCTION: games are played in PAIRS with a shared dice seed — for each
seed we play one game with our agent as white and one as black. This cancels the
seat/first-move asymmetry and correlates the luck. The independent statistical
unit is the PAIR (distinct seeds), so equity CIs use the per-pair mean.

Usage:
    julia --threads 16 --project scripts/headtohead_afterstate_gnubg.jl \\
        --pairs 750 --num-workers 15 --gnubg-ply 0

Options:
    --pairs=750            game PAIRS vs gnubg (total games = 2 x pairs)
    --random-pairs=50      sanity game pairs vs RANDOM first (100 games)
    --num-workers=15
    --gnubg-ply=0          primary gnubg ply
    --mode=a               our move-selection: a (0-ply-A DIRECT) | twoply
    --two-ply-topk=6
    --seed=1
    --extra-gnubg1         also run --pairs-extra games vs gnubg-1ply
    --extra-twoply         also run --pairs-extra games with our 2-ply vs gnubg-0ply
    --pairs-extra=150
    --net-ckpt=sessions/preroll-afterstate-gnubg/preroll_afterstate_gnubg_40k.data
    --race-ckpt=sessions/race-supervised-v2/checkpoints/race_train_latest.data
    --obs-type=min_plus_flat --width=256 --blocks=5 --race-width=128 --race-blocks=3
"""

using ArgParse

function parse_args_hh()
    s = ArgParseSettings(description="Head-to-head afterstate-value vs gnubg", autofix_names=true)
    @add_arg_table! s begin
        "--pairs";        arg_type=Int; default=750
        "--random-pairs"; arg_type=Int; default=50
        "--num-workers";  arg_type=Int; default=15
        "--gnubg-ply";    arg_type=Int; default=0
        "--mode";         arg_type=String; default="a"       # a | twoply
        "--two-ply-topk"; arg_type=Int; default=6
        "--seed";         arg_type=Int; default=1
        "--extra-gnubg1"; action=:store_true
        "--extra-twoply"; action=:store_true
        "--pairs-extra";  arg_type=Int; default=150
        "--net-ckpt";     arg_type=String; default="sessions/preroll-afterstate-gnubg/preroll_afterstate_gnubg_40k.data"
        "--race-ckpt";    arg_type=String; default="sessions/race-supervised-v2/checkpoints/race_train_latest.data"
        "--obs-type";     arg_type=String; default="min_plus_flat"
        "--width";        arg_type=Int; default=256
        "--blocks";       arg_type=Int; default=5
        "--race-width";   arg_type=Int; default=128
        "--race-blocks";  arg_type=Int; default=3
    end
    return ArgParse.parse_args(s)
end
const A = parse_args_hh()

ENV["BACKGAMMON_OBS_TYPE"] = A["obs_type"]

using AlphaZero
using AlphaZero: GI, FluxLib, GameLoop
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
const TERMINAL_WIN = 12.0

# ── value-net equity (mover perspective), routed contact/race ─────────────────
# (helpers copied from afterstate_value_pr.jl — do NOT modify that file)
function value_equities!(out::Vector{Float64}, states::Vector{BackgammonNet.BackgammonGame},
                         idxs::Vector{Int}, contact_net, race_net; chunk::Int=1024)
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

# ── enumerate every legal full-turn afterstate of P (deduped by board) ────────
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

# ── 0-ply-A DIRECT scoring: score = MY equity = -(opponent pre-roll equity) ───
function score_A(afters::Vector{BackgammonNet.BackgammonGame}, cnet, rnet)
    na = length(afters)
    s = fill(-Inf, na)
    live = Int[]
    for i in 1:na
        afters[i].terminated ? (s[i] = TERMINAL_WIN) : push!(live, i)
    end
    if !isempty(live)
        v = fill(NaN, na)
        value_equities!(v, afters, live, cnet, rnet)
        for i in live; s[i] = -v[i]; end
    end
    return s
end

# ── 0-ply-B (roll-expectation) — used only to rank 2-ply root candidates ──────
function score_B(afters::Vector{BackgammonNet.BackgammonGame}, cnet, rnet)
    na = length(afters)
    s = fill(-Inf, na)
    live = [i for i in 1:na if !afters[i].terminated]
    for i in 1:na; afters[i].terminated && (s[i] = TERMINAL_WIN); end
    if !isempty(live)
        rollstates = BackgammonNet.BackgammonGame[]; owner = Int[]; w = Float64[]
        for i in live
            base = afters[i]
            for r in 1:21
                pr = BackgammonNet.DICE_PROBS[r]; pr == 0f0 && continue
                gr = BackgammonNet.clone(base); BackgammonNet.apply_chance!(gr, r)
                push!(rollstates, gr); push!(owner, i); push!(w, Float64(pr))
            end
        end
        vR = fill(NaN, length(rollstates))
        value_equities!(vR, rollstates, collect(1:length(rollstates)), cnet, rnet)
        acc = Dict{Int,Float64}()
        for k in 1:length(rollstates); acc[owner[k]] = get(acc, owner[k], 0.0) + w[k]*vR[k]; end
        for i in live; s[i] = -acc[i]; end
    end
    return s
end

# ── 2-ply (SOUND leaves) — verbatim structure from afterstate_value_pr.jl ─────
function score_twoply(afters::Vector{BackgammonNet.BackgammonGame}, base_scoreB::Vector{Float64},
                      topk::Int, cnet, rnet)
    na = length(afters)
    out = fill(-Inf, na)
    order = sortperm(base_scoreB; rev=true)
    cand = order[1:min(topk, na)]
    for i in cand
        A_m = afters[i]
        if A_m.terminated; out[i] = TERMINAL_WIN; continue; end
        leaf = BackgammonNet.BackgammonGame[]; leaf_w = Float64[]
        rmeta = Int[]; repmeta = Int[]
        term_win = Dict{Tuple{Int,Int},Bool}(); reps_count = Dict{Int,Int}()
        for r in 1:21
            prr = BackgammonNet.DICE_PROBS[r]; prr == 0f0 && continue
            gr = BackgammonNet.clone(A_m); BackgammonNet.apply_chance!(gr, r)
            reps = enumerate_full_turn_afterstates(gr)
            reps_count[r] = length(reps)
            for (ri, rp) in enumerate(reps)
                if rp.terminated
                    term_win[(r, ri)] = true
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
        vL = fill(NaN, length(leaf))
        isempty(leaf) || value_equities!(vL, leaf, collect(1:length(leaf)), cnet, rnet)
        myacc = Dict{Tuple{Int,Int},Float64}()
        for k in 1:length(leaf)
            key = (rmeta[k], repmeta[k]); myacc[key] = get(myacc, key, 0.0) + leaf_w[k]*vL[k]
        end
        total = 0.0
        for r in 1:21
            prr = BackgammonNet.DICE_PROBS[r]; prr == 0f0 && continue
            best_opp = -Inf
            for ri in 1:get(reps_count, r, 0)
                yq = get(term_win, (r, ri), false) ? TERMINAL_WIN : -get(myacc, (r, ri), Inf)
                yq > best_opp && (best_opp = yq)
            end
            total += Float64(prr) * best_opp
        end
        out[i] = -total
    end
    return out
end

#####
##### Afterstate-value agent (plugs into GameLoop.play_game via ExternalAgent path)
#####

struct AfterstateAgent <: GameLoop.GameAgent
    contact_net::Any
    race_net::Any
    mode::Symbol        # :a | :twoply
    topk::Int
end
struct RandomAgent <: GameLoop.GameAgent end

GameLoop.create_player(::AfterstateAgent; rng=Random.default_rng()) = nothing
GameLoop.create_player(::RandomAgent; rng=Random.default_rng()) = nothing

"""Pick the value-net best full-turn target board, then return the FIRST legal
half-move that leads toward it. Re-derived each half-move (consistent: the best
target stays reachable after moving toward it)."""
function afterstate_move(a::AfterstateAgent, g::BackgammonNet.BackgammonGame)
    legal = BackgammonNet.legal_actions(g)
    length(legal) == 1 && return legal[1]
    afters = enumerate_full_turn_afterstates(g)
    length(afters) == 1 && return legal[1]   # single full-turn outcome; any legal step is fine
    scores = if a.mode == :twoply
        sB = score_B(afters, a.contact_net, a.race_net)
        score_twoply(afters, sB, a.topk, a.contact_net, a.race_net)
    else
        score_A(afters, a.contact_net, a.race_net)
    end
    B = afters[argmax(scores)]
    key = (B.p0, B.p1)
    start_player = g.current_player
    for act in legal
        s2 = BackgammonNet.clone(g); BackgammonNet.apply_action!(s2, act)
        att = BackgammonNet.action_type(s2)
        ended = (att == BackgammonNet.ACTION_TYPE_TERMINAL || s2.current_player != start_player ||
                 att == BackgammonNet.ACTION_TYPE_CHANCE)
        if ended
            (s2.p0, s2.p1) == key && return act
            continue
        end
        reach = enumerate_full_turn_afterstates(s2)
        any(r -> (r.p0, r.p1) == key, reach) && return act
    end
    return legal[1]   # fallback (should not happen)
end

function GameLoop.select_action(a::AfterstateAgent, ::Nothing, env)
    return (afterstate_move(a, env.game), Float32[], Int[])
end
function GameLoop.select_action(::RandomAgent, ::Nothing, env)
    avail = GI.available_actions(env)
    isempty(avail) && return (0, Float32[], Int[])
    return (rand(Random.default_rng(), avail), Float32[], Int[])
end

#####
##### Env / game runner
#####

# STANDARD opening (short_game=false), matching the PR benchmark + training.
function fresh_standard_env()
    game = BackgammonNet.initial_state(; short_game=false, doubles_only=false,
                                       cube_enabled=false, jacoby_enabled=false,
                                       obs_type=OBSERVATION_TYPE)
    return GameEnv(game, MersenneTwister(0))   # env rng unused; play_game drives chance
end

function load_net(path, width, blocks)
    isfile(path) || return nothing
    net = FluxLib.FCResNetMultiHead(gspec, FluxLib.FCResNetMultiHeadHP(width=width, num_blocks=blocks))
    FluxLib.load_weights(path, net)
    return Flux.cpu(net)
end

"""Play `n_pairs` seed-paired games (our agent white then black on the same seed).
Returns our-perspective points for every game, indexed [2p-1]=white, [2p]=black."""
function play_config(our_agent, make_opp, n_pairs::Int, num_workers::Int, base_seed::Int; label::String="")
    ngames = 2 * n_pairs
    points = Vector{Float64}(undef, ngames)
    opp_agents = [make_opp() for _ in 1:num_workers]
    done = Threads.Atomic{Int}(0)
    claimed = Threads.Atomic{Int}(0)
    t0 = time()
    Threads.@threads for w in 1:num_workers
        opp = opp_agents[w]
        while true
            k = Threads.atomic_add!(claimed, 1) + 1
            k > ngames && break
            pair = (k + 1) ÷ 2
            our_white = isodd(k)
            env = fresh_standard_env()
            rng = MersenneTwister(base_seed + pair)
            w_ag, b_ag = our_white ? (our_agent, opp) : (opp, our_agent)
            res = GameLoop.play_game(w_ag, b_ag, env; rng=rng, temperature_fn=_ -> 0.0)
            points[k] = our_white ? res.reward : -res.reward
            d = Threads.atomic_add!(done, 1) + 1
            if d % 100 == 0 || d == ngames
                rate = d / max(time() - t0, 1e-6)
                @printf("  [%s] %d/%d games (%.1f g/s, %.0fs elapsed)\n", label, d, ngames, rate, time()-t0)
                flush(stdout)
            end
        end
    end
    return points
end

#####
##### Statistics
#####

function summarize(points::Vector{Float64}, n_pairs::Int)
    ng = length(points)
    wins = count(>(0.0), points)
    losses = count(<(0.0), points)
    winp = wins / ng
    # win% CI (normal approx over all games)
    win_se = sqrt(max(winp*(1-winp), 0.0) / ng)
    win_lo = winp - 1.96*win_se; win_hi = winp + 1.96*win_se
    # equity (points/game) — paired unit is the statistical sample
    pair_val = [ (points[2p-1] + points[2p]) / 2 for p in 1:n_pairs ]
    ppg = mean(points)                       # == mean(pair_val)
    ppg_se = std(pair_val) / sqrt(n_pairs)
    ppg_lo = ppg - 1.96*ppg_se; ppg_hi = ppg + 1.96*ppg_se
    # point-magnitude breakdown (gammons/backgammons)
    g_win  = count(==(2.0), points); bg_win  = count(==(3.0), points); s_win = count(==(1.0), points)
    g_loss = count(==(-2.0), points); bg_loss = count(==(-3.0), points); s_loss = count(==(-1.0), points)
    return (ng=ng, n_pairs=n_pairs, wins=wins, losses=losses,
            winp=winp, win_lo=win_lo, win_hi=win_hi, win_se=win_se,
            ppg=ppg, ppg_lo=ppg_lo, ppg_hi=ppg_hi, ppg_se=ppg_se,
            s_win=s_win, g_win=g_win, bg_win=bg_win,
            s_loss=s_loss, g_loss=g_loss, bg_loss=bg_loss)
end

function report(label, r)
    println("-"^78)
    println(label)
    @printf("  N = %d games (%d seed-pairs) | wins %d  losses %d\n", r.ng, r.n_pairs, r.wins, r.losses)
    @printf("  WIN%%   = %5.1f%%   95%% CI [%5.1f%%, %5.1f%%]   (±%.1f%%)\n",
            100r.winp, 100r.win_lo, 100r.win_hi, 100*1.96*r.win_se)
    @printf("  EQUITY = %+.4f ppg  95%% CI [%+.4f, %+.4f]   (±%.4f)\n",
            r.ppg, r.ppg_lo, r.ppg_hi, 1.96*r.ppg_se)
    @printf("  our wins:  single %d  gammon %d  backgammon %d\n", r.s_win, r.g_win, r.bg_win)
    @printf("  our losses: single %d  gammon %d  backgammon %d\n", r.s_loss, r.g_loss, r.bg_loss)
    flush(stdout)
end

make_gnubg(ply::Int) = (e = BackgammonNet.GnubgCLibBackend(ply=ply, threads=1); BackgammonNet.open!(e); e)

function main()
    try; LinearAlgebra.BLAS.set_num_threads(1); catch; end
    nw = A["num_workers"]; seed = A["seed"]
    mode = Symbol(A["mode"]); topk = A["two_ply_topk"]

    println("="^78)
    println("HEAD-TO-HEAD — pre-roll afterstate-value agent vs gnubg (STANDARD opening)")
    println("="^78)
    println("Net:     $(A["net_ckpt"]) (contact $(A["width"])x$(A["blocks"]))")
    println("Race:    $(A["race_ckpt"]) ($(A["race_width"])x$(A["race_blocks"]))")
    println("Obs:     $(A["obs_type"])  state_dim=$STATE_DIM  actions=$NUM_ACTIONS  workers=$nw")
    println("Mode:    $(mode)$(mode == :twoply ? " (top-$topk)" : " (0-ply-A DIRECT)")")
    println("="^78); flush(stdout)

    cnet = load_net(A["net_ckpt"], A["width"], A["blocks"])
    cnet === nothing && error("contact net not found: $(A["net_ckpt"])")
    rnet = load_net(A["race_ckpt"], A["race_width"], A["race_blocks"])
    println(rnet === nothing ? "WARNING: race net missing — race positions use contact net" :
            "Loaded contact + race nets.")

    # ── SANITY: opening pre-roll DIRECT eval should be ≈ +0.07 (mover's slight edge) ──
    open_state = BackgammonNet.initial_state(; short_game=false, cube_enabled=false,
                                             jacoby_enabled=false, obs_type=OBS_SYM)
    ov = fill(NaN, 1); value_equities!(ov, [open_state], [1], cnet, rnet)
    @printf("\nSANITY(opening) pre-roll DIRECT value-net equity = %+.4f  (expect ≈ +0.07)\n", ov[1])
    flush(stdout)

    our_agent = AfterstateAgent(cnet, rnet, mode, topk)

    # ── SANITY: crush random (>95% expected). Fail-fast if sign/harness is wrong. ──
    println("\n" * "#"^78)
    println("# SANITY vs RANDOM  ($(2*A["random_pairs"]) games) — expect win% > 95%")
    println("#"^78); flush(stdout)
    rp = play_config(our_agent, () -> RandomAgent(), A["random_pairs"], nw, 900_000; label="random")
    rr = summarize(rp, A["random_pairs"])
    report("OUR ($(mode)) vs RANDOM", rr)
    if rr.winp < 0.90
        println("\n*** ABORT: win% vs random is $(round(100rr.winp,digits=1))% (<90%). ")
        println("    Harness/sign is WRONG — not running gnubg. ***")
        return
    end
    println("\n>>> Harness/sign validated (crushes random). Proceeding to gnubg.\n"); flush(stdout)

    # ── PRIMARY: vs gnubg-0ply ──
    gply = A["gnubg_ply"]
    println("#"^78)
    println("# PRIMARY: OUR ($(mode)) vs gnubg-$(gply)ply  ($(2*A["pairs"]) games)")
    println("#"^78); flush(stdout)
    gp = play_config(our_agent, () -> GameLoop.ExternalAgent(make_gnubg(gply)),
                     A["pairs"], nw, seed; label="gnubg$gply")
    gr = summarize(gp, A["pairs"])

    # ── optional extras ──
    er1 = nothing; er2 = nothing
    if A["extra_gnubg1"]
        println("\n#"^0 * "#"^78)
        println("# EXTRA: OUR ($(mode)) vs gnubg-1ply  ($(2*A["pairs_extra"]) games)")
        println("#"^78); flush(stdout)
        ep = play_config(our_agent, () -> GameLoop.ExternalAgent(make_gnubg(1)),
                         A["pairs_extra"], nw, seed + 500_000; label="gnubg1")
        er1 = summarize(ep, A["pairs_extra"])
    end
    if A["extra_twoply"]
        println("\n" * "#"^78)
        println("# EXTRA: OUR 2-PLY (top-$topk) vs gnubg-$(gply)ply  ($(2*A["pairs_extra"]) games)")
        println("#"^78); flush(stdout)
        agent2 = AfterstateAgent(cnet, rnet, :twoply, topk)
        tp = play_config(agent2, () -> GameLoop.ExternalAgent(make_gnubg(gply)),
                         A["pairs_extra"], nw, seed + 700_000; label="2ply-g$gply")
        er2 = summarize(tp, A["pairs_extra"])
    end

    # ── FINAL REPORT ──
    println("\n" * "="^78)
    println("FINAL RESULTS")
    println("="^78)
    report("SANITY  — OUR ($mode) vs RANDOM", rr)
    report("PRIMARY — OUR ($mode) vs gnubg-$(gply)ply", gr)
    er1 !== nothing && report("EXTRA   — OUR ($mode) vs gnubg-1ply", er1)
    er2 !== nothing && report("EXTRA   — OUR 2-ply vs gnubg-$(gply)ply", er2)
    println("="^78)

    # ── verdict ──
    println("\nVERDICT (vs gnubg-$(gply)ply)")
    @printf("  win%% = %.1f%% [%.1f, %.1f]   equity = %+.4f ppg [%+.4f, %+.4f]\n",
            100gr.winp, 100gr.win_lo, 100gr.win_hi, gr.ppg, gr.ppg_lo, gr.ppg_hi)
    if gr.win_hi >= 0.47 && gr.win_lo <= 0.53
        println("  => Within a few % of 50% — game-play CONFIRMS near-gnubg strength (as PR predicted).")
    elseif gr.winp >= 0.45
        println("  => Marginally below 50% (as PR predicted for 0-ply, slightly weaker than gnubg-0ply).")
    elseif gr.winp < 0.40
        println("  => FAR below 50% — per-decision PR OVERSTATED playing strength. FLAG for investigation.")
    else
        println("  => Below break-even; weaker than gnubg-0ply in play.")
    end
    println("="^78); flush(stdout)
end

main()
