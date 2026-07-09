#!/usr/bin/env julia
"""
afterstate_class_regret.jl — TAIL CHECK for the gnubg-relabel experiment.

On the FIXED benchmark set (benchmark_pr seed-1 gnubg-0ply generator — SAME positions
behind the PR baselines), classify each contact decision (BEAR_IN_VS_CONTACT /
DEEP_ANCHOR / BACK_CHECKER_MID / CONTACT_GENERIC — identical heuristics to
build_gnubg_afterstate_labels.jl) and report, PER CLASS, the native gnubg-ply1 regret
(=PR) of the net's greedy 0-ply-B afterstate move, plus the top-10% equity-loss share.

Run once per net (old wildbg-100k vs new gnubg) to see whether the worst classes'
regret dropped and the top-10% loss concentration shrank.

Usage:
  julia --threads 16 --project scripts/afterstate_class_regret.jl \\
     --net-ckpt sessions/preroll-afterstate/preroll_afterstate_ft_100k.data \\
     --n-positions 1500 --num-workers 16
"""

using ArgParse
function parse_args_c()
    s = ArgParseSettings(autofix_names=true)
    @add_arg_table! s begin
        "--n-positions"; arg_type=Int; default=1500
        "--seed";        arg_type=Int; default=1
        "--num-workers"; arg_type=Int; default=16
        "--gnubg-ply";   arg_type=Int; default=1
        "--net-ckpt";    arg_type=String; default="sessions/preroll-afterstate/preroll_afterstate_ft_100k.data"
        "--race-ckpt";   arg_type=String; default="sessions/race-supervised-v2/checkpoints/race_train_latest.data"
        "--obs-type";    arg_type=String; default="min_plus_flat"
        "--width";       arg_type=Int; default=256
        "--blocks";      arg_type=Int; default=5
        "--race-width";  arg_type=Int; default=128
        "--race-blocks"; arg_type=Int; default=3
        "--max-games";   arg_type=Int; default=100000
        "--tag";         arg_type=String; default=""
    end
    return ArgParse.parse_args(s)
end
const A = parse_args_c()
ENV["BACKGAMMON_OBS_TYPE"] = A["obs_type"]

using AlphaZero
using AlphaZero: GI, FluxLib
import Flux
using BackgammonNet
using Random, Statistics, Printf
import LinearAlgebra
try; LinearAlgebra.BLAS.set_num_threads(1); catch; end

include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
const gspec = GameSpec()
const NUM_ACTIONS = GI.num_actions(gspec)
const STATE_DIM = GI.state_dim(gspec)[1]
const OBS_SYM = Symbol(A["obs_type"])
@inline _sigmoid(x) = 1.0f0/(1.0f0+exp(-Float32(x)))

struct MoveDecision; state::BackgammonNet.BackgammonGame; res_p0::UInt128; res_p1::UInt128; is_contact::Bool; player::Int; end
struct BenchPosition; state::BackgammonNet.BackgammonGame; is_contact::Bool; player::Int; end

# ── classifier (identical to build_gnubg_afterstate_labels.jl) ──
function classify(P::BackgammonNet.BackgammonGame)
    cp = Int(P.current_player)
    m = BackgammonNet.to_gnubg_simple(P; perspective=cp)
    mover_home=0; mover_on=max(m[1],0); mover_backdeep=0; mover_back=0; opp_anchor=false
    @inbounds for pt in 1:24
        v = m[pt+1]
        if v > 0
            mover_on += v
            (1<=pt<=6)  && (mover_home += v)
            (22<=pt<=24)&& (mover_backdeep += v)
            (19<=pt<=24)&& (mover_back += v)
        elseif v <= -2 && (1<=pt<=4)
            opp_anchor = true
        end
    end
    mover_off = 15 - mover_on
    if mover_home + mover_off >= 11; return :BEAR_IN_VS_CONTACT
    elseif mover_backdeep >= 2 || opp_anchor; return :DEEP_ANCHOR
    elseif mover_back >= 2; return :BACK_CHECKER_MID
    else; return :CONTACT_GENERIC; end
end

# ── native gnubg regret (verbatim) ──
function native_regret(gnubg, d::MoveDecision)
    g = d.state
    (g.phase == BackgammonNet.PHASE_CHECKER_PLAY) || return (:forced, 0.0)
    BackgammonNet.open!(gnubg)
    md = lock(BackgammonNet._GNUBG_CLIB_LOCK) do
        BackgammonNet._gnubg_clib_move_data(gnubg, g)
    end
    n = length(md); n <= 1 && return (:forced, 0.0)
    player = Int(g.current_player); best_eq = -Inf; our_eq = nothing
    @inbounds for (tsimple, probs) in md
        eq = Float64(BackgammonNet.compute_cubeless_equity(g, probs))
        eq > best_eq && (best_eq = eq)
        tp0,tp1 = BackgammonNet.from_gnubg_simple(tsimple, player)
        (tp0==d.res_p0 && tp1==d.res_p1) && (our_eq = eq)
    end
    our_eq === nothing && return (:unmatched, 0.0)
    return (:ok, max(best_eq - our_eq, 0.0))
end

function _sample_dice(rng)
    r = rand(rng, Float32); c = 0.0f0
    @inbounds for i in 1:length(BackgammonNet.DICE_PROBS)
        c += BackgammonNet.DICE_PROBS[i]; r <= c && return i
    end
    return length(BackgammonNet.DICE_PROBS)
end
function play_turn!(g, engine, fail)
    sp = g.current_player
    while true
        att = BackgammonNet.action_type(g)
        (att==BackgammonNet.ACTION_TYPE_TERMINAL||att==BackgammonNet.ACTION_TYPE_CHANCE)&&break
        g.current_player != sp && break
        acts = BackgammonNet.legal_actions(g); isempty(acts)&&break
        a = length(acts)==1 ? acts[1] : (try BackgammonNet.best_move(engine,g) catch; Threads.atomic_add!(fail,1); acts[1] end)
        BackgammonNet.apply_action!(g, a); g.terminated&&break
    end
    g
end
make_gnubg(ply) = (e=BackgammonNet.GnubgCLibBackend(ply=ply,threads=1); BackgammonNet.open!(e); e)

function gen_positions(gen, n, seed, maxg)
    pos = BenchPosition[]; fail = Threads.Atomic{Int}(0); gi=0
    while length(pos) < n && gi < maxg
        gi += 1; rng = MersenneTwister(seed+gi); g = BackgammonNet.initial_state(obs_type=OBS_SYM)
        while true
            at = BackgammonNet.action_type(g)
            at == BackgammonNet.ACTION_TYPE_TERMINAL && break
            if at == BackgammonNet.ACTION_TYPE_CHANCE; BackgammonNet.apply_chance!(g,_sample_dice(rng)); continue; end
            if BackgammonNet.is_contact_position(g)
                length(pos)<n && push!(pos, BenchPosition(BackgammonNet.clone(g), true, Int(g.current_player)))
            end
            play_turn!(g, gen, fail); g.terminated && break; length(pos)>=n && break
        end
    end
    return pos, gi
end

function enumerate_afters(P)
    sp = P.current_player
    seen = Dict{Tuple{UInt128,UInt128}, BackgammonNet.BackgammonGame}()
    stack = BackgammonNet.BackgammonGame[BackgammonNet.clone(P)]
    while !isempty(stack)
        s = pop!(stack); att = BackgammonNet.action_type(s)
        done = (att==BackgammonNet.ACTION_TYPE_TERMINAL || s.current_player!=sp || att==BackgammonNet.ACTION_TYPE_CHANCE)
        if done; k=(s.p0,s.p1); haskey(seen,k)||(seen[k]=s); continue; end
        acts = BackgammonNet.legal_actions(s)
        if isempty(acts); k=(s.p0,s.p1); haskey(seen,k)||(seen[k]=s); continue; end
        for a in acts; s2=BackgammonNet.clone(s); BackgammonNet.apply_action!(s2,a); push!(stack,s2); end
    end
    collect(values(seen))
end

function value_equities!(out, states, idxs, cnet, rnet; chunk=1024)
    cidx=Int[]; ridx=Int[]
    for i in idxs; BackgammonNet.is_contact_position(states[i]) ? push!(cidx,i) : push!(ridx,i); end
    _eval_group!(out, states, cidx, cnet, chunk); _eval_group!(out, states, ridx, rnet, chunk); out
end
function _eval_group!(out, states, group, net, chunk)
    net===nothing && return
    n=length(group); p=1
    while p<=n
        q=min(p+chunk-1,n); m=q-p+1
        X=Matrix{Float32}(undef,STATE_DIM,m); Amask=ones(Float32,NUM_ACTIONS,m)
        @inbounds for k in 1:m; vectorize_state_into!(view(X,:,k), gspec, states[group[p+k-1]]); end
        _,Lw,Lgw,Lbgw,Lgl,Lbgl,_ = FluxLib.forward_normalized_multihead(net,X,Amask)
        @inbounds for k in 1:m
            gi=group[p+k-1]
            heads=(_sigmoid(Lw[1,k]),_sigmoid(Lgw[1,k]),_sigmoid(Lbgw[1,k]),_sigmoid(Lgl[1,k]),_sigmoid(Lbgl[1,k]))
            out[gi]=Float64(BackgammonNet.compute_cubeless_equity(states[gi],heads))
        end
        p=q+1
    end
end
const TERMINAL_WIN = 12.0
# 0-ply-B: my equity = -E_r[V(A+r)]  (opponent post-roll, in-distribution)
function scoreB(afters, cnet, rnet)
    na=length(afters); sc=fill(-Inf,na); live=Int[]
    for i in 1:na; afters[i].terminated ? (sc[i]=TERMINAL_WIN) : push!(live,i); end
    isempty(live) && return sc
    rolls=BackgammonNet.BackgammonGame[]; owner=Int[]; w=Float64[]
    for i in live
        base=afters[i]
        for r in 1:21
            pr=BackgammonNet.DICE_PROBS[r]; pr==0f0 && continue
            gr=BackgammonNet.clone(base); BackgammonNet.apply_chance!(gr,r)
            push!(rolls,gr); push!(owner,i); push!(w,Float64(pr))
        end
    end
    vR=fill(NaN,length(rolls)); value_equities!(vR,rolls,collect(1:length(rolls)),cnet,rnet)
    acc=Dict{Int,Float64}()
    for k in 1:length(rolls); acc[owner[k]]=get(acc,owner[k],0.0)+w[k]*vR[k]; end
    for i in live; sc[i]=-acc[i]; end
    sc
end

function load_net(path,w,b)
    isfile(path) || return nothing
    net=FluxLib.FCResNetMultiHead(gspec,FluxLib.FCResNetMultiHeadHP(width=w,num_blocks=b))
    FluxLib.load_weights(path,net); Flux.cpu(net)
end

function main()
    nw=A["num_workers"]; seed=A["seed"]
    println("="^88)
    println("AFTERSTATE CLASS REGRET  net=$(A["net_ckpt"])  tag=$(A["tag"])")
    println("="^88); flush(stdout)
    cnet = load_net(A["net_ckpt"],A["width"],A["blocks"]); cnet===nothing && error("net missing")
    rnet = load_net(A["race_ckpt"],A["race_width"],A["race_blocks"])

    print("generating $(A["n_positions"]) benchmark positions (seed-$seed gnubg-0ply)... "); flush(stdout)
    gen = make_gnubg(0); t=time()
    pos, ng = gen_positions(gen, A["n_positions"], seed, A["max_games"]); try BackgammonNet.close(gen) catch end
    @printf("%d from %d games (%.1fs)\n", length(pos), ng, time()-t); flush(stdout)

    n=length(pos)
    greedy = Vector{MoveDecision}(undef,n); cls = Vector{Symbol}(undef,n)
    idx=Threads.Atomic{Int}(0)
    print("scoring greedy 0-ply-B... "); flush(stdout); t=time()
    Threads.@threads for w in 1:nw
        while true
            i=Threads.atomic_add!(idx,1)+1; i>n && break
            bp=pos[i]; cls[i]=classify(bp.state)
            afters=enumerate_afters(bp.state); sB=scoreB(afters,cnet,rnet)
            gB=afters[argmax(sB)]
            greedy[i]=MoveDecision(bp.state,gB.p0,gB.p1,bp.is_contact,bp.player)
        end
    end
    @printf("%.1fs\n", time()-t); flush(stdout)

    # grade
    gbs=[make_gnubg(A["gnubg_ply"]) for _ in 1:nw]
    reg=fill(NaN,n); st=Vector{Symbol}(undef,n); idx2=Threads.Atomic{Int}(0)
    print("grading (gnubg-ply$(A["gnubg_ply"]))... "); flush(stdout); t=time()
    Threads.@threads for w in 1:nw
        gb=gbs[w]
        while true
            i=Threads.atomic_add!(idx2,1)+1; i>n && break
            tag,e = native_regret(gb, greedy[i]); st[i]=tag; tag==:ok && (reg[i]=e)
        end
    end
    for gb in gbs; try BackgammonNet.close(gb) catch end; end
    @printf("%.1fs\n\n", time()-t); flush(stdout)

    ok = st .== :ok
    classes=[:BEAR_IN_VS_CONTACT,:DEEP_ANCHOR,:BACK_CHECKER_MID,:CONTACT_GENERIC]
    println("PER-CLASS greedy-0plyB native regret (PR = 500*mean regret, lower=better):")
    @printf("  %-20s %6s %8s %10s\n","class","n","PR","lossShare%")
    total_loss = sum(reg[ok])
    for c in classes
        sel = ok .& (cls .== c); e = reg[sel]
        if isempty(e); @printf("  %-20s %6d %8s\n", c, 0, "-"); continue; end
        @printf("  %-20s %6d %8.2f %9.1f%%\n", c, length(e), 500*mean(e), 100*sum(e)/total_loss)
    end
    e_all = reg[ok]
    @printf("  %-20s %6d %8.2f\n","ALL contact",length(e_all),500*mean(e_all))
    # top-10% equity-loss concentration
    s = sort(e_all; rev=true); k=max(1,round(Int,0.10*length(s)))
    @printf("\nTOP-10%% decisions carry %.1f%% of total equity loss (n_top=%d)\n",
            100*sum(s[1:k])/sum(s), k)
    println("(forced=$(count(==(:forced),st)) unmatched=$(count(==(:unmatched),st)))")
    println("="^88)
end
main()
