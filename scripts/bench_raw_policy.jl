#!/usr/bin/env julia
# bench_raw_policy.jl — RAW-POLICY (no-search) contact PR on the SAME common,
# seed-deterministic benchmark set that scripts/benchmark_pr.jl uses, graded by
# gnubg ply-1 NATIVE move lists (verbatim grading contract copied from
# scripts/benchmark_pr.jl / analyze_pr_native.jl).
#
# WHY THIS EXISTS: benchmark_pr.jl --mcts-iters 1 is DEGENERATE — with 1–2 MCTS
# simulations the selected move is NET-INDEPENDENT (verified: two different nets
# produce identical resulting boards on 100% of turns at iters≤2, but differ on
# ~37% at iters≥25). So "raw = mcts-1" does NOT measure the policy head. This
# script instead plays each full turn by the network's RAW POLICY ARGMAX
# (Network.forward_normalized, greedy) — the true no-search policy.
#
# Generates the position set ONCE and grades every requested checkpoint on it.
#
# Usage:
#   julia --threads 14 --project scripts/bench_raw_policy.jl \
#       --ckpts i140=<contact>:<race>,cov60=<contact>:<race> \
#       --n-positions 1000 --seed 1 --num-workers 14

using ArgParse, Random, Statistics, Printf

function parse_a()
    s = ArgParseSettings(autofix_names=true)
    @add_arg_table! s begin
        "--ckpts";       arg_type=String; required=true  # tag=contact:race,tag2=contact2:race2
        "--width";       arg_type=Int; default=256
        "--blocks";      arg_type=Int; default=5
        "--race-width";  arg_type=Int; default=128
        "--race-blocks"; arg_type=Int; default=3
        "--n-positions"; arg_type=Int; default=1000
        "--seed";        arg_type=Int; default=1
        "--num-workers"; arg_type=Int; default=12
        "--max-games";   arg_type=Int; default=100_000
        "--obs-type";    arg_type=String; default="min_plus_flat"
    end
    parse_args(s)
end
const A = parse_a()

ENV["BACKGAMMON_OBS_TYPE"] = A["obs_type"]
using AlphaZero
using AlphaZero: GI, FluxLib, Network
import Flux
using BackgammonNet
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
const gspec = GameSpec()
const NA = GI.num_actions(gspec)
const SD = length(vec(GI.vectorize_state(gspec, GI.current_state(GI.init(gspec)))))

# ── verbatim grading contract from benchmark_pr.jl ─────────────────────────────
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
    best_eq = -Inf; our_eq = nothing
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
function _sample_dice(rng)
    r = rand(rng, Float32); c = 0.0f0
    @inbounds for i in 1:length(BackgammonNet.DICE_PROBS)
        c += BackgammonNet.DICE_PROBS[i]; r <= c && return i
    end
    return length(BackgammonNet.DICE_PROBS)
end
function play_turn_with_engine!(g, engine, rng_unused, fail_counter::Threads.Atomic{Int})
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
function generate_common_positions(gen_engine, n_contact::Int, base_seed::Int, max_games::Int)
    contact = BenchPosition[]; fail = Threads.Atomic{Int}(0); gi = 0
    while length(contact) < n_contact && gi < max_games
        gi += 1
        rng = MersenneTwister(base_seed + gi)
        g = BackgammonNet.initial_state()
        while true
            at = BackgammonNet.action_type(g)
            at == BackgammonNet.ACTION_TYPE_TERMINAL && break
            if at == BackgammonNet.ACTION_TYPE_CHANCE
                BackgammonNet.apply_chance!(g, _sample_dice(rng)); continue
            end
            pre = BackgammonNet.clone(g)
            is_contact = BackgammonNet.is_contact_position(g)
            player = Int(g.current_player)
            is_contact && length(contact) < n_contact && push!(contact, BenchPosition(pre, true, player))
            play_turn_with_engine!(g, gen_engine, rng, fail)
            g.terminated && break
            length(contact) >= n_contact && break
        end
    end
    return contact, gi, fail[]
end
function score_moves(gnubg_backends, decisions, num_workers)
    n = length(decisions)
    errors = fill(NaN, n); contact_flag = Vector{Bool}(undef, n)
    player_of = Vector{Int}(undef, n); status = Vector{Symbol}(undef, n)
    idx = Threads.Atomic{Int}(0)
    Threads.@threads for w in 1:num_workers
        gb = gnubg_backends[w]
        while true
            i = Threads.atomic_add!(idx, 1) + 1
            i > n && break
            d = decisions[i]; contact_flag[i] = d.is_contact; player_of[i] = d.player
            local res
            try res = native_regret(gb, d)
            catch e; status[i] = :error; continue end
            tag, err, _ = res; status[i] = tag; tag == :ok && (errors[i] = err)
        end
    end
    return errors, contact_flag, player_of, status
end
function summarize(errors, contact_flag, player_of, status)
    ok = status .== :ok; n_ok = count(ok); n_ok == 0 && return nothing
    e_ok = errors[ok]; mean_err = sum(e_ok)/n_ok
    subset(sel) = begin
        e = errors[ok .& sel]
        isempty(e) ? (n=0, ER=NaN, PR=NaN) : (n=length(e), ER=1000.0*sum(e)/length(e), PR=500.0*sum(e)/length(e))
    end
    return (n_ok=n_ok, n_forced=count(==(:forced),status), n_unmatched=count(==(:unmatched),status),
            n_error=count(==(:error),status), PR=500.0*mean_err, ER=1000.0*mean_err,
            contact=subset(contact_flag), p0=subset(player_of.==0), p1=subset(player_of.==1))
end

# ── raw-policy net play ────────────────────────────────────────────────────────
function loadnet(path, w, b)
    net = FluxLib.FCResNetMultiHead(gspec, FluxLib.FCResNetMultiHeadHP(width=w, num_blocks=b))
    FluxLib.load_weights(path, net); Flux.cpu(net)
end
# raw policy argmax over legal actions of state s (routes contact→cn, race→rn)
function raw_argmax(cn, rn, s)
    net = BackgammonNet.is_contact_position(s) ? cn : rn
    buf = zeros(Float32, SD)
    vectorize_state_into!(buf, gspec, s)   # fixed 350-dim inference encoding (matches oracle)
    X = reshape(buf, :, 1)
    Am = zeros(Float32, NA, 1)
    la = BackgammonNet.legal_actions(s)
    for a in la; (1<=a<=NA) && (Am[a,1]=1f0); end
    Praw,_ = Network.convert_output_tuple(net, Network.forward_normalized(net, X, Am))
    best = la[1]; bp = -Inf32
    for a in la; if Praw[a,1] > bp; bp = Praw[a,1]; best = a; end; end
    best
end
function raw_decisions(cn, rn, positions::Vector{BenchPosition})
    out = Vector{MoveDecision}(undef, length(positions))
    for (i, bp) in enumerate(positions)
        g = BackgammonNet.clone(bp.state); start = g.current_player
        while true
            att = BackgammonNet.action_type(g)
            (att == BackgammonNet.ACTION_TYPE_TERMINAL || att == BackgammonNet.ACTION_TYPE_CHANCE) && break
            g.current_player != start && break
            acts = BackgammonNet.legal_actions(g); isempty(acts) && break
            a = length(acts)==1 ? acts[1] : raw_argmax(cn, rn, BackgammonNet.clone(g))
            BackgammonNet.apply_action!(g, a); g.terminated && break
        end
        out[i] = MoveDecision(bp.state, g.p0, g.p1, bp.is_contact, bp.player)
    end
    out
end

# ── main ───────────────────────────────────────────────────────────────────────
nw = A["num_workers"]
println("Generating common contact set (gnubg-0ply, seed $(A["seed"]), n=$(A["n_positions"]))..."); flush(stdout)
gen = BackgammonNet.GnubgCLibBackend(ply=0, threads=1); BackgammonNet.open!(gen)
positions, ngames, gfail = generate_common_positions(gen, A["n_positions"], A["seed"], A["max_games"])
try BackgammonNet.close(gen) catch end
println("  $(length(positions)) contact positions from $ngames games (gen fallbacks=$gfail)"); flush(stdout)

println("Opening $nw gnubg ply-1 grading backends..."); flush(stdout)
gbs = [begin gb=BackgammonNet.GnubgCLibBackend(ply=1, threads=1); BackgammonNet.open!(gb); gb end for _ in 1:nw]

specs = split(A["ckpts"], ",")
println("\n" * "="^74)
@printf("%-10s %8s %8s   %6s %6s   %s\n", "tag", "PR", "ER", "P0", "P1", "n(scored/forced/unmatched)")
println("="^74)
for spec in specs
    tag, paths = split(spec, "="); cpath, rpath = split(paths, ":")
    cn = loadnet(String(cpath), A["width"], A["blocks"])
    rn = loadnet(String(rpath), A["race_width"], A["race_blocks"])
    decs = raw_decisions(cn, rn, positions)
    errors, cflag, pof, status = score_moves(gbs, decs, nw)
    s = summarize(errors, cflag, pof, status)
    @printf("%-10s %8.2f %8.2f   %6.1f %6.1f   %d/%d/%d\n",
            String(tag), s.contact.PR, s.contact.ER, s.p0.PR, s.p1.PR,
            s.n_ok, s.n_forced, s.n_unmatched)
    flush(stdout)
end
println("="^74)
for gb in gbs; try BackgammonNet.close(gb) catch end; end
println("DONE")
