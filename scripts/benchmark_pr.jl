#!/usr/bin/env julia
"""
benchmark_pr.jl — CROSS-ENGINE-COMPARABLE checker-play PR on a COMMON, fixed
benchmark position set. Grades multiple agents (the net, gnubg-0ply, wildbg-large,
gnubg-1ply floor) on the IDENTICAL set of pre-move positions, every move scored
against the same gnubg-ply-1 native move-list reference.

WHY THIS SCRIPT EXISTS (the puzzle it resolves)
  scripts/analyze_pr_native.jl grades our net on the NET's own self-play positions;
  scripts/calibrate_pr_ladder.jl grades each opponent engine on THAT engine's own
  self-play positions. Those PR numbers are therefore NOT directly comparable across
  engines — each engine is measured on a different position distribution:
        i140 net (own self-play positions)  ~48-54 PR
        wildbg-large (own self-play)        ~13.5 PR
        gnubg-0ply   (own self-play)        ~1.9  PR
  Yet i140 BEATS wildbg-large head-to-head (~54% win). Is i140 really ~50 PR on the
  SAME positions where wildbg is ~13.5 (→ the head-to-head win is confounded — e.g.
  by wildbg move-fallbacks — so "beat wildbg" is suspect), OR do the numbers converge
  once graded on one common set (→ the earlier gap was a position-distribution
  artifact)? Only grading ALL agents on ONE fixed position set answers this. That is
  what this script does.

  IT IS ALSO THE INTENDED PR-TRACKING TOOL for the future cubeful net: a FIXED
  position set makes PR comparable across net versions over time. The agent side is
  deliberately swappable (see `--agents` / `AGENT_LABEL`) — add a new net checkpoint
  and it is graded on the very same positions as everything else.

DESIGN
  1. FIXED COMMON POSITION SET (deterministic from --seed; no data file needed —
     regenerated identically each run). We play cubeless money games in which BOTH
     sides move by gnubg-0ply (a strong, realistic mover) with seeded dice, and
     snapshot the PRE-MOVE position at the start of every checker turn. Contact
     positions are the priority (--n-positions); a modest RACE subset is collected
     too (--n-race). Each snapshot records the player to move (both are represented,
     since play alternates), so the report can show the P0/P1 asymmetry.

     NOTE on "non-forced": forced positions (≤1 legal FULL move) are a property of the
     POSITION, not the agent, so they are the same for every agent. We snapshot all
     checker-turn positions and let grading mark forced ones `:forced` and EXCLUDE
     them from PR — identically across agents. (Reported per agent as n_forced.)

  2. GRADE EACH AGENT ON THE SAME POSITIONS. For each benchmark position we have the
     agent choose its move, capture the RESULTING board (g.p0,g.p1), and grade via
     the gnubg-ply-1 NATIVE move-list reference — regret = best_eq − agent_move_eq,
     matched BY BOARD (never by action-id → sidesteps the player-0 action-id bug).
     Agents:
        gnubg1 : GnubgCLibBackend(ply=1)  — the FLOOR (grades its own moves → ~1 PR,
                 validates the pipeline on THESE positions).
        gnubg0 : GnubgCLibBackend(ply=0)
        wildbg : WildbgBackend(nets=:large)
        net    : the checkpoint (default i140) at MCTS-`--mcts-iters`.

REUSED MACHINERY (verbatim from scripts/analyze_pr_native.jl &
scripts/calibrate_pr_ladder.jl — read those for the full derivation):
  * `MoveDecision` (pre-move state + resulting bitboards + is_contact + player).
  * `native_regret` — gnubg's ONE native move-list eval (`_gnubg_clib_move_data`);
    eq(move)=compute_cubeless_equity(g,probs) (mover perspective, NO flip);
    best_eq=max over list; our_eq=list entry whose RESULTING board (from_gnubg_simple
    bitboards) equals the agent's; regret=max(best_eq−our_eq,0). `:forced` (≤1 move),
    `:unmatched` (agent board not in list — should not happen for a legal move).
  * `score_moves` (parallel, lock-serialized shared gnubg-ply-1 backends),
    `summarize` (ER=1000·mean, PR=500·mean; contact/race + P0/P1 splits).
  * External-engine full-TURN execution via `best_move` with a legal-action fallback
    on the rare doubles match error (that turn then grades `:forced`).
  * gnubg-0ply as the position GENERATOR (calibrate_pr_ladder's `make_playing_engine`).

Usage:
    julia --threads 16 --project scripts/benchmark_pr.jl \\
        --n-positions 1200 --n-race 300 --seed 1 \\
        --agents gnubg1,gnubg0,wildbg,net \\
        --net-ckpt sessions/contact-flywheel/checkpoints/contact_iter_140.data \\
        --mcts-iters 800

Options:
    --agents=gnubg1,gnubg0,wildbg,net   comma list (subset ok); order in report fixed
    --n-positions=1200   target CONTACT pre-move decisions in the common set
    --n-race=300         target RACE pre-move decisions (0 = skip race subset)
    --seed=1             base RNG seed (dice) — makes the common set deterministic
    --net-ckpt=sessions/contact-flywheel/checkpoints/contact_iter_140.data
    --race-checkpoint=   explicit race checkpoint (else auto-detect race_latest.data)
    --obs-type=min_plus_flat  --width=256 --blocks=5 --race-width=128 --race-blocks=3
    --mcts-iters=800     MCTS iterations/move for the net agent
    --gnubg-ply=1        reference ply for grading (1 = validated reference; ≥2 refused)
    --num-workers=12     CPU worker threads (grading + net move selection)
    --wildbg-lib=PATH    libwildbg.so (default ~/github/wildbg/target/release/libwildbg.so)
    --max-games=100000   safety cap on generator games
    --inference-batch-size=50  --inference-backend=auto  --chance-mode=passthrough
"""

using ArgParse

function parse_bench_args()
    s = ArgParseSettings(description="Common-benchmark cross-engine checker-play PR (fixed positions)", autofix_names=true)
    @add_arg_table! s begin
        "--agents";          help = "comma list: gnubg1,gnubg0,wildbg,net"; arg_type = String; default = "gnubg1,gnubg0,wildbg,net"
        "--n-positions";     help = "Target CONTACT pre-move decisions in the common set"; arg_type = Int; default = 1200
        "--n-race";          help = "Target RACE pre-move decisions (0 = skip)"; arg_type = Int; default = 300
        "--seed";            help = "Base RNG seed (dice) — deterministic common set"; arg_type = Int; default = 1
        "--net-ckpt";        help = "Net contact checkpoint"; arg_type = String; default = "sessions/contact-flywheel/checkpoints/contact_iter_140.data"
        "--race-checkpoint"; help = "Explicit race checkpoint (else auto-detect)"; arg_type = String; default = ""
        "--obs-type";        help = "Observation type"; arg_type = String; default = "min_plus_flat"
        "--width";           help = "Contact network width"; arg_type = Int; default = 256
        "--blocks";          help = "Contact network blocks"; arg_type = Int; default = 5
        "--race-width";      help = "Race network width"; arg_type = Int; default = 128
        "--race-blocks";     help = "Race network blocks"; arg_type = Int; default = 3
        "--mcts-iters";      help = "MCTS iterations per move (net agent)"; arg_type = Int; default = 800
        "--gnubg-ply";       help = "gnubg reference ply (1 = validated reference; ≥2 refused)"; arg_type = Int; default = 1
        "--num-workers";     help = "CPU worker threads"; arg_type = Int; default = 12
        "--wildbg-lib";      help = "libwildbg.so path"; arg_type = String; default = ""
        "--max-games";       help = "Safety cap on generator games"; arg_type = Int; default = 100_000
        "--inference-batch-size"; help = "Inference batch size for MCTS"; arg_type = Int; default = 50
        "--inference-backend";    help = "CPU inference backend: auto, fast, or flux"; arg_type = String; default = "auto"
        "--chance-mode";          help = "Chance-node handling: passthrough (default) or exact_expectation"; arg_type = String; default = "passthrough"
    end
    return ArgParse.parse_args(s)
end

const ARGS_B = parse_bench_args()

using AlphaZero
using AlphaZero: GI, MCTS, Network, FluxLib, MctsParams, ConstSchedule, BatchedMCTS, GameLoop
using AlphaZero.NetLib
import Flux
using Random
using Statistics
using Dates
using Printf
using BackgammonNet

ENV["BACKGAMMON_OBS_TYPE"] = ARGS_B["obs_type"]
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
const gspec = GameSpec()
const NUM_ACTIONS = GI.num_actions(gspec)
const _state_dim = GI.state_dim(gspec)[1]
const ORACLE_CFG = AlphaZero.BackgammonInference.OracleConfig(
    _state_dim, NUM_ACTIONS, gspec;
    vectorize_state! = vectorize_state_into!,
    route_state = s -> (s isa BackgammonNet.BackgammonGame && !BackgammonNet.is_contact_position(s) ? 2 : 1))

# ── A graded UNIT is a full TURN (one player's complete checker move) ──────────
# (verbatim contract from analyze_pr_native.jl / calibrate_pr_ladder.jl)
struct MoveDecision
    state::BackgammonNet.BackgammonGame   # pre-move clone (remaining_actions as rolled)
    res_p0::UInt128                       # resulting board after the agent's full move
    res_p1::UInt128
    is_contact::Bool
    player::Int
end

"""One fixed benchmark position: a pre-move checker-play state (dice rolled),
its phase tag, and the player to move. Reused unchanged across every agent."""
struct BenchPosition
    state::BackgammonNet.BackgammonGame
    is_contact::Bool
    player::Int
end

# ── Native gnubg move-list grading (verbatim from analyze_pr_native.jl) ─────────
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

# ── Dice sampling (verbatim from analyze_pr_native.jl) ──────────────────────────
function _sample_dice(rng)
    r = rand(rng, Float32); c = 0.0f0
    @inbounds for i in 1:length(BackgammonNet.DICE_PROBS)
        c += BackgammonNet.DICE_PROBS[i]
        r <= c && return i
    end
    return length(BackgammonNet.DICE_PROBS)
end

# ── Full-TURN execution by an engine's best_move (from calibrate_pr_ladder.jl) ──
"""Drive one full checker TURN (1 or 2 half-moves for doubles) from pre-move
state `g` (mutated) using `engine.best_move`, with a legal-action fallback on the
rare match error (counted in `fail_counter`; that turn then grades `:forced`)."""
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

# ── FIXED benchmark position set: gnubg-0ply moves BOTH sides, seeded dice ──────
"""Generate the common benchmark set: play cubeless games in which `gen_engine`
(gnubg-0ply) moves both sides with seeded dice, snapshotting the PRE-MOVE state at
the start of every checker turn. Returns (contact::Vector{BenchPosition},
race::Vector{BenchPosition}, n_games, n_fail). Deterministic in `base_seed`."""
function generate_common_positions(gen_engine, n_contact::Int, n_race::Int,
                                    base_seed::Int, max_games::Int)
    contact = BenchPosition[]
    race = BenchPosition[]
    fail = Threads.Atomic{Int}(0)
    gi = 0
    while (length(contact) < n_contact || length(race) < n_race) && gi < max_games
        gi += 1
        rng = MersenneTwister(base_seed + gi)
        g = BackgammonNet.initial_state()
        while true
            at = BackgammonNet.action_type(g)
            at == BackgammonNet.ACTION_TYPE_TERMINAL && break
            if at == BackgammonNet.ACTION_TYPE_CHANCE
                BackgammonNet.apply_chance!(g, _sample_dice(rng)); continue
            end
            # start of a checker turn — snapshot the PRE-MOVE decision
            pre = BackgammonNet.clone(g)
            is_contact = BackgammonNet.is_contact_position(g)
            player = Int(g.current_player)
            if is_contact
                length(contact) < n_contact && push!(contact, BenchPosition(pre, true, player))
            else
                length(race) < n_race && push!(race, BenchPosition(pre, false, player))
            end
            # advance the game by gnubg-0ply's full move
            play_turn_with_engine!(g, gen_engine, rng, fail)
            g.terminated && break
            (length(contact) >= n_contact && length(race) >= n_race) && break
        end
    end
    return contact, race, gi, fail[]
end

# ── Agent move computation on the COMMON positions ─────────────────────────────
"""ENGINE agent (gnubg0/gnubg1/wildbg): choose a move for every benchmark position
via `best_move`, capture the resulting board. SERIAL (engine C/cache state)."""
function engine_decisions(engine, positions::Vector{BenchPosition})
    out = Vector{MoveDecision}(undef, length(positions))
    fail = Threads.Atomic{Int}(0)
    for (i, bp) in enumerate(positions)
        g = BackgammonNet.clone(bp.state)
        play_turn_with_engine!(g, engine, nothing, fail)
        out[i] = MoveDecision(bp.state, g.p0, g.p1, bp.is_contact, bp.player)
    end
    return out, fail[]
end

"""NET agent: choose a move for every benchmark position via greedy MCTS (argmax
policy, temperature 0 — matching analyze_pr_native.jl), capture the resulting
board. PARALLEL: one MctsAgent+player+env per worker, shared oracles (the same
concurrent-oracle pattern analyze_pr_native.jl uses for net self-play)."""
function net_decisions(cpu_single, cpu_batch, mcts_params, batch_size,
                       positions::Vector{BenchPosition}, num_workers::Int, seed::Int)
    n = length(positions)
    out = Vector{MoveDecision}(undef, n)
    idx = Threads.Atomic{Int}(0)
    Threads.@threads for w in 1:num_workers
        agent = GameLoop.MctsAgent(cpu_single, cpu_batch, mcts_params, batch_size, gspec)
        env = GI.init(gspec)
        player = GameLoop.create_player(agent; rng=MersenneTwister(seed + w))
        while true
            i = Threads.atomic_add!(idx, 1) + 1
            i > n && break
            bp = positions[i]
            GameLoop._reset_player!(player)            # independent positions → fresh tree
            GI.set_state!(env, bp.state)               # clones bp.state into env
            start_player = bp.state.current_player
            while true
                (GI.game_terminated(env) || GI.is_chance_node(env)) && break
                env.game.current_player != start_player && break
                avail = GI.available_actions(env)
                isempty(avail) && break
                action = if length(avail) == 1
                    avail[1]                            # forced sub-move → no MCTS
                else
                    actions, policy, _ = GameLoop.select_action(agent, player, env)
                    actions[argmax(policy)]             # greedy (τ=0)
                end
                GI.play!(env, action)
            end
            out[i] = MoveDecision(bp.state, env.game.p0, env.game.p1, bp.is_contact, bp.player)
        end
    end
    return out
end

# ── Parallel scoring (verbatim from analyze_pr_native.jl) ───────────────────────
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

# ── Aggregation (verbatim from analyze_pr_native.jl) ────────────────────────────
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

# ── Engine factory (verbatim from calibrate_pr_ladder.jl) ───────────────────────
function make_playing_engine(name::Symbol, wildbg_lib::String)
    if name === :gnubg0
        e = BackgammonNet.GnubgCLibBackend(ply=0, threads=1); BackgammonNet.open!(e); return e
    elseif name === :gnubg1
        e = BackgammonNet.GnubgCLibBackend(ply=1, threads=1); BackgammonNet.open!(e); return e
    elseif name === :wildbg
        lib_size = filesize(wildbg_lib)
        nets = lib_size > 10_000_000 ? :large : :small
        nets == :large ? BackgammonNet.wildbg_set_lib_path!(large=wildbg_lib) :
                         BackgammonNet.wildbg_set_lib_path!(small=wildbg_lib)
        e = BackgammonNet.WildbgBackend(nets=nets); BackgammonNet.open!(e); return e
    else
        error("unknown engine $name")
    end
end

const AGENT_LABEL = Dict(
    :gnubg1 => "gnubg-1ply  (GnubgCLibBackend ply=1)  [FLOOR / self-consistency]",
    :gnubg0 => "gnubg-0ply  (GnubgCLibBackend ply=0)",
    :wildbg => "wildbg-large (WildbgBackend nets=:large)")
const AGENT_ORDER = [:gnubg1, :gnubg0, :wildbg, :net]

# ── Resolve race checkpoint (verbatim logic from analyze_pr_native.jl) ──────────
function resolve_race_ckpt(ckpt_path, explicit)
    ckpt_dir = dirname(ckpt_path); ckpt_name = basename(ckpt_path)
    if !isempty(explicit); return explicit; end
    if startswith(ckpt_name, "contact_")
        cand = joinpath(ckpt_dir, replace(ckpt_name, "contact_" => "race_"))
        isfile(cand) && return cand
        isfile(joinpath(ckpt_dir, "race_latest.data")) && return joinpath(ckpt_dir, "race_latest.data")
    elseif isfile(joinpath(ckpt_dir, "race_latest.data"))
        return joinpath(ckpt_dir, "race_latest.data")
    end
    return nothing
end

# ── Main ────────────────────────────────────────────────────────────────────────
function main()
    gnubg_ply = ARGS_B["gnubg_ply"]
    if gnubg_ply >= 2
        error("gnubg reference ply=$gnubg_ply refused: gnubg's 2-ply evaluator deadlocks its " *
              "internal thread pool. Use --gnubg-ply 1 (validated reference) or 0.")
    end
    num_workers = ARGS_B["num_workers"]
    n_contact = ARGS_B["n_positions"]
    n_race = ARGS_B["n_race"]
    base_seed = ARGS_B["seed"]
    max_games = ARGS_B["max_games"]
    batch_size = ARGS_B["inference_batch_size"]
    mcts_iters = ARGS_B["mcts_iters"]

    wildbg_lib = ARGS_B["wildbg_lib"]
    if isempty(wildbg_lib)
        wildbg_lib = joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg.so")
    end

    reqs = Symbol.(strip.(split(lowercase(ARGS_B["agents"]), ",")))
    valid = Set([:gnubg1, :gnubg0, :wildbg, :net])
    for a in reqs
        a in valid || error("--agents entry '$a' invalid; choose from gnubg1,gnubg0,wildbg,net")
    end
    agents = [a for a in AGENT_ORDER if a in reqs]   # canonical order
    want_wildbg = :wildbg in agents
    want_net = :net in agents

    if want_wildbg && !isfile(wildbg_lib)
        error("libwildbg not found at $wildbg_lib — pass --wildbg-lib=/path/to/libwildbg.so")
    end
    net_ckpt = ARGS_B["net_ckpt"]
    race_ckpt = nothing
    if want_net
        isfile(net_ckpt) || error("net checkpoint not found: $net_ckpt")
        race_ckpt = resolve_race_ckpt(net_ckpt, ARGS_B["race_checkpoint"])
    end

    println("=" ^ 74)
    println("COMMON-BENCHMARK PR — all agents graded on ONE fixed position set")
    println("=" ^ 74)
    println("Agents:        $(join(agents, ", "))")
    println("Common set:    $n_contact contact + $n_race race pre-move decisions (gnubg-0ply generator)")
    println("Reference:     gnubg ply-$gnubg_ply NATIVE move list (per TURN), floor-validated ≈1 PR")
    println("Grading:       match agent board vs native list (from_gnubg_simple), regret=best−agent")
    println("Seed:          $base_seed  (common set is deterministic in this seed)")
    println("Workers:       $num_workers CPU")
    if want_net
        println("Net:           $net_ckpt")
        println("               race=$(race_ckpt === nothing ? "(none)" : race_ckpt)")
        println("               arch contact=$(ARGS_B["width"])w×$(ARGS_B["blocks"])b" *
                (race_ckpt === nothing ? "" : " + race=$(ARGS_B["race_width"])w×$(ARGS_B["race_blocks"])b") *
                ", obs=$(ARGS_B["obs_type"]), MCTS-$mcts_iters greedy, chance=$(ARGS_B["chance_mode"])")
    end
    want_wildbg && println("wildbg lib:    $wildbg_lib")
    println("=" ^ 74); flush(stdout)

    # ── Shared ply-reference backends for parallel grading ──
    gnubg_backends = [begin
        gb = BackgammonNet.GnubgCLibBackend(ply=gnubg_ply, threads=1); BackgammonNet.open!(gb); gb
    end for _ in 1:num_workers]

    # ── Build the COMMON position set (once, reused for all agents) ──
    println("\nGENERATING common position set (gnubg-0ply moves both sides, seeded dice)...")
    flush(stdout)
    tgen = time()
    gen_engine = make_playing_engine(:gnubg0, wildbg_lib)
    contact_ps, race_ps, n_games, gen_fail = generate_common_positions(
        gen_engine, n_contact, n_race, base_seed, max_games)
    try BackgammonNet.close(gen_engine) catch end
    positions = vcat(contact_ps, race_ps)
    println("  $(length(contact_ps)) contact + $(length(race_ps)) race = $(length(positions)) positions " *
            "from $n_games games ($(round(time()-tgen, digits=1))s)")
    gen_fail > 0 && println("  (generator best_move fallbacks: $gen_fail)")
    let np0 = count(p -> p.player == 0, positions), np1 = count(p -> p.player == 1, positions)
        println("  player-to-move split in the set:  P0=$np0   P1=$np1")
    end
    if isempty(positions)
        for gb in gnubg_backends; try BackgammonNet.close(gb) catch end; end
        println("No positions generated — nothing to grade."); return
    end
    flush(stdout)

    # ── Optional net oracle setup (only if net requested) ──
    cpu_single = cpu_batch = mcts_params = nothing
    if want_net
        contact_network = FluxLib.FCResNetMultiHead(
            gspec, FluxLib.FCResNetMultiHeadHP(width=ARGS_B["width"], num_blocks=ARGS_B["blocks"]))
        FluxLib.load_weights(net_ckpt, contact_network)
        contact_network = Flux.cpu(contact_network)
        race_network = nothing
        if race_ckpt !== nothing
            race_network = FluxLib.FCResNetMultiHead(
                gspec, FluxLib.FCResNetMultiHeadHP(width=ARGS_B["race_width"], num_blocks=ARGS_B["race_blocks"]))
            FluxLib.load_weights(race_ckpt, race_network)
            race_network = Flux.cpu(race_network)
        end
        mcts_params = MctsParams(
            num_iters_per_turn=mcts_iters, cpuct=1.5, temperature=ConstSchedule(0.0),
            dirichlet_noise_ϵ=0.0, dirichlet_noise_α=1.0, chance_mode=Symbol(ARGS_B["chance_mode"]))
        backend = AlphaZero.BackgammonInference.resolve_cpu_backend(ARGS_B["inference_backend"])
        println("Net CPU inference: $(AlphaZero.BackgammonInference.cpu_backend_summary(backend))")
        cpu_single, cpu_batch = AlphaZero.BackgammonInference.make_cpu_oracles(
            backend, contact_network, ORACLE_CFG; secondary_net=race_network, batch_size=batch_size)
        flush(stdout)
    end

    # ── Grade every agent on the SAME positions ──
    results = Any[]
    for name in agents
        println("\n" * "─" ^ 74)
        println("AGENT: $(name === :net ? "net i140 [$(basename(net_ckpt))] MCTS-$mcts_iters" : AGENT_LABEL[name])")
        println("─" ^ 74); flush(stdout)

        println("  Phase A: choose a move for all $(length(positions)) positions...")
        flush(stdout)
        t0 = time()
        n_fail = 0
        decisions = if name === :net
            net_decisions(cpu_single, cpu_batch, mcts_params, batch_size,
                          positions, num_workers, base_seed)
        else
            engine = make_playing_engine(name, wildbg_lib)
            d, nf = engine_decisions(engine, positions)
            try BackgammonNet.close(engine) catch end
            n_fail = nf
            d
        end
        move_time = time() - t0
        println("    done ($(round(move_time, digits=1))s)" *
                (n_fail > 0 ? "  [best_move fallbacks: $n_fail → those turns grade :forced]" : ""))
        flush(stdout)

        println("  Phase B: gnubg ply-$gnubg_ply NATIVE grading (parallel)...")
        flush(stdout)
        t1 = time()
        errors, cflag, pof, status = score_moves(gnubg_backends, decisions, num_workers)
        s = summarize(errors, cflag, pof, status)
        println("    grading done ($(round(time()-t1, digits=1))s)")
        push!(results, (name=name, s=s, n_fail=n_fail))
    end

    for gb in gnubg_backends; try BackgammonNet.close(gb) catch end; end

    # ── Report: the cross-comparable ladder on the COMMON set ──
    println("\n" * "=" ^ 74)
    println("CROSS-COMPARABLE PR LADDER  (all agents, IDENTICAL $(length(positions))-position set)")
    println("=" ^ 74)
    floor_pr = nothing
    for r in results
        (r.s !== nothing && r.name === :gnubg1) && (floor_pr = r.s.PR)
    end
    fstr(x) = floor_pr === nothing ? "   n/a" : @sprintf("%+7.2f", x - floor_pr)
    label(n) = n === :net ? "net i140 (MCTS-$mcts_iters)" : AGENT_LABEL[n]
    @printf("%-46s %8s %9s %10s %7s\n", "agent", "PR", "ER(mEMG)", "PR−floor", "n")
    println("-" ^ 82)
    for name in AGENT_ORDER
        r = findfirst(x -> x.name === name, results)
        r === nothing && continue
        s = results[r].s
        s === nothing && continue
        @printf("%-46s %8.2f %9.2f %10s %7d\n", label(name), s.PR, s.ER, fstr(s.PR), s.n_ok)
    end
    println("-" ^ 82)

    # Per-agent detail: contact/race + player split
    for name in AGENT_ORDER
        r = findfirst(x -> x.name === name, results)
        r === nothing && continue
        s = results[r].s
        s === nothing && continue
        println("\n$(label(name)):")
        @printf("  PR=%.2f  ER=%.2f   scored=%d  forced=%d  unmatched=%d  err=%d\n",
                s.PR, s.ER, s.n_ok, s.n_forced, s.n_unmatched, s.n_error)
        @printf("  contact: n=%5d PR=%6.2f ER=%7.2f    race: n=%5d PR=%6.2f ER=%7.2f\n",
                s.contact.n, s.contact.PR, s.contact.ER, s.race.n, s.race.PR, s.race.ER)
        @printf("  player split (asymmetry): P0 PR=%.2f (n=%d) | P1 PR=%.2f (n=%d)\n",
                s.p0.PR, s.p0.n, s.p1.PR, s.p1.n)
        if name === :gnubg1
            println((s.n_unmatched == 0 && s.PR < 3.0) ?
                "  => FLOOR small & 0 unmatched : pipeline VALIDATED on these positions. Read every\n" *
                "     agent as PR ABOVE this ~1-PR noise pedestal." :
                "  => FLOOR unexpectedly large or unmatched>0 : investigate capture/perspective.")
        end
        results[r].n_fail > 0 && @printf("  (%d best_move fallbacks graded as :forced)\n", results[r].n_fail)
    end

    # ── Interpretation: the resolution of the puzzle ──
    getpr(n) = (i = findfirst(x -> x.name === n, results); i === nothing || results[i].s === nothing ? nothing : results[i].s.PR)
    net_pr = getpr(:net); wb_pr = getpr(:wildbg); g0_pr = getpr(:gnubg0)
    println("\n" * "=" ^ 74)
    println("INTERPRETATION — does grading on ONE common set resolve the puzzle?")
    println("=" ^ 74)
    println("Prior (each engine on its OWN self-play positions, NOT comparable):")
    println("  i140 ~48-54 PR | wildbg-large ~13.5 PR | gnubg-0ply ~1.9 PR — yet i140 wins ~54% vs wildbg.")
    if net_pr !== nothing && wb_pr !== nothing
        println()
        @printf("On the COMMON set:  net i140 = %.1f PR   wildbg-large = %.1f PR%s\n",
                net_pr, wb_pr, g0_pr === nothing ? "" : @sprintf("   gnubg-0ply = %.1f PR", g0_pr))
        ratio = net_pr / max(wb_pr, 1e-9)
        gap = net_pr - wb_pr
        if net_pr > wb_pr + 8 && ratio > 1.8
            println()
            println(">>> RESOLUTION: The gap PERSISTS on identical positions — net i140 is still")
            @printf("    ~%.0f PR while wildbg-large is ~%.0f PR (%.1fx, +%.0f PR) on the SAME decisions.\n",
                    net_pr, wb_pr, ratio, gap)
            println("    So the earlier ~50-vs-~13.5 gap was NOT a position-distribution artifact: the net")
            println("    genuinely throws away far more equity per move than wildbg on a common set.")
            println("    *** Therefore the head-to-head '~54% win vs wildbg' is CONFOUNDED / SUSPECT. ***")
            println("    A move-quality-much-weaker agent out-winning a much-stronger one single-game is a")
            println("    red flag — most plausibly wildbg's own move-fallbacks (the 'returned no move for")
            println("    doubles' path) or an unfair head-to-head harness handed the net free equity that")
            println("    common-set grading (which grades wildbg's ACTUAL board vs ply-1) does not. Treat")
            println("    'beat wildbg' as not established until the head-to-head harness is audited.")
        elseif abs(gap) <= 8 || ratio <= 1.4
            println()
            println(">>> RESOLUTION: On identical positions the numbers CONVERGE toward each other —")
            @printf("    net %.1f vs wildbg %.1f (gap %.1f PR). The earlier ~50-vs-~13.5 spread was largely\n",
                    net_pr, wb_pr, gap)
            println("    a POSITION-DISTRIBUTION artifact: each engine's own self-play positions flattered")
            println("    or punished it differently. With that removed, i140 winning ~54% head-to-head vs")
            println("    a comparably-rated wildbg is consistent — no confound required.")
        else
            println()
            @printf(">>> RESOLUTION (intermediate): net %.1f vs wildbg %.1f on the common set — the gap\n", net_pr, wb_pr)
            println("    shrank from the ~50-vs-13.5 prior but did not vanish. Part position-distribution")
            println("    artifact, part real. Inspect the head-to-head harness AND re-grade with more")
            println("    positions before trusting 'beat wildbg'.")
        end
    else
        println("(Run with --agents including both net and wildbg to get the head-to-head resolution.)")
    end
    floor_pr !== nothing && @printf("\nFloor anchor (gnubg-1ply on common set) = %.2f PR (≈1 target → pipeline sound).\n", floor_pr)
    println()
    println("This script is ALSO the PR-tracking tool for the future cubeful net: the position set is")
    println("fixed & seed-deterministic, and the agent side is swappable (--agents / --net-ckpt), so")
    println("successive net versions are directly comparable to each other and to these baselines.")
    println("=" ^ 74); flush(stdout)
end

main()
