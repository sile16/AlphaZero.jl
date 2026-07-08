#!/usr/bin/env julia
"""
diag_policy_gap.jl — WHY is the contact net's POLICY head weak?

THE QUESTION (informs the cubeful redesign)
  The raw net (policy argmax, no search) wins only ~6.7% vs wildbg, but the SAME
  net at MCTS-800 wins ~52%. The value head is well-calibrated (~0.95 corr). So
  the weakness is the POLICY PRIOR. Two competing explanations:

    (A) POLICY-LEARNING failure — the head does NOT reproduce even its OWN
        search's decisions. The target signal (the MCTS visit distribution) exists
        but the head fails to capture it. Since capacity was ruled out (512x8 did
        not help), that points to a TRAINING issue (policy loss weight, target
        temperature/sharpening, # policy steps) — fixable in the cubeful net.

    (B) WEAK TARGETS — the head faithfully matches its search, but the search's
        chosen moves are themselves weak vs gnubg. Then the fix is STRONGER search
        during self-play (deeper MCTS / better value) to generate better targets.

  We distinguish (A) vs (B) directly. On ~1000 COMMON contact positions (the same
  seeded gnubg-0ply generator as scripts/benchmark_pr.jl), for each position we
  compute THREE full-turn moves and grade each vs gnubg-ply1 (native move list):

    R = RAW POLICY argmax   — the policy head read directly, NO search (per-
                              half-move greedy argmax of the masked policy prior).
    M = MCTS-<iters> move   — the net's actual searched choice (greedy, tau=0).
    G = gnubg-ply1 best move — the reference (argmax of the native move list).

  Then:
    * Agreement:  R==M  (does the prior reproduce its OWN search's move?),
                  R==G  (prior picks gnubg's best),  M==G (search picks it).
    * PR of R and PR of M vs gnubg (true-scale per-move regret, match-by-board).

  VERDICT LOGIC:
    R==M LOW  (<~50%)                     -> (A) policy-LEARNING failure dominates.
    R==M HIGH but BOTH R,M high PR        -> (B) targets are weak; deepen search.
  The R-vs-M PR gap quantifies how much search improves per-move quality.

REUSED MACHINERY (verbatim contract from scripts/benchmark_pr.jl &
scripts/analyze_pr_native.jl — read those for the full derivation):
  * gnubg-0ply seeded COMMON contact-position generator (benchmark_pr.jl).
  * dual-model net load (contact 256x5 + race 128x3, min_plus_flat) + CPU oracles.
  * MctsAgent greedy move (tau=0) for M — the net's searched choice.
  * The single CPU oracle `cpu_single(state) -> (policy_over_legal, value)` read
    DIRECTLY for R (policy prior, no tree) — policy is aligned to the state's
    SORTED legal actions (GI.available_actions), matching the oracle's P indexing.
  * gnubg native move-list regret, matched BY BOARD (from_gnubg_simple) so the
    player-0 action-id quirk never enters; best_eq/our_eq from the SAME native
    eval -> gnubg-self floor ~0. One native eval per position yields G, regret(R)
    and regret(M) together.
  * Doubles handled per-turn (two half-moves combined to one resulting board);
    forced turns (<=1 legal full move) excluded from PR & agreement, identically
    for R, M and G (a property of the position, not the agent).

Usage:
    julia --threads 16 --project scripts/diag_policy_gap.jl <contact_ckpt> \\
        [--n 1000] [--seed 1] [--mcts-iters 800] [options]

Options (defaults match benchmark_pr.jl for the i140 dual net):
    --n=1000             target COMMON contact pre-move decisions
    --seed=1             base RNG seed (dice) — deterministic common set
    --mcts-iters=800     MCTS iterations/move for M (the net's searched choice)
    --obs-type=min_plus_flat  --width=256 --blocks=5 --race-width=128 --race-blocks=3
    --race-checkpoint=   explicit race checkpoint (else auto-detect race_latest.data)
    --gnubg-ply=1        reference ply (1 = validated; >=2 refused)
    --num-workers=12     CPU worker threads
    --max-games=100000   safety cap on generator games
    --inference-batch-size=50  --inference-backend=auto  --chance-mode=passthrough
"""

using ArgParse

function parse_diag_args()
    s = ArgParseSettings(description="Diagnose the contact net's policy gap: learning failure vs weak targets", autofix_names=true)
    @add_arg_table! s begin
        "checkpoint"
            help = "Contact/main checkpoint file"; arg_type = String; required = true
        "--n";               help = "Target COMMON contact pre-move decisions"; arg_type = Int; default = 1000
        "--seed";            help = "Base RNG seed (dice) — deterministic common set"; arg_type = Int; default = 1
        "--mcts-iters";      help = "MCTS iterations per move for M (the net's searched choice)"; arg_type = Int; default = 800
        "--obs-type";        help = "Observation type"; arg_type = String; default = "min_plus_flat"
        "--width";           help = "Contact network width"; arg_type = Int; default = 256
        "--blocks";          help = "Contact network blocks"; arg_type = Int; default = 5
        "--race-width";      help = "Race network width"; arg_type = Int; default = 128
        "--race-blocks";     help = "Race network blocks"; arg_type = Int; default = 3
        "--race-checkpoint"; help = "Explicit race checkpoint (else auto-detect)"; arg_type = String; default = ""
        "--gnubg-ply";       help = "gnubg reference ply (1 = validated; >=2 refused)"; arg_type = Int; default = 1
        "--num-workers";     help = "CPU worker threads"; arg_type = Int; default = 12
        "--max-games";       help = "Safety cap on generator games"; arg_type = Int; default = 100_000
        "--inference-batch-size"; help = "Inference batch size for MCTS"; arg_type = Int; default = 50
        "--inference-backend";    help = "CPU inference backend: auto, fast, or flux"; arg_type = String; default = "auto"
        "--chance-mode";          help = "Chance-node handling: passthrough (default) or exact_expectation"; arg_type = String; default = "passthrough"
    end
    return ArgParse.parse_args(s)
end

const ARGS_D = parse_diag_args()

using AlphaZero
using AlphaZero: GI, MCTS, Network, FluxLib, MctsParams, ConstSchedule, BatchedMCTS, GameLoop
using AlphaZero.NetLib
import Flux
using Random
using Statistics
using Printf
using BackgammonNet

ENV["BACKGAMMON_OBS_TYPE"] = ARGS_D["obs_type"]
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
const gspec = GameSpec()
const NUM_ACTIONS = GI.num_actions(gspec)
const _state_dim = GI.state_dim(gspec)[1]
const ORACLE_CFG = AlphaZero.BackgammonInference.OracleConfig(
    _state_dim, NUM_ACTIONS, gspec;
    vectorize_state! = vectorize_state_into!,
    route_state = s -> (s isa BackgammonNet.BackgammonGame && !BackgammonNet.is_contact_position(s) ? 2 : 1))

# ── Fixed benchmark position (verbatim from benchmark_pr.jl) ───────────────────
struct BenchPosition
    state::BackgammonNet.BackgammonGame
    is_contact::Bool
    player::Int
end

# ── A full-turn move decision: pre-move state + the resulting board ────────────
struct MoveDecision
    res_p0::UInt128
    res_p1::UInt128
    matched::Bool   # did we compute a resulting board at all (always true here)
end

# ── Dice sampling (verbatim from benchmark_pr.jl) ──────────────────────────────
function _sample_dice(rng)
    r = rand(rng, Float32); c = 0.0f0
    @inbounds for i in 1:length(BackgammonNet.DICE_PROBS)
        c += BackgammonNet.DICE_PROBS[i]
        r <= c && return i
    end
    return length(BackgammonNet.DICE_PROBS)
end

# ── Full-TURN execution by an engine's best_move (from benchmark_pr.jl) ────────
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

# ── FIXED common CONTACT position set (verbatim from benchmark_pr.jl) ──────────
function generate_common_positions(gen_engine, n_contact::Int, base_seed::Int, max_games::Int)
    contact = BenchPosition[]
    fail = Threads.Atomic{Int}(0)
    gi = 0
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
            if is_contact
                length(contact) < n_contact && push!(contact, BenchPosition(pre, true, player))
            end
            play_turn_with_engine!(g, gen_engine, fail)
            g.terminated && break
            length(contact) >= n_contact && break
        end
    end
    return contact, gi, fail[]
end

# ── R: RAW POLICY argmax move (policy head, NO search) ─────────────────────────
"""For each benchmark position play the FULL turn by greedily taking the masked
policy prior's argmax at each half-move (no MCTS). The single oracle returns the
policy over the state's SORTED legal actions (GI.available_actions), so the argmax
index maps back to the action directly. Parallel: cpu_single uses task-local
buffers (safe under Threads.@threads)."""
function raw_policy_decisions(cpu_single, positions::Vector{BenchPosition}, num_workers::Int)
    n = length(positions)
    out = Vector{MoveDecision}(undef, n)
    idx = Threads.Atomic{Int}(0)
    Threads.@threads for _ in 1:num_workers
        while true
            i = Threads.atomic_add!(idx, 1) + 1
            i > n && break
            bp = positions[i]
            g = BackgammonNet.clone(bp.state)
            start_player = g.current_player
            while true
                att = BackgammonNet.action_type(g)
                (att == BackgammonNet.ACTION_TYPE_TERMINAL || att == BackgammonNet.ACTION_TYPE_CHANCE) && break
                g.current_player != start_player && break
                avail = GI.available_actions(gspec, g)   # SORTED — matches oracle P index
                isempty(avail) && break
                action = if length(avail) == 1
                    avail[1]
                else
                    pv, _ = cpu_single(g)                 # policy prior over avail (no search)
                    avail[argmax(pv)]
                end
                BackgammonNet.apply_action!(g, action)
                g.terminated && break
            end
            out[i] = MoveDecision(g.p0, g.p1, true)
        end
    end
    return out
end

# ── M: MCTS move (the net's searched choice, greedy tau=0) ─────────────────────
"""Verbatim net move contract from benchmark_pr.jl: one MctsAgent+player+env per
worker, greedy MCTS (argmax visit policy), forced sub-moves skip MCTS."""
function mcts_decisions(cpu_single, cpu_batch, mcts_params, batch_size,
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
            GameLoop._reset_player!(player)
            GI.set_state!(env, bp.state)
            start_player = bp.state.current_player
            while true
                (GI.game_terminated(env) || GI.is_chance_node(env)) && break
                env.game.current_player != start_player && break
                avail = GI.available_actions(env)
                isempty(avail) && break
                action = if length(avail) == 1
                    avail[1]
                else
                    actions, policy, _ = GameLoop.select_action(agent, player, env)
                    actions[argmax(policy)]
                end
                GI.play!(env, action)
            end
            out[i] = MoveDecision(env.game.p0, env.game.p1, true)
        end
    end
    return out
end

# ── Combined native grading of R and M (one native eval per position) ──────────
"""For one position return, from gnubg's SINGLE native move-list eval:
  status  :ok | :forced | :error
  reg_R, reg_M   per-move regret vs best (best_eq - our_eq, >=0)
  r_best, m_best whether R / M board == the argmax (gnubg's best) board
  r_eq_m         whether R board == M board (raw prior reproduces its own search)
  r_matched, m_matched  whether the R / M board was found in the native list
Match-by-board via from_gnubg_simple (mover perspective, no flip)."""
function grade_pair(gnubg::BackgammonNet.GnubgCLibBackend, bp::BenchPosition,
                    rd::MoveDecision, md::MoveDecision)
    g = bp.state
    r_eq_m = (rd.res_p0 == md.res_p0 && rd.res_p1 == md.res_p1)
    (g.phase == BackgammonNet.PHASE_CHECKER_PLAY) ||
        return (:forced, 0.0, 0.0, false, false, r_eq_m, false, false)
    BackgammonNet.open!(gnubg)
    mvd = lock(BackgammonNet._GNUBG_CLIB_LOCK) do
        BackgammonNet._gnubg_clib_move_data(gnubg, g)
    end
    n = length(mvd)
    n <= 1 && return (:forced, 0.0, 0.0, false, false, r_eq_m, false, false)

    player = Int(g.current_player)
    best_eq = -Inf
    best_p0 = UInt128(0); best_p1 = UInt128(0)
    r_eq = nothing; m_eq = nothing
    @inbounds for (tsimple, probs) in mvd
        eq = Float64(BackgammonNet.compute_cubeless_equity(g, probs))  # mover perspective, NO flip
        tp0, tp1 = BackgammonNet.from_gnubg_simple(tsimple, player)
        if eq > best_eq
            best_eq = eq; best_p0 = tp0; best_p1 = tp1
        end
        (tp0 == rd.res_p0 && tp1 == rd.res_p1) && (r_eq = eq)
        (tp0 == md.res_p0 && tp1 == md.res_p1) && (m_eq = eq)
    end
    r_matched = r_eq !== nothing
    m_matched = m_eq !== nothing
    reg_R = r_matched ? max(best_eq - r_eq, 0.0) : 0.0
    reg_M = m_matched ? max(best_eq - m_eq, 0.0) : 0.0
    r_best = (rd.res_p0 == best_p0 && rd.res_p1 == best_p1)
    m_best = (md.res_p0 == best_p0 && md.res_p1 == best_p1)
    return (:ok, reg_R, reg_M, r_best, m_best, r_eq_m, r_matched, m_matched)
end

function grade_all(gnubg_backends, positions, r_decs, m_decs, num_workers)
    n = length(positions)
    status  = Vector{Symbol}(undef, n)
    reg_R   = fill(NaN, n); reg_M = fill(NaN, n)
    r_best  = falses(n); m_best = falses(n); r_eq_m = falses(n)
    r_match = falses(n); m_match = falses(n)
    player  = Vector{Int}(undef, n)
    idx = Threads.Atomic{Int}(0)
    Threads.@threads for w in 1:num_workers
        gb = gnubg_backends[w]
        while true
            i = Threads.atomic_add!(idx, 1) + 1
            i > n && break
            bp = positions[i]
            player[i] = bp.player
            local res
            try
                res = grade_pair(gb, bp, r_decs[i], m_decs[i])
            catch e
                @warn "grade_pair failed on decision $i; skipping" exception=(e, catch_backtrace()) maxlog=5
                status[i] = :error; continue
            end
            st, rR, rM, rb, mb, rem, rmt, mmt = res
            status[i] = st
            reg_R[i] = rR; reg_M[i] = rM
            r_best[i] = rb; m_best[i] = mb; r_eq_m[i] = rem
            r_match[i] = rmt; m_match[i] = mmt
        end
    end
    return (; status, reg_R, reg_M, r_best, m_best, r_eq_m, r_match, m_match, player)
end

# ── Resolve race checkpoint (verbatim logic from benchmark_pr.jl) ──────────────
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

# ── PR helpers ─────────────────────────────────────────────────────────────────
pr_of(regs) = isempty(regs) ? NaN : 500.0 * sum(regs) / length(regs)
er_of(regs) = isempty(regs) ? NaN : 1000.0 * sum(regs) / length(regs)

function main()
    gnubg_ply = ARGS_D["gnubg_ply"]
    if gnubg_ply >= 2
        error("gnubg reference ply=$gnubg_ply refused: gnubg's 2-ply evaluator deadlocks its " *
              "internal thread pool. Use --gnubg-ply 1 (validated reference) or 0.")
    end
    num_workers = ARGS_D["num_workers"]
    n_contact = ARGS_D["n"]
    base_seed = ARGS_D["seed"]
    max_games = ARGS_D["max_games"]
    batch_size = ARGS_D["inference_batch_size"]
    mcts_iters = ARGS_D["mcts_iters"]

    net_ckpt = ARGS_D["checkpoint"]
    isfile(net_ckpt) || error("net checkpoint not found: $net_ckpt")
    race_ckpt = resolve_race_ckpt(net_ckpt, ARGS_D["race_checkpoint"])

    println("=" ^ 78)
    println("POLICY-GAP DIAGNOSIS — is the weak policy a LEARNING failure or WEAK TARGETS?")
    println("=" ^ 78)
    println("Contact net:   $net_ckpt")
    println("Race net:      $(race_ckpt === nothing ? "(none — single model)" : race_ckpt)")
    println("Arch:          contact=$(ARGS_D["width"])w×$(ARGS_D["blocks"])b" *
            (race_ckpt === nothing ? "" : " + race=$(ARGS_D["race_width"])w×$(ARGS_D["race_blocks"])b") *
            ", obs=$(ARGS_D["obs_type"])")
    println("Common set:    $n_contact contact pre-move decisions (gnubg-0ply generator, seed=$base_seed)")
    println("Moves graded:  R=raw policy argmax (NO search) | M=MCTS-$mcts_iters greedy | G=gnubg-ply$gnubg_ply best")
    println("Reference:     gnubg ply-$gnubg_ply NATIVE move list, match-by-board (floor≈0)")
    println("Workers:       $num_workers CPU   (chance=$(ARGS_D["chance_mode"]))")
    println("=" ^ 78); flush(stdout)

    # ── Shared gnubg backends for parallel grading ──
    gnubg_backends = [begin
        gb = BackgammonNet.GnubgCLibBackend(ply=gnubg_ply, threads=1); BackgammonNet.open!(gb); gb
    end for _ in 1:num_workers]

    # ── Build the COMMON contact set (gnubg-0ply moves both sides) ──
    println("\nGENERATING common contact position set (gnubg-0ply moves both sides, seeded dice)...")
    flush(stdout)
    tgen = time()
    gen_engine = begin
        e = BackgammonNet.GnubgCLibBackend(ply=0, threads=1); BackgammonNet.open!(e); e
    end
    positions, n_games, gen_fail = generate_common_positions(gen_engine, n_contact, base_seed, max_games)
    try BackgammonNet.close(gen_engine) catch end
    println("  $(length(positions)) contact positions from $n_games games ($(round(time()-tgen, digits=1))s)")
    gen_fail > 0 && println("  (generator best_move fallbacks: $gen_fail)")
    let np0 = count(p -> p.player == 0, positions), np1 = count(p -> p.player == 1, positions)
        println("  player-to-move split:  P0=$np0   P1=$np1")
    end
    if isempty(positions)
        for gb in gnubg_backends; try BackgammonNet.close(gb) catch end; end
        println("No positions generated — nothing to grade."); return
    end
    flush(stdout)

    # ── Net oracles (contact + race) ──
    contact_network = FluxLib.FCResNetMultiHead(
        gspec, FluxLib.FCResNetMultiHeadHP(width=ARGS_D["width"], num_blocks=ARGS_D["blocks"]))
    FluxLib.load_weights(net_ckpt, contact_network)
    contact_network = Flux.cpu(contact_network)
    race_network = nothing
    if race_ckpt !== nothing
        race_network = FluxLib.FCResNetMultiHead(
            gspec, FluxLib.FCResNetMultiHeadHP(width=ARGS_D["race_width"], num_blocks=ARGS_D["race_blocks"]))
        FluxLib.load_weights(race_ckpt, race_network)
        race_network = Flux.cpu(race_network)
    end
    mcts_params = MctsParams(
        num_iters_per_turn=mcts_iters, cpuct=1.5, temperature=ConstSchedule(0.0),
        dirichlet_noise_ϵ=0.0, dirichlet_noise_α=1.0, chance_mode=Symbol(ARGS_D["chance_mode"]))
    backend = AlphaZero.BackgammonInference.resolve_cpu_backend(ARGS_D["inference_backend"])
    println("\nNet CPU inference: $(AlphaZero.BackgammonInference.cpu_backend_summary(backend))")
    cpu_single, cpu_batch = AlphaZero.BackgammonInference.make_cpu_oracles(
        backend, contact_network, ORACLE_CFG; secondary_net=race_network, batch_size=batch_size)
    flush(stdout)

    # ── R: raw policy argmax (no search) ──
    println("\nComputing R = RAW POLICY argmax (policy head, NO search) for $(length(positions)) positions...")
    flush(stdout)
    t0 = time()
    r_decs = raw_policy_decisions(cpu_single, positions, num_workers)
    println("  done ($(round(time()-t0, digits=1))s)"); flush(stdout)

    # ── M: MCTS-<iters> greedy move ──
    println("\nComputing M = MCTS-$mcts_iters greedy move for $(length(positions)) positions...")
    flush(stdout)
    t1 = time()
    m_decs = mcts_decisions(cpu_single, cpu_batch, mcts_params, batch_size, positions, num_workers, base_seed)
    println("  done ($(round(time()-t1, digits=1))s)"); flush(stdout)

    # ── Grade both vs gnubg (one native eval per position) ──
    println("\nGRADING R and M vs gnubg ply-$gnubg_ply native move list (parallel)...")
    flush(stdout)
    t2 = time()
    G = grade_all(gnubg_backends, positions, r_decs, m_decs, num_workers)
    println("  done ($(round(time()-t2, digits=1))s)"); flush(stdout)
    for gb in gnubg_backends; try BackgammonNet.close(gb) catch end; end

    # ── Aggregate ──
    ok = G.status .== :ok
    n_ok = count(ok)
    n_forced = count(==(:forced), G.status)
    n_error = count(==(:error), G.status)
    n_r_unmatched = count(ok .& .!G.r_match)
    n_m_unmatched = count(ok .& .!G.m_match)

    # Restrict PR/agreement to positions where BOTH R and M matched a native board
    scored = ok .& G.r_match .& G.m_match
    ns = count(scored)

    regR = G.reg_R[scored]; regM = G.reg_M[scored]
    pr_R = pr_of(regR); er_R = er_of(regR)
    pr_M = pr_of(regM); er_M = er_of(regM)

    rEqM  = count(G.r_eq_m[scored]);  frac_RM = ns == 0 ? NaN : 100.0 * rEqM / ns
    rBest = count(G.r_best[scored]);  frac_RG = ns == 0 ? NaN : 100.0 * rBest / ns
    mBest = count(G.m_best[scored]);  frac_MG = ns == 0 ? NaN : 100.0 * mBest / ns

    # Player split
    function split_stats(mask)
        idxs = findall(scored .& mask)
        isempty(idxs) && return (n=0, prR=NaN, prM=NaN, rm=NaN, rg=NaN, mg=NaN)
        rr = G.reg_R[idxs]; rm_ = G.reg_M[idxs]
        (n=length(idxs),
         prR=pr_of(rr), prM=pr_of(rm_),
         rm=100.0*count(G.r_eq_m[idxs])/length(idxs),
         rg=100.0*count(G.r_best[idxs])/length(idxs),
         mg=100.0*count(G.m_best[idxs])/length(idxs))
    end
    p0 = split_stats(G.player .== 0)
    p1 = split_stats(G.player .== 1)

    # ── Report ──
    println("\n" * "=" ^ 78)
    println("RESULTS")
    println("=" ^ 78)
    println("Positions:  total=$(length(positions))  scored(unforced, both matched)=$ns")
    println("            forced=$n_forced  error=$n_error  R-unmatched=$n_r_unmatched  M-unmatched=$n_m_unmatched")
    println()
    println("1) AGREEMENT RATES (over $ns scored positions)")
    @printf("   R==M  (raw policy reproduces its OWN search's move) : %6.2f%%   (%d/%d)\n", frac_RM, rEqM, ns)
    @printf("   R==G  (raw policy picks gnubg's best move)          : %6.2f%%   (%d/%d)\n", frac_RG, rBest, ns)
    @printf("   M==G  (MCTS-%d picks gnubg's best move)           : %6.2f%%   (%d/%d)\n", mcts_iters, frac_MG, mBest, ns)
    println()
    println("2) TRUE-SCALE PR vs gnubg ply-$gnubg_ply  (500 * mean per-move regret)")
    @printf("   R  raw policy   : PR = %7.2f   ER(mEMG) = %7.2f\n", pr_R, er_R)
    @printf("   M  MCTS-%-5d    : PR = %7.2f   ER(mEMG) = %7.2f\n", mcts_iters, pr_M, er_M)
    @printf("   search gain (PR_R - PR_M)  = %7.2f  PR  (equity/move search recovers)\n", pr_R - pr_M)
    println()
    println("3) PLAYER SPLIT (asymmetry check)")
    @printf("   P0 (n=%4d):  PR_R=%7.2f  PR_M=%7.2f   R==M=%5.1f%%  R==G=%5.1f%%  M==G=%5.1f%%\n",
            p0.n, p0.prR, p0.prM, p0.rm, p0.rg, p0.mg)
    @printf("   P1 (n=%4d):  PR_R=%7.2f  PR_M=%7.2f   R==M=%5.1f%%  R==G=%5.1f%%  M==G=%5.1f%%\n",
            p1.n, p1.prR, p1.prM, p1.rm, p1.rg, p1.mg)

    # ── Verdict ──
    println("\n" * "=" ^ 78)
    println("VERDICT — learning failure (A) vs weak targets (B)?")
    println("=" ^ 78)
    @printf("R==M agreement = %.1f%%  |  PR_R = %.1f  PR_M = %.1f  (search gain %.1f PR)\n",
            frac_RM, pr_R, pr_M, pr_R - pr_M)
    println()
    if frac_RM < 50.0
        println(">>> (A) POLICY-LEARNING FAILURE DOMINATES.")
        @printf("    The raw policy reproduces its OWN search's move only %.1f%% of the time — the\n", frac_RM)
        println("    training SIGNAL (the MCTS choice) exists but the head is NOT capturing it. Search")
        @printf("    slashes per-move regret from PR %.1f (raw) to %.1f (MCTS): the head is throwing\n", pr_R, pr_M)
        println("    away decisions the very same net's search already found. Capacity was ruled out")
        println("    (512x8 did not help), so this is a TRAINING problem — for the cubeful net, get the")
        println("    policy loss weighting, target temperature/sharpening, and # policy steps right so")
        println("    the prior actually absorbs the search distribution. (Also re-examine whether the")
        println("    policy target is the visit distribution vs a one-hot / mis-scaled target.)")
    elseif pr_M > 20.0
        println(">>> (B) WEAK TARGETS DOMINATE.")
        @printf("    The raw policy DOES track its search (R==M %.1f%%), but the search itself is weak:\n", frac_RM)
        @printf("    even MCTS-%d is PR %.1f vs gnubg. The head faithfully learns a mediocre teacher.\n", mcts_iters, pr_M)
        println("    Fix = STRONGER search during self-play (deeper MCTS / better value) to generate")
        println("    better policy targets — a bigger/longer-trained policy head cannot exceed them.")
    else
        println(">>> MIXED / INCONCLUSIVE.")
        @printf("    R==M is high (%.1f%%) yet MCTS-%d is already strong (PR %.1f). The prior tracks a\n",
                frac_RM, mcts_iters, pr_M)
        @printf("    good teacher but still loses PR %.1f at the raw level — modest learning slack plus\n", pr_R - pr_M)
        println("    solid targets. Both levers help; prioritize by which PR gap is larger.")
    end
    println()
    println("Note: G (and thus regret/agreement) = gnubg ply-$gnubg_ply NATIVE move-list best, matched")
    println("by resulting board (from_gnubg_simple, mover perspective) — the same floor≈0 reference as")
    println("scripts/analyze_pr_native.jl. Doubles graded per-turn; forced turns excluded identically.")
    println("=" ^ 78); flush(stdout)
end

main()
