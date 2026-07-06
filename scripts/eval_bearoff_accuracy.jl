#!/usr/bin/env julia
"""
Evaluate race checkpoints against the exact k=7 bearoff table.

Metrics:
- Value error on post-dice bearoff decision states
- Tie-aware exact top-1 move accuracy vs table-optimal move set
- Raw policy quality: top-k coverage, optimal-move rank, optimal prior mass
- Move regret (optimal table value - chosen move's exact table value)
- Both raw NN one-ply move choice and NN+MCTS move choice

The evaluation set is a fixed sampled subset of unique post-dice bearoff
decision states from a raw bootstrap artifact, cached to disk so all checkpoints
see the exact same positions.

Example:
  julia --threads 10 --project scripts/eval_bearoff_accuracy.jl \
      /home/sile/alphazero-server-race-v12/checkpoints/race_iter_50.data \
      --width=256 --blocks=5 --mcts-iters=600 --num-positions=2000
"""

using ArgParse

function parse_args()
    s = ArgParseSettings(description="Evaluate bearoff value and move accuracy", autofix_names=true)
    @add_arg_table! s begin
        "checkpoints"
            nargs = '*'
            help = "Checkpoint files to evaluate"
        "--checkpoints-dir"
            arg_type = String
            default = ""
            help = "Optional checkpoint directory (used with --iters)"
        "--iters"
            arg_type = String
            default = ""
            help = "Comma-separated checkpoint iterations, e.g. 5,10,20,30,40,50"
        "--width"
            arg_type = Int
            default = 256
        "--blocks"
            arg_type = Int
            default = 5
        "--obs-type"
            arg_type = String
            default = "minimal_flat"
        "--num-positions"
            arg_type = Int
            default = 2000
            help = "Number of unique bearoff decision states to evaluate"
        "--positions-cache"
            arg_type = String
            default = "/tmp/bearoff_decision_eval_2000_seed42.jls"
            help = "Cache file for sampled bearoff decision states"
        "--bootstrap-file"
            arg_type = String
            default = "/home/sile/github/BackgammonNet.jl/data/bootstrap/bootstrap_5000g_bgblitz1ply.jls"
            help = "Raw bootstrap with BackgammonGame states (fallback source)"
        "--start-positions-file"
            arg_type = String
            default = "eval_data/race_eval_2000.jls"
            help = "Canonical race start positions used to generate bearoff eval states"
        "--rollouts-per-start"
            arg_type = Int
            default = 2
            help = "Number of random rollouts to generate per race start"
        "--bearoff-dir"
            arg_type = String
            default = ""
            help = "Optional explicit path to bearoff_k7_twosided directory"
        "--mcts-iters"
            arg_type = Int
            default = 600
        "--inference-batch-size"
            arg_type = Int
            default = 50
        "--inference-backend"
            arg_type = String
            default = "auto"
        "--num-workers"
            arg_type = Int
            default = max(1, Threads.nthreads() - 1)
        "--seed"
            arg_type = Int
            default = 42
        "--log-interval"
            arg_type = Int
            default = 250
        "--output"
            arg_type = String
            default = ""
        "--dump-failures"
            arg_type = Int
            default = 0
            help = "Dump up to this many worst MCTS failures"
        "--dump-path"
            arg_type = String
            default = ""
            help = "Optional path for detailed root diagnostics"
    end
    return ArgParse.parse_args(s)
end

const ARGS = parse_args()

using AlphaZero
using AlphaZero: GI, Network, FluxLib, MctsParams, ConstSchedule, MctsPlayer, think, reset_player!
using AlphaZero.BackgammonInference
import Flux
import AlphaZero.MCTS
using Serialization
using Statistics
using Random
using Dates
using Printf
using StaticArrays
using BackgammonNet
using BackgammonNet: BearoffK7, bearoff_turn_value

ENV["BACKGAMMON_OBS_TYPE"] = ARGS["obs_type"]
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
const gspec = GameSpec()
const NUM_ACTIONS = GI.num_actions(gspec)
const STATE_DIM = GI.state_dim(gspec)[1]
const ORACLE_CFG = AlphaZero.BackgammonInference.OracleConfig(
    STATE_DIM, NUM_ACTIONS, gspec; vectorize_state! = vectorize_state_into!)

const BACKGAMMONNET_REPO = dirname(dirname(pathof(BackgammonNet)))

@inline normalized_points(v::Real) = Float64(v) / 3.0

function parse_checkpoint_list()
    ckpts = String[]
    append!(ckpts, ARGS["checkpoints"])
    if !isempty(ARGS["checkpoints_dir"]) && !isempty(ARGS["iters"])
        for item in split(ARGS["iters"], ",")
            iter = strip(item)
            isempty(iter) && continue
            push!(ckpts, joinpath(ARGS["checkpoints_dir"], "race_iter_$(iter).data"))
        end
    end
    isempty(ckpts) && error("Provide checkpoints as positional args or use --checkpoints-dir with --iters")
    return ckpts
end

function find_bearoff_dir()
    if !isempty(ARGS["bearoff_dir"])
        return ARGS["bearoff_dir"]
    end
    candidates = [
        joinpath(BACKGAMMONNET_REPO, "tools", "bearoff_twosided", "bearoff_k7_twosided"),
        joinpath(homedir(), "bearoff_k7_twosided"),
        "/homeshare/projects/AlphaZero.jl/eval_data/bearoff_k7_twosided",
    ]
    for dir in candidates
        if isdir(dir) && isfile(joinpath(dir, "bearoff_k7_c14.bin"))
            return dir
        end
    end
    error("bearoff_k7_twosided not found")
end

function make_state_key(g::BackgammonNet.BackgammonGame)
    return (g.p0, g.p1, g.dice[1], g.dice[2], g.remaining_actions, g.current_player)
end

function build_positions_cache(cache_path::String)
    if isfile(ARGS["start_positions_file"])
        return build_positions_cache_from_rollouts(cache_path)
    end
    return build_positions_cache_from_bootstrap(cache_path)
end

function build_positions_cache_from_bootstrap(cache_path::String)
    println("Building bearoff decision cache from $(ARGS["bootstrap_file"])")
    raw = deserialize(ARGS["bootstrap_file"])
    states = raw.states
    wanted = ARGS["num_positions"]
    seen = Set{Tuple{UInt128, UInt128, Int8, Int8, Int8, Int8}}()
    candidates = BackgammonNet.BackgammonGame[]

    for s in states
        BackgammonNet.is_chance_node(s) && continue
        s.phase == BackgammonNet.PHASE_CHECKER_PLAY || continue
        s.remaining_actions == Int8(1) || continue
        BearoffK7.is_bearoff_position(s.p0, s.p1) || continue
        length(BackgammonNet.legal_actions(s)) > 1 || continue
        key = make_state_key(s)
        key in seen && continue
        push!(seen, key)
        push!(candidates, BackgammonNet.clone(s))
    end

    println("Found $(length(candidates)) unique post-dice bearoff decision states")
    if length(candidates) < wanted
        error("Only found $(length(candidates)) unique states, fewer than requested $wanted")
    end

    rng = MersenneTwister(ARGS["seed"])
    order = randperm(rng, length(candidates))[1:wanted]
    positions = [candidates[i] for i in order]
    mkpath(dirname(cache_path))
    serialize(cache_path, positions)
    println("Saved $wanted cached positions to $cache_path")
    return positions
end

function start_game_from_tuple(position_data::Tuple{UInt128, UInt128, Int8}, seed::Int)
    p0, p1, cp = position_data
    game = BackgammonNet.BackgammonGame(
        p0, p1, SVector{2, Int8}(0, 0), Int8(0), cp, false, 0.0f0;
        obs_type=:minimal_flat)
    return GameEnv(game, MersenneTwister(seed))
end

function build_positions_cache_from_rollouts(cache_path::String)
    starts = deserialize(ARGS["start_positions_file"])
    wanted = ARGS["num_positions"]
    rng = MersenneTwister(ARGS["seed"])
    seen = Set{Tuple{UInt128, UInt128, Int8, Int8, Int8, Int8}}()
    candidates = BackgammonNet.BackgammonGame[]

    println("Building bearoff decision cache from canonical rollouts on $(ARGS["start_positions_file"])")
    println("  Starts: $(length(starts)) | Rollouts/start: $(ARGS["rollouts_per_start"])")

    for (i, start_pos) in enumerate(starts)
        for r in 1:ARGS["rollouts_per_start"]
            env = start_game_from_tuple(start_pos, ARGS["seed"] + 100_000 * r + i)
            while !GI.game_terminated(env)
                if GI.is_chance_node(env)
                    BackgammonNet.sample_chance!(env.game, env.rng)
                    continue
                end

                state = GI.current_state(env)
                if state.phase == BackgammonNet.PHASE_CHECKER_PLAY &&
                   state.remaining_actions == Int8(1) &&
                   BearoffK7.is_bearoff_position(state.p0, state.p1) &&
                   length(BackgammonNet.legal_actions(state)) > 1
                    key = make_state_key(state)
                    if !(key in seen)
                        push!(seen, key)
                        push!(candidates, state)
                    end
                end

                acts = BackgammonNet.legal_actions(env.game)
                isempty(acts) && break
                GI.play!(env, rand(rng, acts))
            end
        end
        if i % 200 == 0
            println("  scanned $i / $(length(starts)) starts, found $(length(candidates)) candidates")
        end
    end

    println("Found $(length(candidates)) unique canonical post-dice bearoff decision states")
    if length(candidates) < wanted
        error("Only found $(length(candidates)) unique states, fewer than requested $wanted")
    end

    order = randperm(rng, length(candidates))[1:wanted]
    positions = [candidates[i] for i in order]
    mkpath(dirname(cache_path))
    serialize(cache_path, positions)
    println("Saved $wanted cached positions to $cache_path")
    return positions
end

function load_positions()
    cache_path = ARGS["positions_cache"]
    if isfile(cache_path)
        positions = deserialize(cache_path)
        if length(positions) == ARGS["num_positions"]
            println("Loaded $(length(positions)) cached bearoff positions from $cache_path")
            return positions
        end
        println("Cache size mismatch ($(length(positions)) vs $(ARGS["num_positions"])) — rebuilding")
    end
    return build_positions_cache(cache_path)
end

function exact_action_values(state::BackgammonNet.BackgammonGame, table)
    actions = BackgammonNet.legal_actions(state)
    mover = Int(state.current_player)
    work = BackgammonNet.clone(state)
    action_values = Dict{Int, Float64}()

    for action in actions
        BackgammonNet.copy_state!(work, state)
        BackgammonNet.apply_action!(work, action)

        # Turn-aware exact value: handles terminal rewards (gammon multiplier),
        # completed turns (opponent pre-dice lookup), and doubles mid-turn states
        # (recursion) — see BackgammonNet.bearoff_turn_value for the doubles pitfall.
        move_val = normalized_points(bearoff_turn_value(table, work, mover))
        action_values[action] = move_val
    end

    isempty(action_values) && error("No exact bearoff move values computed")
    vals = collect(Base.values(action_values))
    best = maximum(vals)
    tol = 1e-8
    optimal = sort([a for (a, v) in action_values if best - v <= tol])
    nonbest = [v for v in vals if best - v > tol]
    second_best = isempty(nonbest) ? best : maximum(nonbest)
    margin = best - second_best
    return (action_values=action_values, best_value=best, optimal_actions=optimal, margin=margin)
end

function nn_greedy_action(state, value_oracle)
    actions = BackgammonNet.legal_actions(state)
    mover = Int(state.current_player)
    # A4: terminal children (immediate win/loss) scored from exact game.reward, not
    # the NN — scoring decisive moves with the value net corrupts the metric.
    values = Vector{Float64}(undef, length(actions))
    succs = BackgammonNet.BackgammonGame[]
    succ_idx = Int[]
    for (i, action) in enumerate(actions)
        g = BackgammonNet.clone(state)
        BackgammonNet.apply_action!(g, action)
        if g.terminated
            white_r = Float64(g.reward)
            mover_r = mover == 0 ? white_r : -white_r
            values[i] = normalized_points(mover_r)
        else
            push!(succs, g); push!(succ_idx, i)
        end
    end
    if !isempty(succs)
        evals = value_oracle(succs)
        for (k, i) in enumerate(succ_idx)
            v = Float64(evals[k][2])
            Int(succs[k].current_player) != mover && (v = -v)
            values[i] = v
        end
    end
    best_action = actions[1]
    best_value = -Inf
    for (i, action) in enumerate(actions)
        if values[i] > best_value
            best_value = values[i]
            best_action = action
        end
    end
    return (action=best_action, value=best_value)
end

function policy_diagnostics(state, policy_oracle, exact)
    actions = BackgammonNet.legal_actions(state)
    policy, _ = policy_oracle(state)
    length(policy) == length(actions) || error("Policy/action mismatch: $(length(policy)) vs $(length(actions))")

    order = sortperm(policy; rev=true)
    ranked_actions = actions[order]
    ranked_probs = Float64.(policy[order])
    optimal_set = Set(exact.optimal_actions)

    top1_hit = ranked_actions[1] in optimal_set
    top3_hit = any(a in optimal_set for a in ranked_actions[1:min(3, end)])
    top5_hit = any(a in optimal_set for a in ranked_actions[1:min(5, end)])

    best_rank = length(actions) + 1
    opt_mass = 0.0
    expected_regret = 0.0
    for (a, p) in zip(actions, policy)
        p64 = Float64(p)
        if a in optimal_set
            opt_mass += p64
        end
        expected_regret += p64 * (exact.best_value - exact.action_values[a])
    end
    for (rank, a) in enumerate(ranked_actions)
        if a in optimal_set
            best_rank = rank
            break
        end
    end

    return (
        top1_hit = top1_hit,
        top3_hit = top3_hit,
        top5_hit = top5_hit,
        best_rank = best_rank,
        opt_mass = opt_mass,
        expected_regret = expected_regret,
        top1_prob = ranked_probs[1],
    )
end

function mcts_action(state, player, rngseed::Int)
    env = GameEnv(BackgammonNet.clone(state), MersenneTwister(rngseed))
    try
        actions, policy = think(player, env)
        return actions[argmax(policy)]
    catch
        return nothing
    finally
        reset_player!(player)
    end
end

function child_nn_value(state, action, value_oracle)
    g = BackgammonNet.clone(state)
    mover = Int(state.current_player)
    BackgammonNet.apply_action!(g, action)
    # A4/F4: a terminal child is a known win/loss — score it from exact game.reward,
    # not the NN (same fix as nn_greedy_action; this is the diagnostic-dump path).
    if g.terminated
        white_r = Float64(g.reward)
        return normalized_points(mover == 0 ? white_r : -white_r)
    end
    v = Float64(value_oracle([g])[1][2])
    if Int(g.current_player) != mover
        v = -v
    end
    return v
end

function dump_root_diagnostics(io, checkpoint, idx, state, exact, value_oracle, player)
    env = GameEnv(BackgammonNet.clone(state), MersenneTwister(ARGS["seed"] + idx))
    MCTS.explore!(player.mcts, env, ARGS["mcts_iters"])
    actions, pi = MCTS.policy(player.mcts, env)
    info = player.mcts.tree[GI.current_state(env)]

    println(io, "checkpoint=$(checkpoint_label(checkpoint)) idx=$idx current_player=$(state.current_player) dice=$(Tuple(state.dice)) n_actions=$(length(actions))")
    println(io, "best_value=$(round(exact.best_value, digits=6)) optimal_actions=$(join(exact.optimal_actions, ',')) margin=$(round(exact.margin, digits=6))")

    rows = NamedTuple[]
    for (j, action) in enumerate(actions)
        stats = info.stats[j]
        q = stats.W / max(stats.N, 1)
        exact_val = exact.action_values[action]
        child_v = child_nn_value(state, action, value_oracle)
        push!(rows, (
            action=action,
            prior=Float64(stats.P),
            visits=stats.N,
            visit_policy=Float64(pi[j]),
            q=Float64(q),
            child_nn_value=child_v,
            exact_value=Float64(exact_val),
            regret=Float64(exact.best_value - exact_val),
            is_optimal=action in exact.optimal_actions,
        ))
    end
    sort!(rows; by=r -> (-r.visit_policy, -r.prior))
    for row in rows
        println(io,
            "  action=$(row.action) prior=$(round(row.prior, digits=4)) visits=$(row.visits) visit_policy=$(round(row.visit_policy, digits=4)) q=$(round(row.q, digits=4)) child_v=$(round(row.child_nn_value, digits=4)) exact=$(round(row.exact_value, digits=4)) regret=$(round(row.regret, digits=4)) optimal=$(row.is_optimal)")
    end
    println(io)
    reset_player!(player)
end

function corr_or_zero(x::Vector{Float64}, y::Vector{Float64})
    (length(x) >= 3 && std(x) > 0 && std(y) > 0) ? cor(x, y) : 0.0
end

function pctl(v::Vector{Float64}, p::Float64)
    isempty(v) && return 0.0
    s = sort(v)
    idx = clamp(Int(ceil(p * length(s))), 1, length(s))
    return s[idx]
end

function checkpoint_label(path::String)
    base = basename(path)
    if occursin(r"race_iter_\d+\.data", base)
        return replace(base, ".data" => "")
    elseif base == "race_latest.data"
        return "race_latest"
    else
        return base
    end
end

function evaluate_checkpoint(checkpoint::String, positions, table)
    width = ARGS["width"]
    blocks = ARGS["blocks"]
    batch_size = ARGS["inference_batch_size"]
    n_workers = min(ARGS["num_workers"], Threads.nthreads())
    backend = AlphaZero.BackgammonInference.resolve_cpu_backend(ARGS["inference_backend"])

    network = FluxLib.FCResNetMultiHead(
        gspec, FluxLib.FCResNetMultiHeadHP(width=width, num_blocks=blocks))
    FluxLib.load_weights(checkpoint, network)
    network = Flux.cpu(network)

    run_mcts = ARGS["mcts_iters"] > 0
    policy_oracles = Vector{Any}(undef, n_workers)
    value_oracles = Vector{Any}(undef, n_workers)
    players = Vector{Any}(undef, n_workers)
    mcts_params = MctsParams(
        num_iters_per_turn=ARGS["mcts_iters"],
        cpuct=1.5,
        temperature=ConstSchedule(0.0),
        dirichlet_noise_ϵ=0.0,
        dirichlet_noise_α=1.0,
    )
    for tid in 1:n_workers
        so, _ = AlphaZero.BackgammonInference.make_cpu_oracles(
            backend, network, ORACLE_CFG; batch_size=batch_size, nslots=1)
        _, vo = AlphaZero.BackgammonInference.make_cpu_oracles(
            :flux, network, ORACLE_CFG; batch_size=batch_size, nslots=1)
        policy_oracles[tid] = so
        value_oracles[tid] = vo
        players[tid] = run_mcts ? MctsPlayer(gspec, so, mcts_params) : nothing
    end

    n = length(positions)
    nn_values = zeros(Float64, n)
    exact_values = zeros(Float64, n)
    margins = zeros(Float64, n)
    n_opt_actions = zeros(Int, n)
    nn_wrong = falses(n)
    mcts_wrong = falses(n)
    mcts_valid = falses(n)
    nn_regret = zeros(Float64, n)
    policy_top1 = falses(n)
    policy_top3 = falses(n)
    policy_top5 = falses(n)
    policy_best_rank = zeros(Int, n)
    policy_opt_mass = zeros(Float64, n)
    policy_expected_regret = zeros(Float64, n)
    policy_top1_prob = zeros(Float64, n)
    mcts_regret = zeros(Float64, n)
    claimed = Threads.Atomic{Int}(0)
    done = Threads.Atomic{Int}(0)
    t0 = time()

    Threads.@threads for tid in 1:n_workers
        po = policy_oracles[tid]
        vo = value_oracles[tid]
        player = players[tid]
        while true
            idx = Threads.atomic_add!(claimed, 1) + 1
            idx > n && break
            state = positions[idx]
            exact = exact_action_values(state, table)
            exact_values[idx] = exact.best_value
            margins[idx] = exact.margin
            n_opt_actions[idx] = length(exact.optimal_actions)

            nn_v = Float64(vo([state])[1][2])
            nn_values[idx] = nn_v

            nn_move = nn_greedy_action(state, vo).action
            nn_move_val = exact.action_values[nn_move]
            nn_regret[idx] = exact.best_value - nn_move_val
            nn_wrong[idx] = !(nn_move in exact.optimal_actions)

            pd = policy_diagnostics(state, po, exact)
            policy_top1[idx] = pd.top1_hit
            policy_top3[idx] = pd.top3_hit
            policy_top5[idx] = pd.top5_hit
            policy_best_rank[idx] = pd.best_rank
            policy_opt_mass[idx] = pd.opt_mass
            policy_expected_regret[idx] = pd.expected_regret
            policy_top1_prob[idx] = pd.top1_prob

            if run_mcts
                mcts_move = mcts_action(state, player, ARGS["seed"] + idx)
                if mcts_move !== nothing
                    mcts_valid[idx] = true
                    mcts_move_val = exact.action_values[mcts_move]
                    mcts_regret[idx] = exact.best_value - mcts_move_val
                    mcts_wrong[idx] = !(mcts_move in exact.optimal_actions)
                end
            end

            d = Threads.atomic_add!(done, 1) + 1
            if d % ARGS["log_interval"] == 0
                elapsed = time() - t0
                rate = d / elapsed * 60
                eta = (n - d) / max(d / elapsed, 1e-9)
                @printf("  %s: %d/%d positions (%.1f pos/min, ETA %.0fs)\n",
                        checkpoint_label(checkpoint), d, n, rate, eta)
                flush(stdout)
            end
        end
    end

    value_diff = nn_values .- exact_values
    valid_idx = findall(mcts_valid)
    mcts_wrong_mean = isempty(valid_idx) ? NaN : mean(mcts_wrong[valid_idx])
    mcts_regret_mean = isempty(valid_idx) ? NaN : mean(mcts_regret[valid_idx])
    mcts_regret_p95 = isempty(valid_idx) ? NaN : pctl(mcts_regret[valid_idx], 0.95)
    mcts_regret_gt_001 = isempty(valid_idx) ? NaN : mean(mcts_regret[valid_idx] .> 0.01)
    mcts_regret_gt_005 = isempty(valid_idx) ? NaN : mean(mcts_regret[valid_idx] .> 0.05)

    if run_mcts && ARGS["dump_failures"] > 0 && !isempty(ARGS["dump_path"])
        dump_idx = sortperm(collect(1:n); by=i -> mcts_valid[i] ? mcts_regret[i] : -Inf, rev=true)
        selected = Int[]
        for idx in dump_idx
            mcts_valid[idx] || continue
            mcts_regret[idx] > 1e-8 || continue
            push!(selected, idx)
            length(selected) >= ARGS["dump_failures"] && break
        end
        if !isempty(selected)
            open(ARGS["dump_path"], "a") do io
                println(io, "# checkpoint=$(checkpoint_label(checkpoint))")
                for idx in selected
                    dump_root_diagnostics(io, checkpoint, idx, positions[idx], exact_action_values(positions[idx], table), value_oracles[1], players[1])
                end
            end
        end
    end

    result = (
        checkpoint = checkpoint,
        label = checkpoint_label(checkpoint),
        n = n,
        mcts_iters = ARGS["mcts_iters"],
        value_mae = mean(abs.(value_diff)),
        value_rmse = sqrt(mean(value_diff .^ 2)),
        value_bias = mean(value_diff),
        value_corr = corr_or_zero(nn_values, exact_values),
        policy_top1 = mean(policy_top1),
        policy_top3 = mean(policy_top3),
        policy_top5 = mean(policy_top5),
        policy_best_rank = mean(policy_best_rank),
        policy_median_best_rank = Float64(median(policy_best_rank)),
        policy_opt_mass = mean(policy_opt_mass),
        policy_expected_regret = mean(policy_expected_regret),
        policy_top1_prob = mean(policy_top1_prob),
        nn_top1 = 1 - mean(nn_wrong),
        nn_wrong = mean(nn_wrong),
        nn_regret = mean(nn_regret),
        nn_regret_p95 = pctl(nn_regret, 0.95),
        nn_regret_gt_001 = mean(nn_regret .> 0.01),
        nn_regret_gt_005 = mean(nn_regret .> 0.05),
        mcts_n = length(valid_idx),
        mcts_skip_rate = run_mcts ? 1 - length(valid_idx) / n : 1.0,
        mcts_top1 = isnan(mcts_wrong_mean) ? NaN : 1 - mcts_wrong_mean,
        mcts_wrong = mcts_wrong_mean,
        mcts_regret = mcts_regret_mean,
        mcts_regret_p95 = mcts_regret_p95,
        mcts_regret_gt_001 = mcts_regret_gt_001,
        mcts_regret_gt_005 = mcts_regret_gt_005,
        avg_opt_actions = mean(n_opt_actions),
        tie_rate = mean(n_opt_actions .> 1),
        avg_margin = mean(margins),
        hard_rate_001 = mean(margins .> 0.01),
        hard_rate_005 = mean(margins .> 0.05),
    )
    return result
end

function print_result(r)
    println("=" ^ 88)
    println("Checkpoint: $(r.label)")
    @printf("  Positions: %d | MCTS iters: %d\n", r.n, r.mcts_iters)
    @printf("  Value:  MAE=%.4f  RMSE=%.4f  bias=%+.4f  corr=%.4f\n",
            r.value_mae, r.value_rmse, r.value_bias, r.value_corr)
    @printf("  Policy prior: top1=%.2f%%  top3=%.2f%%  top5=%.2f%%  opt-mass=%.2f%%\n",
            100 * r.policy_top1, 100 * r.policy_top3, 100 * r.policy_top5, 100 * r.policy_opt_mass)
    @printf("  Policy rank:  mean best-opt rank=%.2f  median=%.1f  top1-prob=%.2f%%  exp-regret=%.4f\n",
            r.policy_best_rank, r.policy_median_best_rank, 100 * r.policy_top1_prob, r.policy_expected_regret)
    @printf("  NN top-1 exact:        %.2f%%  wrong=%.2f%%  regret=%.4f  p95=%.4f\n",
            100 * r.nn_top1, 100 * r.nn_wrong, r.nn_regret, r.nn_regret_p95)
    @printf("  NN regret > 0.01:      %.2f%%  > 0.05: %.2f%%\n",
            100 * r.nn_regret_gt_001, 100 * r.nn_regret_gt_005)
    @printf("  NN+MCTS top-1 exact:   %.2f%%  wrong=%.2f%%  regret=%.4f  p95=%.4f\n",
            100 * r.mcts_top1, 100 * r.mcts_wrong, r.mcts_regret, r.mcts_regret_p95)
    @printf("  NN+MCTS regret > 0.01: %.2f%%  > 0.05: %.2f%%  (valid=%d, skipped=%.2f%%)\n",
            100 * r.mcts_regret_gt_001, 100 * r.mcts_regret_gt_005, r.mcts_n, 100 * r.mcts_skip_rate)
    @printf("  Table difficulty: avg optimal moves=%.2f  tie-rate=%.2f%%  avg margin=%.4f\n",
            r.avg_opt_actions, 100 * r.tie_rate, r.avg_margin)
    @printf("  Hard states: margin > 0.01 = %.2f%%, > 0.05 = %.2f%%\n",
            100 * r.hard_rate_001, 100 * r.hard_rate_005)
end

function write_summary(results)
    isempty(ARGS["output"]) && return
    open(ARGS["output"], "w") do io
        println(io, "checkpoint,n,mcts_iters,value_mae,value_rmse,value_bias,value_corr,policy_top1,policy_top3,policy_top5,policy_best_rank,policy_median_best_rank,policy_opt_mass,policy_expected_regret,policy_top1_prob,nn_top1,nn_wrong,nn_regret,nn_regret_p95,nn_regret_gt_001,nn_regret_gt_005,mcts_n,mcts_skip_rate,mcts_top1,mcts_wrong,mcts_regret,mcts_regret_p95,mcts_regret_gt_001,mcts_regret_gt_005,avg_opt_actions,tie_rate,avg_margin,hard_rate_001,hard_rate_005")
        for r in results
            @printf(io,
                "%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                r.label, r.n, r.mcts_iters, r.value_mae, r.value_rmse, r.value_bias, r.value_corr,
                r.policy_top1, r.policy_top3, r.policy_top5, r.policy_best_rank, r.policy_median_best_rank, r.policy_opt_mass, r.policy_expected_regret, r.policy_top1_prob,
                r.nn_top1, r.nn_wrong, r.nn_regret, r.nn_regret_p95, r.nn_regret_gt_001, r.nn_regret_gt_005,
                r.mcts_n, r.mcts_skip_rate, r.mcts_top1, r.mcts_wrong, r.mcts_regret, r.mcts_regret_p95, r.mcts_regret_gt_001, r.mcts_regret_gt_005,
                r.avg_opt_actions, r.tie_rate, r.avg_margin, r.hard_rate_001, r.hard_rate_005)
        end
    end
    println("Wrote summary to $(ARGS["output"])")
end

function main()
    checkpoints = parse_checkpoint_list()
    println("Bearoff accuracy eval")
    println("  Checkpoints: $(length(checkpoints))")
    println("  Bearoff positions: $(ARGS["num_positions"])")
    println("  MCTS iters: $(ARGS["mcts_iters"])")
    println("  Workers: $(min(ARGS["num_workers"], Threads.nthreads()))")
    println("  Inference backend: $(AlphaZero.BackgammonInference.cpu_backend_summary(ARGS["inference_backend"]))")

    positions = load_positions()
    table_dir = find_bearoff_dir()
    println("Loading bearoff table from $table_dir")
    table = BearoffK7.BearoffTable(table_dir)

    results = NamedTuple[]
    for ckpt in checkpoints
        println("\nEvaluating $ckpt")
        t0 = time()
        result = evaluate_checkpoint(ckpt, positions, table)
        print_result(result)
        @printf("  Elapsed: %.1fs\n", time() - t0)
        push!(results, result)
    end

    write_summary(results)
end

main()
