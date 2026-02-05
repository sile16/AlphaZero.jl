#!/usr/bin/env julia
"""
Distributed AlphaZero training using Julia's Distributed stdlib.

Architecture:
- Master process: Training loop, replay buffer, weight updates
- Worker processes: Self-play game generation with MCTS

Usage:
    # Local (spawns worker processes on same machine)
    julia --project scripts/train_distributed.jl \\
        --num-workers=6 \\
        --total-iterations=50

    # Remote workers (add machines via --worker-hosts)
    julia --project scripts/train_distributed.jl \\
        --num-workers=0 \\
        --worker-hosts="worker1,worker2,worker3"
"""

using ArgParse
using Dates
using Distributed
using Random
using Statistics

function parse_args()
    s = ArgParseSettings(
        description="Distributed AlphaZero training",
        autofix_names=true
    )

    @add_arg_table! s begin
        "--game"
            help = "Game to train"
            arg_type = String
            default = "backgammon-deterministic"
        "--network-type"
            help = "Network type"
            arg_type = String
            default = "fcresnet-multihead"
        "--network-width"
            help = "Network width"
            arg_type = Int
            default = 128
        "--network-blocks"
            help = "Number of residual blocks"
            arg_type = Int
            default = 3
        "--num-workers"
            help = "Number of local worker processes"
            arg_type = Int
            default = 4
        "--worker-hosts"
            help = "Comma-separated list of remote worker hostnames"
            arg_type = String
            default = ""
        "--total-iterations"
            help = "Total training iterations"
            arg_type = Int
            default = 50
        "--games-per-iteration"
            help = "Games to collect per iteration"
            arg_type = Int
            default = 50
        "--batch-size"
            help = "Training batch size"
            arg_type = Int
            default = 256
        "--buffer-capacity"
            help = "Replay buffer capacity"
            arg_type = Int
            default = 100000
        "--mcts-iters"
            help = "MCTS iterations per move"
            arg_type = Int
            default = 100
        "--learning-rate"
            help = "Learning rate"
            arg_type = Float64
            default = 0.001
        "--l2-reg"
            help = "L2 regularization"
            arg_type = Float64
            default = 0.0001
        "--eval-interval"
            help = "Evaluation interval"
            arg_type = Int
            default = 10
        "--eval-games"
            help = "Games per evaluation"
            arg_type = Int
            default = 50
        "--final-eval-games"
            help = "Games for final evaluation (0 to disable)"
            arg_type = Int
            default = 1000
        "--checkpoint-interval"
            help = "Checkpoint save interval"
            arg_type = Int
            default = 10
        "--seed"
            help = "Random seed"
            arg_type = Int
            default = nothing
        "--no-wandb"
            help = "Disable WandB logging"
            action = :store_true
        "--session-dir"
            help = "Session directory (auto-generated if not specified)"
            arg_type = String
            default = ""
    end

    return ArgParse.parse_args(s)
end

# Parse args before adding workers
const ARGS_PARSED = parse_args()

# Setup session directory
const SESSION_DIR = if isempty(ARGS_PARSED["session_dir"])
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    joinpath("sessions", "distributed_$(timestamp)")
else
    ARGS_PARSED["session_dir"]
end
mkpath(SESSION_DIR)
mkpath(joinpath(SESSION_DIR, "checkpoints"))

println("=" ^ 60)
println("Distributed AlphaZero Training")
println("=" ^ 60)
println("Session: $SESSION_DIR")
println("Game: $(ARGS_PARSED["game"])")
println("Network: $(ARGS_PARSED["network_type"]) ($(ARGS_PARSED["network_width"])x$(ARGS_PARSED["network_blocks"]))")
println("Workers: $(ARGS_PARSED["num_workers"]) local processes")
println("Iterations: $(ARGS_PARSED["total_iterations"])")
println("Games/iteration: $(ARGS_PARSED["games_per_iteration"])")
println("MCTS iterations: $(ARGS_PARSED["mcts_iters"])")
println("Eval interval: $(ARGS_PARSED["eval_interval"]) iterations")
println("Eval games: $(ARGS_PARSED["eval_games"])")
println("Final eval games: $(ARGS_PARSED["final_eval_games"])")
println("=" ^ 60)
flush(stdout)

# Add worker processes
num_local_workers = ARGS_PARSED["num_workers"]
if num_local_workers > 0
    println("\nSpawning $num_local_workers worker processes...")
    flush(stdout)
    addprocs(num_local_workers; exeflags=`--project=$(Base.active_project())`)
end

# Add remote workers if specified
worker_hosts = ARGS_PARSED["worker_hosts"]
if !isempty(worker_hosts)
    hosts = split(worker_hosts, ",")
    println("Adding remote workers: $hosts")
    for host in hosts
        addprocs([(strip(String(host)), 1)]; exeflags=`--project`)
    end
end

println("Total workers: $(nworkers())")
flush(stdout)

# Configuration to broadcast to workers
const GAME_NAME = ARGS_PARSED["game"]
const NET_WIDTH = ARGS_PARSED["network_width"]
const NET_BLOCKS = ARGS_PARSED["network_blocks"]
const MCTS_ITERS = ARGS_PARSED["mcts_iters"]
const MAIN_SEED = ARGS_PARSED["seed"]

# Load packages on all workers
@everywhere begin
    using Random
    using Statistics
end

@everywhere using AlphaZero
@everywhere using AlphaZero: GI, Network, FluxLib, MctsParams, MctsPlayer, ConstSchedule
@everywhere using AlphaZero.NetLib: serialize_weights, deserialize_weights
@everywhere import Flux

# Set game-specific environment on all workers
if GAME_NAME == "backgammon-deterministic"
    @everywhere ENV["BACKGAMMON_OBS_TYPE"] = "minimal"
end

# Include game module on all workers
const SCRIPT_DIR = @__DIR__
@everywhere SCRIPT_DIR_REMOTE = $SCRIPT_DIR

@everywhere begin
    # Include game based on name
    const GAMES_DIR = joinpath(SCRIPT_DIR_REMOTE, "..", "games")

    if $GAME_NAME == "backgammon-deterministic"
        include(joinpath(GAMES_DIR, "backgammon-deterministic", "game.jl"))
    elseif $GAME_NAME == "backgammon"
        include(joinpath(GAMES_DIR, "backgammon", "game.jl"))
    else
        error("Unknown game: $($GAME_NAME)")
    end
end

# Broadcast configuration to workers
@everywhere begin
    const NET_WIDTH_W = $NET_WIDTH
    const NET_BLOCKS_W = $NET_BLOCKS
    const MCTS_ITERS_W = $MCTS_ITERS
    const MAIN_SEED_W = $MAIN_SEED
end

# Define helper functions FIRST to avoid world age issues
@everywhere begin
    """Sample an index from a probability distribution."""
    function sample_from_policy(policy::AbstractVector{<:Real}, rng)
        r = rand(rng)
        cumsum = 0.0
        for i in 1:length(policy)
            cumsum += policy[i]
            if r <= cumsum
                return i
            end
        end
        return length(policy)
    end

    """Convert game trace to training samples."""
    function convert_trace_to_samples(gspec, states, policies, rewards, is_chance, final_reward, outcome)
        n = length(states)
        samples = []

        gamma = 0.99f0

        # Compute cumulative rewards
        cumulative = zeros(Float32, n)
        cr = 0.0f0
        for i in n:-1:1
            cr = gamma * cr + rewards[i]
            cumulative[i] = cr
        end
        # Add final reward
        cumulative[end] += final_reward

        num_actions = GI.num_actions(gspec)

        for i in 1:n
            state = states[i]
            policy = policies[i]
            is_ch = is_chance[i]

            # Get white playing
            wp = GI.white_playing(gspec, state)
            z = wp ? cumulative[i] : -cumulative[i]

            # Vectorize state
            state_arr = GI.vectorize_state(gspec, state)
            state_vec = Vector{Float32}(vec(state_arr))

            # Expand policy
            full_policy = zeros(Float32, num_actions)
            if !is_ch && !isempty(policy)
                env_temp = GI.init(gspec, state)
                actions_mask = GI.actions_mask(env_temp)
                full_policy[actions_mask] = policy
            end

            # Equity targets
            eq = zeros(Float32, 5)
            has_eq = false
            if !isnothing(outcome)
                has_eq = true
                won = outcome.white_won == wp
                if won
                    eq[1] = 1.0f0  # P(win)
                    eq[2] = outcome.is_gammon ? 1.0f0 : 0.0f0  # P(gammon|win)
                    eq[3] = outcome.is_backgammon ? 1.0f0 : 0.0f0  # P(bg|win)
                else
                    eq[4] = outcome.is_gammon ? 1.0f0 : 0.0f0  # P(gammon|loss)
                    eq[5] = outcome.is_backgammon ? 1.0f0 : 0.0f0  # P(bg|loss)
                end
            end

            push!(samples, (
                state=state_vec,
                policy=full_policy,
                value=z,
                equity=eq,
                has_equity=has_eq,
                is_chance=is_ch
            ))
        end

        return samples
    end
end

# Worker initialization
@everywhere begin
    global gspec = GameSpec()
    global worker_network = nothing
    global worker_mcts_params = nothing
    global worker_player = nothing
    global worker_rng = nothing

    function init_worker(worker_id::Int, weight_bytes::Vector{UInt8})
        global worker_network, worker_mcts_params, worker_player, worker_rng

        # Create network
        worker_network = FluxLib.FCResNetMultiHead(
            gspec,
            FluxLib.FCResNetMultiHeadHP(width=NET_WIDTH_W, num_blocks=NET_BLOCKS_W)
        )

        # Load weights
        weights = deserialize_weights(weight_bytes)
        for (p, w) in zip(Flux.params(worker_network), weights)
            copyto!(p, w)
        end

        # Create MCTS params
        worker_mcts_params = MctsParams(
            num_iters_per_turn=MCTS_ITERS_W,
            cpuct=2.0,
            temperature=ConstSchedule(1.0),
            dirichlet_noise_ϵ=0.25,
            dirichlet_noise_α=1.0
        )

        # Create player
        worker_player = MctsPlayer(gspec, worker_network, worker_mcts_params)

        # Initialize RNG with worker-specific seed
        seed = isnothing(MAIN_SEED_W) ? nothing : MAIN_SEED_W + worker_id * 104729
        worker_rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)

        return (worker_id=worker_id, params=sum(length, Flux.params(worker_network)))
    end

    function update_weights(weight_bytes::Vector{UInt8})
        global worker_network
        weights = deserialize_weights(weight_bytes)
        for (p, w) in zip(Flux.params(worker_network), weights)
            copyto!(p, w)
        end
        return true
    end

    function play_game_and_get_samples()
        global gspec, worker_player, worker_rng

        # Play game
        env = GI.init(gspec)
        if hasproperty(env, :rng)
            env.rng = worker_rng
        end

        trace_states = []
        trace_policies = []
        trace_rewards = Float32[]
        trace_is_chance = Bool[]

        while !GI.game_terminated(env)
            state = GI.current_state(env)
            push!(trace_states, state)

            # Check if chance node
            if GI.is_chance_node(env)
                push!(trace_policies, Float32[])
                push!(trace_is_chance, true)
                GI.play!(env, rand(worker_rng, GI.available_actions(env)))
            else
                actions, policy = AlphaZero.think(worker_player, env)
                push!(trace_policies, Float32.(policy))
                push!(trace_is_chance, false)

                # Sample action from policy
                action = actions[sample_from_policy(policy, worker_rng)]
                GI.play!(env, action)
            end

            push!(trace_rewards, 0.0f0)
        end

        AlphaZero.reset_player!(worker_player)

        # Get final reward
        final_reward = Float32(GI.white_reward(env))

        # Get game outcome for multi-head training
        outcome = GI.game_outcome(env)

        # Convert trace to samples (pass gspec explicitly)
        samples = convert_trace_to_samples(
            gspec, trace_states, trace_policies, trace_rewards, trace_is_chance,
            final_reward, outcome
        )

        return samples
    end

    """
    Play evaluation games on worker against random player.
    Returns (white_rewards, black_rewards) arrays.
    """
    function play_eval_games(weight_bytes::Vector{UInt8}, num_games::Int, play_as_white::Bool)
        global gspec, worker_rng

        # Create evaluation network (fresh copy with given weights)
        eval_network = FluxLib.FCResNetMultiHead(
            gspec,
            FluxLib.FCResNetMultiHeadHP(width=NET_WIDTH_W, num_blocks=NET_BLOCKS_W)
        )
        weights = deserialize_weights(weight_bytes)
        for (p, w) in zip(Flux.params(eval_network), weights)
            copyto!(p, w)
        end

        # Create evaluation player (lower temperature, no noise)
        eval_mcts_params = MctsParams(
            num_iters_per_turn=MCTS_ITERS_W,
            cpuct=1.5,
            temperature=ConstSchedule(0.0),
            dirichlet_noise_ϵ=0.0,
            dirichlet_noise_α=1.0
        )
        eval_player = MctsPlayer(gspec, eval_network, eval_mcts_params)

        rewards = Float64[]

        for _ in 1:num_games
            env = GI.init(gspec)
            if hasproperty(env, :rng)
                env.rng = worker_rng
            end

            while !GI.game_terminated(env)
                if GI.is_chance_node(env)
                    GI.play!(env, rand(worker_rng, GI.available_actions(env)))
                elseif play_as_white == GI.white_playing(env)
                    # Our turn
                    actions, policy = AlphaZero.think(eval_player, env)
                    GI.play!(env, actions[argmax(policy)])
                else
                    # Random player's turn
                    GI.play!(env, rand(worker_rng, GI.available_actions(env)))
                end
            end

            # Get reward from our perspective
            reward = GI.white_reward(env)
            if !play_as_white
                reward = -reward
            end
            push!(rewards, reward)

            AlphaZero.reset_player!(eval_player)
        end

        return rewards
    end
end

println("\nInitializing workers...")
flush(stdout)

# Load master packages
using AlphaZero
using AlphaZero: GI, Network, FluxLib, MctsParams
using AlphaZero.NetLib: serialize_weights, deserialize_weights
import Flux
import CUDA

# Check GPU
use_gpu = CUDA.functional()
if use_gpu
    CUDA.allowscalar(false)
    println("GPU: $(CUDA.name(CUDA.device()))")
else
    println("GPU: Not available, using CPU")
end
flush(stdout)

# Include game on master
include(joinpath(SCRIPT_DIR, "..", "games", GAME_NAME, "game.jl"))

# Create master network
gspec_master = GameSpec()
network = FluxLib.FCResNetMultiHead(
    gspec_master,
    FluxLib.FCResNetMultiHeadHP(width=NET_WIDTH, num_blocks=NET_BLOCKS)
)

if use_gpu
    network = Network.to_gpu(network)
end

println("Network parameters: $(sum(length, Flux.params(network)))")

# Serialize weights for workers
weight_bytes = serialize_weights(network)
println("Serialized weights: $(length(weight_bytes)) bytes")
flush(stdout)

# Initialize all workers
worker_ids = workers()
init_results = pmap(enumerate(worker_ids)) do (idx, wid)
    remotecall_fetch(init_worker, wid, idx, weight_bytes)
end
println("Workers initialized: $(length(init_results))")
for r in init_results
    println("  Worker $(r.worker_id): $(r.params) params")
end
flush(stdout)

"""
Run parallel evaluation on workers.
Distributes games across all workers and collects results.
Returns (white_avg, black_avg, combined_avg).
"""
function parallel_eval_on_workers(weight_bytes::Vector{UInt8}, num_games::Int; verbose::Bool=true)
    games_per_side = num_games ÷ 2
    games_per_worker = max(1, games_per_side ÷ nworkers())

    # Play as white - distribute across workers
    if verbose
        println("  Playing $(games_per_side) games as white...")
        flush(stdout)
    end

    white_results = pmap(worker_ids) do wid
        remotecall_fetch(play_eval_games, wid, weight_bytes, games_per_worker, true)
    end
    white_rewards = reduce(vcat, white_results)

    # Play as black - distribute across workers
    if verbose
        println("  Playing $(games_per_side) games as black...")
        flush(stdout)
    end

    black_results = pmap(worker_ids) do wid
        remotecall_fetch(play_eval_games, wid, weight_bytes, games_per_worker, false)
    end
    black_rewards = reduce(vcat, black_results)

    # Compute averages
    white_avg = isempty(white_rewards) ? 0.0 : mean(white_rewards)
    black_avg = isempty(black_rewards) ? 0.0 : mean(black_rewards)
    combined = (white_avg + black_avg) / 2

    actual_games = length(white_rewards) + length(black_rewards)

    return (
        white_avg=white_avg,
        black_avg=black_avg,
        combined=combined,
        actual_games=actual_games
    )
end

# Training parameters
const LEARNING_RATE = Float32(ARGS_PARSED["learning_rate"])
const L2_REG = Float32(ARGS_PARSED["l2_reg"])

# Create optimizer
opt = Flux.Adam(LEARNING_RATE)
opt_state = Flux.setup(opt, network)

# Replay buffer
const BUFFER_CAPACITY = ARGS_PARSED["buffer_capacity"]
const BATCH_SIZE = ARGS_PARSED["batch_size"]
replay_buffer = []

# Training metrics
total_games = 0
total_samples = 0
start_time = time()

# Save run info
open(joinpath(SESSION_DIR, "run_info.txt"), "w") do f
    println(f, "# Distributed AlphaZero Training")
    println(f, "timestamp: $(Dates.format(now(), "yyyymmdd_HHMMSS"))")
    println(f, "num_workers: $(nworkers())")
    println(f, "game: $GAME_NAME")
    println(f, "network: $(ARGS_PARSED["network_type"]) $(NET_WIDTH)x$(NET_BLOCKS)")
    println(f, "mcts_iters: $MCTS_ITERS")
    println(f, "seed: $(isnothing(MAIN_SEED) ? "none" : MAIN_SEED)")
end

println("\n" * "=" ^ 60)
println("Starting training...")
println("=" ^ 60)
flush(stdout)

# Training loop
for iter in 1:ARGS_PARSED["total_iterations"]
    iter_start = time()

    # Collect games from workers
    games_needed = ARGS_PARSED["games_per_iteration"]
    games_per_worker = max(1, games_needed ÷ nworkers())

    # Parallel game collection
    all_samples = pmap(worker_ids) do wid
        samples = []
        for _ in 1:games_per_worker
            game_samples = remotecall_fetch(play_game_and_get_samples, wid)
            append!(samples, game_samples)
        end
        return samples
    end

    # Flatten and add to buffer
    new_samples = reduce(vcat, all_samples)
    append!(replay_buffer, new_samples)

    games_this_iter = games_per_worker * nworkers()
    global total_games += games_this_iter
    global total_samples += length(new_samples)

    # Trim buffer
    if length(replay_buffer) > BUFFER_CAPACITY
        deleteat!(replay_buffer, 1:(length(replay_buffer) - BUFFER_CAPACITY))
    end

    # Training
    avg_loss = 0.0
    if length(replay_buffer) >= BATCH_SIZE
        num_batches = max(1, length(replay_buffer) ÷ BATCH_SIZE)
        total_loss = 0.0

        for _ in 1:num_batches
            # Sample batch
            indices = rand(1:length(replay_buffer), BATCH_SIZE)
            batch = [replay_buffer[i] for i in indices]

            # Prepare batch tensors
            states = hcat([s.state for s in batch]...)
            policies = hcat([s.policy for s in batch]...)
            values = Float32[s.value for s in batch]
            equities = hcat([s.equity for s in batch]...)
            has_eq = [s.has_equity for s in batch]

            if use_gpu
                states = CUDA.cu(states)
                policies = CUDA.cu(policies)
                values = CUDA.cu(values)
                equities = CUDA.cu(equities)
            end

            # Training step
            loss, grads = Flux.withgradient(network) do net
                preds_policy, preds_value = Network.forward(net, states)

                # Policy loss (cross-entropy)
                policy_loss = -mean(sum(policies .* log.(preds_policy .+ 1f-8), dims=1))

                # Value loss (MSE)
                value_loss = mean((values .- vec(preds_value[1,:])).^2)

                policy_loss + value_loss
            end

            Flux.update!(opt_state, network, grads[1])

            # Apply weight decay (L2 regularization) manually after gradient update
            for p in Flux.params(network)
                p .*= (1.0f0 - L2_REG * LEARNING_RATE)
            end

            total_loss += loss
        end

        avg_loss = total_loss / num_batches
    end

    # Update worker weights
    global weight_bytes = serialize_weights(network)
    @sync for wid in worker_ids
        @async remotecall_fetch(update_weights, wid, weight_bytes)
    end

    iter_time = time() - iter_start
    elapsed = time() - start_time
    games_per_min = total_games / (elapsed / 60)

    @info "Iteration $iter" avg_loss buffer_size=length(replay_buffer) total_games games_per_min iter_time
    flush(stdout)

    # Evaluation (runs on workers in parallel)
    if iter % ARGS_PARSED["eval_interval"] == 0
        eval_games = ARGS_PARSED["eval_games"]
        @info "Running parallel evaluation on workers ($eval_games games)..."
        flush(stdout)

        eval_start = time()
        eval_results = parallel_eval_on_workers(weight_bytes, eval_games; verbose=true)
        eval_time = time() - eval_start

        @info "Eval results: white=$(round(eval_results.white_avg, digits=3)), " *
              "black=$(round(eval_results.black_avg, digits=3)), " *
              "combined=$(round(eval_results.combined, digits=3)) " *
              "($(eval_results.actual_games) games in $(round(eval_time, digits=1))s)"
    end

    # Checkpoint
    if iter % ARGS_PARSED["checkpoint_interval"] == 0
        checkpoint_path = joinpath(SESSION_DIR, "checkpoints", "iter_$iter.data")
        FluxLib.save_weights(checkpoint_path, network)

        # Also save as latest
        latest_path = joinpath(SESSION_DIR, "checkpoints", "latest.data")
        FluxLib.save_weights(latest_path, network)

        # Save iteration number
        open(joinpath(SESSION_DIR, "checkpoints", "iter.txt"), "w") do f
            println(f, iter)
        end

        @info "Saved checkpoint at iteration $iter"
    end
end

# Final evaluation (before cleanup, so workers are still available)
final_eval_games = ARGS_PARSED["final_eval_games"]
if final_eval_games > 0
    println("\n" * "=" ^ 60)
    println("Running Final Evaluation")
    println("=" ^ 60)
    println("Games: $final_eval_games")
    println("Workers: $(nworkers())")
    println("MCTS iterations: $MCTS_ITERS")
    flush(stdout)

    final_eval_start = time()
    final_results = parallel_eval_on_workers(weight_bytes, final_eval_games; verbose=true)
    final_eval_time = time() - final_eval_start

    println("=" ^ 60)
    println("Final Evaluation Results:")
    println("  White:    $(round(final_results.white_avg, digits=3))")
    println("  Black:    $(round(final_results.black_avg, digits=3))")
    println("  Combined: $(round(final_results.combined, digits=3))")
    println("  Games:    $(final_results.actual_games)")
    println("  Time:     $(round(final_eval_time / 60, digits=2)) minutes")
    println("  Speed:    $(round(final_results.actual_games / final_eval_time * 60, digits=1)) games/min")
    println("=" ^ 60)
    flush(stdout)

    # Save final evaluation results to session directory
    open(joinpath(SESSION_DIR, "final_eval_results.txt"), "w") do f
        println(f, "# Final Evaluation Results")
        println(f, "timestamp: $(Dates.format(now(), "yyyymmdd_HHMMSS"))")
        println(f, "games: $(final_results.actual_games)")
        println(f, "white_avg: $(final_results.white_avg)")
        println(f, "black_avg: $(final_results.black_avg)")
        println(f, "combined: $(final_results.combined)")
        println(f, "time_seconds: $final_eval_time")
        println(f, "workers: $(nworkers())")
        println(f, "mcts_iters: $MCTS_ITERS")
    end
    println("Results saved to: $(joinpath(SESSION_DIR, "final_eval_results.txt"))")
end

# Training complete
elapsed = time() - start_time
println("\n" * "=" ^ 60)
println("Training Complete!")
println("=" ^ 60)
println("Total iterations: $(ARGS_PARSED["total_iterations"])")
println("Total games: $total_games")
println("Total samples: $total_samples")
println("Total time: $(round(elapsed/60, digits=1)) minutes")
println("Session: $SESSION_DIR")
println("=" ^ 60)

# Cleanup workers
rmprocs(workers())
