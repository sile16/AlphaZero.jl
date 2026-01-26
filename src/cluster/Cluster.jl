#####
##### Cluster Module
#####
##### Distributed training using Julia threads for single-machine parallelism.
##### Each worker thread has its own network copy for local inference.
#####

module Cluster

using Base.Threads
using Statistics: mean
using Random

import Flux

using ..AlphaZero
using ..AlphaZero: GI, Network, AbstractNetwork, AbstractGameSpec
using ..AlphaZero: MctsParams, LearningParams
using ..AlphaZero: MctsPlayer, play_game, reset_player!
using ..AlphaZero: Trace, total_reward
using ..AlphaZero.NetLib: FluxNetwork, serialize_weights, deserialize_weights, load_weights!, save_weights

# Import losses function from learning.jl
using ..AlphaZero: losses

#####
##### Exports
#####

# Types
export ClusterSample, GameBatch, WeightUpdate, WorkerStatus, TrainingMetrics, EvalResults

# Worker
export ClusterWorker, update_worker_weights!, stop_worker!

# Coordinator
export ClusterCoordinator, add_samples!, sample_batch, prepare_training_batch
export training_step!, get_network_weights, broadcast_weights!, save_checkpoint!
export collect_samples!, get_stats

# High-level API
export start_cluster_training, run_local_cluster

#####
##### Include submodules
#####

include("types.jl")
include("worker.jl")
include("coordinator.jl")

#####
##### Thread-based training implementation
#####

"""
Thread-safe sample buffer for worker-trainer communication.
"""
mutable struct ThreadedSampleBuffer
    samples::Vector{ClusterSample}
    games_completed::Int
    lock::ReentrantLock
end

ThreadedSampleBuffer() = ThreadedSampleBuffer(ClusterSample[], 0, ReentrantLock())

function add_game!(buffer::ThreadedSampleBuffer, samples::Vector{ClusterSample})
    lock(buffer.lock) do
        append!(buffer.samples, samples)
        buffer.games_completed += 1
    end
end

function drain_samples!(buffer::ThreadedSampleBuffer)
    lock(buffer.lock) do
        result = copy(buffer.samples)
        empty!(buffer.samples)
        return result
    end
end

function get_game_count(buffer::ThreadedSampleBuffer)
    lock(buffer.lock) do
        return buffer.games_completed
    end
end

"""
Thread-based self-play worker.
Each worker has its own RNG seeded with a unique seed derived from the main seed.
"""
function run_threaded_worker!(
    worker_id::Int,
    gspec::AbstractGameSpec,
    network::AbstractNetwork,
    mcts_params::MctsParams,
    sample_buffer::ThreadedSampleBuffer,
    running::Ref{Bool},
    network_lock::ReentrantLock,
    network_version::Ref{Int};
    worker_seed::Union{Nothing, Int}=nothing
)
    # Initialize thread-local RNG with worker-specific seed
    if !isnothing(worker_seed)
        Random.seed!(worker_seed)
        @debug "Worker $worker_id: seeded RNG with $worker_seed"
    end

    # Create worker's own network copy (CPU, test mode)
    worker_network = Network.copy(network, on_gpu=false, test_mode=true)
    local_version = 0

    @debug "Worker $worker_id started"

    while running[]
        try
            # Sync weights if version changed
            if network_version[] > local_version
                lock(network_lock) do
                    for (wp, mp) in zip(Network.params(worker_network), Network.params(network))
                        copyto!(wp, Array(mp))
                    end
                    local_version = network_version[]
                end
                @debug "Worker $worker_id: synced to version $local_version"
            end

            # Play a game
            player = MctsPlayer(gspec, worker_network, mcts_params)
            trace = play_game(gspec, player)
            reset_player!(player)

            # Convert to samples
            samples = trace_to_cluster_samples(gspec, trace, mcts_params)

            # Submit samples
            add_game!(sample_buffer, samples)

        catch e
            if !(e isa InterruptException)
                @error "Worker $worker_id error" exception=(e, catch_backtrace())
            end
            break
        end
    end

    @debug "Worker $worker_id stopped"
end

"""
Convert a trace to cluster samples.
"""
function trace_to_cluster_samples(gspec::AbstractGameSpec, trace::Trace, mcts_params::MctsParams)
    gamma = mcts_params.gamma
    n = length(trace)
    samples = ClusterSample[]
    sizehint!(samples, n)

    # Compute cumulative rewards
    wr = 0.0
    cumulative_rewards = zeros(n)
    for i in reverse(1:n)
        wr = gamma * wr + trace.rewards[i]
        cumulative_rewards[i] = wr
    end

    has_outcome = !isnothing(trace.outcome)
    num_actions = GI.num_actions(gspec)

    for i in 1:n
        state = trace.states[i]
        policy = trace.policies[i]
        is_chance = trace.is_chance[i]

        wp = GI.white_playing(gspec, state)
        z = wp ? cumulative_rewards[i] : -cumulative_rewards[i]
        t = Float32(n - i + 1)

        # Vectorize state
        state_arr = GI.vectorize_state(gspec, state)
        state_vec = Vector{Float32}(vec(state_arr))

        # Expand sparse policy to full action space
        actions_mask = GI.actions_mask(GI.init(gspec, state))
        full_policy = zeros(Float32, num_actions)
        if !is_chance && !isempty(policy)
            full_policy[actions_mask] = Float32.(policy)
        end

        # Equity targets
        eq_win, eq_gw, eq_bgw, eq_gl, eq_bgl = 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0
        has_eq = false

        if has_outcome
            outcome = trace.outcome
            won = outcome.white_won == wp
            has_eq = true
            if won
                eq_win = 1.0f0
                eq_gw = outcome.is_gammon ? 1.0f0 : 0.0f0
                eq_bgw = outcome.is_backgammon ? 1.0f0 : 0.0f0
            else
                eq_gl = outcome.is_gammon ? 1.0f0 : 0.0f0
                eq_bgl = outcome.is_backgammon ? 1.0f0 : 0.0f0
            end
        end

        push!(samples, ClusterSample(
            state_vec, full_policy, Float32(z), t, is_chance,
            eq_win, eq_gw, eq_bgw, eq_gl, eq_bgl, has_eq
        ))
    end

    return samples
end

"""
    start_cluster_training(...)

Start training using Julia threads for parallel self-play.
Each worker thread has its own network copy for inference.
Each worker gets a unique seed derived from the main seed for reproducibility.

Returns the coordinator after training completes.
"""
function start_cluster_training(
    gspec::AbstractGameSpec,
    network_constructor,
    learning_params::LearningParams,
    mcts_params::MctsParams;
    num_workers::Int = 4,
    buffer_capacity::Int = 100_000,
    batch_size::Int = 2048,
    checkpoint_interval::Int = 10,
    total_iterations::Int = 100,
    games_per_iteration::Int = 50,
    use_gpu::Bool = true,
    checkpoint_dir::String = "checkpoints",
    wandb_log::Union{Nothing, Function} = nothing,
    eval_fn::Union{Nothing, Function} = nothing,
    eval_interval::Int = 10,
    seed::Union{Nothing, Int} = nothing
)
    # Check threads
    num_threads = nthreads()
    if num_threads < num_workers + 1
        @warn "Recommend at least $(num_workers + 1) threads. Got $num_threads. Run with: julia --threads=$(num_workers + 1)"
    end

    # Create coordinator
    network = network_constructor(gspec)
    coord = ClusterCoordinator(
        gspec, network, learning_params, mcts_params;
        buffer_capacity=buffer_capacity,
        use_gpu=use_gpu,
        checkpoint_dir=checkpoint_dir
    )

    # Shared state for thread communication
    sample_buffer = ThreadedSampleBuffer()
    running = Ref(true)
    network_lock = ReentrantLock()
    network_version = Ref(1)

    # Generate worker seeds from main seed
    worker_seeds = if !isnothing(seed)
        # Derive unique seeds for each worker: seed + worker_id * large_prime
        [seed + i * 104729 for i in 1:num_workers]
    else
        fill(nothing, num_workers)
    end

    # Start worker threads
    if !isnothing(seed)
        @info "Starting $num_workers worker threads with seeds derived from $seed"
    else
        @info "Starting $num_workers worker threads (unseeded)"
    end
    worker_tasks = Task[]
    for i in 1:num_workers
        ws = worker_seeds[i]
        task = Threads.@spawn run_threaded_worker!(
            i, gspec, coord.network, mcts_params,
            sample_buffer, running, network_lock, network_version;
            worker_seed=ws
        )
        push!(worker_tasks, task)
    end

    # Training loop
    start_time = time()
    games_at_iter_start = 0

    try
        for iter in 1:total_iterations
            coord.iteration = iter
            iter_start = time()

            # Collect samples from workers
            target_games = games_per_iteration
            wait_start = time()
            max_wait = 300.0  # 5 minute timeout
            while get_game_count(sample_buffer) - games_at_iter_start < target_games
                sleep(0.1)
                if time() - wait_start > max_wait
                    @warn "Timeout waiting for games"
                    break
                end
            end

            # Drain samples and add to coordinator buffer
            new_samples = drain_samples!(sample_buffer)
            games_this_iter = get_game_count(sample_buffer) - games_at_iter_start
            games_at_iter_start = get_game_count(sample_buffer)

            # Add samples to coordinator buffer
            for sample in new_samples
                push!(coord.buffer, sample)
            end
            coord.total_games += games_this_iter
            coord.total_samples += length(new_samples)

            # Trim buffer to capacity
            if length(coord.buffer) > coord.buffer_capacity
                deleteat!(coord.buffer, 1:(length(coord.buffer) - coord.buffer_capacity))
            end

            # Training steps
            if length(coord.buffer) >= batch_size
                num_batches = max(1, length(coord.buffer) ÷ batch_size)
                total_loss = 0.0
                loss_count = 0
                for _ in 1:num_batches
                    loss = training_step!(coord, batch_size)
                    if !isnothing(loss)
                        total_loss += loss
                        loss_count += 1
                    end
                end
                avg_loss = loss_count > 0 ? total_loss / loss_count : 0.0
            else
                avg_loss = 0.0
            end

            # Update network version to trigger worker sync
            lock(network_lock) do
                network_version[] += 1
            end

            iter_time = time() - iter_start
            elapsed = time() - start_time
            games_per_min = coord.total_games / (elapsed / 60)

            @info "Iteration $iter" avg_loss buffer_size=length(coord.buffer) total_games=coord.total_games games_per_min iter_time

            # WandB logging
            if !isnothing(wandb_log)
                metrics = Dict{String, Any}(
                    "train/loss" => avg_loss,
                    "train/iteration" => iter,
                    "train/iteration_time_s" => iter_time,
                    "buffer/size" => length(coord.buffer),
                    "buffer/capacity_pct" => 100.0 * length(coord.buffer) / buffer_capacity,
                    "games/total" => coord.total_games,
                    "games/per_minute" => games_per_min,
                    "samples/total" => coord.total_samples,
                    "workers/active" => num_workers
                )
                wandb_log(metrics)
            end

            # Evaluation
            if !isnothing(eval_fn) && iter % eval_interval == 0
                eval_results = eval_fn(coord.network)
                if !isnothing(wandb_log)
                    wandb_log(eval_results)
                end
            end

            # Checkpoint
            if iter % checkpoint_interval == 0
                save_checkpoint!(coord)
                @info "Saved checkpoint at iteration $iter"
            end
        end
    finally
        # Stop workers
        running[] = false
        @info "Waiting for workers to finish..."
        for task in worker_tasks
            try
                wait(task)
            catch end
        end
    end

    # Final checkpoint
    save_checkpoint!(coord)

    return coord
end

"""
    run_local_cluster(...)

Convenience wrapper for `start_cluster_training` with sensible defaults
for local single-machine training.
"""
function run_local_cluster(
    gspec::AbstractGameSpec,
    network_constructor,
    learning_params::LearningParams,
    mcts_params::MctsParams;
    num_workers::Int = max(1, nthreads() - 1),
    kwargs...
)
    return start_cluster_training(
        gspec, network_constructor, learning_params, mcts_params;
        num_workers=num_workers,
        kwargs...
    )
end

end # module
