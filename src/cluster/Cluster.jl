#####
##### Cluster Module
#####
##### Distributed training using Julia threads for single-machine parallelism.
##### Each worker thread has its own network copy for local inference.
#####

module Cluster

using Base.Threads
using Statistics: mean, std
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

# Import reanalyze functions
using ..AlphaZero: ReanalyzeConfig, ReanalyzeStats
using ..AlphaZero: reanalyze_step!, get_reanalyze_metrics

#####
##### Weight Statistics
#####

"""
Compute per-layer weight statistics for monitoring.
Returns a Dict with weight RMS, max, and gradient info.
"""
function compute_weight_statistics(network::AbstractNetwork)
    stats = Dict{String, Any}()

    # Get all parameters
    params = Flux.params(network)

    # Overall statistics
    all_weights = Float64[]
    total_params = 0

    for (i, p) in enumerate(params)
        p_cpu = collect(p)  # Move to CPU if on GPU
        p_flat = vec(p_cpu)
        append!(all_weights, Float64.(p_flat))
        total_params += length(p_flat)

        # Per-layer stats (only log first few and last few layers to avoid clutter)
        if i <= 3 || i > length(params) - 3
            rms = sqrt(mean(p_flat .^ 2))
            stats["weights/layer$(i)_rms"] = rms
            stats["weights/layer$(i)_max"] = maximum(abs.(p_flat))
        end
    end

    # Global weight statistics
    stats["weights/total_rms"] = sqrt(mean(all_weights .^ 2))
    stats["weights/total_max"] = maximum(abs.(all_weights))
    stats["weights/total_sum_sq"] = sum(all_weights .^ 2)
    stats["weights/num_params"] = total_params

    return stats
end

#####
##### Exports
#####

# Types
export ClusterSample, GameBatch, WeightUpdate, WorkerStatus, TrainingMetrics, EvalResults
export PrioritizedSamplingConfig, get_beta

# Worker
export ClusterWorker, update_worker_weights!, stop_worker!

# Coordinator
export ClusterCoordinator, add_samples!, sample_batch, prepare_training_batch
export training_step!, training_step_prioritized!, get_network_weights, broadcast_weights!, save_checkpoint!
export sample_batch_prioritized, update_priorities!, compute_td_errors
export collect_samples!, get_stats

# Async workers
export AsyncReanalyzeWorker, start_reanalyze_worker!, stop_reanalyze_worker!, notify_model_update!
export AsyncEvalWorker, AsyncEvalResult, start_eval_worker!, stop_eval_worker!
export request_eval!, get_eval_result

# High-level API
export start_cluster_training, run_local_cluster, start_distributed_training

#####
##### Include submodules
#####

include("types.jl")
include("worker.jl")
include("coordinator.jl")
include("async_workers.jl")

#####
##### PER Metrics
#####

"""
Compute Prioritized Experience Replay metrics for logging.
"""
function compute_per_metrics(
    buffer::Vector{ClusterSample},
    config::PrioritizedSamplingConfig,
    current_step::Int
)
    metrics = Dict{String, Any}()

    if isempty(buffer)
        return metrics
    end

    priorities = [s.priority for s in buffer]

    metrics["per/enabled"] = 1.0
    metrics["per/alpha"] = Float64(config.alpha)
    metrics["per/beta"] = Float64(get_beta(config, current_step))
    metrics["per/priority_mean"] = mean(priorities)
    metrics["per/priority_max"] = maximum(priorities)
    metrics["per/priority_min"] = minimum(priorities)
    metrics["per/priority_std"] = length(priorities) > 1 ? std(priorities) : 0.0

    # Count samples with high priority (> mean + std)
    threshold = mean(priorities) + std(priorities)
    high_priority_count = count(p -> p > threshold, priorities)
    metrics["per/high_priority_pct"] = 100.0 * high_priority_count / length(priorities)

    return metrics
end

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
    seed::Union{Nothing, Int} = nothing,
    load_network_path::Union{Nothing, String} = nothing,
    reanalyze_config::ReanalyzeConfig = ReanalyzeConfig(enabled=false),
    per_config::PrioritizedSamplingConfig = PrioritizedSamplingConfig(enabled=false)
)
    # Check threads
    num_threads = nthreads()
    if num_threads < num_workers + 1
        @warn "Recommend at least $(num_workers + 1) threads. Got $num_threads. Run with: julia --threads=$(num_workers + 1)"
    end

    # Create coordinator
    network = network_constructor(gspec)

    # Load weights from checkpoint if specified
    if !isnothing(load_network_path)
        @info "Loading network weights from: $load_network_path"
        Network.load_weights(load_network_path, network)
        @info "Weights loaded successfully"
    end

    coord = ClusterCoordinator(
        gspec, network, learning_params, mcts_params;
        buffer_capacity=buffer_capacity,
        use_gpu=use_gpu,
        checkpoint_dir=checkpoint_dir
    )

    # Initialize reanalyze stats
    reanalyze_stats = ReanalyzeStats()
    if reanalyze_config.enabled
        @info "Reanalysis enabled" batch_size=reanalyze_config.batch_size alpha=reanalyze_config.reanalyze_alpha
    end

    # Log PER config
    if per_config.enabled
        @info "Prioritized Experience Replay enabled" alpha=per_config.alpha beta=per_config.beta epsilon=per_config.epsilon
    end

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

            # Add samples to coordinator buffer with added_step tracking
            for sample in new_samples
                # Create new sample with added_step set to current iteration
                # New samples are generated with current model, so mark as up-to-date
                sample_with_step = ClusterSample(
                    sample.state, sample.policy, sample.value, sample.turn, sample.is_chance,
                    sample.equity_p_win, sample.equity_p_gw, sample.equity_p_bgw,
                    sample.equity_p_gl, sample.equity_p_bgl, sample.has_equity,
                    sample.priority, iter, sample.last_reanalyze_step, sample.reanalyze_count,
                    iter  # model_iter_reanalyzed = current iteration
                )
                push!(coord.buffer, sample_with_step)
            end
            coord.total_games += games_this_iter
            coord.total_samples += length(new_samples)

            # Trim buffer to capacity
            if length(coord.buffer) > coord.buffer_capacity
                deleteat!(coord.buffer, 1:(length(coord.buffer) - coord.buffer_capacity))
            end

            # Training steps (use PER if enabled, otherwise uniform sampling)
            if length(coord.buffer) >= batch_size
                num_batches = max(1, length(coord.buffer) ÷ batch_size)
                total_loss = 0.0
                total_loss_policy = 0.0
                total_loss_value = 0.0
                total_loss_reg = 0.0
                loss_count = 0
                for _ in 1:num_batches
                    loss_result = if per_config.enabled
                        training_step_prioritized!(coord, batch_size, per_config, iter)
                    else
                        training_step!(coord, batch_size)
                    end
                    if !isnothing(loss_result)
                        L, Lp, Lv, Lreg = loss_result
                        total_loss += L
                        total_loss_policy += Lp
                        total_loss_value += Lv
                        total_loss_reg += Lreg
                        loss_count += 1
                    end
                end
                avg_loss = loss_count > 0 ? total_loss / loss_count : 0.0
                avg_loss_policy = loss_count > 0 ? total_loss_policy / loss_count : 0.0
                avg_loss_value = loss_count > 0 ? total_loss_value / loss_count : 0.0
                avg_loss_reg = loss_count > 0 ? total_loss_reg / loss_count : 0.0
            else
                avg_loss = 0.0
                avg_loss_policy = 0.0
                avg_loss_value = 0.0
                avg_loss_reg = 0.0
            end

            # Run reanalysis step if enabled
            if reanalyze_config.enabled && iter % reanalyze_config.update_interval == 0
                if length(coord.buffer) >= reanalyze_config.batch_size
                    reanalyze_step!(
                        coord.buffer,
                        coord.network,
                        gspec,
                        reanalyze_config,
                        iter,
                        reanalyze_stats,
                        coord.use_gpu
                    )
                end
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
                    "train/loss_policy" => avg_loss_policy,
                    "train/loss_value" => avg_loss_value,
                    "train/loss_reg" => avg_loss_reg,
                    "train/iteration" => iter,
                    "train/iteration_time_s" => iter_time,
                    "buffer/size" => length(coord.buffer),
                    "buffer/capacity_pct" => 100.0 * length(coord.buffer) / buffer_capacity,
                    "games/total" => coord.total_games,
                    "games/per_minute" => games_per_min,
                    "samples/total" => coord.total_samples,
                    "workers/active" => num_workers
                )

                # Add weight statistics every 10 iterations
                if iter % 10 == 0
                    weight_stats = compute_weight_statistics(coord.network)
                    merge!(metrics, weight_stats)
                end

                # Add reanalyze metrics if enabled
                if reanalyze_config.enabled && iter % reanalyze_config.log_interval == 0
                    reanalyze_metrics = get_reanalyze_metrics(reanalyze_stats)
                    merge!(metrics, reanalyze_metrics)
                end

                # Add PER metrics if enabled
                if per_config.enabled && iter % 10 == 0
                    per_metrics = compute_per_metrics(coord.buffer, per_config, iter)
                    merge!(metrics, per_metrics)
                end

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

#####
##### Truly Distributed Training with Concurrent Workers
#####

"""
    start_distributed_training(...)

Distributed training with fully concurrent architecture:
- Self-play workers run continuously on separate threads
- Reanalyze worker runs continuously on separate thread (if enabled)
- Eval worker runs continuously on separate thread (if enabled)
- Training loop runs on main thread, never blocking on workers

This architecture is designed for large-scale training where we want
maximum GPU utilization and parallel work.
"""
function start_distributed_training(
    gspec::AbstractGameSpec,
    network_constructor,
    learning_params::LearningParams,
    mcts_params::MctsParams;
    num_workers::Int = 4,
    buffer_capacity::Int = 100_000,
    batch_size::Int = 256,
    checkpoint_interval::Int = 10,
    total_iterations::Int = 100,
    games_per_iteration::Int = 50,
    use_gpu::Bool = true,
    checkpoint_dir::String = "checkpoints",
    wandb_log::Union{Nothing, Function} = nothing,
    eval_games::Int = 50,
    eval_interval::Int = 10,
    seed::Union{Nothing, Int} = nothing,
    load_network_path::Union{Nothing, String} = nothing,
    reanalyze_config::ReanalyzeConfig = ReanalyzeConfig(enabled=false),
    per_config::PrioritizedSamplingConfig = PrioritizedSamplingConfig(enabled=false)
)
    @info "=" ^ 60
    @info "Starting Distributed Training (Concurrent Architecture)"
    @info "=" ^ 60

    # Check threads
    num_threads = nthreads()
    min_threads = num_workers + 3  # workers + training + reanalyze + eval
    if num_threads < min_threads
        @warn "Recommend at least $min_threads threads. Got $num_threads."
    end

    # Create coordinator
    network = network_constructor(gspec)
    if !isnothing(load_network_path)
        @info "Loading network weights from: $load_network_path"
        Network.load_weights(load_network_path, network)
    end

    coord = ClusterCoordinator(
        gspec, network, learning_params, mcts_params;
        buffer_capacity=buffer_capacity,
        use_gpu=use_gpu,
        checkpoint_dir=checkpoint_dir
    )

    # Shared state with locks
    buffer_lock = ReentrantLock()
    network_lock = ReentrantLock()
    network_version = Ref(1)
    running = Ref(true)

    # Thread-safe network getter (returns a copy for workers)
    function get_network_copy()
        lock(network_lock) do
            return Base.copy(coord.network)
        end
    end

    # Sample buffer for workers
    sample_buffer = ThreadedSampleBuffer()

    # Generate worker seeds
    worker_seeds = if !isnothing(seed)
        [seed + i * 1000 for i in 1:num_workers]
    else
        [nothing for _ in 1:num_workers]
    end

    @info "Starting components:"
    @info "  Self-play workers: $num_workers"
    @info "  Reanalyze: $(reanalyze_config.enabled)"
    @info "  PER: $(per_config.enabled)"
    @info "  Eval interval: $eval_interval iterations"

    # Start self-play worker threads
    worker_tasks = Task[]
    for i in 1:num_workers
        worker_network = Base.copy(network)
        if use_gpu
            worker_network = Network.to_cpu(worker_network)
        end

        task = @spawn run_threaded_worker!(
            i, gspec, worker_network, mcts_params, sample_buffer,
            running, network_lock, network_version;
            worker_seed=worker_seeds[i]
        )
        push!(worker_tasks, task)
    end

    # Start async reanalyze worker (if enabled)
    reanalyze_worker = nothing
    if reanalyze_config.enabled
        reanalyze_worker = AsyncReanalyzeWorker(reanalyze_config)
        start_reanalyze_worker!(
            reanalyze_worker,
            coord.buffer,
            buffer_lock,
            get_network_copy,
            gspec,
            use_gpu
        )
        @info "Reanalyze worker started (async)"
    end

    # Start async eval worker
    eval_worker = AsyncEvalWorker()
    start_eval_worker!(
        eval_worker,
        gspec,
        get_network_copy,
        eval_games,
        mcts_params;
        use_gpu=false  # Eval on CPU to not compete with training
    )
    @info "Eval worker started (async)"

    # Request initial eval
    request_eval!(eval_worker, 0)

    start_time = time()
    last_eval_iter = 0

    try
        for iter in 1:total_iterations
            iter_start = time()
            games_at_iter_start = coord.total_games

            # Wait for minimum games this iteration
            while running[]
                new_samples = drain_samples!(sample_buffer)
                games_this_drain = get_game_count(sample_buffer) - coord.total_games

                # Add samples to buffer (with lock for reanalyze worker)
                if !isempty(new_samples)
                    lock(buffer_lock) do
                        for sample in new_samples
                            # New samples are generated with current model, so mark as up-to-date
                            sample_with_step = ClusterSample(
                                sample.state, sample.policy, sample.value, sample.turn,
                                sample.is_chance, sample.equity_p_win, sample.equity_p_gw,
                                sample.equity_p_bgw, sample.equity_p_gl, sample.equity_p_bgl,
                                sample.has_equity,
                                per_config.enabled ? per_config.initial_priority : 0.0f0,
                                iter, 0, 0,
                                iter  # model_iter_reanalyzed = current iteration
                            )
                            push!(coord.buffer, sample_with_step)
                        end
                        coord.total_games += games_this_drain
                        coord.total_samples += length(new_samples)

                        # Trim buffer to capacity
                        if length(coord.buffer) > coord.buffer_capacity
                            deleteat!(coord.buffer, 1:(length(coord.buffer) - coord.buffer_capacity))
                        end
                    end
                end

                # Check if we have enough games this iteration
                games_this_iter = coord.total_games - games_at_iter_start
                if games_this_iter >= games_per_iteration
                    break
                end

                # Small sleep while waiting for more games
                sleep(0.1)
            end

            # Wait for minimum samples before training
            buffer_size = lock(buffer_lock) do
                length(coord.buffer)
            end

            if buffer_size < batch_size
                @info "Waiting for buffer to fill..." buffer_size batch_size
                sleep(1.0)
                continue
            end

            # Update iteration counter
            coord.iteration = iter

            # Training steps (use buffer lock)
            num_batches = max(1, buffer_size ÷ batch_size)
            total_loss = 0.0
            loss_count = 0

            for _ in 1:num_batches
                # Sample and train (with lock)
                loss_result = lock(buffer_lock) do
                    if per_config.enabled
                        training_step_prioritized!(coord, batch_size, per_config, iter)
                    else
                        training_step!(coord, batch_size)
                    end
                end

                if !isnothing(loss_result)
                    L, Lp, Lv, Lreg = loss_result
                    total_loss += L
                    loss_count += 1
                end
            end

            avg_loss = loss_count > 0 ? total_loss / loss_count : 0.0

            # Update network version (triggers worker sync)
            lock(network_lock) do
                network_version[] += 1
            end

            # Notify reanalyze worker that model is updated
            if !isnothing(reanalyze_worker)
                notify_model_update!(reanalyze_worker, iter)
            end

            iter_time = time() - iter_start
            elapsed = time() - start_time
            games_per_min = coord.total_games / (elapsed / 60)

            # Check for eval results (non-blocking)
            eval_result = get_eval_result(eval_worker)
            if !isnothing(eval_result)
                @info "Eval result (iter $(eval_result.iteration))" combined=round(eval_result.vs_random_combined, digits=3)
                if !isnothing(wandb_log)
                    wandb_log(Dict(
                        "eval/vs_random_white" => eval_result.vs_random_white,
                        "eval/vs_random_black" => eval_result.vs_random_black,
                        "eval/vs_random_combined" => eval_result.vs_random_combined,
                        "eval/iteration" => eval_result.iteration
                    ))
                end
            end

            # Request new eval if interval reached
            if iter % eval_interval == 0 && iter > last_eval_iter
                request_eval!(eval_worker, iter)
                last_eval_iter = iter
            end

            # Log progress
            reanalyze_count = isnothing(reanalyze_worker) ? 0 : reanalyze_worker.samples_reanalyzed[]
            @info "Iteration $iter" avg_loss buffer_size games=coord.total_games games_per_min reanalyze=reanalyze_count

            # Logging
            if !isnothing(wandb_log)
                metrics = Dict{String, Any}(
                    "train/loss" => avg_loss,
                    "train/iteration" => iter,
                    "buffer/size" => buffer_size,
                    "games/total" => coord.total_games,
                    "games/per_minute" => games_per_min,
                    "workers/active" => num_workers
                )

                if reanalyze_config.enabled && !isnothing(reanalyze_worker)
                    metrics["reanalyze/total"] = reanalyze_worker.samples_reanalyzed[]
                end

                wandb_log(metrics)
            end

            # Checkpoint
            if iter % checkpoint_interval == 0
                save_checkpoint!(coord)
                @info "Saved checkpoint at iteration $iter"
            end
        end
    finally
        # Stop all workers
        @info "Stopping workers..."
        running[] = false

        if !isnothing(reanalyze_worker)
            stop_reanalyze_worker!(reanalyze_worker)
        end
        stop_eval_worker!(eval_worker)

        for task in worker_tasks
            try wait(task) catch end
        end
    end

    # Final checkpoint
    save_checkpoint!(coord)

    total_time = time() - start_time
    @info "Training complete" iterations=total_iterations games=coord.total_games time_min=round(total_time/60, digits=2)

    return coord
end

end # module
