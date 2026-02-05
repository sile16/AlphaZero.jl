#####
##### Async Workers for Distributed Training
#####
##### Provides non-blocking reanalyze and evaluation workers that run
##### concurrently with the main training loop.
#####

using Base.Threads: @spawn, nthreads, Atomic
using Statistics: mean
import Flux

using ..AlphaZero: GI, Network, AbstractNetwork, AbstractGameSpec
using ..AlphaZero: MctsParams, LearningParams
using ..AlphaZero: ReanalyzeConfig, ReanalyzeStats
using ..AlphaZero: sample_for_reanalysis, sample_for_smart_reanalysis, count_stale_samples

# Import MCTS player and related functions
import ..AlphaZero: MctsPlayer, think, reset_player!

#####
##### Async Reanalyze Worker
#####

"""
    AsyncReanalyzeWorker

Background worker that continuously reanalyzes buffer samples.
Runs on a separate thread and doesn't block training.

Smart reanalysis strategy:
- Tracks which model iteration each sample was last reanalyzed with
- Prioritizes samples with oldest model version (most stale)
- Stops when all samples are up-to-date with current model
- Resumes when model is updated
"""
mutable struct AsyncReanalyzeWorker
    running::Ref{Bool}
    task::Union{Nothing, Task}
    stats::ReanalyzeStats
    config::ReanalyzeConfig
    samples_reanalyzed::Atomic{Int}
    current_model_iteration::Atomic{Int}  # Track latest model version
    samples_up_to_date::Atomic{Bool}      # True when all samples are current

    function AsyncReanalyzeWorker(config::ReanalyzeConfig)
        new(Ref(false), nothing, ReanalyzeStats(), config,
            Atomic{Int}(0), Atomic{Int}(0), Atomic{Bool}(false))
    end
end

"""
Notify reanalyze worker that model has been updated.
"""
function notify_model_update!(worker::AsyncReanalyzeWorker, iteration::Int)
    worker.current_model_iteration[] = iteration
    worker.samples_up_to_date[] = false  # Need to reanalyze with new model
end

"""
Start the async reanalyze worker.

Smart reanalysis strategy:
- Uses model iteration tracking to prioritize stale samples
- Stops when all samples are up-to-date with current model
- Resumes automatically when model is updated via notify_model_update!
"""
function start_reanalyze_worker!(
    worker::AsyncReanalyzeWorker,
    buffer::Vector{ClusterSample},
    buffer_lock::ReentrantLock,
    get_network::Function,  # Function that returns current network (thread-safe)
    gspec::AbstractGameSpec,
    use_gpu::Bool
)
    worker.running[] = true

    worker.task = @spawn begin
        @info "Reanalyze worker started (smart mode)"
        last_log_time = time()

        while worker.running[]
            current_model_iter = worker.current_model_iteration[]

            # Check if buffer has enough samples
            buffer_size = lock(buffer_lock) do
                length(buffer)
            end

            if buffer_size < worker.config.batch_size
                sleep(0.1)  # Wait for more samples
                continue
            end

            # Check how many samples need reanalysis
            stale_count = lock(buffer_lock) do
                count_stale_samples(buffer, current_model_iter)
            end

            # If all samples are up-to-date, mark as done and wait
            if stale_count == 0
                if !worker.samples_up_to_date[]
                    @info "Reanalyze: buffer up-to-date with model iter $current_model_iter"
                    worker.samples_up_to_date[] = true
                end
                sleep(0.5)  # Wait for model update
                continue
            end

            # Log progress periodically
            if time() - last_log_time > 10.0
                @info "Reanalyze: $stale_count samples stale (model iter $current_model_iter)"
                last_log_time = time()
            end

            try
                # Get current network (coordinator provides thread-safe copy)
                network = get_network()
                current_step = worker.stats.total_steps

                # Sample indices using smart selection (prioritize oldest model iteration)
                indices = lock(buffer_lock) do
                    sample_for_smart_reanalysis(
                        buffer,
                        worker.config.batch_size,
                        current_model_iter
                    )
                end

                if isempty(indices)
                    sleep(0.1)
                    continue
                end

                # Get samples (with lock)
                samples_to_reanalyze = lock(buffer_lock) do
                    [buffer[i] for i in indices if i <= length(buffer)]
                end

                if isempty(samples_to_reanalyze)
                    continue
                end

                # Prepare batch for network evaluation
                state_shape = GI.state_dim(gspec)
                states = [reshape(s.state, state_shape) for s in samples_to_reanalyze]
                X = Flux.batch(states)

                if use_gpu
                    X = Network.convert_input(network, X)
                end

                # Get network predictions (batched)
                P, V = Network.forward(network, X)

                # Move to CPU
                cpu_fn = use_gpu ? Flux.cpu : identity
                new_values = cpu_fn(V)[1, :]

                # Update samples in buffer (with lock)
                lock(buffer_lock) do
                    for (j, idx) in enumerate(indices)
                        if idx > length(buffer)
                            continue
                        end

                        sample = buffer[idx]
                        old_value = sample.value
                        new_value = new_values[j]

                        # Compute TD-error
                        td_error = abs(new_value - old_value)

                        # Blend old and new values
                        blended_value = (1 - worker.config.reanalyze_alpha) * old_value +
                                        worker.config.reanalyze_alpha * new_value

                        # Update sample with new model iteration marker
                        buffer[idx] = ClusterSample(
                            sample.state, sample.policy, Float32(blended_value),
                            sample.turn, sample.is_chance,
                            sample.equity_p_win, sample.equity_p_gw, sample.equity_p_bgw,
                            sample.equity_p_gl, sample.equity_p_bgl, sample.has_equity,
                            Float32(td_error),       # Updated priority
                            sample.added_step,
                            current_step,            # Updated last_reanalyze_step
                            sample.reanalyze_count + 1,
                            current_model_iter       # Mark as reanalyzed with this model
                        )
                    end
                end

                # Update stats
                worker.samples_reanalyzed[] += length(indices)
                worker.stats.total_reanalyzed += length(indices)
                worker.stats.total_steps += 1

            catch e
                if !(e isa InterruptException)
                    @warn "Reanalyze worker error" exception=(e, catch_backtrace())
                end
                sleep(0.5)
            end

            # Small sleep to prevent busy-waiting
            sleep(0.01)
        end

        @info "Reanalyze worker stopped" total_reanalyzed=worker.stats.total_reanalyzed
    end

    return worker
end

function stop_reanalyze_worker!(worker::AsyncReanalyzeWorker)
    worker.running[] = false
    if !isnothing(worker.task)
        try
            wait(worker.task)
        catch end
    end
end

#####
##### Async Eval Worker
#####

"""
    AsyncEvalResult

Result from an async evaluation run.
"""
struct AsyncEvalResult
    iteration::Int
    vs_random_white::Float64
    vs_random_black::Float64
    vs_random_combined::Float64
    num_games::Int
    eval_time::Float64
    timestamp::Float64
end

"""
    AsyncEvalWorker

Background worker that runs evaluation without blocking training.
"""
mutable struct AsyncEvalWorker
    running::Ref{Bool}
    task::Union{Nothing, Task}
    results_channel::Channel{AsyncEvalResult}
    pending_eval::Atomic{Bool}
    current_iteration::Atomic{Int}

    function AsyncEvalWorker(; channel_size::Int = 10)
        new(
            Ref(false),
            nothing,
            Channel{AsyncEvalResult}(channel_size),
            Atomic{Bool}(false),
            Atomic{Int}(0)
        )
    end
end

"""
Start the async eval worker.
"""
function start_eval_worker!(
    worker::AsyncEvalWorker,
    gspec::AbstractGameSpec,
    get_network::Function,  # Returns current network copy
    eval_games::Int,
    mcts_params::MctsParams;
    use_gpu::Bool = false  # Eval usually runs on CPU
)
    worker.running[] = true

    worker.task = @spawn begin
        @info "Eval worker started"

        while worker.running[]
            # Check if evaluation is requested
            if !worker.pending_eval[]
                sleep(0.1)
                continue
            end

            iter = worker.current_iteration[]
            worker.pending_eval[] = false

            try
                eval_start = time()

                # Get network copy for evaluation
                network = get_network()
                if !use_gpu
                    network = Network.to_cpu(network)
                end

                # Create MCTS player for evaluation
                player = MctsPlayer(gspec, network, mcts_params)

                # Play games vs random
                white_rewards = Float64[]
                black_rewards = Float64[]

                games_per_side = eval_games ÷ 2

                # As white
                for _ in 1:games_per_side
                    game = GI.init(gspec)
                    reward = play_vs_random(game, player, true)
                    push!(white_rewards, reward)
                end

                # As black
                for _ in 1:games_per_side
                    game = GI.init(gspec)
                    reward = play_vs_random(game, player, false)
                    push!(black_rewards, reward)
                end

                # Compute results
                white_avg = isempty(white_rewards) ? 0.0 : mean(white_rewards)
                black_avg = isempty(black_rewards) ? 0.0 : mean(black_rewards)
                combined = (white_avg + black_avg) / 2

                eval_time = time() - eval_start

                result = AsyncEvalResult(
                    iter, white_avg, black_avg, combined,
                    eval_games, eval_time, time()
                )

                # Send result (non-blocking)
                if isready(worker.results_channel)
                    try take!(worker.results_channel) catch end  # Remove old if full
                end
                put!(worker.results_channel, result)

                @info "Eval completed" iteration=iter combined=round(combined, digits=3) time_s=round(eval_time, digits=1)

            catch e
                if !(e isa InterruptException)
                    @warn "Eval worker error" exception=(e, catch_backtrace())
                end
            end
        end

        @info "Eval worker stopped"
    end

    return worker
end

"""
Request an evaluation at the given iteration.
Non-blocking - returns immediately.
"""
function request_eval!(worker::AsyncEvalWorker, iteration::Int)
    worker.current_iteration[] = iteration
    worker.pending_eval[] = true
end

"""
Get latest eval result if available (non-blocking).
"""
function get_eval_result(worker::AsyncEvalWorker)
    if isready(worker.results_channel)
        return take!(worker.results_channel)
    end
    return nothing
end

function stop_eval_worker!(worker::AsyncEvalWorker)
    worker.running[] = false
    close(worker.results_channel)
    if !isnothing(worker.task)
        try
            wait(worker.task)
        catch end
    end
end

#####
##### Helper: Play vs Random
#####

"""
Play a game against random player.
Returns reward from perspective of the MCTS player.
"""
function play_vs_random(game, player::MctsPlayer, player_is_white::Bool)
    reset_player!(player)

    while !GI.game_terminated(game)
        if GI.white_playing(game) == player_is_white
            # MCTS player's turn
            action = think(player, game)
            GI.play!(game, action)
        else
            # Random player's turn
            actions = GI.available_actions(game)
            action = rand(actions)
            GI.play!(game, action)
        end
    end

    reward = GI.white_reward(game)
    return player_is_white ? reward : -reward
end
