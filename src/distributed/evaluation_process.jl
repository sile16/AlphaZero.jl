#####
##### Evaluation Process
#####

"""
Periodic evaluation of training progress.

The evaluation process:
- Loads checkpoints periodically
- Runs games against baseline opponents
- Reports metrics to WandB
- Supports reward histogram tracking
"""

using ZMQ
using Statistics

#####
##### Evaluation State
#####

"""
    EvaluationProcess

Process for evaluating trained networks.

# Fields
- `config`: Evaluation configuration
- `gspec`: Game specification
- `current_network`: Network being evaluated
- `previous_network`: Previous best network for comparison
- `mcts_params`: MCTS parameters for evaluation games
- `context`: ZMQ context
- `running`: Process running flag
- `stats`: Evaluation statistics
"""
mutable struct EvaluationProcess
    config::EvaluationConfig
    gspec::AbstractGameSpec
    current_network::Union{Nothing, AbstractNetwork}
    previous_network::Union{Nothing, AbstractNetwork}
    mcts_params::MctsParams
    context::ZMQ.Context
    running::Bool
    stats::Dict{String, Any}
    last_evaluated_iteration::Int

    function EvaluationProcess(
        config::EvaluationConfig,
        gspec::AbstractGameSpec,
        mcts_params::MctsParams
    )
        ctx = ZMQ.Context()

        stats = Dict{String, Any}(
            "evaluations_completed" => 0,
            "start_time" => time(),
            "results" => EvaluationResult[],
        )

        return new(
            config, gspec, nothing, nothing, mcts_params,
            ctx, false, stats, -1
        )
    end
end

#####
##### Baseline players
#####

"""
    create_baseline_player(process::EvaluationProcess, baseline::Symbol) -> AbstractPlayer

Create a baseline player for evaluation.
"""
function create_baseline_player(process::EvaluationProcess, baseline::Symbol)
    if baseline == :random
        return RandomPlayer()
    elseif baseline == :previous
        if isnothing(process.previous_network)
            @warn "No previous network available, using random baseline"
            return RandomPlayer()
        end
        nn = if process.config.use_gpu
            Network.to_gpu(process.previous_network)
        else
            process.previous_network
        end
        Network.set_test_mode!(nn, true)
        return MctsPlayer(process.gspec, nn, process.mcts_params)
    else
        # Check for iteration-specific baseline (e.g., :iteration_10)
        baseline_str = String(baseline)
        if startswith(baseline_str, "iteration_")
            iter_num = parse(Int, baseline_str[11:end])
            # Would need to load from checkpoint
            @warn "Loading specific iteration checkpoint not yet implemented"
            return RandomPlayer()
        end
        @warn "Unknown baseline: $baseline, using random"
        return RandomPlayer()
    end
end

"""
    create_evaluation_player(process::EvaluationProcess) -> AbstractPlayer

Create a player using the current network for evaluation.
"""
function create_evaluation_player(process::EvaluationProcess)
    if isnothing(process.current_network)
        error("No network loaded for evaluation")
    end

    nn = if process.config.use_gpu
        Network.to_gpu(process.current_network)
    else
        process.current_network
    end
    Network.set_test_mode!(nn, true)

    return MctsPlayer(process.gspec, nn, process.mcts_params)
end

#####
##### Evaluation games
#####

"""
    run_evaluation_games(
        process::EvaluationProcess,
        baseline::Symbol,
        num_games::Int
    ) -> EvaluationResult

Run evaluation games against a baseline and compute statistics.
"""
function run_evaluation_games(
    process::EvaluationProcess,
    baseline::Symbol,
    num_games::Int
)
    eval_player = create_evaluation_player(process)
    baseline_player = create_baseline_player(process, baseline)

    rewards = Float64[]
    sizehint!(rewards, num_games)

    wins = 0
    losses = 0
    draws = 0

    for i in 1:num_games
        # Alternate colors
        if i % 2 == 0
            player = TwoPlayers(eval_player, baseline_player)
            colors_flipped = false
        else
            player = TwoPlayers(baseline_player, eval_player)
            colors_flipped = true
        end

        trace = play_game(process.gspec, player)
        reward = total_reward(trace)

        # Adjust for color flip
        if colors_flipped
            reward = -reward
        end

        push!(rewards, reward)

        if reward > 0
            wins += 1
        elseif reward < 0
            losses += 1
        else
            draws += 1
        end

        # Reset players periodically
        if i % 10 == 0
            reset_player!(eval_player)
            reset_player!(baseline_player)
        end
    end

    # Compute statistics
    win_rate = wins / num_games
    avg_reward = mean(rewards)

    # Build reward histogram
    reward_histogram = Dict{Float64, Int}()
    for r in rewards
        # Round to one decimal
        rounded = round(r, digits=1)
        reward_histogram[rounded] = get(reward_histogram, rounded, 0) + 1
    end

    return EvaluationResult(
        iteration=process.last_evaluated_iteration,
        baseline=baseline,
        win_rate=win_rate,
        avg_reward=avg_reward,
        reward_histogram=reward_histogram,
        num_games=num_games
    )
end

"""
    run_full_evaluation(process::EvaluationProcess) -> Vector{EvaluationResult}

Run evaluation against all configured baselines.
"""
function run_full_evaluation(process::EvaluationProcess)
    results = EvaluationResult[]

    for baseline in process.config.baselines
        @info "Evaluating against $baseline..."
        result = run_evaluation_games(process, baseline, process.config.num_games)
        push!(results, result)

        @info "  Win rate: $(round(result.win_rate * 100, digits=1))%"
        @info "  Avg reward: $(round(result.avg_reward, digits=3))"
    end

    return results
end

#####
##### Network loading
#####

"""
    load_network_for_evaluation!(process::EvaluationProcess, weights_data::Vector{UInt8})

Load network weights for evaluation.
"""
function load_network_for_evaluation!(
    process::EvaluationProcess,
    weights_data::Vector{UInt8},
    iteration::Int
)
    # Move current to previous
    if !isnothing(process.current_network)
        process.previous_network = Network.copy(process.current_network)
    end

    # Load new weights
    weight_arrays = deserialize_network_weights(weights_data)
    if isnothing(process.current_network)
        error("Network must be initialized before loading weights")
    end
    load_weights_into_network!(process.current_network, weight_arrays)

    process.last_evaluated_iteration = iteration
end

"""
    initialize_network!(process::EvaluationProcess, network::AbstractNetwork)

Initialize the evaluation process with a network architecture.
"""
function initialize_network!(process::EvaluationProcess, network::AbstractNetwork)
    process.current_network = Network.copy(network)
    Network.set_test_mode!(process.current_network, true)
end

#####
##### Checkpoint watching
#####

"""
    watch_checkpoints(
        process::EvaluationProcess,
        checkpoint_dir::String;
        check_interval_s::Float64=30.0
    )

Watch a checkpoint directory and evaluate new checkpoints as they appear.
"""
function watch_checkpoints(
    process::EvaluationProcess,
    checkpoint_dir::String;
    check_interval_s::Float64=30.0
)
    process.running = true
    @info "Evaluation process watching: $checkpoint_dir"

    last_checkpoint_time = 0.0

    while process.running
        try
            # Check for new checkpoint
            meta_file = joinpath(checkpoint_dir, "latest.json")

            if isfile(meta_file)
                meta_time = mtime(meta_file)

                if meta_time > last_checkpoint_time
                    # New checkpoint available
                    metadata = JSON3.read(read(meta_file, String))
                    iteration = metadata["iteration"]
                    checkpoint_file = metadata["checkpoint_file"]

                    if iteration > process.last_evaluated_iteration
                        @info "Found new checkpoint: iteration $iteration"

                        # Load network
                        if isnothing(process.current_network)
                            process.current_network = Network.load(checkpoint_file)
                        else
                            # Move current to previous
                            process.previous_network = Network.copy(process.current_network)
                            # Load new checkpoint
                            new_network = Network.load(checkpoint_file)
                            # Copy weights
                            for (p_old, p_new) in zip(
                                Network.params(process.current_network),
                                Network.params(new_network)
                            )
                                copyto!(p_old, p_new)
                            end
                        end

                        process.last_evaluated_iteration = iteration

                        # Run evaluation
                        results = run_full_evaluation(process)

                        # Store results
                        append!(process.stats["results"], results)
                        process.stats["evaluations_completed"] += 1

                        last_checkpoint_time = meta_time
                    end
                end
            end

            sleep(check_interval_s)

        catch e
            if e isa InterruptException
                @info "Evaluation interrupted"
                break
            else
                @error "Evaluation error" exception=(e, catch_backtrace())
                sleep(check_interval_s)
            end
        end
    end

    @info "Evaluation process stopped"
    return process.stats
end

"""
    watch_checkpoints_async(process::EvaluationProcess, checkpoint_dir::String; kwargs...) -> Task

Watch checkpoints in a background task.
"""
function watch_checkpoints_async(process::EvaluationProcess, checkpoint_dir::String; kwargs...)
    return @async watch_checkpoints(process, checkpoint_dir; kwargs...)
end

#####
##### WandB integration
#####

"""
    log_evaluation_wandb(wandb, results::Vector{EvaluationResult})

Log evaluation results to WandB.
"""
function log_evaluation_wandb(wandb, results::Vector{EvaluationResult})
    if isnothing(wandb)
        return
    end

    try
        for result in results
            prefix = "eval/$(result.baseline)"
            wandb.log(Dict(
                "iteration" => result.iteration,
                "$(prefix)/win_rate" => result.win_rate,
                "$(prefix)/avg_reward" => result.avg_reward,
                "$(prefix)/num_games" => result.num_games,
            ))
        end
    catch e
        @debug "WandB log failed" exception=e
    end
end

#####
##### Lifecycle
#####

"""
    shutdown_evaluation(process::EvaluationProcess)

Shutdown the evaluation process.
"""
function shutdown_evaluation(process::EvaluationProcess)
    process.running = false
    try
        ZMQ.close(process.context)
    catch e
        @warn "Error during evaluation shutdown" exception=e
    end
end

#####
##### Statistics
#####

"""
    get_evaluation_stats(process::EvaluationProcess) -> Dict

Get evaluation statistics.
"""
function get_evaluation_stats(process::EvaluationProcess)
    stats = copy(process.stats)
    stats["last_evaluated_iteration"] = process.last_evaluated_iteration
    stats["uptime_seconds"] = time() - stats["start_time"]
    return stats
end

"""
    get_latest_results(process::EvaluationProcess) -> Dict{Symbol, EvaluationResult}

Get the most recent evaluation results for each baseline.
"""
function get_latest_results(process::EvaluationProcess)
    results = Dict{Symbol, EvaluationResult}()
    for r in reverse(process.stats["results"])
        if !haskey(results, r.baseline)
            results[r.baseline] = r
        end
    end
    return results
end

#####
##### Convenience constructor
#####

"""
    create_evaluation_process(
        gspec::AbstractGameSpec,
        mcts_params::MctsParams;
        num_games::Int=100,
        baselines::Vector{Symbol}=[:random, :previous],
        use_gpu::Bool=true
    ) -> EvaluationProcess

Create an evaluation process with simple configuration.
"""
function create_evaluation_process(
    gspec::AbstractGameSpec,
    mcts_params::MctsParams;
    num_games::Int=100,
    baselines::Vector{Symbol}=[:random, :previous],
    use_gpu::Bool=true
)
    config = EvaluationConfig(
        coordinator_endpoint=EndpointConfig(port=5557),
        num_games=num_games,
        baselines=baselines,
        use_gpu=use_gpu
    )

    return EvaluationProcess(config, gspec, mcts_params)
end
