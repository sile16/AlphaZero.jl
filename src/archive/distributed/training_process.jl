#####
##### Training Process
#####

"""
GPU training process for distributed AlphaZero.

The training process:
- Requests training batches from replay buffer
- Runs gradient updates on GPU
- Periodically saves checkpoints
- Broadcasts weight updates to workers
- Logs metrics to WandB
"""

using ZMQ
import Flux

#####
##### Training State
#####

"""
    TrainingProcess

Training process for distributed learning.

# Fields
- `config`: Training configuration
- `network`: Neural network being trained
- `gspec`: Game specification
- `learning_params`: Learning hyperparameters
- `context`: ZMQ context
- `replay_socket`: Socket for requesting batches
- `weight_pub_socket`: Socket for publishing weights
- `running`: Process running flag
- `iteration`: Current training iteration
- `stats`: Training statistics
"""
mutable struct TrainingProcess
    config::TrainingConfig
    network::AbstractNetwork
    gspec::AbstractGameSpec
    learning_params::LearningParams
    context::ZMQ.Context
    replay_socket::ZMQ.Socket
    weight_pub_socket::Union{Nothing, ZMQ.Socket}
    running::Bool
    iteration::Int
    stats::Dict{String, Any}
    checkpoint_path::String

    function TrainingProcess(
        config::TrainingConfig,
        network::AbstractNetwork,
        gspec::AbstractGameSpec,
        learning_params::LearningParams;
        weight_pub_endpoint::Union{Nothing, EndpointConfig}=nothing
    )
        ctx = ZMQ.Context()

        # REQ socket for requesting batches from replay buffer
        replay_socket = ZMQ.Socket(ctx, ZMQ.REQ)
        ZMQ.connect(replay_socket, endpoint_string(config.replay_endpoint, bind=false))

        # Optional PUB socket for weight updates
        weight_pub_socket = if !isnothing(weight_pub_endpoint)
            s = ZMQ.Socket(ctx, ZMQ.PUB)
            ZMQ.bind(s, endpoint_string(weight_pub_endpoint))
            s
        else
            nothing
        end

        # Move network to GPU if configured
        if config.use_gpu
            network = Network.to_gpu(network)
        end

        # Create checkpoint directory
        checkpoint_path = joinpath(config.checkpoint_dir, "training")
        mkpath(checkpoint_path)

        stats = Dict{String, Any}(
            "total_batches" => 0,
            "total_samples" => 0,
            "total_loss" => 0.0,
            "start_time" => time(),
            "losses" => Float64[],
            "policy_losses" => Float64[],
            "value_losses" => Float64[],
        )

        return new(
            config, network, gspec, learning_params, ctx,
            replay_socket, weight_pub_socket, false, 0, stats, checkpoint_path
        )
    end
end

#####
##### Batch fetching
#####

"""
    fetch_training_batch(process::TrainingProcess) -> TrainingSampleBatch

Request and receive a training batch from the replay buffer.
"""
function fetch_training_batch(process::TrainingProcess)
    # Send batch request
    request = TrainingSampleBatch(
        batch_id=UInt64(0),
        samples=SerializedSample[],  # Empty samples indicates request
        total_buffer_size=process.config.batch_size
    )

    request_bytes = serialize_message(request)
    ZMQ.send(process.replay_socket, request_bytes)

    # Receive batch
    response_bytes = ZMQ.recv(process.replay_socket)
    batch = deserialize_message(Vector{UInt8}(response_bytes), TrainingSampleBatch)

    return batch
end

#####
##### Sample conversion
#####

"""
    convert_batch_for_training(process::TrainingProcess, batch::TrainingSampleBatch)

Convert a batch of serialized samples to training-ready tensors.
"""
function convert_batch_for_training(process::TrainingProcess, batch::TrainingSampleBatch)
    samples = batch.samples
    n = length(samples)

    if n == 0
        return nothing
    end

    # Stack all data
    state_dim = length(samples[1].state)
    X = zeros(Float32, state_dim, n)
    V = zeros(Float32, 1, n)
    IsChance = zeros(Float32, 1, n)
    W = ones(Float32, 1, n)  # Uniform weights for now

    # Determine policy dimension from first non-chance sample
    policy_dim = 0
    for s in samples
        if !s.is_chance && length(s.policy) > 0
            policy_dim = length(s.policy)
            break
        end
    end

    if policy_dim == 0
        # Fallback: try to get from game spec
        policy_dim = length(GI.actions(process.gspec))
    end

    P = zeros(Float32, policy_dim, n)
    A = ones(Float32, policy_dim, n)  # All actions valid (masks come from states)

    # Multi-head equity targets
    has_equity = any(s -> !isnothing(s.equity), samples)
    EqWin = zeros(Float32, 1, n)
    EqGW = zeros(Float32, 1, n)
    EqBGW = zeros(Float32, 1, n)
    EqGL = zeros(Float32, 1, n)
    EqBGL = zeros(Float32, 1, n)
    HasEquity = zeros(Float32, 1, n)

    for (i, s) in enumerate(samples)
        X[:, i] = s.state
        V[1, i] = s.value
        IsChance[1, i] = s.is_chance ? 1.0f0 : 0.0f0

        if !s.is_chance && length(s.policy) == policy_dim
            P[:, i] = s.policy
        end

        if !isnothing(s.equity)
            EqWin[1, i] = s.equity.p_win
            EqGW[1, i] = s.equity.p_gammon_win
            EqBGW[1, i] = s.equity.p_bg_win
            EqGL[1, i] = s.equity.p_gammon_loss
            EqBGL[1, i] = s.equity.p_bg_loss
            HasEquity[1, i] = 1.0f0
        end
    end

    # Move to GPU if needed
    data = (; W, X, A, P, V, IsChance, EqWin, EqGW, EqBGW, EqGL, EqBGL, HasEquity)

    if process.config.use_gpu
        data = Network.convert_input_tuple(process.network, data)
    end

    return data
end

#####
##### Training step
#####

"""
    training_step!(process::TrainingProcess, batch_data) -> NamedTuple

Run a single training step on the batch data.
Returns loss metrics.
"""
function training_step!(process::TrainingProcess, batch_data)
    if isnothing(batch_data)
        return (loss=0.0, policy_loss=0.0, value_loss=0.0)
    end

    params = process.learning_params
    Wmean = mean(batch_data.W)
    Hp = 0.0f0  # Entropy placeholder

    # Compute losses using existing loss function
    L, Lp, Lv, Lreg, Linv = losses(
        process.network, params, Wmean, Hp, batch_data
    )

    # Gradient update
    opt = params.optimiser
    opt_state = Flux.setup(Flux.Adam(opt.lr), process.network)

    loss_fn(nn) = losses(nn, params, Wmean, Hp, batch_data)[1]
    l, grads = Flux.withgradient(loss_fn, process.network)
    Flux.update!(opt_state, process.network, grads[1])

    # Convert to CPU values for logging
    loss = Float64(Network.convert_output(process.network, L))
    policy_loss = Float64(Network.convert_output(process.network, Lp))
    value_loss = Float64(Network.convert_output(process.network, Lv))

    return (loss=loss, policy_loss=policy_loss, value_loss=value_loss)
end

#####
##### Weight broadcasting
#####

"""
    broadcast_weights!(process::TrainingProcess)

Broadcast current network weights to all workers.
"""
function broadcast_weights!(process::TrainingProcess)
    if isnothing(process.weight_pub_socket)
        return
    end

    # Get network on CPU for serialization
    cpu_network = Network.to_cpu(process.network)
    weights_data = serialize_network_weights(cpu_network)

    update = WeightUpdate(
        iteration=process.iteration,
        weights_data=weights_data,
        timestamp=time(),
        checksum=compute_checksum(weights_data)
    )

    msg_bytes = serialize_message(update)
    ZMQ.send(process.weight_pub_socket, msg_bytes)

    @debug "Broadcast weights for iteration $(process.iteration)"
end

#####
##### Checkpointing
#####

"""
    save_checkpoint(process::TrainingProcess)

Save a training checkpoint.
"""
function save_checkpoint(process::TrainingProcess)
    checkpoint_file = joinpath(
        process.checkpoint_path,
        "checkpoint_iter$(process.iteration).data"
    )

    # Save network
    cpu_network = Network.to_cpu(process.network)
    Network.save(checkpoint_file, cpu_network)

    # Save metadata
    meta_file = joinpath(process.checkpoint_path, "latest.json")
    metadata = Dict(
        "iteration" => process.iteration,
        "checkpoint_file" => checkpoint_file,
        "timestamp" => time(),
        "stats" => process.stats
    )
    open(meta_file, "w") do io
        JSON3.write(io, metadata)
    end

    @info "Saved checkpoint at iteration $(process.iteration)"
end

"""
    load_checkpoint(process::TrainingProcess, checkpoint_file::String)

Load a training checkpoint.
"""
function load_checkpoint(process::TrainingProcess, checkpoint_file::String)
    process.network = Network.load(checkpoint_file)
    if process.config.use_gpu
        process.network = Network.to_gpu(process.network)
    end
end

#####
##### WandB integration
#####

"""
    init_wandb(process::TrainingProcess)

Initialize WandB logging if configured.
"""
function init_wandb(process::TrainingProcess)
    if isnothing(process.config.wandb_project)
        return nothing
    end

    try
        wandb = pyimport("wandb")
        run_name = something(process.config.wandb_run_name, "distributed_training_$(Dates.now())")
        wandb.init(
            project=process.config.wandb_project,
            name=run_name,
            config=Dict(
                "batch_size" => process.config.batch_size,
                "learning_rate" => process.learning_params.optimiser.lr,
                "l2_regularization" => process.learning_params.l2_regularization,
            )
        )
        return wandb
    catch e
        @warn "Failed to initialize WandB" exception=e
        return nothing
    end
end

"""
    log_wandb(wandb, process::TrainingProcess, metrics)

Log metrics to WandB.
"""
function log_wandb(wandb, process::TrainingProcess, metrics)
    if isnothing(wandb)
        return
    end

    try
        wandb.log(Dict(
            "iteration" => process.iteration,
            "train/loss" => metrics.loss,
            "train/policy_loss" => metrics.policy_loss,
            "train/value_loss" => metrics.value_loss,
            "train/total_samples" => process.stats["total_samples"],
        ))
    catch e
        @debug "WandB log failed" exception=e
    end
end

#####
##### Main training loop
#####

"""
    run_training(process::TrainingProcess; max_iterations::Int=1000)

Run the training process main loop.
"""
function run_training(process::TrainingProcess; max_iterations::Int=1000)
    process.running = true
    @info "Training process starting"
    @info "  Replay buffer: $(endpoint_string(process.config.replay_endpoint, bind=false))"
    @info "  Batch size: $(process.config.batch_size)"

    wandb = init_wandb(process)

    while process.running && process.iteration < max_iterations
        try
            # Fetch training batch
            batch = fetch_training_batch(process)

            if isempty(batch.samples)
                @debug "Empty batch received, waiting..."
                sleep(1.0)
                continue
            end

            # Convert batch to training format
            batch_data = convert_batch_for_training(process, batch)

            # Training step
            metrics = training_step!(process, batch_data)

            # Update stats
            process.iteration += 1
            process.stats["total_batches"] += 1
            process.stats["total_samples"] += length(batch.samples)
            process.stats["total_loss"] += metrics.loss
            push!(process.stats["losses"], metrics.loss)
            push!(process.stats["policy_losses"], metrics.policy_loss)
            push!(process.stats["value_losses"], metrics.value_loss)

            # Log to WandB
            log_wandb(wandb, process, metrics)

            # Periodic checkpoint and weight broadcast
            if process.iteration % process.config.batches_per_checkpoint == 0
                save_checkpoint(process)
                broadcast_weights!(process)
            end

            # Log progress
            if process.iteration % 10 == 0
                avg_loss = process.stats["total_loss"] / process.stats["total_batches"]
                @info "Iteration $(process.iteration): loss=$(round(metrics.loss, digits=4)), avg_loss=$(round(avg_loss, digits=4))"
            end

        catch e
            if e isa InterruptException
                @info "Training interrupted"
                break
            else
                @error "Training error" exception=(e, catch_backtrace())
                sleep(1.0)
            end
        end
    end

    # Final checkpoint
    save_checkpoint(process)

    # Cleanup
    shutdown_training(process)

    if !isnothing(wandb)
        try
            wandb.finish()
        catch end
    end

    @info "Training process stopped at iteration $(process.iteration)"

    return process.stats
end

"""
    run_training_async(process::TrainingProcess; kwargs...) -> Task

Run training in a background task.
"""
function run_training_async(process::TrainingProcess; kwargs...)
    return @async run_training(process; kwargs...)
end

#####
##### Lifecycle management
#####

"""
    shutdown_training(process::TrainingProcess)

Shutdown the training process.
"""
function shutdown_training(process::TrainingProcess)
    process.running = false

    try
        ZMQ.close(process.replay_socket)
        if !isnothing(process.weight_pub_socket)
            ZMQ.close(process.weight_pub_socket)
        end
        ZMQ.close(process.context)
    catch e
        @warn "Error during training shutdown" exception=e
    end
end

#####
##### Statistics
#####

"""
    get_training_stats(process::TrainingProcess) -> Dict

Get current training statistics.
"""
function get_training_stats(process::TrainingProcess)
    stats = copy(process.stats)
    uptime = time() - stats["start_time"]
    stats["uptime_seconds"] = uptime
    stats["current_iteration"] = process.iteration

    if stats["total_batches"] > 0
        stats["avg_loss"] = stats["total_loss"] / stats["total_batches"]
        stats["batches_per_second"] = stats["total_batches"] / uptime
        stats["samples_per_second"] = stats["total_samples"] / uptime
    end

    # Recent metrics (last 100)
    n = min(100, length(stats["losses"]))
    if n > 0
        stats["recent_avg_loss"] = mean(stats["losses"][end-n+1:end])
        stats["recent_avg_policy_loss"] = mean(stats["policy_losses"][end-n+1:end])
        stats["recent_avg_value_loss"] = mean(stats["value_losses"][end-n+1:end])
    end

    return stats
end

#####
##### Convenience constructor
#####

"""
    create_training_process(
        network::AbstractNetwork,
        gspec::AbstractGameSpec,
        learning_params::LearningParams;
        replay_host::String="localhost",
        replay_port::Int=5556,
        checkpoint_dir::String="checkpoints",
        use_gpu::Bool=true,
        kwargs...
    ) -> TrainingProcess

Create a training process with simple configuration.
"""
function create_training_process(
    network::AbstractNetwork,
    gspec::AbstractGameSpec,
    learning_params::LearningParams;
    replay_host::String="localhost",
    replay_port::Int=5556,
    checkpoint_dir::String="checkpoints",
    use_gpu::Bool=true,
    wandb_project::Union{Nothing, String}=nothing,
    kwargs...
)
    config = TrainingConfig(
        replay_endpoint=EndpointConfig(host=replay_host, port=replay_port),
        coordinator_endpoint=EndpointConfig(host=replay_host, port=5557),
        learning_params=learning_params,
        checkpoint_dir=checkpoint_dir,
        use_gpu=use_gpu,
        wandb_project=wandb_project;
        kwargs...
    )

    return TrainingProcess(config, network, gspec, learning_params)
end
