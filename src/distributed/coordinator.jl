#####
##### Distributed Training Coordinator
#####

"""
Main coordinator for distributed AlphaZero training.

The coordinator:
- Manages all distributed components (inference server, replay buffer, training, workers)
- Handles weight distribution to workers
- Monitors system health via heartbeats
- Logs overall progress to WandB
"""

using ZMQ
using Dates

#####
##### Coordinator State
#####

"""
    DistributedCoordinator

Main coordinator for distributed training.

# Fields
- `config`: Coordinator configuration
- `gspec`: Game specification
- `network`: Current best network
- `inference_server`: GPU inference server
- `replay_manager`: Replay buffer manager
- `training_process`: GPU training process
- `evaluation_process`: Periodic evaluation
- `workers`: Managed worker processes
- `context`: ZMQ context
- `command_socket`: PUB socket for broadcasting commands
- `running`: Coordinator running flag
- `iteration`: Current training iteration
- `stats`: Coordinator statistics
"""
mutable struct DistributedCoordinator
    config::CoordinatorConfig
    gspec::AbstractGameSpec
    network::AbstractNetwork
    inference_server::Union{Nothing, ZMQInferenceServer}
    replay_manager::Union{Nothing, ReplayBufferManager}
    training_task::Union{Nothing, Task}
    evaluation_task::Union{Nothing, Task}
    worker_tasks::Vector{Task}
    context::ZMQ.Context
    command_socket::ZMQ.Socket
    running::Bool
    iteration::Int
    stats::Dict{String, Any}
    session_dir::String

    function DistributedCoordinator(
        config::CoordinatorConfig,
        gspec::AbstractGameSpec,
        network::AbstractNetwork
    )
        # Validate configuration
        validate_config(config)

        ctx = ZMQ.Context()

        # PUB socket for broadcasting commands/weights to workers
        command_socket = ZMQ.Socket(ctx, ZMQ.PUB)
        ZMQ.bind(command_socket, endpoint_string(config.command_endpoint))

        # Create session directory
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        session_dir = joinpath(config.session_dir, "distributed_$(timestamp)")
        mkpath(session_dir)

        stats = Dict{String, Any}(
            "start_time" => time(),
            "total_games" => 0,
            "total_samples" => 0,
            "worker_heartbeats" => Dict{String, WorkerHeartbeat}(),
        )

        return new(
            config, gspec, network,
            nothing, nothing, nothing, nothing, Task[],
            ctx, command_socket, false, 0, stats, session_dir
        )
    end
end

#####
##### Component initialization
#####

"""
    start_inference_server!(coordinator::DistributedCoordinator)

Start the inference server component.
"""
function start_inference_server!(coordinator::DistributedCoordinator)
    @info "Starting inference server..."

    coordinator.inference_server = ZMQInferenceServer(
        coordinator.network,
        coordinator.gspec,
        coordinator.config.inference_config
    )

    # Run in background
    @async begin
        try
            run_inference_server(coordinator.inference_server)
        catch e
            @error "Inference server error" exception=(e, catch_backtrace())
        end
    end

    @info "Inference server started on port $(coordinator.config.inference_config.endpoint.port)"
end

"""
    start_replay_manager!(coordinator::DistributedCoordinator)

Start the replay buffer manager component.
"""
function start_replay_manager!(coordinator::DistributedCoordinator)
    @info "Starting replay buffer manager..."

    coordinator.replay_manager = ReplayBufferManager(coordinator.config.replay_config)

    # Run in background
    @async begin
        try
            run_replay_manager(coordinator.replay_manager)
        catch e
            @error "Replay manager error" exception=(e, catch_backtrace())
        end
    end

    @info "Replay manager started on port $(coordinator.config.replay_config.endpoint.port)"
end

"""
    start_training!(coordinator::DistributedCoordinator)

Start the training process component.
"""
function start_training!(coordinator::DistributedCoordinator)
    @info "Starting training process..."

    # Get learning params from config or use defaults
    learning_params = if !isnothing(coordinator.config.training_config.learning_params)
        coordinator.config.training_config.learning_params
    else
        LearningParams()  # Default params
    end

    training = TrainingProcess(
        coordinator.config.training_config,
        Network.copy(coordinator.network),
        coordinator.gspec,
        learning_params
    )

    coordinator.training_task = @async begin
        try
            run_training(training; max_iterations=coordinator.config.total_iterations)
        catch e
            @error "Training error" exception=(e, catch_backtrace())
        end
    end

    @info "Training process started"
end

"""
    start_evaluation!(coordinator::DistributedCoordinator)

Start the evaluation process component.
"""
function start_evaluation!(coordinator::DistributedCoordinator)
    if isnothing(coordinator.config.evaluation_config)
        @info "Evaluation disabled"
        return
    end

    @info "Starting evaluation process..."

    mcts_params = MctsParams()  # Use default or from config
    eval_process = EvaluationProcess(
        coordinator.config.evaluation_config,
        coordinator.gspec,
        mcts_params
    )

    # Initialize with current network
    initialize_network!(eval_process, coordinator.network)

    checkpoint_dir = joinpath(coordinator.session_dir, "checkpoints")
    coordinator.evaluation_task = @async begin
        try
            watch_checkpoints(eval_process, checkpoint_dir)
        catch e
            @error "Evaluation error" exception=(e, catch_backtrace())
        end
    end

    @info "Evaluation process started"
end

"""
    start_local_workers!(coordinator::DistributedCoordinator, mcts_params::MctsParams)

Start local self-play workers.
"""
function start_local_workers!(coordinator::DistributedCoordinator, mcts_params::MctsParams)
    num_workers = coordinator.config.num_local_workers
    if num_workers <= 0
        @info "No local workers configured"
        return
    end

    @info "Starting $num_workers local workers..."

    for i in 1:num_workers
        worker_id = "local_worker_$i"

        config = WorkerConfig(
            worker_id=worker_id,
            inference_endpoint=EndpointConfig(
                host="localhost",
                port=coordinator.config.inference_config.endpoint.port
            ),
            replay_endpoint=EndpointConfig(
                host="localhost",
                port=coordinator.config.replay_config.endpoint.port
            ),
            coordinator_endpoint=EndpointConfig(
                host="localhost",
                port=coordinator.config.command_endpoint.port
            ),
            mcts_params=mcts_params,
        )

        worker = SelfPlayWorker(config, coordinator.gspec)

        task = @async begin
            try
                run_worker(worker)
            catch e
                @error "Worker $worker_id error" exception=(e, catch_backtrace())
            end
        end

        push!(coordinator.worker_tasks, task)
    end

    @info "Started $num_workers local workers"
end

#####
##### Weight distribution
#####

"""
    broadcast_weights!(coordinator::DistributedCoordinator)

Broadcast current network weights to all workers.
"""
function broadcast_weights!(coordinator::DistributedCoordinator)
    # Serialize network weights
    weights_data = serialize_network_weights(coordinator.network)

    update = WeightUpdate(
        iteration=coordinator.iteration,
        weights_data=weights_data,
        timestamp=time(),
        checksum=compute_checksum(weights_data)
    )

    # Wrap in command
    command = WorkerCommand(
        command=:update_weights,
        payload=update
    )

    msg_bytes = serialize_message(command)
    ZMQ.send(coordinator.command_socket, msg_bytes)

    @debug "Broadcast weights for iteration $(coordinator.iteration)"
end

"""
    broadcast_command!(coordinator::DistributedCoordinator, command::Symbol)

Broadcast a command to all workers.
"""
function broadcast_command!(coordinator::DistributedCoordinator, command::Symbol)
    cmd = WorkerCommand(command=command)
    msg_bytes = serialize_message(cmd)
    ZMQ.send(coordinator.command_socket, msg_bytes)
end

#####
##### Main coordination loop
#####

"""
    run_coordinator(coordinator::DistributedCoordinator, mcts_params::MctsParams)

Run the main coordinator loop.
"""
function run_coordinator(coordinator::DistributedCoordinator, mcts_params::MctsParams)
    coordinator.running = true
    @info "Starting distributed training coordinator"
    @info "Session directory: $(coordinator.session_dir)"

    # Start all components
    start_inference_server!(coordinator)
    sleep(0.5)  # Allow server to bind

    start_replay_manager!(coordinator)
    sleep(0.5)

    start_local_workers!(coordinator, mcts_params)
    sleep(0.5)

    # Wait for replay buffer to fill before starting training
    @info "Waiting for replay buffer to fill..."
    while coordinator.running && !isnothing(coordinator.replay_manager)
        if is_ready_for_training(coordinator.replay_manager)
            @info "Replay buffer ready with $(length(coordinator.replay_manager.buffer)) samples"
            break
        end
        sleep(1.0)
    end

    start_training!(coordinator)
    start_evaluation!(coordinator)

    # Save initial configuration
    save_config(joinpath(coordinator.session_dir, "config.json"), coordinator.config)

    # Main monitoring loop
    last_log_time = time()
    log_interval = 30.0  # Log every 30 seconds

    while coordinator.running
        try
            # Check component health
            check_component_health!(coordinator)

            # Periodic logging
            if time() - last_log_time > log_interval
                log_progress(coordinator)
                last_log_time = time()
            end

            # Check if training is complete
            if !isnothing(coordinator.training_task) && istaskdone(coordinator.training_task)
                @info "Training completed"
                break
            end

            sleep(1.0)

        catch e
            if e isa InterruptException
                @info "Coordinator interrupted"
                break
            else
                @error "Coordinator error" exception=(e, catch_backtrace())
            end
        end
    end

    # Shutdown
    shutdown_coordinator(coordinator)

    return coordinator.stats
end

"""
    run_coordinator_async(coordinator::DistributedCoordinator, mcts_params::MctsParams) -> Task

Run coordinator in a background task.
"""
function run_coordinator_async(coordinator::DistributedCoordinator, mcts_params::MctsParams)
    return @async run_coordinator(coordinator, mcts_params)
end

#####
##### Health monitoring
#####

"""
    check_component_health!(coordinator::DistributedCoordinator)

Check health of all components and restart if needed.
"""
function check_component_health!(coordinator::DistributedCoordinator)
    # Check inference server
    if !isnothing(coordinator.inference_server) && !coordinator.inference_server.running
        @warn "Inference server stopped, restarting..."
        start_inference_server!(coordinator)
    end

    # Check replay manager
    if !isnothing(coordinator.replay_manager) && !coordinator.replay_manager.running
        @warn "Replay manager stopped, restarting..."
        start_replay_manager!(coordinator)
    end

    # Check workers
    active_workers = count(t -> !istaskdone(t), coordinator.worker_tasks)
    if active_workers < coordinator.config.num_local_workers && coordinator.running
        @debug "Some workers stopped: $active_workers / $(coordinator.config.num_local_workers) active"
    end
end

"""
    log_progress(coordinator::DistributedCoordinator)

Log current training progress.
"""
function log_progress(coordinator::DistributedCoordinator)
    uptime = time() - coordinator.stats["start_time"]

    # Get buffer stats
    buffer_stats = if !isnothing(coordinator.replay_manager)
        get_buffer_stats(coordinator.replay_manager)
    else
        Dict()
    end

    # Get inference stats
    inference_stats = if !isnothing(coordinator.inference_server)
        get_server_stats(coordinator.inference_server)
    else
        Dict()
    end

    @info "Training Progress" uptime_hours=round(uptime/3600, digits=2) buffer_size=get(buffer_stats, "current_buffer_size", 0) games_per_minute=round(get(buffer_stats, "games_per_minute", 0.0), digits=2) states_per_second=round(get(inference_stats, "states_per_second", 0.0), digits=1)
end

#####
##### Lifecycle management
#####

"""
    shutdown_coordinator(coordinator::DistributedCoordinator)

Shutdown all components gracefully.
"""
function shutdown_coordinator(coordinator::DistributedCoordinator)
    coordinator.running = false
    @info "Shutting down coordinator..."

    # Signal workers to stop
    broadcast_command!(coordinator, :shutdown)
    sleep(0.5)

    # Stop inference server
    if !isnothing(coordinator.inference_server)
        shutdown_inference_server(coordinator.inference_server)
    end

    # Stop replay manager
    if !isnothing(coordinator.replay_manager)
        shutdown_replay_manager(coordinator.replay_manager)
    end

    # Wait for workers to finish
    for task in coordinator.worker_tasks
        try
            wait(task)
        catch end
    end

    # Close ZMQ
    try
        ZMQ.close(coordinator.command_socket)
        ZMQ.close(coordinator.context)
    catch e
        @warn "Error closing coordinator sockets" exception=e
    end

    @info "Coordinator shutdown complete"
end

#####
##### WandB integration
#####

"""
    init_coordinator_wandb(coordinator::DistributedCoordinator)

Initialize WandB logging for the coordinator.
"""
function init_coordinator_wandb(coordinator::DistributedCoordinator)
    if isnothing(coordinator.config.wandb_project)
        return nothing
    end

    try
        wandb = pyimport("wandb")
        run_name = "distributed_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
        wandb.init(
            project=coordinator.config.wandb_project,
            name=run_name,
            config=Dict(
                "num_workers" => coordinator.config.num_local_workers,
                "buffer_capacity" => coordinator.config.replay_config.capacity,
                "inference_batch_size" => coordinator.config.inference_config.batch_size,
                "total_iterations" => coordinator.config.total_iterations,
            )
        )
        return wandb
    catch e
        @warn "Failed to initialize WandB" exception=e
        return nothing
    end
end

#####
##### Convenience functions
#####

"""
    create_coordinator(
        gspec::AbstractGameSpec,
        network::AbstractNetwork;
        num_workers::Int=4,
        inference_port::Int=5555,
        replay_port::Int=5556,
        command_port::Int=5557,
        session_dir::String="sessions",
        total_iterations::Int=100,
        wandb_project::Union{Nothing, String}=nothing
    ) -> DistributedCoordinator

Create a coordinator with simple configuration.
"""
function create_coordinator(
    gspec::AbstractGameSpec,
    network::AbstractNetwork;
    num_workers::Int=4,
    inference_port::Int=5555,
    replay_port::Int=5556,
    command_port::Int=5557,
    session_dir::String="sessions",
    total_iterations::Int=100,
    wandb_project::Union{Nothing, String}=nothing
)
    config = default_local_config(
        inference_port=inference_port,
        replay_port=replay_port,
        command_port=command_port,
        num_workers=num_workers,
        session_dir=session_dir,
        total_iterations=total_iterations,
        wandb_project=wandb_project
    )

    return DistributedCoordinator(config, gspec, network)
end

"""
    train_distributed!(
        gspec::AbstractGameSpec,
        network::AbstractNetwork,
        mcts_params::MctsParams;
        kwargs...
    )

High-level function to run distributed training.
"""
function train_distributed!(
    gspec::AbstractGameSpec,
    network::AbstractNetwork,
    mcts_params::MctsParams;
    kwargs...
)
    coordinator = create_coordinator(gspec, network; kwargs...)
    return run_coordinator(coordinator, mcts_params)
end
