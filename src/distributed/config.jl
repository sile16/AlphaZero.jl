#####
##### Configuration types for distributed training
#####

"""
Configuration structs for distributed AlphaZero training components.
"""

using Base: @kwdef

#####
##### Network endpoints
#####

"""
    EndpointConfig

Configuration for a ZMQ endpoint.

# Fields
- `host`: Hostname or IP address
- `port`: Port number
- `protocol`: Transport protocol (tcp, ipc, inproc)
"""
@kwdef struct EndpointConfig
    host::String = "*"
    port::Int
    protocol::String = "tcp"
end

"""
    endpoint_string(config::EndpointConfig) -> String

Convert endpoint config to ZMQ connection string.
"""
function endpoint_string(config::EndpointConfig; bind::Bool=true)
    host = bind ? config.host : (config.host == "*" ? "localhost" : config.host)
    return "$(config.protocol)://$(host):$(config.port)"
end

#####
##### Inference server configuration
#####

"""
    InferenceServerConfig

Configuration for the GPU inference server.

# Fields
- `endpoint`: ZMQ endpoint configuration
- `batch_size`: Target batch size for GPU inference
- `timeout_ms`: Max time to wait for batch to fill (milliseconds)
- `use_gpu`: Whether to use GPU for inference
- `warmup_batches`: Number of warmup batches on startup
"""
@kwdef struct InferenceServerConfig
    endpoint::EndpointConfig = EndpointConfig(port=5555)
    batch_size::Int = 64
    timeout_ms::Int = 50
    use_gpu::Bool = true
    warmup_batches::Int = 3
end

#####
##### Replay buffer configuration
#####

"""
    ReplayBufferConfig

Configuration for the replay buffer manager.

# Fields
- `endpoint`: ZMQ endpoint for receiving samples
- `capacity`: Maximum number of samples to store
- `min_samples_for_training`: Minimum samples before training starts
- `prioritized`: Whether to use prioritized sampling
- `priority_alpha`: Priority exponent (0 = uniform, 1 = full priority)
- `priority_beta`: Importance sampling correction factor
"""
@kwdef struct ReplayBufferConfig
    endpoint::EndpointConfig = EndpointConfig(port=5556)
    capacity::Int = 500_000
    min_samples_for_training::Int = 10_000
    prioritized::Bool = false
    priority_alpha::Float64 = 0.6
    priority_beta::Float64 = 0.4
end

#####
##### Worker configuration
#####

"""
    WorkerConfig

Configuration for a self-play worker.

# Fields
- `worker_id`: Unique identifier for this worker
- `inference_endpoint`: Connection to inference server
- `replay_endpoint`: Connection to replay buffer
- `coordinator_endpoint`: Connection to coordinator (for commands/weights)
- `games_per_batch`: Games to complete before submitting samples
- `mcts_params`: MCTS configuration
- `heartbeat_interval_s`: Seconds between heartbeats
- `num_parallel_games`: Games to run in parallel per worker
"""
@kwdef struct WorkerConfig
    worker_id::String
    inference_endpoint::EndpointConfig
    replay_endpoint::EndpointConfig
    coordinator_endpoint::EndpointConfig
    games_per_batch::Int = 10
    mcts_params::Union{Nothing, MctsParams} = nothing
    gumbel_params::Union{Nothing, GumbelMctsParams} = nothing
    heartbeat_interval_s::Float64 = 30.0
    num_parallel_games::Int = 1
    use_gpu::Bool = true  # Whether to use GPU for local inference
end

#####
##### Training configuration
#####

"""
    TrainingConfig

Configuration for the training process.

# Fields
- `replay_endpoint`: Connection to replay buffer
- `coordinator_endpoint`: Connection to coordinator
- `learning_params`: Learning hyperparameters
- `batch_size`: Training batch size
- `batches_per_checkpoint`: Batches between checkpoints
- `checkpoint_dir`: Directory for saving checkpoints
- `use_gpu`: Whether to use GPU for training
- `wandb_project`: WandB project name (nothing to disable)
- `wandb_run_name`: WandB run name
"""
@kwdef struct TrainingConfig
    replay_endpoint::EndpointConfig
    coordinator_endpoint::EndpointConfig
    learning_params::Union{Nothing, LearningParams} = nothing
    batch_size::Int = 2048
    batches_per_checkpoint::Int = 100
    checkpoint_dir::String = "checkpoints"
    use_gpu::Bool = true
    wandb_project::Union{Nothing, String} = nothing
    wandb_run_name::Union{Nothing, String} = nothing
end

#####
##### Evaluation configuration
#####

"""
    EvaluationConfig

Configuration for the evaluation process.

# Fields
- `coordinator_endpoint`: Connection to coordinator
- `num_games`: Games per evaluation
- `baselines`: Baseline opponents to evaluate against
- `interval_iterations`: Iterations between evaluations
- `use_gpu`: Whether to use GPU for evaluation
- `sim_params`: Simulation parameters for evaluation games
"""
@kwdef struct EvaluationConfig
    coordinator_endpoint::EndpointConfig
    num_games::Int = 100
    baselines::Vector{Symbol} = [:random, :previous]
    interval_iterations::Int = 5
    use_gpu::Bool = true
    sim_params::Union{Nothing, SimParams} = nothing
end

#####
##### Coordinator configuration
#####

"""
    CoordinatorConfig

Configuration for the main coordinator process.

# Fields
- `inference_config`: Inference server configuration
- `replay_config`: Replay buffer configuration
- `training_config`: Training configuration
- `evaluation_config`: Evaluation configuration
- `command_endpoint`: Endpoint for worker commands
- `num_local_workers`: Number of workers to spawn locally
- `session_dir`: Directory for session data
- `total_iterations`: Target training iterations
- `games_per_iteration`: Games per training iteration
- `wandb_project`: WandB project for logging
"""
@kwdef struct CoordinatorConfig
    inference_config::InferenceServerConfig = InferenceServerConfig()
    replay_config::ReplayBufferConfig = ReplayBufferConfig()
    training_config::TrainingConfig
    evaluation_config::Union{Nothing, EvaluationConfig} = nothing
    command_endpoint::EndpointConfig = EndpointConfig(port=5557)
    num_local_workers::Int = 4
    session_dir::String = "sessions"
    total_iterations::Int = 100
    games_per_iteration::Int = 500
    wandb_project::Union{Nothing, String} = nothing
end

#####
##### Full distributed experiment configuration
#####

"""
    DistributedExperiment

Complete configuration for a distributed training experiment.

# Fields
- `game_module`: Module containing game implementation
- `game_name`: Name of the game
- `network_params`: Network hyperparameters
- `mcts_params`: MCTS hyperparameters
- `coordinator`: Coordinator configuration
- `description`: Experiment description
"""
@kwdef struct DistributedExperiment
    game_module::Module
    game_name::String
    network_params::Any  # Network hyperparameters (e.g., FCResNetMultiHeadHP)
    mcts_params::MctsParams
    coordinator::CoordinatorConfig
    description::String = ""
end

#####
##### Preset configurations
#####

"""
    default_local_config(;
        inference_port=5555,
        replay_port=5556,
        command_port=5557,
        num_workers=4,
        kwargs...
    ) -> CoordinatorConfig

Create a default configuration for local single-machine training.
"""
function default_local_config(;
    inference_port::Int=5555,
    replay_port::Int=5556,
    command_port::Int=5557,
    num_workers::Int=4,
    session_dir::String="sessions",
    kwargs...
)
    inference_config = InferenceServerConfig(
        endpoint=EndpointConfig(port=inference_port),
    )
    replay_config = ReplayBufferConfig(
        endpoint=EndpointConfig(port=replay_port),
    )
    training_config = TrainingConfig(
        replay_endpoint=EndpointConfig(host="localhost", port=replay_port),
        coordinator_endpoint=EndpointConfig(host="localhost", port=command_port),
    )

    return CoordinatorConfig(
        inference_config=inference_config,
        replay_config=replay_config,
        training_config=training_config,
        command_endpoint=EndpointConfig(port=command_port),
        num_local_workers=num_workers,
        session_dir=session_dir;
        kwargs...
    )
end

"""
    worker_config_from_coordinator(
        coord_host::String,
        coord_config::CoordinatorConfig,
        worker_id::String
    ) -> WorkerConfig

Create a worker configuration that connects to a coordinator.
"""
function worker_config_from_coordinator(
    coord_host::String,
    coord_config::CoordinatorConfig,
    worker_id::String
)
    return WorkerConfig(
        worker_id=worker_id,
        inference_endpoint=EndpointConfig(
            host=coord_host,
            port=coord_config.inference_config.endpoint.port
        ),
        replay_endpoint=EndpointConfig(
            host=coord_host,
            port=coord_config.replay_config.endpoint.port
        ),
        coordinator_endpoint=EndpointConfig(
            host=coord_host,
            port=coord_config.command_endpoint.port
        ),
    )
end

#####
##### Configuration validation
#####

"""
    validate_config(config) -> Bool

Validate a configuration, throwing errors for invalid settings.
"""
function validate_config(config::CoordinatorConfig)
    # Check port conflicts
    ports = [
        config.inference_config.endpoint.port,
        config.replay_config.endpoint.port,
        config.command_endpoint.port,
    ]
    if !isnothing(config.training_config)
        push!(ports, config.training_config.coordinator_endpoint.port)
    end

    if length(unique(ports)) != length(ports)
        error("Port conflict in configuration")
    end

    # Check positive values
    if config.num_local_workers < 0
        error("num_local_workers must be non-negative")
    end

    if config.total_iterations <= 0
        error("total_iterations must be positive")
    end

    return true
end

function validate_config(config::WorkerConfig)
    if isempty(config.worker_id)
        error("worker_id cannot be empty")
    end
    return true
end

#####
##### Configuration serialization
#####

"""
    save_config(path::String, config)

Save configuration to a JSON file.
"""
function save_config(path::String, config)
    open(path, "w") do io
        JSON3.write(io, config)
    end
end

"""
    load_config(path::String, ::Type{T}) -> T

Load configuration from a JSON file.
"""
function load_config(path::String, ::Type{T}) where T
    json_str = read(path, String)
    return JSON3.read(json_str, T)
end
