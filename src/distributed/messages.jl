#####
##### Message types for distributed training
#####

"""
Message types for ZMQ-based distributed AlphaZero training.

These structs define the protocol for communication between:
- Self-play workers and inference server
- Self-play workers and replay buffer manager
- Coordinator and all components
"""

using Base: @kwdef

#####
##### Inference messages
#####

"""
    InferenceRequest

Request from a worker to the inference server for batch evaluation.

# Fields
- `worker_id`: Unique identifier for the worker
- `request_id`: Unique request ID for matching responses
- `states`: Vector of serialized board states to evaluate
"""
@kwdef struct InferenceRequest
    worker_id::String
    request_id::UInt64
    states::Vector{Vector{Float32}}
end

"""
    InferenceResponse

Response from inference server with policy and value predictions.

For multi-head networks, values contains the full equity components.

# Fields
- `request_id`: Matches the corresponding request
- `policies`: Vector of policy distributions (one per state)
- `values`: Vector of value predictions (scalar or multi-head)
- `is_multihead`: Whether values are multi-head equity outputs
"""
@kwdef struct InferenceResponse
    request_id::UInt64
    policies::Vector{Vector{Float32}}
    values::Vector{Any}  # Float32 for single-head, NamedTuple for multi-head
    is_multihead::Bool = false
end

"""
    MultiHeadValue

Multi-head value output for backgammon-style equity networks.
"""
@kwdef struct MultiHeadValue
    p_win::Float32
    p_gammon_win::Float32
    p_bg_win::Float32
    p_gammon_loss::Float32
    p_bg_loss::Float32
end

#####
##### Sample submission messages
#####

"""
    SerializedSample

A single training sample in serialized form for network transmission.

# Fields
- `state`: Vectorized board state
- `policy`: MCTS policy distribution
- `value`: Game outcome value (-1 to 1)
- `turn`: Turn number in the game
- `is_chance`: Whether this was a chance node
- `equity`: Optional multi-head equity targets
"""
@kwdef struct SerializedSample
    state::Vector{Float32}
    policy::Vector{Float32}
    value::Float32
    turn::Float32
    is_chance::Bool = false
    equity::Union{Nothing, MultiHeadValue} = nothing
end

"""
    GameSamples

Collection of samples from a completed game, sent from worker to replay buffer.

# Fields
- `worker_id`: Identifier of the worker that generated the game
- `game_id`: Unique identifier for this game
- `samples`: Vector of training samples from this game
- `metadata`: Game metadata (length, outcome, etc.)
"""
@kwdef struct GameSamples
    worker_id::String
    game_id::UInt64
    samples::Vector{SerializedSample}
    metadata::Dict{String,Any} = Dict{String,Any}()
end

#####
##### Weight distribution messages
#####

"""
    WeightUpdate

Notification of new network weights, sent from coordinator to workers.

# Fields
- `iteration`: Training iteration number
- `weights_data`: Serialized network weights (bytes)
- `timestamp`: Unix timestamp of update
- `checksum`: Optional checksum for verification
"""
@kwdef struct WeightUpdate
    iteration::Int
    weights_data::Vector{UInt8}
    timestamp::Float64
    checksum::UInt64 = 0
end

"""
    WeightRequest

Request for current weights, typically from a new worker joining.

# Fields
- `worker_id`: Identifier of requesting worker
"""
@kwdef struct WeightRequest
    worker_id::String
end

#####
##### Training control messages
#####

"""
    TrainingSampleBatch

Batch of samples for training, sent from replay buffer to training process.

# Fields
- `batch_id`: Unique batch identifier
- `samples`: Vector of training samples
- `total_buffer_size`: Current size of replay buffer
"""
@kwdef struct TrainingSampleBatch
    batch_id::UInt64
    samples::Vector{SerializedSample}
    total_buffer_size::Int
end

"""
    TrainingMetrics

Metrics from a training step, for logging.

# Fields
- `iteration`: Current training iteration
- `loss`: Total loss
- `policy_loss`: Policy component of loss
- `value_loss`: Value component of loss
- `samples_used`: Number of samples in this step
- `timestamp`: Unix timestamp
"""
@kwdef struct TrainingMetrics
    iteration::Int
    loss::Float64
    policy_loss::Float64
    value_loss::Float64
    samples_used::Int
    timestamp::Float64
end

#####
##### Worker control messages
#####

"""
    WorkerHeartbeat

Periodic heartbeat from worker to coordinator.

# Fields
- `worker_id`: Worker identifier
- `games_completed`: Total games completed by this worker
- `samples_generated`: Total samples generated
- `current_iteration`: Iteration of weights currently loaded
- `timestamp`: Unix timestamp
"""
@kwdef struct WorkerHeartbeat
    worker_id::String
    games_completed::Int
    samples_generated::Int
    current_iteration::Int
    timestamp::Float64
end

"""
    WorkerCommand

Command from coordinator to worker.

# Fields
- `command`: Command type (:continue, :pause, :shutdown, :update_weights)
- `payload`: Optional command-specific data
"""
@kwdef struct WorkerCommand
    command::Symbol
    payload::Any = nothing
end

#####
##### Evaluation messages
#####

"""
    EvaluationRequest

Request to evaluate current network against a baseline.

# Fields
- `iteration`: Current training iteration
- `weights_data`: Serialized network weights to evaluate
- `baseline`: Baseline opponent (:random, :previous, :iteration_N)
- `num_games`: Number of evaluation games
"""
@kwdef struct EvaluationRequest
    iteration::Int
    weights_data::Vector{UInt8}
    baseline::Symbol
    num_games::Int = 100
end

"""
    EvaluationResult

Results from an evaluation run.

# Fields
- `iteration`: Training iteration that was evaluated
- `baseline`: Baseline that was used
- `win_rate`: Win rate against baseline (0 to 1)
- `avg_reward`: Average reward per game
- `reward_histogram`: Histogram of rewards
- `num_games`: Number of games played
"""
@kwdef struct EvaluationResult
    iteration::Int
    baseline::Symbol
    win_rate::Float64
    avg_reward::Float64
    reward_histogram::Dict{Float64,Int} = Dict{Float64,Int}()
    num_games::Int
end

#####
##### Message envelope for routing
#####

"""
    MessageEnvelope

Wrapper for routing messages through ZMQ.

# Fields
- `msg_type`: Type identifier for deserialization
- `payload`: Serialized message data
- `sender_id`: ID of sender
- `timestamp`: Unix timestamp
"""
@kwdef struct MessageEnvelope
    msg_type::Symbol
    payload::Vector{UInt8}
    sender_id::String
    timestamp::Float64 = time()
end

# Message type registry
const MESSAGE_TYPES = Dict{Symbol, DataType}(
    :inference_request => InferenceRequest,
    :inference_response => InferenceResponse,
    :game_samples => GameSamples,
    :weight_update => WeightUpdate,
    :weight_request => WeightRequest,
    :training_batch => TrainingSampleBatch,
    :training_metrics => TrainingMetrics,
    :worker_heartbeat => WorkerHeartbeat,
    :worker_command => WorkerCommand,
    :evaluation_request => EvaluationRequest,
    :evaluation_result => EvaluationResult,
)
