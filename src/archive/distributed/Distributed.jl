#####
##### Distributed Training Module
#####

"""
    Distributed

Module for distributed AlphaZero training with:
- N Self-Play Workers (can run on different servers)
- 1 Inference Server (centralized GPU for batched neural network evaluation)
- 1 Replay Buffer Manager (manages training samples)
- 1 Training Process (GPU-based learning)
- 1 Evaluation Process (periodic strength testing)

## Quick Start

### Local single-machine training:

```julia
using AlphaZero
using AlphaZero.Distributed

# Setup game and network
gspec = Examples.games["connect-four"]
network = NetLib.SimpleNet(gspec)
mcts_params = MctsParams(num_iters_per_turn=400)

# Run distributed training
train_distributed!(gspec, network, mcts_params;
    num_workers=4,
    total_iterations=100,
    wandb_project="alphazero-distributed"
)
```

### Multi-server setup:

On coordinator server:
```julia
coordinator = create_coordinator(gspec, network;
    num_workers=0,  # Remote workers only
    inference_port=5555,
    replay_port=5556,
)
run_coordinator(coordinator, mcts_params)
```

On worker servers:
```julia
worker = create_worker(gspec, "worker_1", "coordinator_ip", "coordinator_ip";
    inference_port=5555,
    replay_port=5556,
    mcts_params=mcts_params
)
run_worker(worker)
```

## Components

- `ZMQInferenceServer`: GPU inference server
- `ZMQRemoteOracle`: Client for remote inference
- `ReplayBufferManager`: Centralized sample storage
- `SelfPlayWorker`: MCTS self-play worker
- `TrainingProcess`: GPU training loop
- `EvaluationProcess`: Periodic evaluation
- `DistributedCoordinator`: Main orchestration
"""
module Distributed

using ..AlphaZero
using ..AlphaZero: GI, Network, MCTS, Util, Trace, TrainingSample
using ..AlphaZero: MctsParams, GumbelMctsParams, SimParams, LearningParams
using ..AlphaZero: AbstractNetwork, AbstractGameSpec, AbstractPlayer
using ..AlphaZero: MctsPlayer, GumbelMctsPlayer, RandomPlayer, TwoPlayers
using ..AlphaZero: play_game, reset_player!, total_reward
using ..AlphaZero: EquityTargets, MemoryBuffer
using ..AlphaZero.FluxLib

using JSON3
using Statistics: mean, median

# Message types
include("messages.jl")
export InferenceRequest, InferenceResponse, MultiHeadValue
export SerializedSample, GameSamples
export WeightUpdate, WeightRequest
export TrainingSampleBatch, TrainingMetrics
export WorkerHeartbeat, WorkerCommand
export EvaluationRequest, EvaluationResult
export MessageEnvelope, MESSAGE_TYPES

# Serialization utilities
include("serialization.jl")
export serialize_message, deserialize_message
export serialize_network_weights, deserialize_network_weights, load_weights_into_network!
export serialize_state, serialize_states_batch
export serialize_training_sample, deserialize_training_sample
export wrap_message, unwrap_message
export send_zmq_message, recv_zmq_message
export compute_checksum

# Configuration
include("config.jl")
export EndpointConfig, endpoint_string
export InferenceServerConfig
export ReplayBufferConfig
export WorkerConfig
export TrainingConfig
export EvaluationConfig
export CoordinatorConfig
export DistributedExperiment
export default_local_config, worker_config_from_coordinator
export validate_config
export save_config, load_config

# Inference server
include("inference_server.jl")
export ZMQInferenceServer
export evaluate_batch, handle_request
export run_inference_server, run_inference_server_async
export shutdown_inference_server
export update_network!, update_network_weights!
export get_server_stats
export create_inference_server

# Remote oracle
include("remote_oracle.jl")
export ZMQRemoteOracle
export connect!, disconnect!, reconnect!
export batch_inference
export compute_scalar_value
export MCTSOracleAdapter
export BatchingRemoteOracle, queue_inference!, flush_batch!

# Replay buffer manager
include("replay_manager.jl")
export ReplayBufferManager
export add_samples!, sample_batch, get_batch_for_training
export is_ready_for_training
export run_replay_manager, run_replay_manager_async
export shutdown_replay_manager
export get_buffer_stats
export ReplayManagerWithREP, run_replay_manager_with_rep

# Self-play worker
include("self_play_worker.jl")
export SelfPlayWorker
export create_mcts_player
export play_self_play_game, trace_to_samples
export submit_game_samples
export check_weight_updates!, apply_weight_update!, request_initial_weights!
export run_worker, run_worker_async
export shutdown_worker
export get_worker_stats
export create_worker, create_local_worker, create_remote_worker

# Training process
include("training_process.jl")
export TrainingProcess
export fetch_training_batch, convert_batch_for_training
export training_step!
export broadcast_weights!, save_checkpoint, load_checkpoint
export run_training, run_training_async
export shutdown_training
export get_training_stats
export create_training_process

# Evaluation process
include("evaluation_process.jl")
export EvaluationProcess
export create_baseline_player, create_evaluation_player
export run_evaluation_games, run_full_evaluation
export load_network_for_evaluation!, initialize_network!
export watch_checkpoints, watch_checkpoints_async
export shutdown_evaluation
export get_evaluation_stats, get_latest_results
export create_evaluation_process

# Coordinator
include("coordinator.jl")
export DistributedCoordinator
export start_inference_server!, start_replay_manager!
export start_training!, start_evaluation!
export start_local_workers!
export broadcast_weights!, broadcast_command!
export run_coordinator, run_coordinator_async
export check_component_health!, log_progress
export shutdown_coordinator
export create_coordinator
export train_distributed!

end  # module Distributed
