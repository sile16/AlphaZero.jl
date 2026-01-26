#####
##### Cluster Types
#####
##### Shared data types for coordinator-worker communication.
#####

"""
Serialized game sample for transfer between processes.
"""
struct ClusterSample
    state::Vector{Float32}      # Flattened state
    policy::Vector{Float32}     # Full policy (all actions)
    value::Float32              # Target value
    turn::Float32               # Turn number
    is_chance::Bool             # Is this a chance node
    # Multi-head equity targets (optional)
    equity_p_win::Float32
    equity_p_gw::Float32        # P(gammon|win)
    equity_p_bgw::Float32       # P(backgammon|win)
    equity_p_gl::Float32        # P(gammon|loss)
    equity_p_bgl::Float32       # P(backgammon|loss)
    has_equity::Bool
end

"""
Batch of samples from a completed game.
"""
struct GameBatch
    worker_id::Int
    samples::Vector{ClusterSample}
    game_length::Int
    outcome::Float32  # Final reward
end

"""
Weight update message from coordinator to workers.
"""
struct WeightUpdate
    iteration::Int
    weights::Vector{UInt8}  # Serialized network weights
    timestamp::Float64
end

"""
Worker status report.
"""
struct WorkerStatus
    worker_id::Int
    games_played::Int
    samples_generated::Int
    last_update_time::Float64
end

"""
Training metrics from coordinator.
"""
struct TrainingMetrics
    iteration::Int
    loss::Float64
    buffer_size::Int
    total_games::Int
    total_samples::Int
    games_per_minute::Float64
end

"""
Evaluation results.
"""
struct EvalResults
    iteration::Int
    vs_random_white::Float64
    vs_random_black::Float64
    vs_random_combined::Float64
    num_games::Int
    eval_time::Float64
end
