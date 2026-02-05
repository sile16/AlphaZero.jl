#####
##### Cluster Types
#####
##### Shared data types for coordinator-worker communication.
#####

"""
Serialized game sample for transfer between processes.

Includes reanalysis tracking fields for MuZero-style sample updates.
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
    # Reanalysis tracking
    priority::Float32           # TD-error based priority for sampling
    added_step::Int             # Training step when sample was added
    last_reanalyze_step::Int    # Last reanalysis step (worker internal counter)
    reanalyze_count::Int        # Times this sample has been reanalyzed
    # Smart reanalysis - track model iteration for freshness
    model_iter_reanalyzed::Int  # Model iteration this sample was last reanalyzed with (0 = never)
end

# Constructor with default reanalyze values (for new samples)
function ClusterSample(
    state, policy, value, turn, is_chance,
    equity_p_win, equity_p_gw, equity_p_bgw, equity_p_gl, equity_p_bgl, has_equity;
    priority::Float32 = 0.0f0,
    added_step::Int = 0,
    last_reanalyze_step::Int = 0,
    reanalyze_count::Int = 0,
    model_iter_reanalyzed::Int = 0
)
    return ClusterSample(
        state, policy, value, turn, is_chance,
        equity_p_win, equity_p_gw, equity_p_bgw, equity_p_gl, equity_p_bgl, has_equity,
        priority, added_step, last_reanalyze_step, reanalyze_count, model_iter_reanalyzed
    )
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

"""
    PrioritizedSamplingConfig

Configuration for Prioritized Experience Replay (PER) sampling.

Based on Schaul et al. 2016 "Prioritized Experience Replay" (ICLR 2016).
https://arxiv.org/abs/1511.05952

| Field | Description |
|:------|:------------|
| `enabled` | Whether prioritized sampling is enabled |
| `alpha` | Priority exponent (0=uniform, 1=full prioritization) |
| `beta` | Importance sampling exponent (0=no correction, 1=full correction) |
| `beta_annealing_steps` | Steps to anneal beta from initial to 1.0 |
| `epsilon` | Small constant added to priorities to ensure non-zero probability |
| `initial_priority` | Priority for new samples (before TD-error is computed) |
"""
@kwdef struct PrioritizedSamplingConfig
    enabled::Bool = false
    alpha::Float32 = 0.6f0           # Standard value from PER paper
    beta::Float32 = 0.4f0            # Initial beta, anneals to 1.0
    beta_annealing_steps::Int = 100  # Anneal over this many iterations
    epsilon::Float32 = 0.01f0        # Small constant for numerical stability
    initial_priority::Float32 = 1.0f0  # High priority for new samples
end

"""
Get the current beta value with annealing.
Beta anneals linearly from initial value to 1.0.
"""
function get_beta(config::PrioritizedSamplingConfig, current_step::Int)
    if current_step >= config.beta_annealing_steps
        return 1.0f0
    end
    progress = Float32(current_step) / Float32(config.beta_annealing_steps)
    return config.beta + (1.0f0 - config.beta) * progress
end
