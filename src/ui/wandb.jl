#####
##### Weights & Biases (wandb) Integration
#####

module Wandb

using PythonCall
using ..AlphaZero: Report, Params, Network

# Lazy-loaded wandb module
const wandb = Ref{Py}()

function ensure_wandb_loaded()
  if !isassigned(wandb)
    wandb[] = pyimport("wandb")
  end
  return wandb[]
end

"""
    wandb_available() -> Bool

Check if wandb Python package is available.
"""
function wandb_available()
  try
    ensure_wandb_loaded()
    return true
  catch
    return false
  end
end

"""
    wandb_init(; project, name=nothing, config=Dict())

Initialize a new wandb run.

# Arguments
- `project::String`: The wandb project name
- `name::String=nothing`: Optional run name (auto-generated if not provided)
- `config::Dict`: Configuration dictionary to log
"""
function wandb_init(; project::String, name=nothing, config=Dict())
  wb = ensure_wandb_loaded()
  wb.init(
    project=project,
    name=name,
    config=pydict(config)
  )
end

"""
    wandb_log(metrics::Dict; step=nothing)

Log metrics to wandb.

# Arguments
- `metrics::Dict`: Dictionary of metric names to values
- `step::Int=nothing`: Optional step number
"""
function wandb_log(metrics::Dict; step=nothing)
  wb = ensure_wandb_loaded()
  if isnothing(step)
    wb.log(pydict(metrics))
  else
    wb.log(pydict(metrics), step=step)
  end
end

"""
    wandb_finish()

Finish the current wandb run.
"""
function wandb_finish()
  wb = ensure_wandb_loaded()
  wb.finish()
end

#####
##### Report conversion helpers
#####

"""
Convert training parameters to a wandb config dictionary.
"""
function params_to_config(params::Params, network)
  config = Dict{String, Any}(
    "num_iters" => params.num_iters,
    "use_symmetries" => params.use_symmetries,
    "ternary_outcome" => params.ternary_outcome,
  )

  # Self-play parameters
  sp = params.self_play
  config["self_play/num_games"] = sp.sim.num_games
  config["self_play/num_workers"] = sp.sim.num_workers
  config["self_play/batch_size"] = sp.sim.batch_size
  config["self_play/use_gpu"] = sp.sim.use_gpu

  # MCTS parameters
  mcts = sp.mcts
  config["mcts/num_iters_per_turn"] = mcts.num_iters_per_turn
  config["mcts/cpuct"] = mcts.cpuct
  config["mcts/gamma"] = mcts.gamma
  config["mcts/dirichlet_noise_alpha"] = mcts.dirichlet_noise_ϵ > 0 ? mcts.dirichlet_noise_α : 0
  config["mcts/dirichlet_noise_epsilon"] = mcts.dirichlet_noise_ϵ

  # Learning parameters
  lp = params.learning
  config["learning/batch_size"] = lp.batch_size
  config["learning/l2_regularization"] = lp.l2_regularization
  config["learning/nonvalidity_penalty"] = lp.nonvalidity_penalty
  config["learning/num_checkpoints"] = lp.num_checkpoints
  config["learning/max_batches_per_checkpoint"] = lp.max_batches_per_checkpoint

  # Optimizer info
  opt = lp.optimiser
  config["learning/optimizer"] = string(typeof(opt).name.name)

  # Arena parameters (if present)
  if !isnothing(params.arena)
    ap = params.arena
    config["arena/num_games"] = ap.sim.num_games
    config["arena/update_threshold"] = ap.update_threshold
  end

  # Progressive simulation parameters (if present)
  if !isnothing(params.progressive_sim)
    ps = params.progressive_sim
    config["progressive_sim/sim_min"] = ps.sim_min
    config["progressive_sim/sim_max"] = ps.sim_max
  end

  if !isnothing(params.turn_progressive_sim)
    tps = params.turn_progressive_sim
    config["turn_progressive_sim/turn_sim_min"] = tps.turn_sim_min
    config["turn_progressive_sim/turn_sim_target"] = tps.turn_sim_target
    config["turn_progressive_sim/ramp_turns_initial"] = tps.ramp_turns_initial
    config["turn_progressive_sim/ramp_turns_final"] = tps.ramp_turns_final
  end

  # Network parameters
  config["network/num_parameters"] = Network.num_parameters(network)
  config["network/num_regularized_parameters"] = Network.num_regularized_parameters(network)

  return config
end

"""
Convert a SelfPlay report to wandb metrics.
"""
function self_play_metrics(report::Report.SelfPlay)
  return Dict{String, Any}(
    "self_play/samples_gen_speed" => report.samples_gen_speed,
    "self_play/avg_exploration_depth" => report.average_exploration_depth,
    "self_play/mcts_memory_footprint" => report.mcts_memory_footprint,
    "self_play/memory_size" => report.memory_size,
    "self_play/memory_num_distinct_boards" => report.memory_num_distinct_boards,
    "self_play/num_sims_per_turn" => report.num_sims_per_turn,
  )
end

"""
Convert a LearningStatus report to wandb metrics.
"""
function learning_status_metrics(status::Report.LearningStatus; prefix="")
  p = isempty(prefix) ? "" : prefix * "/"
  return Dict{String, Any}(
    "$(p)loss/total" => status.loss.L,
    "$(p)loss/policy" => status.loss.Lp,
    "$(p)loss/value" => status.loss.Lv,
    "$(p)loss/regularization" => status.loss.Lreg,
    "$(p)loss/invalid" => status.loss.Linv,
    "$(p)entropy/mcts_policy" => status.Hp,
    "$(p)entropy/network_policy" => status.Hpnet,
  )
end

"""
Convert a Learning report to wandb metrics.
"""
function learning_metrics(report::Report.Learning)
  metrics = Dict{String, Any}(
    "learning/time_convert" => report.time_convert,
    "learning/time_loss" => report.time_loss,
    "learning/time_train" => report.time_train,
    "learning/time_eval" => report.time_eval,
    "learning/nn_replaced" => report.nn_replaced ? 1 : 0,
    "learning/num_batches" => length(report.losses),
  )

  # Initial status metrics
  merge!(metrics, learning_status_metrics(report.initial_status, prefix="initial"))

  # Final loss from last checkpoint or initial if no checkpoints
  if !isempty(report.checkpoints)
    final_status = report.checkpoints[end].status_after
    merge!(metrics, learning_status_metrics(final_status, prefix="final"))
  end

  # Loss statistics over batches
  if !isempty(report.losses)
    metrics["learning/loss_mean"] = sum(report.losses) / length(report.losses)
    metrics["learning/loss_final"] = report.losses[end]
  end

  return metrics
end

"""
Convert a Checkpoint report to wandb metrics.
"""
function checkpoint_metrics(report::Report.Checkpoint; checkpoint_idx::Int=1)
  metrics = Dict{String, Any}(
    "checkpoint/batch_id" => report.batch_id,
    "checkpoint/avg_reward" => report.evaluation.avgr,
    "checkpoint/redundancy" => report.evaluation.redundancy,
    "checkpoint/nn_replaced" => report.nn_replaced ? 1 : 0,
  )
  merge!(metrics, learning_status_metrics(report.status_after, prefix="checkpoint"))
  return metrics
end

"""
Convert an Iteration report to wandb metrics.
"""
function iteration_metrics(report::Report.Iteration, iteration::Int)
  metrics = Dict{String, Any}(
    "iteration" => iteration,
    "time/self_play" => report.perfs_self_play.time,
    "time/memory_analysis" => report.perfs_memory_analysis.time,
    "time/learning" => report.perfs_learning.time,
    "time/total" => report.perfs_self_play.time + report.perfs_memory_analysis.time + report.perfs_learning.time,
    "memory/self_play_allocated" => report.perfs_self_play.allocated,
    "memory/learning_allocated" => report.perfs_learning.allocated,
  )

  # Self-play metrics
  merge!(metrics, self_play_metrics(report.self_play))

  # Learning metrics
  merge!(metrics, learning_metrics(report.learning))

  # Memory analysis metrics (if present)
  if !isnothing(report.memory)
    mem = report.memory
    metrics["memory_analysis/all_samples_count"] = mem.all_samples.num_samples
    metrics["memory_analysis/all_boards_count"] = mem.all_samples.num_boards
    metrics["memory_analysis/latest_batch_count"] = mem.latest_batch.num_samples
  end

  return metrics
end

"""
Convert benchmark evaluation to wandb metrics.
"""
function benchmark_metrics(evaluations::Vector{Report.Evaluation})
  metrics = Dict{String, Any}()
  for (i, eval) in enumerate(evaluations)
    prefix = length(evaluations) == 1 ? "benchmark" : "benchmark_$i"
    metrics["$(prefix)/avg_reward"] = eval.avgr
    metrics["$(prefix)/redundancy"] = eval.redundancy
    if !isnothing(eval.rewards)
      metrics["$(prefix)/num_games"] = length(eval.rewards)
    end
  end
  return metrics
end

export wandb_available, wandb_init, wandb_log, wandb_finish
export params_to_config, self_play_metrics, learning_status_metrics
export learning_metrics, checkpoint_metrics, iteration_metrics, benchmark_metrics

end # module Wandb
