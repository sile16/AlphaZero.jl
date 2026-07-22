#####
##### AlphaZero Parameters
#####

"""
Parameters of an MCTS player.

| Parameter              | Type                         | Default             |
|:-----------------------|:-----------------------------|:--------------------|
| `num_iters_per_turn`   | `Int`                        |  -                  |
| `gamma`                | `Float64`                    | `1.`                |
| `cpuct`                | `Float64`                    | `1.`                |
| `temperature`          | `AbstractSchedule{Float64}`  | `ConstSchedule(1.)` |
| `dirichlet_noise_ϵ`    | `Float64`                    |  -                  |
| `dirichlet_noise_α`    | `Float64`                    |  -                  |
| `prior_temperature`    | `Float64`                    | `1.`                |

# Explanation

An MCTS player picks an action as follows. Given a game state, it launches
`num_iters_per_turn` MCTS iterations, with UCT exploration constant `cpuct`.
Rewards are discounted using the `gamma` factor.

Then, an action is picked according to the distribution ``π`` where
``π_i ∝ n_i^{1/τ}`` with ``n_i`` the number of times that the ``i^{\\text{th}}``
action was visited and ``τ`` the `temperature` parameter.

It is typical to use a high value of the temperature parameter ``τ``
during the first moves of a game to increase exploration and then switch to
a small value. Therefore, `temperature` is am [`AbstractSchedule`](@ref).

For information on parameters `cpuct`, `dirichlet_noise_ϵ`,
`dirichlet_noise_α` and `prior_temperature`, see [`MCTS.Env`](@ref).

# AlphaGo Zero Parameters

In the original AlphaGo Zero paper:

+ The discount factor `gamma` is set to 1.
+ The number of MCTS iterations per move is 1600, which
  corresponds to 0.4s of computation time.
+ The temperature is set to 1 for the 30 first moves and then to an
  infinitesimal value.
+ The ``ϵ`` parameter for the Dirichlet noise is set to ``0.25`` and
  the ``α`` parameter to ``0.03``, which is consistent with the heuristic
  of using ``α = 10/n`` with ``n`` the maximum number of possibles moves,
  which is ``19 × 19 + 1 = 362`` in the case of Go.
"""
@kwdef struct MctsParams
  gamma :: Float64 = 1.
  cpuct :: Float64 = 1.
  num_iters_per_turn :: Int
  temperature :: AbstractSchedule{Float64} = ConstSchedule(1.)
  dirichlet_noise_ϵ :: Float64
  dirichlet_noise_α :: Float64
  prior_temperature :: Float64 = 1.
  # Chance node handling mode for stochastic games:
  #   :full - expand all outcomes (full expectimax, expensive; classic recursive engine)
  #   :sampling - sample one outcome per visit (Monte Carlo, fast)
  #   :progressive - progressive widening with prior integration
  #   :passthrough - sample one dice outcome, continue (batched player; training default path)
  #   :exact_expectation - EVAL-ONLY batched expectimax over dice outcomes (first-class
  #                        chance_tree entries). Must be requested explicitly; :full defaults
  #                        everywhere, so honoring it in the batched player would flip training.
  chance_mode :: Symbol = :full
  # Progressive widening parameters (for :progressive mode)
  # Expand new outcome when N^α > num_expanded (α=0.5 expands at 1,4,9,16,25,36...)
  progressive_widening_alpha :: Float64 = 0.5
  # Virtual visits to weight NN prior (higher = prior persists longer)
  prior_virtual_visits :: Float64 = 1.0
end

"""
    SamplesWeighingPolicy

During self-play, early board positions are possibly encountered many
times across several games. The corresponding samples can be merged
together and given a weight ``W`` that is a nondecreasing function of the
number ``n`` of merged samples:

  - `CONSTANT_WEIGHT`: ``W(n) = 1``
  - `LOG_WEIGHT`: ``W(n) = \\log_2(n) + 1``
  - `LINEAR_WEIGHT`: ``W(n) = n``
"""
@enum SamplesWeighingPolicy CONSTANT_WEIGHT LOG_WEIGHT LINEAR_WEIGHT

"""
    LearningParams

Parameters governing the learning phase of a training iteration, where
the neural network is updated to fit the data in the memory buffer.

| Parameter                     | Type                            | Default    |
|:------------------------------|:--------------------------------|:-----------|
| `use_gpu`                     | `Bool`                          | `false`    |
| `use_position_averaging`      | `Bool`                          | `true`     |
| `samples_weighing_policy`     | [`SamplesWeighingPolicy`](@ref) |  -         |
| `optimiser`                   | [`OptimiserSpec`](@ref)         |  -         |
| `l2_regularization`           | `Float32`                       |  -         |
| `rewards_renormalization`     | `Float32`                       | `1f0`      |
| `nonvalidity_penalty`         | `Float32`                       | `1f0`      |
| `batch_size`                  | `Int`                           |  -         |
| `loss_computation_batch_size` | `Int`                           |  -         |
| `min_checkpoints_per_epoch`   | `Float64`                       |  -         |
| `max_batches_per_checkpoint`  | `Int`                           |  -         |
| `num_checkpoints`             | `Int`                           |  -         |

# Description

The neural network goes through `num_checkpoints` series of `n` updates using
batches of size `batch_size` drawn from memory, where `n` is defined as follows:

```
n = min(max_batches_per_checkpoint, ntotal ÷ min_checkpoints_per_epoch)
```

with `ntotal` the total number of batches in memory.

+ `nonvalidity_penalty` is the multiplicative constant of a loss term that
   corresponds to the average probability weight that the network puts on
   invalid actions.
+ `batch_size` is the batch size used for gradient descent.
+ `loss_computation_batch_size` is the batch size that is used to compute
  the loss between each epochs.
+ All rewards are divided by `rewards_renormalization` before the MSE loss is computed.
+ If `use_position_averaging` is set to true, samples in memory that correspond
  to the same board position are averaged together. The merged sample is
  reweighted according to `samples_weighing_policy`.

# AlphaGo Zero Parameters

In the original AlphaGo Zero paper:
+ The batch size for gradient updates is ``2048``.
+ The L2 regularization parameter is set to ``10^{-4}``.
+ Checkpoints are produced every 1000 training steps, which corresponds
  to seeing about 20% of the samples in the memory buffer:
  ``(1000 × 2048) / 10^7  ≈ 0.2``.
+ It is unclear how many checkpoints are taken or how many training steps
  are performed in total.
"""
@kwdef struct LearningParams
  use_gpu :: Bool = false
  use_position_averaging :: Bool = true
  samples_weighing_policy :: SamplesWeighingPolicy
  optimiser :: OptimiserSpec
  l2_regularization :: Float32
  rewards_renormalization :: Float32 = 1f0
  nonvalidity_penalty :: Float32 = 1f0
  batch_size :: Int
  loss_computation_batch_size :: Int
  min_checkpoints_per_epoch :: Int
  max_batches_per_checkpoint :: Int
  num_checkpoints :: Int
end

"""
    MemAnalysisParams

Parameters governing the analysis of the memory buffer
(for debugging and profiling purposes).

| Parameter           | Type           | Default   |
|:--------------------|:---------------|:----------|
| `num_game_stages`   | `Int`          |  -        |

# Explanation

The memory analysis consists in partitioning the memory buffer in
`num_game_stages` parts of equal size, according to the number of
remaining moves until the end of the game for each sample. Then,
the quality of the predictions of the current neural network is
evaluated on each subset (see [`Report.Memory`](@ref)).

This is useful to get an idea of how the neural network performance
varies depending on the game stage (typically, good value estimates for
endgame board positions are available earlier in the training process
than good values for middlegame positions).
"""
@kwdef struct MemAnalysisParams
  num_game_stages :: Int
end

"""
    ProgressiveSimParams

Parameters for progressive simulation budget during training.

| Parameter    | Type   | Description                                      |
|:-------------|:-------|:-------------------------------------------------|
| `sim_min`    | `Int`  | Minimum simulations per turn (early iterations)  |
| `sim_max`    | `Int`  | Maximum simulations per turn (late iterations)   |

# Explanation

Progressive simulation allocates fewer MCTS simulations in early iterations
when the neural network is still weak, and gradually increases the budget
as training progresses. This improves training efficiency by avoiding
wasted computation on unreliable early-stage evaluations.

The simulation budget for iteration `i` out of `num_iters` total iterations
is computed using linear interpolation: `sim_min + (sim_max - sim_min) * i / num_iters`

Reference: "MiniZero: Comparative Analysis of AlphaZero and MuZero on Go, Othello, and Atari Games"
(arXiv:2310.11305)
"""
@kwdef struct ProgressiveSimParams
  sim_min :: Int
  sim_max :: Int
end

"""
Compute the simulation budget for a given iteration.

    compute_sim_budget(params::ProgressiveSimParams, iter::Int, num_iters::Int) :: Int

Returns the number of MCTS simulations to use at iteration `iter`.
"""
function compute_sim_budget(params::ProgressiveSimParams, iter::Int, num_iters::Int)
  t = iter / num_iters
  return round(Int, params.sim_min + (params.sim_max - params.sim_min) * t)
end

"""
    TurnProgressiveSimParams

Parameters for turn-based progressive simulation budget during training.
This varies simulation count based on BOTH the turn number within a game AND
the training iteration.

| Parameter             | Type   | Description                                           |
|:----------------------|:-------|:------------------------------------------------------|
| `turn_sim_min`        | `Int`  | Minimum sims at game start (default: 2)               |
| `turn_sim_target`     | `Int`  | Target sims to ramp up to within game                 |
| `ramp_turns_initial`  | `Int`  | Turns to reach target at iteration 1 (default: 30)    |
| `ramp_turns_final`    | `Int`  | Turns to reach target at final iteration (default: 3) |

# Explanation

This approach is based on the intuition that:
1. Early moves in a game may be more pattern-based and need fewer simulations
2. Mid/late game positions need more tactical depth (more simulations)
3. As training progresses, the network gets better at openings, so we can
   ramp up simulation count faster

At iteration 1, simulations ramp from `turn_sim_min` to `turn_sim_target` over
`ramp_turns_initial` turns. At the final iteration, the ramp completes in just
`ramp_turns_final` turns (nearly all moves at target sims).

The total simulation budget is designed to match baseline constant simulations
when averaged over all iterations.
"""
@kwdef struct TurnProgressiveSimParams
  turn_sim_min :: Int = 2
  turn_sim_target :: Int = 600
  ramp_turns_initial :: Int = 30
  ramp_turns_final :: Int = 3
end

"""
Compute the number of turns needed to ramp up to target sims at a given iteration.

    compute_ramp_turns(params::TurnProgressiveSimParams, iter::Int, num_iters::Int) :: Float64

Returns the number of game turns after which simulations reach the target.
"""
function compute_ramp_turns(params::TurnProgressiveSimParams, iter::Int, num_iters::Int)
  if num_iters <= 1
    return Float64(params.ramp_turns_final)
  end
  t = (iter - 1) / (num_iters - 1)
  return params.ramp_turns_initial - (params.ramp_turns_initial - params.ramp_turns_final) * t
end

"""
Compute the simulation budget for a given turn within a game.

    compute_turn_sim_budget(params::TurnProgressiveSimParams, turn::Int, iter::Int, num_iters::Int) :: Int

Returns the number of MCTS simulations to use at game turn `turn` during iteration `iter`.
"""
function compute_turn_sim_budget(params::TurnProgressiveSimParams, turn::Int, iter::Int, num_iters::Int)
  ramp_turns = compute_ramp_turns(params, iter, num_iters)
  if turn >= ramp_turns
    return params.turn_sim_target
  else
    progress = turn / ramp_turns
    return round(Int, params.turn_sim_min + (params.turn_sim_target - params.turn_sim_min) * progress)
  end
end


for T in [MctsParams, LearningParams]
  Util.generate_update_constructor(T) |> eval
end

#####
##### Utilities
#####

"""
    necessary_samples(ϵ, β) = log(1 / β) / (2 * ϵ^2)

Compute the number of times ``N`` that a random variable
``X \\sim \\text{Ber}(p)`` has to be sampled so that if the
empirical average of ``X`` is greather than
``1/2 + ϵ``, then ``p > 1/2`` with probability at least ``1-β``.

This bound is based on [Hoeffding's inequality
](https://en.wikipedia.org/wiki/Hoeffding%27s_inequality).
"""
necessary_samples(ϵ, β) = log(1 / β) / (2 * ϵ^2)

