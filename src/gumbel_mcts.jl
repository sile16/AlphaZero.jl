"""
Gumbel MCTS implementation based on "Policy improvement by planning with Gumbel"
(Danihelka et al., 2022).

This implementation uses:
- Gumbel-max trick for action selection at the root
- Sequential halving (progressive widening) to allocate simulations
- Q-value completion for unvisited actions

## Oracle Interface

Same as standard MCTS - an oracle evaluates states and returns (P, V) where:
- P is a probability vector over available actions
- V is a scalar value estimate for white
"""
module GumbelMCTS

using Distributions: Gumbel
using ..AlphaZero: GI, Util, MCTS

#####
##### State Statistics (reuse from MCTS)
#####

const ActionStats = MCTS.ActionStats
const StateInfo = MCTS.StateInfo

#####
##### Gumbel MCTS Environment
#####

"""
    GumbelMCTS.Env(game_spec, oracle; <keyword args>)

Create and initialize a Gumbel MCTS environment.

## Keyword Arguments

  - `gamma=1.`: reward discount factor
  - `num_simulations`: total simulation budget per move
  - `max_considered_actions=16`: maximum number of actions to consider (top-k)
  - `prior_temperature=1.`: temperature to apply to oracle's policy output
  - `c_scale=1.`: scaling factor for completed Q-values (from paper)
  - `c_visit=50.`: visit count offset for completed Q-values (from paper)
"""
mutable struct Env{State, Oracle}
  # Store state statistics (can be shared/reused)
  tree :: Dict{State, StateInfo}
  # External oracle
  oracle :: Oracle
  # Parameters
  gamma :: Float64
  num_simulations :: Int
  max_considered_actions :: Int
  prior_temperature :: Float64
  c_scale :: Float64  # Scaling for Q completion
  c_visit :: Float64  # Visit offset for Q completion
  # Performance statistics
  total_simulations :: Int64
  total_nodes_traversed :: Int64
  # Game specification
  gspec :: GI.AbstractGameSpec
  # Root state info (cached for current position)
  root_gumbels :: Vector{Float64}
  root_q_values :: Vector{Float64}
  root_visits :: Vector{Int}

  function Env(gspec, oracle;
      gamma=1.,
      num_simulations=50,
      max_considered_actions=16,
      prior_temperature=1.,
      c_scale=1.,
      c_visit=50.)
    S = GI.state_type(gspec)
    tree = Dict{S, StateInfo}()
    n_actions = GI.num_actions(gspec)
    new{S, typeof(oracle)}(
      tree, oracle, gamma, num_simulations, max_considered_actions,
      prior_temperature, c_scale, c_visit, 0, 0, gspec,
      Float64[], Float64[], Int[])
  end
end

#####
##### Gumbel Sampling
#####

"""
Sample n values from Gumbel(0, 1) distribution.
"""
function sample_gumbel(n::Int)
  return rand(Gumbel(0, 1), n)
end

#####
##### Q-value Completion
#####

"""
    completed_q(q_sum, n_visits, total_visits, value_estimate, c_scale, c_visit)

Compute the "completed" Q-value that interpolates between the value estimate
(for unvisited actions) and the empirical Q-value (for visited actions).

From the paper: Q_completed = V + (N(a) / (c_visit + N_total)) * (Q(a) - V)

This ensures unvisited actions have Q ≈ V, while visited actions use their
empirical Q-values.
"""
function completed_q(q_sum::Float64, n_visits::Int, total_visits::Int,
                     value_estimate::Float64, c_scale::Float64, c_visit::Float64)
  if n_visits == 0
    return value_estimate
  end
  q_empirical = q_sum / n_visits
  # Interpolation factor based on visit count
  mix = (c_scale * n_visits) / (c_visit + total_visits)
  return value_estimate + mix * (q_empirical - value_estimate)
end

#####
##### Sequential Halving
#####

"""
    compute_simulation_budget(num_actions, total_sims)

Compute how to allocate simulations across phases of sequential halving.
Returns a vector of (num_actions_this_phase, sims_per_action) tuples.
"""
function compute_sequential_halving_schedule(num_actions::Int, total_sims::Int)
  schedule = Tuple{Int, Int}[]
  n = num_actions
  remaining_sims = total_sims

  while n > 1 && remaining_sims > 0
    # Number of phases remaining (including this one)
    phases_remaining = max(1, ceil(Int, log2(n)))
    # Simulations per action this phase
    sims_per_action = max(1, remaining_sims ÷ (n * phases_remaining))
    push!(schedule, (n, sims_per_action))
    remaining_sims -= n * sims_per_action
    n = max(1, n ÷ 2)  # Halve the number of actions
  end

  # If we still have budget, give remaining to last action
  if remaining_sims > 0 && !isempty(schedule)
    n_last, sims_last = schedule[end]
    schedule[end] = (n_last, sims_last + remaining_sims ÷ max(1, n_last))
  end

  return schedule
end

#####
##### State Info Access
#####

function init_state_info(P, V, prior_temperature)
  P = Util.apply_temperature(P, prior_temperature)
  stats = [ActionStats(p, 0, 0) for p in P]
  return StateInfo(stats, V)
end

function state_info(env::Env, state)
  if haskey(env.tree, state)
    return (env.tree[state], false)
  else
    (P, V) = env.oracle(state)
    info = init_state_info(P, V, env.prior_temperature)
    env.tree[state] = info
    return (info, true)
  end
end

#####
##### UCT for Non-Root Nodes
#####

"""
UCT scores for internal nodes (same as standard MCTS).
"""
function uct_scores(info::StateInfo, cpuct::Float64)
  Ntot = sum(a.N for a in info.stats)
  sqrtNtot = sqrt(Ntot)
  return map(info.stats) do a
    Q = a.N > 0 ? a.W / a.N : 0.0
    Q + cpuct * a.P * sqrtNtot / (a.N + 1)
  end
end

#####
##### Simulation
#####

"""
    run_simulation!(env, game, root_action_idx)

Run a single simulation starting with the given root action.
After the root action, uses UCT for action selection in the subtree.
Returns the Q-value for the root player.
"""
function run_simulation!(env::Env, game, root_action_idx::Int; cpuct::Float64=1.0)
  state = GI.current_state(game)
  actions = GI.available_actions(game)

  # Play the root action
  root_action = actions[root_action_idx]
  wp = GI.white_playing(game)
  GI.play!(game, root_action)
  wr = GI.white_reward(game)
  r = wp ? wr : -wr

  if GI.game_terminated(game)
    return r
  end

  # Continue with UCT in subtree
  pswitch = wp != GI.white_playing(game)
  qnext = run_subtree_simulation!(env, game, cpuct)
  qnext = pswitch ? -qnext : qnext
  q = r + env.gamma * qnext

  # Update root action statistics
  update_root_stats!(env, root_action_idx, q)

  env.total_simulations += 1
  return q
end

"""
Run simulation in subtree using standard UCT.
"""
function run_subtree_simulation!(env::Env, game, cpuct::Float64)
  if GI.game_terminated(game)
    return 0.0
  end

  state = GI.current_state(game)
  actions = GI.available_actions(game)
  info, new_node = state_info(env, state)

  if new_node
    return info.Vest
  end

  # UCT selection
  scores = uct_scores(info, cpuct)
  action_id = argmax(scores)
  action = actions[action_id]

  wp = GI.white_playing(game)
  GI.play!(game, action)
  wr = GI.white_reward(game)
  r = wp ? wr : -wr

  pswitch = wp != GI.white_playing(game)
  qnext = run_subtree_simulation!(env, game, cpuct)
  qnext = pswitch ? -qnext : qnext
  q = r + env.gamma * qnext

  # Update state info
  update_state_info!(env, state, action_id, q)
  env.total_nodes_traversed += 1

  return q
end

function update_state_info!(env::Env, state, action_id::Int, q::Float64)
  stats = env.tree[state].stats
  astats = stats[action_id]
  stats[action_id] = ActionStats(astats.P, astats.W + q, astats.N + 1)
end

function update_root_stats!(env::Env, action_idx::Int, q::Float64)
  env.root_q_values[action_idx] += q
  env.root_visits[action_idx] += 1
end

#####
##### Main Exploration Function
#####

"""
    explore!(env, game)

Run Gumbel MCTS from the current game state using sequential halving.
"""
function explore!(env::Env, game)
  if GI.game_terminated(game)
    return
  end

  state = GI.current_state(game)
  actions = GI.available_actions(game)
  n_actions = length(actions)

  # Get prior and value from oracle
  info, _ = state_info(env, state)
  priors = [a.P for a in info.stats]
  value_est = info.Vest

  # Initialize root tracking
  env.root_gumbels = sample_gumbel(n_actions)
  env.root_q_values = zeros(Float64, n_actions)
  env.root_visits = zeros(Int, n_actions)

  # Compute initial Gumbel scores: σ(a) = log(π(a)) + g(a)
  # Use log of prior (add small epsilon to avoid -Inf)
  log_priors = log.(max.(priors, 1e-10))
  initial_scores = log_priors .+ env.root_gumbels

  # Select top-k actions
  k = min(n_actions, env.max_considered_actions)
  sorted_indices = sortperm(initial_scores, rev=true)
  considered_actions = sorted_indices[1:k]

  # Compute sequential halving schedule
  schedule = compute_sequential_halving_schedule(k, env.num_simulations)

  # Run sequential halving
  cpuct = 1.0  # UCT constant for subtree
  for (n_actions_phase, sims_per_action) in schedule
    if length(considered_actions) <= 1
      break
    end

    # Simulate each considered action
    for action_idx in considered_actions
      for _ in 1:sims_per_action
        run_simulation!(env, GI.clone(game), action_idx, cpuct=cpuct)
      end
    end

    # Compute completed Q-values and updated scores
    total_visits = sum(env.root_visits)
    scores = map(considered_actions) do idx
      q_completed = completed_q(
        env.root_q_values[idx],
        env.root_visits[idx],
        total_visits,
        Float64(value_est),
        env.c_scale,
        env.c_visit
      )
      # Updated score: σ(a) + Q_completed(a)
      initial_scores[idx] + q_completed
    end

    # Keep top half
    n_keep = max(1, length(considered_actions) ÷ 2)
    sorted_by_score = sortperm(scores, rev=true)
    considered_actions = considered_actions[sorted_by_score[1:n_keep]]
  end
end

#####
##### Policy Extraction
#####

"""
    policy(env, game)

Return the recommended stochastic policy based on visit counts.
A call to `explore!` must precede this.
"""
function policy(env::Env, game)
  actions = GI.available_actions(game)

  if isempty(env.root_visits)
    error("GumbelMCTS.explore! must be called before policy")
  end

  # Policy proportional to visit counts (same as standard MCTS)
  total = sum(env.root_visits)
  if total == 0
    # Fallback to uniform if no simulations ran
    n = length(actions)
    return actions, ones(n) ./ n
  end

  π = env.root_visits ./ total
  return actions, π
end

"""
    selected_action(env, game)

Return the action selected by Gumbel MCTS (highest Gumbel score + Q).
This is the action that would be played with temperature=0.
"""
function selected_action(env::Env, game)
  actions = GI.available_actions(game)
  state = GI.current_state(game)
  info = env.tree[state]
  value_est = info.Vest
  priors = [a.P for a in info.stats]

  # Compute final scores
  log_priors = log.(max.(priors, 1e-10))
  total_visits = sum(env.root_visits)

  final_scores = map(1:length(actions)) do idx
    q_completed = completed_q(
      env.root_q_values[idx],
      env.root_visits[idx],
      total_visits,
      Float64(value_est),
      env.c_scale,
      env.c_visit
    )
    log_priors[idx] + env.root_gumbels[idx] + q_completed
  end

  return actions[argmax(final_scores)]
end

#####
##### Utilities
#####

"""
    reset!(env)

Empty the MCTS tree and reset root state.
"""
function reset!(env::Env)
  empty!(env.tree)
  env.root_gumbels = Float64[]
  env.root_q_values = Float64[]
  env.root_visits = Int[]
end

"""
    average_exploration_depth(env)

Return the average number of nodes traversed per simulation.
"""
function average_exploration_depth(env::Env)
  env.total_simulations == 0 && return 0
  return env.total_nodes_traversed / env.total_simulations
end

"""
    memory_footprint_per_node(gspec)

Estimate memory per node (same as standard MCTS).
"""
function memory_footprint_per_node(gspec)
  return MCTS.memory_footprint_per_node(gspec)
end

"""
    approximate_memory_footprint(env)

Estimate total tree memory footprint.
"""
function approximate_memory_footprint(env::Env)
  return memory_footprint_per_node(env.gspec) * length(env.tree)
end

end  # module GumbelMCTS
