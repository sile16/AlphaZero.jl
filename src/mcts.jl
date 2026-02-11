"""
A generic, standalone implementation of Monte Carlo Tree Search.
It can be used on any game that implements `GameInterface`
and with any external oracle.

## Oracle Interface

An oracle can be any function or callable object.
  
   oracle(state)

evaluates a single state from the current player's perspective and returns 
a pair `(P, V)` where:

  - `P` is a probability vector on `GI.available_actions(GI.init(gspec, state))`
  - `V` is a scalar estimating the value or win probability for white.
"""
module MCTS

using Distributions: Categorical, Dirichlet

using ..AlphaZero: GI, Util

#####
##### Standard Oracles
#####

"""
    MCTS.RolloutOracle(game_spec::AbstractGameSpec, γ=1.) <: Function

This oracle estimates the value of a position by simulating a random game
from it (a rollout). Moreover, it puts a uniform prior on available actions.
Therefore, it can be used to implement the "vanilla" MCTS algorithm.
"""
struct RolloutOracle{GameSpec} <: Function
  gspec :: GameSpec
  gamma :: Float64
  RolloutOracle(gspec, γ=1.) = new{typeof(gspec)}(gspec, γ)
end

function rollout!(game, γ=1.)
  if GI.is_chance_node(game)
    # At chance node: sample outcome by probability
    outcomes = GI.chance_outcomes(game)
    probs = [p for (_, p) in outcomes]
    idx = Util.rand_categorical(probs)
    outcome, _ = outcomes[idx]
    GI.apply_chance!(game, outcome)
    wr = GI.white_reward(game)
  else
    # At decision node: pick random action
    action = rand(GI.available_actions(game))
    GI.play!(game, action)
    wr = GI.white_reward(game)
  end
  if GI.game_terminated(game)
    return wr
  else
    return wr + γ * rollout!(game, γ)
  end
end

function (r::RolloutOracle)(state)
  g = GI.init(r.gspec, state)
  wp = GI.white_playing(g)
  n = length(GI.available_actions(g))
  P = ones(n) ./ n
  wr = rollout!(g, r.gamma)
  V = wp ? wr : -wr
  return P, V
end

struct RandomOracle{GameSpec}
  gspec :: GameSpec
end

function (r::RandomOracle)(state)
  g = GI.init(r.gspec, state)
  n = length(GI.available_actions(g))
  P = ones(n) ./ n
  V = 0.
  return P, V
end

#####
##### State Statistics
#####

struct ActionStats
  P :: Float32 # Prior probability as given by the oracle
  W :: Float64 # Cumulated Q-value for the action (Q = W/N)
  N :: Int # Number of times the action has been visited
end

struct StateInfo
  stats :: Vector{ActionStats}
  actions :: Vector{Int} # Available actions (cached to avoid recomputing during traversal)
  Vest  :: Float32 # Value estimate given by the oracle
end

Ntot(b::StateInfo) = sum(s.N for s in b.stats)

#####
##### Chance Node Statistics (for stochastic games)
#####

"""
Statistics for a single chance outcome edge.
"""
struct ChanceOutcomeStats
  prob :: Float64   # Probability of this outcome
  W :: Float64      # Cumulated value for this outcome
  N :: Float64      # Visit count (Float64 to support virtual visits for prior integration)
end

"""
Information stored for a chance node in the MCTS tree.
Unlike decision nodes, chance nodes use expectimax aggregation.
"""
mutable struct ChanceNodeInfo
  outcomes :: Vector{ChanceOutcomeStats}  # Stats for each outcome (sorted by prob for progressive)
  outcome_order :: Vector{Int}            # Original indices (for mapping back to game outcomes)
  Vest :: Float32                          # Initial NN value estimate (pre-dice)
  expanded :: Bool                         # Whether all outcomes have been expanded
  num_expanded :: Int                      # Number of outcomes expanded so far (for progressive mode)
end

# Constructor for backward compatibility
function ChanceNodeInfo(outcomes::Vector{ChanceOutcomeStats}, Vest, expanded::Bool)
  return ChanceNodeInfo(outcomes, collect(1:length(outcomes)), Float32(Vest), expanded, expanded ? length(outcomes) : 0)
end

Ntot(c::ChanceNodeInfo) = sum(o.N for o in c.outcomes)
is_expanded(c::ChanceNodeInfo) = c.expanded

#####
##### MCTS Environment
#####

"""
    MCTS.Env(game_spec::AbstractGameSpec, oracle; <keyword args>)

Create and initialize an MCTS environment with a given `oracle`.

## Keyword Arguments

  - `gamma=1.`: the reward discount factor
  - `cpuct=1.`: exploration constant in the UCT formula
  - `noise_ϵ=0., noise_α=1.`: parameters for the dirichlet exploration noise
     (see below)
  - `prior_temperature=1.`: temperature to apply to the oracle's output
     to get the prior probability vector used by MCTS.

## Dirichlet Noise

A naive way to ensure exploration during training is to adopt an ϵ-greedy
policy, playing a random move at every turn instead of using the policy
prescribed by [`MCTS.policy`](@ref) with probability ϵ.
The problem with this naive strategy is that it may lead the player to make
terrible moves at critical moments, thereby biasing the policy evaluation
mechanism.

A superior alternative is to add a random bias to the neural prior for the root
node during MCTS exploration: instead of considering the policy ``p`` output
by the neural network in the UCT formula, one uses ``(1-ϵ)p + ϵη`` where ``η``
is drawn once per call to [`MCTS.explore!`](@ref) from a Dirichlet distribution
of parameter ``α``.
"""
mutable struct Env{State, Oracle}
  # Store (nonterminal) state statistics assuming the white player is to play
  tree :: Dict{State, StateInfo}
  # Store chance node statistics (for stochastic games)
  chance_tree :: Dict{State, ChanceNodeInfo}
  # External oracle to evaluate positions
  oracle :: Oracle
  # Parameters
  gamma :: Float64 # Discount factor
  cpuct :: Float64
  noise_ϵ :: Float64
  noise_α :: Float64
  prior_temperature :: Float64
  chance_mode :: Symbol  # :full, :sampling, :stratified, or :progressive
  # Progressive widening parameters (for :progressive mode)
  progressive_widening_alpha :: Float64  # Expand new outcome when N^α > num_expanded
  prior_virtual_visits :: Float64        # Virtual visits to weight NN prior (like PUCT prior)
  # Performance statistics
  total_simulations :: Int64
  total_nodes_traversed :: Int64
  total_chance_nodes_expanded :: Int64
  # Game specification
  gspec :: GI.AbstractGameSpec

  function Env(gspec, oracle;
      gamma=1., cpuct=1., noise_ϵ=0., noise_α=1., prior_temperature=1.,
      chance_mode=:full, progressive_widening_alpha=0.5, prior_virtual_visits=1.0)
    S = GI.state_type(gspec)
    tree = Dict{S, StateInfo}()
    chance_tree = Dict{S, ChanceNodeInfo}()
    total_simulations = 0
    total_nodes_traversed = 0
    total_chance_nodes_expanded = 0
    new{S, typeof(oracle)}(
      tree, chance_tree, oracle, gamma, cpuct, noise_ϵ, noise_α, prior_temperature,
      chance_mode, progressive_widening_alpha, prior_virtual_visits,
      total_simulations, total_nodes_traversed, total_chance_nodes_expanded, gspec)
  end
end

#####
##### Access and initialize state information
#####

function init_state_info(P, V, prior_temperature, actions::Vector{Int})
  P = Util.apply_temperature(P, prior_temperature)
  stats = [ActionStats(p, 0, 0) for p in P]
  return StateInfo(stats, actions, V)
end

# Returns statistics for the current player, along with a boolean indicating
# whether or not a new node has been created.
function state_info(env, state, actions::Vector{Int})
  if haskey(env.tree, state)
    return (env.tree[state], false)
  else
    (P, V) = env.oracle(state)
    info = init_state_info(P, V, env.prior_temperature, actions)
    env.tree[state] = info
    return (info, true)
  end
end

#####
##### Main algorithm
#####

function uct_scores(info::StateInfo, cpuct, ϵ, η)
  @assert iszero(ϵ) || length(η) == length(info.stats)
  sqrtNtot = sqrt(Ntot(info))
  return map(enumerate(info.stats)) do (i, a)
    Q = a.W / max(a.N, 1)
    P = iszero(ϵ) ? a.P : (1-ϵ) * a.P + ϵ * η[i]
    Q + cpuct * P * sqrtNtot / (a.N + 1)
  end
end

"""Allocation-free UCT action selection: returns the action_id with highest UCT score."""
function best_uct_action(info::StateInfo, cpuct, ϵ, η)
  sqrtNtot = sqrt(Ntot(info))
  best_score = -Inf
  best_id = 1
  @inbounds for (i, a) in enumerate(info.stats)
    Q = a.W / max(a.N, 1)
    P = iszero(ϵ) ? a.P : (1-ϵ) * a.P + ϵ * η[i]
    score = Q + cpuct * P * sqrtNtot / (a.N + 1)
    if score > best_score
      best_score = score
      best_id = i
    end
  end
  return best_id
end

function update_state_info!(env, state, action_id, q)
  stats = env.tree[state].stats
  astats = stats[action_id]
  stats[action_id] = ActionStats(astats.P, astats.W + q, astats.N + 1)
end

# Maximum depth for MCTS simulation to prevent stack overflow in long games
const MAX_SIMULATION_DEPTH = 500

# Run a single MCTS simulation, updating the statistics of all traversed states.
# Return the estimated Q-value for the current player.
# Modifies the state of the game environment.
# Dispatches to chance node handler for stochastic games.
function run_simulation!(env::Env, game; η, root=true, depth=0)
  if GI.game_terminated(game)
    return 0.
  elseif depth >= MAX_SIMULATION_DEPTH
    # Depth limit reached: use oracle value estimate to cut off
    state = GI.current_state(game)
    (_, V) = env.oracle(state)
    return V
  elseif GI.is_chance_node(game)
    return run_simulation_chance!(env, game, η=η, root=root, depth=depth)
  else
    return run_simulation_decision!(env, game, η=η, root=root, depth=depth)
  end
end

# Handle simulation at a decision node (original MCTS logic).
function run_simulation_decision!(env::Env, game; η, root, depth)
  state = GI.current_state(game)
  if haskey(env.tree, state)
    # Existing node: use cached actions (no allocation)
    info = env.tree[state]
    actions = info.actions
  else
    # New node: compute available actions and create tree entry
    actions = GI.available_actions(game)
    (P, V) = env.oracle(state)
    info = init_state_info(P, V, env.prior_temperature, actions)
    env.tree[state] = info
    return info.Vest
  end
  ϵ = root ? env.noise_ϵ : 0.
  action_id = best_uct_action(info, env.cpuct, ϵ, η)
  action = actions[action_id]
  wp = GI.white_playing(game)
  GI.play!(game, action)
  wr = GI.white_reward(game)
  r = wp ? wr : -wr
  pswitch = wp != GI.white_playing(game)
  qnext = run_simulation!(env, game, η=η, root=false, depth=depth+1)
  qnext = pswitch ? -qnext : qnext
  q = r + env.gamma * qnext
  update_state_info!(env, state, action_id, q)
  env.total_nodes_traversed += 1
  return q
end

#####
##### Chance Node Handling (Stochastic MCTS)
#####

"""
Handle simulation at a chance node.

Four modes controlled by env.chance_mode:

:full (default) - Full expectimax:
  - First visit: Query NN, return V
  - Second visit: Expand ALL outcomes at once
  - Subsequent visits: Sample by visit deficit, return weighted average

:sampling - Monte Carlo sampling (faster):
  - First visit: Query NN, return V, initialize stats with prior
  - Subsequent visits: Sample ONE outcome by probability, update running average

:stratified - Stratified sampling (guaranteed coverage):
  - First visit: Query NN, return V, initialize stats with prior
  - Next K visits: Visit each outcome exactly once (random order)
  - After all visited: Sample by probability (same as :sampling)
  - Better variance than pure sampling in early visits

:progressive - Progressive widening with prior integration:
  - First visit: Query NN, return V, initialize with prior virtual visits
  - Subsequent visits: Progressively expand outcomes using N^α > num_expanded
  - Outcomes expanded in order of probability (highest first)
  - Unexpanded outcomes use prior value; expanded use actual backed-up values
  - Prior value treated as virtual visits that get diluted with real samples
"""
function run_simulation_chance!(env::Env, game; η, root, depth)
  state = GI.current_state(game)
  wp = GI.white_playing(game)  # Player who will act AFTER the chance event

  if env.chance_mode == :passthrough
    # Passthrough: sample one outcome, continue as if deterministic.
    # No tree entry, no NN eval at chance node. Equivalent to old deterministic wrapper.
    outcomes = GI.chance_outcomes(game)
    idx = Util.rand_categorical([p for (_, p) in outcomes])
    GI.apply_chance!(game, outcomes[idx][1])
    return run_simulation!(env, game, η=η, root=false, depth=depth)
  elseif env.chance_mode == :sampling
    return run_simulation_chance_sampling!(env, game, state, η, depth)
  elseif env.chance_mode == :stratified
    return run_simulation_chance_stratified!(env, game, state, η, depth)
  elseif env.chance_mode == :progressive
    return run_simulation_chance_progressive!(env, game, state, η, depth)
  else
    return run_simulation_chance_full!(env, game, state, η, depth)
  end
end

"""
Full expectimax mode: expand all outcomes on second visit.
"""
function run_simulation_chance_full!(env::Env, game, state, η, depth)
  if !haskey(env.chance_tree, state)
    # FIRST VISIT: Query NN on pre-dice state for value estimate
    (_, V) = env.oracle(state)
    outcomes = GI.chance_outcomes(game)
    outcome_stats = [ChanceOutcomeStats(prob, 0.0, 0) for (_, prob) in outcomes]
    info = ChanceNodeInfo(outcome_stats, V, false)
    env.chance_tree[state] = info
    return V
  end

  info = env.chance_tree[state]

  if !info.expanded
    # SECOND VISIT: Expand ALL outcomes at once
    return expand_chance_node!(env, game, state, info, η, depth)
  else
    # SUBSEQUENT VISITS: Use expectimax over expanded outcomes
    return expectimax_chance!(env, game, state, info, η, depth)
  end
end

"""
Sampling mode: sample one outcome per visit, use NN prior as starting point.
Much faster than full expectimax - O(1) per visit instead of O(num_outcomes).
"""
function run_simulation_chance_sampling!(env::Env, game, state, η, depth)
  outcomes = GI.chance_outcomes(game)

  if !haskey(env.chance_tree, state)
    # FIRST VISIT: Query NN, use prior value, initialize with virtual visits
    (_, V) = env.oracle(state)
    # Initialize each outcome with prior: mean = V, weight = virtual_N
    # This way: Σ prob × (W/N) = Σ prob × V = V (correct expectation)
    virtual_N = env.prior_virtual_visits
    # Protect against NaN from neural network
    V_safe = isnan(V) ? 0.0f0 : V
    outcome_stats = [ChanceOutcomeStats(prob, V_safe * virtual_N, virtual_N)
                     for (_, prob) in outcomes]
    info = ChanceNodeInfo(outcome_stats, collect(1:length(outcomes)), V_safe, true, length(outcomes))
    env.chance_tree[state] = info
    return V_safe
  end

  info = env.chance_tree[state]

  # Sample ONE outcome according to probability distribution
  probs = [o.prob for o in info.outcomes]
  outcome_idx = Util.rand_categorical(probs)

  outcome, _ = outcomes[outcome_idx]
  game_copy = GI.clone(game)
  GI.apply_chance!(game_copy, outcome)

  if GI.game_terminated(game_copy)
    v = 0.0
  else
    v = run_simulation!(env, game_copy, η=η, root=false, depth=depth+1)
  end

  # Protect against NaN values from simulation
  v_safe = isnan(v) ? 0.0 : v

  # Update statistics for the sampled outcome
  update_chance_outcome!(info, outcome_idx, v_safe)

  # Return expectimax value with NaN protection
  total = 0.0
  for o in info.outcomes
    val = o.W / max(o.N, 1)
    if !isnan(val)
      total += o.prob * val
    end
  end
  return total
end

"""
Stratified sampling mode: visit all outcomes once before random sampling.

This ensures guaranteed coverage of all outcomes before switching to
probability-proportional sampling. Better than pure random sampling because
it reduces variance in the early visits.

Algorithm:
1. First visit: Initialize with NN prior (same as sampling mode)
2. While unvisited outcomes exist: pick a random unvisited outcome
3. After all visited: sample by probability (same as sampling mode)

Uses num_expanded to track how many unique outcomes have been visited.
An outcome is "unvisited" if N == virtual_N (only has prior, no real samples).
"""
function run_simulation_chance_stratified!(env::Env, game, state, η, depth)
  outcomes = GI.chance_outcomes(game)
  virtual_N = env.prior_virtual_visits

  if !haskey(env.chance_tree, state)
    # FIRST VISIT: Query NN, use prior value, initialize with virtual visits
    (_, V) = env.oracle(state)
    # Protect against NaN from neural network
    V_safe = isnan(V) ? 0.0f0 : V
    outcome_stats = [ChanceOutcomeStats(prob, V_safe * virtual_N, virtual_N)
                     for (_, prob) in outcomes]
    # outcome_order maps to original indices (identity for stratified)
    info = ChanceNodeInfo(outcome_stats, collect(1:length(outcomes)), V_safe, false, 0)
    env.chance_tree[state] = info
    return V_safe
  end

  info = env.chance_tree[state]
  num_outcomes = length(info.outcomes)

  # Choose which outcome to visit
  if info.num_expanded < num_outcomes
    # STRATIFIED PHASE: Find unvisited outcomes and pick one randomly
    # An outcome is unvisited if N <= virtual_N (only has prior)
    unvisited_indices = Int[]
    for i in 1:num_outcomes
      if info.outcomes[i].N <= virtual_N
        push!(unvisited_indices, i)
      end
    end

    if !isempty(unvisited_indices)
      # Pick random unvisited outcome
      outcome_idx = unvisited_indices[rand(1:length(unvisited_indices))]
      info.num_expanded += 1  # Track that we're visiting a new outcome
    else
      # All visited, switch to sampling
      info.num_expanded = num_outcomes  # Sync num_expanded
      probs = [o.prob for o in info.outcomes]
      outcome_idx = Util.rand_categorical(probs)
    end
  else
    # SAMPLING PHASE: Sample by probability (all outcomes visited at least once)
    probs = [o.prob for o in info.outcomes]
    outcome_idx = Util.rand_categorical(probs)
  end

  outcome, _ = outcomes[outcome_idx]
  game_copy = GI.clone(game)
  GI.apply_chance!(game_copy, outcome)

  if GI.game_terminated(game_copy)
    v = 0.0
  else
    v = run_simulation!(env, game_copy, η=η, root=false, depth=depth+1)
  end

  # Protect against NaN values from simulation
  v_safe = isnan(v) ? 0.0 : v

  # Update statistics for the visited outcome
  update_chance_outcome!(info, outcome_idx, v_safe)

  # Return expectimax value with NaN protection
  total = 0.0
  for o in info.outcomes
    val = o.W / max(o.N, 1)
    if !isnan(val)
      total += o.prob * val
    end
  end
  return total
end

"""
Progressive widening mode: expand outcomes one at a time, ordered by probability.

Uses progressive widening formula: expand new outcome when N^α > num_expanded
Integrates NN prior with observed values using virtual visits.

Key insight: The NN prior gives us a value estimate before seeing any outcomes.
As we expand and visit outcomes, we gradually replace the prior with real data.
Virtual visits determine how quickly the prior gets "washed out" by real samples.

Reference: Coulom (2007) "Efficient Selectivity and Backup Operators in MCTS"
"""
function run_simulation_chance_progressive!(env::Env, game, state, η, depth)
  outcomes = GI.chance_outcomes(game)
  α = env.progressive_widening_alpha
  virtual_visits = env.prior_virtual_visits

  if !haskey(env.chance_tree, state)
    # FIRST VISIT: Initialize with NN prior, sort outcomes by probability
    (_, V) = env.oracle(state)

    # Sort outcomes by probability (descending) for expansion order
    probs = [p for (_, p) in outcomes]
    sorted_indices = sortperm(probs, rev=true)

    # Initialize stats: all outcomes start with virtual visits from prior
    # W = V * virtual_visits (so initial mean = V)
    # N = virtual_visits (treated as "prior observations")
    outcome_stats = [ChanceOutcomeStats(probs[i], V * virtual_visits, virtual_visits)
                     for i in sorted_indices]

    info = ChanceNodeInfo(
      outcome_stats,
      sorted_indices,      # Store mapping to original indices
      V,
      false,               # Not fully expanded yet
      0                    # No outcomes expanded yet (only have prior)
    )
    env.chance_tree[state] = info
    return V
  end

  info = env.chance_tree[state]
  total_visits = Ntot(info) - length(info.outcomes) * virtual_visits  # Real visits only

  # Progressive widening: should we expand a new outcome?
  num_outcomes = length(info.outcomes)
  should_expand = !info.expanded && (total_visits + 1)^α > info.num_expanded

  if should_expand && info.num_expanded < num_outcomes
    # Expand the next outcome (already sorted by probability)
    return expand_next_progressive!(env, game, outcomes, state, info, η, depth, virtual_visits)
  else
    # Visit an already-expanded outcome (or fall back to highest prob if none expanded)
    return visit_expanded_progressive!(env, game, outcomes, state, info, η, depth, virtual_visits)
  end
end

"""
Expand the next outcome in progressive mode.
Outcomes are expanded in order of probability (highest first).

IMPORTANT: We must mark the outcome as expanded BEFORE the recursive call,
because the recursive simulation can visit the same chance node again.
Without this, multiple recursive calls could all try to expand the same slot.
"""
function expand_next_progressive!(env::Env, game, outcomes, state, info::ChanceNodeInfo, η, depth, virtual_visits)
  # Compute and CLAIM the next slot BEFORE the recursive call
  expand_idx = info.num_expanded + 1  # Next outcome to expand (1-indexed in sorted order)

  # Safety check: ensure we haven't exceeded the outcomes
  if expand_idx > length(info.outcomes)
    # All outcomes already expanded by reentrant calls, just use visit logic
    return visit_expanded_progressive!(env, game, outcomes, state, info, η, depth, virtual_visits)
  end

  # Mark this slot as claimed BEFORE the recursive call (reentrancy safety)
  info.num_expanded += 1
  if info.num_expanded >= length(info.outcomes)
    info.expanded = true
  end
  env.total_chance_nodes_expanded += 1

  original_idx = info.outcome_order[expand_idx]  # Map back to game's outcome order
  outcome, _ = outcomes[original_idx]
  game_copy = GI.clone(game)
  GI.apply_chance!(game_copy, outcome)

  if GI.game_terminated(game_copy)
    v = 0.0
  else
    v = run_simulation!(env, game_copy, η=η, root=false, depth=depth+1)
  end

  # Update statistics AFTER the recursive call
  old = info.outcomes[expand_idx]
  info.outcomes[expand_idx] = ChanceOutcomeStats(old.prob, old.W + v, old.N + 1)

  # Return probability-weighted value estimate
  return compute_progressive_value(info, virtual_visits)
end

"""
Visit an already-expanded outcome using visit-deficit selection.
If no outcomes expanded yet, expand the first (highest probability) one.
"""
function visit_expanded_progressive!(env::Env, game, outcomes, state, info::ChanceNodeInfo, η, depth, virtual_visits)
  if info.num_expanded == 0
    # Edge case: no outcomes expanded yet, expand the first one
    return expand_next_progressive!(env, game, outcomes, state, info, η, depth, virtual_visits)
  end

  # Select among expanded outcomes using visit deficit (like expectimax_chance!)
  # Only consider the first num_expanded outcomes (which are sorted by probability)
  expanded_outcomes = @view info.outcomes[1:info.num_expanded]
  total_expanded_visits = sum(o.N for o in expanded_outcomes) - info.num_expanded * virtual_visits

  # Select outcome with largest deficit: prob - visit_fraction
  # This ensures visits are distributed according to probability
  expanded_probs = [o.prob for o in expanded_outcomes]
  total_expanded_prob = sum(expanded_probs)

  visit_idx = argmax(
    (expanded_outcomes[i].prob / total_expanded_prob) -
    ((expanded_outcomes[i].N - virtual_visits) / max(total_expanded_visits, 1))
    for i in 1:info.num_expanded
  )

  original_idx = info.outcome_order[visit_idx]
  outcome, _ = outcomes[original_idx]
  game_copy = GI.clone(game)
  GI.apply_chance!(game_copy, outcome)

  if GI.game_terminated(game_copy)
    v = 0.0
  else
    v = run_simulation!(env, game_copy, η=η, root=false, depth=depth+1)
  end

  # Update statistics
  old = info.outcomes[visit_idx]
  info.outcomes[visit_idx] = ChanceOutcomeStats(old.prob, old.W + v, old.N + 1)

  return compute_progressive_value(info, virtual_visits)
end

"""
Compute the expected value for a progressive chance node.

For expanded outcomes: use (W / N) as the mean value
For unexpanded outcomes: use the prior value (Vest)

All outcomes are weighted by their probability.
The virtual visits are included in W and N, providing smooth prior integration.
"""
function compute_progressive_value(info::ChanceNodeInfo, virtual_visits)
  total_value = 0.0

  for (i, o) in enumerate(info.outcomes)
    if i <= info.num_expanded
      # Expanded outcome: use actual mean (includes prior via virtual visits)
      mean_value = o.W / max(o.N, 1)
    else
      # Unexpanded outcome: use prior value
      mean_value = info.Vest
    end
    total_value += o.prob * mean_value
  end

  return total_value
end

"""
Expand all chance outcomes and compute expectimax value.
This is called on the SECOND visit to a chance node.
"""
function expand_chance_node!(env::Env, game, state, info::ChanceNodeInfo, η, depth)
  outcomes = GI.chance_outcomes(game)
  wp = GI.white_playing(game)

  new_outcome_stats = ChanceOutcomeStats[]

  for (i, (outcome, prob)) in enumerate(outcomes)
    game_copy = GI.clone(game)
    GI.apply_chance!(game_copy, outcome)

    # Check if game terminated after chance
    if GI.game_terminated(game_copy)
      v = 0.0
    else
      # Recursive simulation from post-chance state
      v = run_simulation!(env, game_copy, η=η, root=false, depth=depth+1)
    end

    push!(new_outcome_stats, ChanceOutcomeStats(prob, v, 1))
  end

  # Compute expectimax value: weighted average over all outcomes
  expectimax_value = sum(s.prob * s.W for s in new_outcome_stats)

  # Mark as expanded and update tree
  info.outcomes = new_outcome_stats
  info.expanded = true
  env.total_chance_nodes_expanded += 1

  return expectimax_value
end

"""
Select the outcome with the largest visit deficit relative to its probability.
Called on subsequent visits after the chance node is expanded.

Instead of random sampling, we select the outcome that is most under-visited
relative to its probability: argmax(prob - visit_fraction). This ensures
the visit distribution converges to match the true probability distribution.
"""
function expectimax_chance!(env::Env, game, state, info::ChanceNodeInfo, η, depth)
  outcomes = GI.chance_outcomes(game)

  # Select outcome with largest delta: probability - actual visit fraction
  total_visits = Ntot(info)
  outcome_idx = argmax(
    o.prob - (o.N / max(total_visits, 1)) for o in info.outcomes
  )

  outcome, _ = outcomes[outcome_idx]
  game_copy = GI.clone(game)
  GI.apply_chance!(game_copy, outcome)

  if GI.game_terminated(game_copy)
    v = 0.0
  else
    v = run_simulation!(env, game_copy, η=η, root=false, depth=depth+1)
  end

  # Update the selected outcome's statistics
  update_chance_outcome!(info, outcome_idx, v)

  # Return expectimax value: weighted average of mean values across ALL outcomes
  return sum(o.prob * (o.W / max(o.N, 1)) for o in info.outcomes)
end

"""
Update statistics for a specific chance outcome.
"""
function update_chance_outcome!(info::ChanceNodeInfo, outcome_idx, value)
  old = info.outcomes[outcome_idx]
  info.outcomes[outcome_idx] = ChanceOutcomeStats(old.prob, old.W + value, old.N + 1)
end

function dirichlet_noise(game, α)
  actions = GI.available_actions(game)
  n = length(actions)
  return rand(Dirichlet(n, α))
end

"""
    MCTS.explore!(env, game, nsims)

Run `nsims` MCTS simulations from the current state.
"""
function explore!(env::Env, game, nsims)
  η = dirichlet_noise(game, env.noise_α)
  for i in 1:nsims
    env.total_simulations += 1
    run_simulation!(env, GI.clone(game), η=η)
  end
end

"""
    MCTS.policy(env, game)

Return the recommended stochastic policy on the current state.

A call to this function must always be preceded by
a call to [`MCTS.explore!`](@ref).
"""
function policy(env::Env, game)
  state = GI.current_state(game)
  info =
    try env.tree[state]
    catch e
      if isa(e, KeyError)
        error("MCTS.explore! must be called before MCTS.policy")
      else
        rethrow(e)
      end
    end
  Ntot = sum(a.N for a in info.stats)
  if Ntot == 0
    # No visits — fall back to uniform policy over available actions
    n = length(info.stats)
    π = fill(1.0 / n, n)
    return info.actions, π
  end
  π = [a.N / Ntot for a in info.stats]
  π ./= sum(π)
  return info.actions, π
end

"""
    MCTS.reset!(env)

Empty the MCTS tree (both decision nodes and chance nodes).
"""
function reset!(env)
  empty!(env.tree)
  empty!(env.chance_tree)
  #GC.gc(true)
end

#####
##### Profiling Utilities
#####

"""
    MCTS.average_exploration_depth(env)

Return the average number of nodes that are traversed during an
MCTS simulation, not counting the root.
"""
function average_exploration_depth(env)
  env.total_simulations == 0 && (return 0)
  return env.total_nodes_traversed / env.total_simulations
end

"""
    MCTS.memory_footprint_per_node(gspec)

Return an estimate of the memory footprint of a single MCTS decision node
for the given game (in bytes).
"""
function memory_footprint_per_node(gspec)
  # The hashtable is at most twice the number of stored elements
  # For every element, a state and a pointer are stored
  size_key = 2 * (GI.state_memsize(gspec) + sizeof(Int))
  n = GI.num_actions(gspec)
  dummy_stats = StateInfo([
    ActionStats(0, 0, 0) for i in 1:n], collect(1:n), 0)
  size_stats = Base.summarysize(dummy_stats)
  return size_key + size_stats
end

"""
    MCTS.memory_footprint_per_chance_node(gspec)

Return an estimate of the memory footprint of a single MCTS chance node
for the given game (in bytes).
"""
function memory_footprint_per_chance_node(gspec)
  num_outcomes = GI.num_chance_outcomes(gspec)
  if num_outcomes == 0
    return 0
  end
  size_key = 2 * (GI.state_memsize(gspec) + sizeof(Int))
  dummy_stats = ChanceNodeInfo(
    [ChanceOutcomeStats(0.0, 0.0, 0.0) for i in 1:num_outcomes],
    collect(1:num_outcomes),  # outcome_order
    Float32(0),               # Vest
    false,                    # expanded
    0)                        # num_expanded
  size_stats = Base.summarysize(dummy_stats)
  return size_key + size_stats
end

"""
    MCTS.approximate_memory_footprint(env)

Return an estimate of the memory footprint of the MCTS tree (in bytes),
including both decision nodes and chance nodes.
"""
function approximate_memory_footprint(env::Env)
  decision_size = memory_footprint_per_node(env.gspec) * length(env.tree)
  chance_size = memory_footprint_per_chance_node(env.gspec) * length(env.chance_tree)
  return decision_size + chance_size
end

# Possibly very slow for large trees
memory_footprint(env::Env) = Base.summarysize(env.tree) + Base.summarysize(env.chance_tree)

end
