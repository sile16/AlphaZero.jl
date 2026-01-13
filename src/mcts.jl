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
  action = rand(GI.available_actions(game))
  GI.play!(game, action)
  wr = GI.white_reward(game)
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
  N :: Int          # Visit count for this outcome
end

"""
Information stored for a chance node in the MCTS tree.
Unlike decision nodes, chance nodes use expectimax aggregation.
"""
mutable struct ChanceNodeInfo
  outcomes :: Vector{ChanceOutcomeStats}  # Stats for each outcome
  Vest :: Float32                          # Initial NN value estimate (pre-dice)
  expanded :: Bool                         # Whether all outcomes have been expanded
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
  # Performance statistics
  total_simulations :: Int64
  total_nodes_traversed :: Int64
  total_chance_nodes_expanded :: Int64
  # Game specification
  gspec :: GI.AbstractGameSpec

  function Env(gspec, oracle;
      gamma=1., cpuct=1., noise_ϵ=0., noise_α=1., prior_temperature=1.)
    S = GI.state_type(gspec)
    tree = Dict{S, StateInfo}()
    chance_tree = Dict{S, ChanceNodeInfo}()
    total_simulations = 0
    total_nodes_traversed = 0
    total_chance_nodes_expanded = 0
    new{S, typeof(oracle)}(
      tree, chance_tree, oracle, gamma, cpuct, noise_ϵ, noise_α, prior_temperature,
      total_simulations, total_nodes_traversed, total_chance_nodes_expanded, gspec)
  end
end

#####
##### Access and initialize state information
#####

function init_state_info(P, V, prior_temperature)
  P = Util.apply_temperature(P, prior_temperature)
  stats = [ActionStats(p, 0, 0) for p in P]
  return StateInfo(stats, V)
end

# Returns statistics for the current player, along with a boolean indicating
# whether or not a new node has been created.
function state_info(env, state)
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

function update_state_info!(env, state, action_id, q)
  stats = env.tree[state].stats
  astats = stats[action_id]
  stats[action_id] = ActionStats(astats.P, astats.W + q, astats.N + 1)
end

# Run a single MCTS simulation, updating the statistics of all traversed states.
# Return the estimated Q-value for the current player.
# Modifies the state of the game environment.
# Dispatches to chance node handler for stochastic games.
function run_simulation!(env::Env, game; η, root=true)
  if GI.game_terminated(game)
    return 0.
  elseif GI.is_chance_node(game)
    return run_simulation_chance!(env, game, η=η, root=root)
  else
    return run_simulation_decision!(env, game, η=η, root=root)
  end
end

# Handle simulation at a decision node (original MCTS logic).
function run_simulation_decision!(env::Env, game; η, root)
  state = GI.current_state(game)
  actions = GI.available_actions(game)
  info, new_node = state_info(env, state)
  if new_node
    return info.Vest
  else
    ϵ = root ? env.noise_ϵ : 0.
    scores = uct_scores(info, env.cpuct, ϵ, η)
    action_id = argmax(scores)
    action = actions[action_id]
    wp = GI.white_playing(game)
    GI.play!(game, action)
    wr = GI.white_reward(game)
    r = wp ? wr : -wr
    pswitch = wp != GI.white_playing(game)
    qnext = run_simulation!(env, game, η=η, root=false)
    qnext = pswitch ? -qnext : qnext
    q = r + env.gamma * qnext
    update_state_info!(env, state, action_id, q)
    env.total_nodes_traversed += 1
    return q
  end
end

#####
##### Chance Node Handling (Stochastic MCTS)
#####

"""
Handle simulation at a chance node.

First visit: Query NN on pre-dice state, return Vest, store ChanceNodeInfo.
Second visit: Expand ALL outcomes, use expectimax.
Subsequent visits: Sample by probability, continue with expectimax.
"""
function run_simulation_chance!(env::Env, game; η, root)
  state = GI.current_state(game)
  wp = GI.white_playing(game)  # Player who will act AFTER the chance event

  if !haskey(env.chance_tree, state)
    # FIRST VISIT: Query NN on pre-dice state for value estimate
    # Don't expand outcomes yet - just return the value estimate
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
    return expand_chance_node!(env, game, state, info, η)
  else
    # SUBSEQUENT VISITS: Use expectimax over expanded outcomes
    return expectimax_chance!(env, game, state, info, η)
  end
end

"""
Expand all chance outcomes and compute expectimax value.
This is called on the SECOND visit to a chance node.
"""
function expand_chance_node!(env::Env, game, state, info::ChanceNodeInfo, η)
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
      v = run_simulation!(env, game_copy, η=η, root=false)
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
function expectimax_chance!(env::Env, game, state, info::ChanceNodeInfo, η)
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
    v = run_simulation!(env, game_copy, η=η, root=false)
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
  actions = GI.available_actions(game)
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
  π = [a.N / Ntot for a in info.stats]
  π ./= sum(π)
  return actions, π
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
  dummy_stats = StateInfo([
    ActionStats(0, 0, 0) for i in 1:GI.num_actions(gspec)], 0)
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
  dummy_stats = ChanceNodeInfo([
    ChanceOutcomeStats(0, 0, 0) for i in 1:num_outcomes], 0, false)
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
