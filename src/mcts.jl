"""
A generic, standalone implementation of Monte Carlo Tree Search.
It can be used on any game that implements `GameInterface`
and with any external oracle.

## Oracle Interface

An oracle can be any function or callable object.
  
   oracle(state)

evaluates a single state from the current player's perspective and returns 
a pair `(P, V)` where:

  - `P` is a probability vector on `GI.available_actions(gspec, state)`
  - `V` is a scalar estimating the value or win probability for white.
"""
module MCTS

using Distributions: Categorical, Dirichlet
import Random

using ..AlphaZero: GI, Util

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

function Ntot(b::StateInfo)
  n = 0
  @inbounds for s in b.stats
    n += s.N
  end
  return n
end

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
function ChanceNodeInfo(outcomes::AbstractVector{<:ChanceOutcomeStats}, Vest, expanded::Bool)
  stored_outcomes = collect(ChanceOutcomeStats, outcomes)
  return ChanceNodeInfo(stored_outcomes, collect(1:length(outcomes)), Float32(Vest),
                        expanded, expanded ? length(outcomes) : 0)
end

Ntot(c::ChanceNodeInfo) = sum(o.N for o in c.outcomes)
is_expanded(c::ChanceNodeInfo) = c.expanded

#####
##### MCTS Environment
#####

"""Dictionary-like state index that stores game-defined immutable keys."""
struct StateIndex{State, Key, Value, Spec}
  data :: Dict{Key, Value}
  gspec :: Spec
end

function StateIndex{State, Value}(gspec) where {State, Value}
  sample_state = GI.current_state(GI.init(gspec))
  Key = typeof(GI.state_key(gspec, sample_state))
  return StateIndex{State, Key, Value, typeof(gspec)}(Dict{Key, Value}(), gspec)
end

@inline _state_index_key(index::StateIndex, state) = GI.state_key(index.gspec, state)
Base.getindex(index::StateIndex, state) = index.data[_state_index_key(index, state)]
Base.setindex!(index::StateIndex, value, state) =
  (index.data[_state_index_key(index, state)] = value)
Base.haskey(index::StateIndex, state) = haskey(index.data, _state_index_key(index, state))
Base.get(index::StateIndex, state, default) = get(index.data, _state_index_key(index, state), default)
Base.length(index::StateIndex) = length(index.data)
Base.isempty(index::StateIndex) = isempty(index.data)
Base.empty!(index::StateIndex) = (empty!(index.data); index)

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
mutable struct Env{State, Tree, ChanceTree, Oracle, R, Spec}
  # Store (nonterminal) state statistics assuming the white player is to play
  tree :: Tree
  # Store chance node statistics (for stochastic games)
  chance_tree :: ChanceTree
  # External oracle to evaluate positions
  oracle :: Oracle
  # Random source for search stochasticity (chance sampling and root noise)
  rng :: R
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
  gspec :: Spec
  # 1 / GI.reward_scale(gspec): rewards are multiplied by this so tree Q-values
  # stay on the same [-1,1] scale as NN value outputs (backgammon: 1/3)
  inv_reward_scale :: Float64

  function Env(gspec, oracle;
      gamma=1., cpuct=1., noise_ϵ=0., noise_α=1., prior_temperature=1.,
      chance_mode=:full, progressive_widening_alpha=0.5, prior_virtual_visits=1.0,
      rng::Random.AbstractRNG=Random.Xoshiro(rand(UInt)))
    S = GI.state_type(gspec)
    tree = StateIndex{S, StateInfo}(gspec)
    chance_tree = StateIndex{S, ChanceNodeInfo}(gspec)
    total_simulations = 0
    total_nodes_traversed = 0
    total_chance_nodes_expanded = 0
    new{S, typeof(tree), typeof(chance_tree), typeof(oracle), typeof(rng), typeof(gspec)}(
      tree, chance_tree, oracle, rng, gamma, cpuct, noise_ϵ, noise_α, prior_temperature,
      chance_mode, progressive_widening_alpha, prior_virtual_visits,
      total_simulations, total_nodes_traversed, total_chance_nodes_expanded, gspec,
      1.0 / GI.reward_scale(gspec))
  end
end

#####
##### Access and initialize state information
#####

function init_state_info(P, V, prior_temperature, actions::Vector{Int})
  length(P) == length(actions) || error("Oracle policy/action mismatch: length(P)=$(length(P)) length(actions)=$(length(actions))")
  P = Util.apply_temperature(P, prior_temperature)
  stats = [ActionStats(p, 0, 0) for p in P]
  return StateInfo(stats, actions, V)
end

"""Allocation-free UCT action selection: returns the action_id with highest UCT score."""
function best_uct_action(info::StateInfo, cpuct, ϵ, η)
  sqrtNtot = sqrt(Ntot(info))
  best_score = -Inf
  best_id = 1
  @inbounds for (i, a) in enumerate(info.stats)
    Q = a.W / max(a.N, 1)
    P = (iszero(ϵ) || isempty(η)) ? a.P : (1-ϵ) * a.P + ϵ * η[i]
    score = Q + cpuct * P * sqrtNtot / (a.N + 1)
    if score > best_score
      best_score = score
      best_id = i
    end
  end
  return best_id
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


end
