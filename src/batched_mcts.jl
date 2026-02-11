#####
##### Batched MCTS Implementation
##### Batches multiple tree traversals before neural network evaluation
#####

"""
Batched MCTS that accumulates leaf nodes and evaluates them in a single batch.

Instead of:
  for sim in 1:N
    traverse → evaluate (blocking) → backpropagate
  end

We do:
  Phase 1: for sim in 1:batch_size
    traverse → record leaf (non-blocking)
  end
  Phase 2: batch_evaluate all leaves
  Phase 3: for sim in 1:batch_size
    backpropagate
  end

This reduces neural network calls from N to ceil(N/batch_size) and uses
larger batch sizes for better GPU utilization.

Chance nodes (stochastic events like dice rolls) are stored as explicit nodes
as passthrough: dice are sampled immediately and traversal continues to decision node
is sampled. On revisit, an outcome is sampled by probability and traversal
continues into the child. No NN evaluation at chance nodes — values propagate
up from child decision nodes.
"""
module BatchedMCTS

using Distributions: Dirichlet
using ..AlphaZero: GI, Util, MCTS

export BatchedEnv, batched_explore!, BatchedMctsPlayer, think, reset_player!, player_temperature

#####
##### Pending simulation state
#####

"""
Stores the state of a simulation that's waiting for neural network evaluation.
"""
mutable struct PendingSimulation{S}
    leaf_state::S           # State to be evaluated
    leaf_actions::Vector{Int}    # Available actions at leaf (for new node creation)
    path::Vector{Tuple{S, Int}}  # (state, action_id) pairs for backpropagation
    rewards::Vector{Float64}     # Rewards accumulated along path
    player_switches::Vector{Bool}  # Whether player switched at each step
    is_new_node::Bool       # Whether this is a new node (needs oracle) or terminal
    terminal_value::Float64 # Value if terminal (no oracle needed)
end

#####
##### Chance node storage
#####

#####
##### Batched MCTS Environment
#####

"""
    BatchedEnv(mcts_env, batch_size)

Wrapper around MCTS.Env that enables batched evaluation.
"""
mutable struct BatchedEnv{S, O}
    env::MCTS.Env{S, O}     # Underlying MCTS environment
    batch_size::Int          # Number of simulations to batch
    pending::Vector{PendingSimulation{S}}  # Simulations waiting for evaluation
    batch_oracle::Any        # Optional: (Vector{S}) -> Vector{(P,V)} for batched GPU eval
    game_pool::Vector{Any}   # Pre-allocated game clones for zero-alloc traversal
    sim_pool::Vector{PendingSimulation{S}}  # Pre-allocated sim objects (reuse vectors)
    # Pre-allocated buffers for batch_evaluate_pending! (avoid per-call allocation)
    _eval_states::Vector{S}
    _eval_indices::Vector{Int}
    # Bear-off evaluator: (game) -> Union{Float64, Nothing}
    bearoff_evaluator::Any
end

function BatchedEnv(env::MCTS.Env{S, O}, batch_size::Int; batch_oracle=nothing, bearoff_evaluator=nothing) where {S, O}
    eval_states = S[]; sizehint!(eval_states, batch_size)
    eval_indices = Int[]; sizehint!(eval_indices, batch_size)
    BatchedEnv{S, O}(env, batch_size, PendingSimulation{S}[], batch_oracle, [],
                     PendingSimulation{S}[], eval_states, eval_indices,
                     bearoff_evaluator)
end

"""Pre-allocate sim pool with sizehinted vectors (called once, reused forever)."""
function _init_sim_pool!(benv::BatchedEnv{S}, game) where S
    dummy_state = GI.current_state(game)
    for _ in 1:benv.batch_size
        path = Tuple{S, Int}[]; sizehint!(path, 12)
        rewards = Float64[]; sizehint!(rewards, 12)
        pswitches = Bool[]; sizehint!(pswitches, 12)
        push!(benv.sim_pool, PendingSimulation{S}(
            dummy_state, Int[], path, rewards, pswitches, false, 0.0
        ))
    end
end

#####
##### Virtual loss for concurrent tree traversal
#####

const VIRTUAL_LOSS = 1  # Penalty applied to in-flight nodes

function apply_virtual_loss!(env::MCTS.Env, state, action_id)
    if haskey(env.tree, state)
        stats = env.tree[state].stats
        astats = stats[action_id]
        # Add virtual loss: decrease W, increase N
        stats[action_id] = MCTS.ActionStats(astats.P, astats.W - VIRTUAL_LOSS, astats.N + 1)
    end
end

function remove_virtual_loss!(env::MCTS.Env, state, action_id)
    if haskey(env.tree, state)
        stats = env.tree[state].stats
        astats = stats[action_id]
        # Remove virtual loss: restore W, keep N (will be updated properly in backprop)
        stats[action_id] = MCTS.ActionStats(astats.P, astats.W + VIRTUAL_LOSS, astats.N - 1)
    end
end

#####
##### Non-blocking tree traversal
#####

"""
In-place traversal: fills a pre-allocated PendingSimulation, reusing its vectors.
Eliminates 3 vector allocations per simulation (path, rewards, player_switches).
Uses get() + inlined virtual loss to reduce Dict lookups from 4 to 1 per node visit.

Chance nodes use passthrough: sample one outcome, apply, continue to decision node.
No tree entry or backprop path entry for chance nodes. Equivalent to old deterministic wrapper.
"""
function traverse_to_leaf!(sim::PendingSimulation{S}, benv::BatchedEnv{S}, game, η) where S
    env = benv.env
    empty!(sim.path)
    empty!(sim.rewards)
    empty!(sim.player_switches)

    depth = 0
    max_depth = 500

    while depth < max_depth
        if GI.game_terminated(game)
            sim.leaf_state = GI.current_state(game)
            sim.leaf_actions = Int[]
            sim.is_new_node = false
            sim.terminal_value = 0.0
            return sim
        end

        if GI.is_chance_node(game)
            # Bear-off table: return exact value if available at chance node
            if benv.bearoff_evaluator !== nothing
                val = benv.bearoff_evaluator(game)
                if val !== nothing
                    sim.leaf_state = GI.current_state(game)
                    sim.leaf_actions = Int[]
                    sim.is_new_node = false
                    sim.terminal_value = val
                    return sim
                end
            end

            # Passthrough: sample one outcome, continue to decision node.
            # No tree entry, no backprop path entry. Equivalent to old deterministic wrapper.
            outcomes = GI.chance_outcomes(game)
            r_val = rand()
            idx = length(outcomes)
            acc = 0.0
            @inbounds for i in eachindex(outcomes)
                acc += outcomes[i][2]
                if r_val <= acc
                    idx = i
                    break
                end
            end
            GI.apply_chance!(game, outcomes[idx][1])
            continue
        end

        state = GI.current_state(game)

        # Single Dict lookup (was: haskey + env.tree[state] = 2 lookups)
        info = get(env.tree, state, nothing)
        if info === nothing
            leaf_actions = GI.available_actions(game)

            # Single-option states (e.g., forced PASS): create trivial tree entry
            # and continue traversal. No oracle evaluation needed since P=[1.0]
            # and the value will come from the child via backpropagation.
            if length(leaf_actions) == 1
                stats = [MCTS.ActionStats(Float32(1.0), Float64(0), 0)]
                info = MCTS.StateInfo(stats, leaf_actions, Float32(0))
                env.tree[state] = info
                # Fall through to normal traversal below (info is now set)
            else
                sim.leaf_state = state
                sim.leaf_actions = leaf_actions
                sim.is_new_node = true
                sim.terminal_value = 0.0
                return sim
            end
        end

        actions = info.actions
        ϵ = isempty(sim.path) ? env.noise_ϵ : 0.0
        action_id = MCTS.best_uct_action(info, env.cpuct, ϵ, η)
        action = actions[action_id]

        # Inline virtual loss (was: apply_virtual_loss! with haskey + env.tree[state] = 2 lookups)
        @inbounds begin
            astats = info.stats[action_id]
            info.stats[action_id] = MCTS.ActionStats(astats.P, astats.W - VIRTUAL_LOSS, astats.N + 1)
        end

        wp = GI.white_playing(game)
        push!(sim.path, (state, action_id))

        GI.play!(game, action)
        wr = GI.white_reward(game)
        r = wp ? wr : -wr
        push!(sim.rewards, r)

        pswitch = wp != GI.white_playing(game)
        push!(sim.player_switches, pswitch)

        depth += 1
        env.total_nodes_traversed += 1
    end

    sim.leaf_state = GI.current_state(game)
    sim.leaf_actions = Int[]
    sim.is_new_node = false
    sim.terminal_value = 0.0
    return sim
end

#####
##### Batch evaluation
#####

"""
Evaluate all pending simulations in a single batch.
Returns vector of (P, V) pairs.

When a `batch_oracle` is set, all states are evaluated in a single call
(e.g. one GPU forward pass via RPC). Otherwise falls back to sequential oracle calls.
"""
function batch_evaluate(benv::BatchedEnv, states::Vector)
    if isempty(states)
        return Tuple{Vector{Float32}, Float32}[]
    end

    if benv.batch_oracle !== nothing
        # Use batched oracle for GPU evaluation (all states in one call)
        return benv.batch_oracle(states)
    else
        # Fall back to sequential oracle calls
        return [benv.env.oracle(state) for state in states]
    end
end

"""Evaluate pending simulations using a batched oracle call."""
function batch_evaluate_pending!(benv::BatchedEnv{S}) where S
    env = benv.env

    # Reuse pre-allocated vectors (avoid per-call allocation)
    states_to_eval = benv._eval_states
    eval_indices = benv._eval_indices
    empty!(states_to_eval)
    empty!(eval_indices)

    for (i, sim) in enumerate(benv.pending)
        if sim.is_new_node
            push!(states_to_eval, sim.leaf_state)
            push!(eval_indices, i)
        end
    end

    if isempty(states_to_eval)
        return
    end

    # Batch evaluate
    results = batch_evaluate(benv, states_to_eval)

    # Initialize tree nodes with results
    for (i, result_idx) in enumerate(eval_indices)
        sim = benv.pending[result_idx]
        P, V = results[i]
        info = MCTS.init_state_info(P, V, env.prior_temperature, sim.leaf_actions)
        env.tree[sim.leaf_state] = info
        sim.terminal_value = V  # Store for backpropagation
    end
end

#####
##### Backpropagation
#####

"""
Backpropagate value through the path, removing virtual losses.
Combined remove_virtual_loss + update_state_info into single Dict lookup per node
(was: 3 lookups per node = haskey + index for remove_VL + index for update).

Chance nodes use passthrough (no tree entries, not in backprop path).
"""
function backpropagate!(benv::BatchedEnv, sim::PendingSimulation)
    env = benv.env

    # Get leaf value (single lookup via get instead of haskey + index)
    if sim.is_new_node
        leaf_info = get(env.tree, sim.leaf_state, nothing)
        q = leaf_info !== nothing ? Float64(leaf_info.Vest) : sim.terminal_value
    else
        q = sim.terminal_value
    end

    # Backpropagate through path in reverse
    for i in length(sim.path):-1:1
        state, action_id = sim.path[i]
        reward = sim.rewards[i]
        pswitch = sim.player_switches[i]

        # Compute Q-value for this step
        q = pswitch ? -q : q
        q = reward + env.gamma * q

        # Decision node: remove virtual loss + update (1 lookup instead of 3)
        # VL removal: W += VL, N -= 1; Update: W += q, N += 1; Net: W += (VL + q), N unchanged
        info = get(env.tree, state, nothing)
        if info !== nothing
            @inbounds begin
                astats = info.stats[action_id]
                info.stats[action_id] = MCTS.ActionStats(astats.P, astats.W + VIRTUAL_LOSS + q, astats.N)
            end
        end
    end
end

#####
##### Main batched exploration
#####

"""
    batched_explore!(benv::BatchedEnv, game, nsims)

Run `nsims` MCTS simulations using batched neural network evaluation.

Simulations are grouped into batches of size `benv.batch_size`.
Each batch:
1. Traverses `batch_size` simulations to leaf nodes
2. Evaluates all leaves in one neural network call
3. Backpropagates all results
"""
function batched_explore!(benv::BatchedEnv, game, nsims)
    env = benv.env

    # Ensure root is in the tree before generating noise (need action count).
    # Without this, if batch_size >= nsims, ALL simulations hit an empty tree,
    # no actions are selected, visit counts stay at 0, and policy() returns NaN.
    state = GI.current_state(game)
    if !GI.game_terminated(game) && !GI.is_chance_node(game) && !haskey(env.tree, state)
        empty!(benv.pending)
        env.total_simulations += 1
        # Use pool sim for root init too
        if isempty(benv.sim_pool)
            _init_sim_pool!(benv, game)
        end
        traverse_to_leaf!(benv.sim_pool[1], benv, GI.clone(game), Float64[])
        push!(benv.pending, benv.sim_pool[1])
        batch_evaluate_pending!(benv)
        for s in benv.pending
            backpropagate!(benv, s)
        end
        nsims -= 1
    end

    # Generate Dirichlet noise using cached actions from tree (avoids GI.available_actions)
    η = if env.noise_α != 0 && haskey(env.tree, state)
        n_actions = length(env.tree[state].actions)
        rand(Dirichlet(n_actions, Float64(env.noise_α)))
    elseif env.noise_α != 0
        n_actions = length(GI.available_actions(game))
        rand(Dirichlet(n_actions, Float64(env.noise_α)))
    else
        Float64[]
    end

    batch_size = benv.batch_size
    num_batches = cld(nsims, batch_size)  # Ceiling division

    # Initialize pools on first call (reused across moves/games)
    if isempty(benv.game_pool)
        benv.game_pool = [GI.clone(game) for _ in 1:batch_size]
    end
    if isempty(benv.sim_pool)
        _init_sim_pool!(benv, game)
    end

    sims_done = 0
    for batch_idx in 1:num_batches
        # Determine batch size for this iteration
        remaining = nsims - sims_done
        current_batch_size = min(batch_size, remaining)

        # Clear pending simulations
        empty!(benv.pending)

        # Phase 1: Traverse to leaves (reuse pool games + pool sims)
        for sim_idx in 1:current_batch_size
            env.total_simulations += 1
            if sim_idx <= length(benv.game_pool)
                game_clone = GI.clone_into!(benv.game_pool[sim_idx], game)
            else
                game_clone = GI.clone(game)
            end
            sim = benv.sim_pool[sim_idx]
            traverse_to_leaf!(sim, benv, game_clone, η)
            push!(benv.pending, sim)
        end

        # Phase 2: Batch evaluate
        batch_evaluate_pending!(benv)

        # Phase 3: Backpropagate
        for sim in benv.pending
            backpropagate!(benv, sim)
        end

        sims_done += current_batch_size
    end
end

#####
##### Convenience functions
#####

"""
Create a batched MCTS environment from a regular one.
"""
function make_batched(env::MCTS.Env, batch_size::Int; batch_oracle=nothing)
    return BatchedEnv(env, batch_size; batch_oracle=batch_oracle)
end

"""
Get the underlying MCTS environment.
"""
get_env(benv::BatchedEnv) = benv.env

"""
Reset the batched environment for a new game.
"""
function reset!(benv::BatchedEnv)
    MCTS.reset!(benv.env)
    empty!(benv.pending)
end

#####
##### Batched MCTS Player
#####

"""
    BatchedMctsPlayer{B} <: AbstractPlayer

A player that uses batched MCTS for improved GPU utilization.
Similar to MctsPlayer but batches neural network evaluations.
"""
struct BatchedMctsPlayer{B} <: Function
    benv::B
    niters::Int
    batch_size::Int
    τ::Any  # Temperature schedule
end

"""
    BatchedMctsPlayer(game_spec, oracle, params; batch_size=32, batch_oracle=nothing, bearoff_evaluator=nothing)

Create a batched MCTS player.

# Arguments
- `game_spec`: Game specification
- `oracle`: Neural network oracle (single-state fallback)
- `params`: MctsParams with MCTS hyperparameters
- `batch_size`: Number of simulations to batch together (default: 32)
- `batch_oracle`: Optional batched oracle `(Vector{S}) -> Vector{(P,V)}` for GPU eval
- `bearoff_evaluator`: Optional `(game) -> Union{Float64, Nothing}` for exact bear-off values at chance nodes
"""
function BatchedMctsPlayer(game_spec, oracle, params; batch_size=32, batch_oracle=nothing, bearoff_evaluator=nothing)
    mcts = MCTS.Env(game_spec, oracle,
        gamma=params.gamma,
        cpuct=params.cpuct,
        noise_ϵ=params.dirichlet_noise_ϵ,
        noise_α=params.dirichlet_noise_α,
        prior_temperature=params.prior_temperature,
        chance_mode=params.chance_mode,
        progressive_widening_alpha=params.progressive_widening_alpha,
        prior_virtual_visits=params.prior_virtual_visits)
    benv = BatchedEnv(mcts, batch_size; batch_oracle=batch_oracle, bearoff_evaluator=bearoff_evaluator)
    return BatchedMctsPlayer(benv, params.num_iters_per_turn, batch_size, params.temperature)
end

function think(p::BatchedMctsPlayer, game)
    batched_explore!(p.benv, game, p.niters)
    return MCTS.policy(p.benv.env, game)
end

function player_temperature(p::BatchedMctsPlayer, game, turn)
    return p.τ[turn]
end

function reset_player!(p::BatchedMctsPlayer)
    reset!(p.benv)
end

end  # module
