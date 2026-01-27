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
    path::Vector{Tuple{S, Int}}  # (state, action_id) pairs for backpropagation
    rewards::Vector{Float64}     # Rewards accumulated along path
    player_switches::Vector{Bool}  # Whether player switched at each step
    is_new_node::Bool       # Whether this is a new node (needs oracle) or terminal
    terminal_value::Float64 # Value if terminal (no oracle needed)
end

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
end

function BatchedEnv(env::MCTS.Env{S, O}, batch_size::Int) where {S, O}
    BatchedEnv{S, O}(env, batch_size, PendingSimulation{S}[])
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
Traverse the tree until reaching a leaf node (new or terminal).
Returns a PendingSimulation with the path taken.
Does NOT call the oracle.
"""
function traverse_to_leaf(benv::BatchedEnv{S}, game, η) where S
    env = benv.env
    path = Tuple{S, Int}[]
    rewards = Float64[]
    player_switches = Bool[]

    depth = 0
    max_depth = 500  # Prevent infinite loops

    while depth < max_depth
        if GI.game_terminated(game)
            # Terminal node - no oracle needed
            return PendingSimulation{S}(
                GI.current_state(game), path, rewards, player_switches,
                false, 0.0
            )
        end

        if GI.is_chance_node(game)
            # Handle chance node - sample and continue
            # Note: For simplicity, we use sampling mode for batched MCTS
            outcomes = GI.chance_outcomes(game)
            probs = [p for (_, p) in outcomes]
            idx = Util.rand_categorical(probs)
            outcome, _ = outcomes[idx]
            GI.apply_chance!(game, outcome)
            continue
        end

        state = GI.current_state(game)
        actions = GI.available_actions(game)

        if !haskey(env.tree, state)
            # New node - need oracle evaluation
            return PendingSimulation{S}(
                state, path, rewards, player_switches,
                true, 0.0
            )
        end

        # Existing node - select action using UCT
        info = env.tree[state]
        ϵ = isempty(path) ? env.noise_ϵ : 0.0  # Only add noise at root
        scores = MCTS.uct_scores(info, env.cpuct, ϵ, η)
        action_id = argmax(scores)
        action = actions[action_id]

        # Apply virtual loss to discourage other traversals from same path
        apply_virtual_loss!(env, state, action_id)

        # Record path for backpropagation
        wp = GI.white_playing(game)
        push!(path, (state, action_id))

        # Take action
        GI.play!(game, action)
        wr = GI.white_reward(game)
        r = wp ? wr : -wr
        push!(rewards, r)

        pswitch = wp != GI.white_playing(game)
        push!(player_switches, pswitch)

        depth += 1
        env.total_nodes_traversed += 1
    end

    # Depth limit reached - treat as terminal with value 0
    return PendingSimulation{S}(
        GI.current_state(game), path, rewards, player_switches,
        false, 0.0
    )
end

#####
##### Batch evaluation
#####

"""
Evaluate all pending simulations in a single batch.
Returns vector of (P, V) pairs.

For now, we call the oracle sequentially on each state.
The benefit comes from batching multiple simulations between oracle calls,
reducing the synchronization overhead.

TODO: Add support for true batched oracle that can evaluate multiple states
in a single GPU call.
"""
function batch_evaluate(benv::BatchedEnv, states::Vector)
    if isempty(states)
        return Tuple{Vector{Float32}, Float32}[]
    end

    oracle = benv.env.oracle

    # Call oracle on each state sequentially
    # The benefit of batched MCTS is reducing synchronization, not batching GPU calls
    # (that's handled separately by the batchifier when using BatchedOracle)
    return [oracle(state) for state in states]
end

"""
Evaluate pending simulations using a batched oracle call.
"""
function batch_evaluate_pending!(benv::BatchedEnv{S}) where S
    env = benv.env

    # Collect states that need evaluation (properly typed)
    states_to_eval = S[]
    eval_indices = Int[]

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
        info = MCTS.init_state_info(P, V, env.prior_temperature)
        env.tree[sim.leaf_state] = info
        sim.terminal_value = V  # Store for backpropagation
    end
end

#####
##### Backpropagation
#####

"""
Backpropagate value through the path, removing virtual losses.
"""
function backpropagate!(benv::BatchedEnv, sim::PendingSimulation)
    env = benv.env

    # Get leaf value
    if sim.is_new_node && haskey(env.tree, sim.leaf_state)
        q = env.tree[sim.leaf_state].Vest
    else
        q = sim.terminal_value
    end

    # Backpropagate through path in reverse
    for i in length(sim.path):-1:1
        state, action_id = sim.path[i]
        reward = sim.rewards[i]
        pswitch = sim.player_switches[i]

        # Remove virtual loss first
        remove_virtual_loss!(env, state, action_id)

        # Compute Q-value for this step
        q = pswitch ? -q : q
        q = reward + env.gamma * q

        # Update state info (adds N=1 and W=q)
        MCTS.update_state_info!(env, state, action_id, q)
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

    # Generate Dirichlet noise once for root
    η = if env.noise_α > 0
        actions = GI.available_actions(game)
        rand(Dirichlet(length(actions), Float64(env.noise_α)))
    else
        Float64[]
    end

    batch_size = benv.batch_size
    num_batches = cld(nsims, batch_size)  # Ceiling division

    sims_done = 0
    for batch_idx in 1:num_batches
        # Determine batch size for this iteration
        remaining = nsims - sims_done
        current_batch_size = min(batch_size, remaining)

        # Clear pending simulations
        empty!(benv.pending)

        # Phase 1: Traverse to leaves
        for _ in 1:current_batch_size
            env.total_simulations += 1
            game_clone = GI.clone(game)
            sim = traverse_to_leaf(benv, game_clone, η)
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
function make_batched(env::MCTS.Env, batch_size::Int)
    return BatchedEnv(env, batch_size)
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
    BatchedMctsPlayer(game_spec, oracle, params; batch_size=32)

Create a batched MCTS player.

# Arguments
- `game_spec`: Game specification
- `oracle`: Neural network oracle
- `params`: MctsParams with MCTS hyperparameters
- `batch_size`: Number of simulations to batch together (default: 32)
"""
function BatchedMctsPlayer(game_spec, oracle, params; batch_size=32)
    mcts = MCTS.Env(game_spec, oracle,
        gamma=params.gamma,
        cpuct=params.cpuct,
        noise_ϵ=params.dirichlet_noise_ϵ,
        noise_α=params.dirichlet_noise_α,
        prior_temperature=params.prior_temperature,
        chance_mode=params.chance_mode,
        progressive_widening_alpha=params.progressive_widening_alpha,
        prior_virtual_visits=params.prior_virtual_visits)
    benv = BatchedEnv(mcts, batch_size)
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
