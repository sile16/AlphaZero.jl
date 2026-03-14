"""
    EvalMCTS - GPU-friendly MCTS for evaluation with full chance node expansion

Unlike training MCTS (batched_mcts.jl) which samples through chance nodes for speed,
EvalMCTS fully expands all dice outcomes at chance nodes using expectimax.

Key design for GPU/batched inference:
- At chance nodes, expand ALL 21 dice outcomes simultaneously
- Collect all leaf positions needing NN eval into a single large batch
- One GPU kernel call evaluates hundreds of positions at once
- Expectimax backpropagation weights values by dice probability
- Virtual loss ensures batched traversals explore diverse paths
- Bear-off positions use exact table values (no NN needed)

This is fundamentally different from training MCTS:
- Training: narrow/deep, sample chance nodes, millions of fast games for value signal
- Eval: wide/deliberate, full chance expansion, fewer games played as well as possible
"""
module EvalMCTS

using ..AlphaZero: GI, MCTS
using Distributions: Dirichlet
using Statistics

export EvalMctsPlayer, eval_think!, eval_reset!

# ============================================================================
# Tree structures
# ============================================================================

const VIRTUAL_LOSS = 1.0  # Penalty applied to in-flight nodes

mutable struct DecisionNode
    # UCT statistics per action
    actions::Vector{Int}
    P::Vector{Float32}      # Prior probabilities from NN
    W::Vector{Float64}      # Cumulative value
    N::Vector{Int}           # Visit counts
    Vest::Float32            # NN value estimate
end

mutable struct ChanceNode
    # One entry per dice outcome (21 for backgammon)
    probs::Vector{Float64}   # Probability of each outcome
    W::Vector{Float64}       # Cumulative value per outcome
    N::Vector{Int}           # Visit counts per outcome
    outcomes::Vector{Any}    # Dice outcome identifiers
    expanded::Bool           # Whether all outcomes have been visited at least once
end

# ============================================================================
# Eval MCTS Environment
# ============================================================================

mutable struct EvalEnv{S}
    decision_tree::Dict{S, DecisionNode}
    chance_tree::Dict{S, ChanceNode}
    cpuct::Float64
    # Batched oracle: Vector{State} -> Vector{(policy, value)}
    batch_oracle::Any
    # Single oracle fallback
    single_oracle::Any
    # Game spec
    gspec::Any
    # Bear-off evaluator: (game) -> Union{Float64, Nothing}
    bearoff_evaluator::Any
    # Stats
    total_sims::Int
    total_batch_evals::Int
    total_positions_evaluated::Int
end

function EvalEnv(gspec, single_oracle; batch_oracle=nothing, cpuct=1.5, bearoff_evaluator=nothing)
    S = GI.state_type(gspec)
    EvalEnv{S}(
        Dict{S, DecisionNode}(),
        Dict{S, ChanceNode}(),
        cpuct,
        batch_oracle,
        single_oracle,
        gspec,
        bearoff_evaluator,
        0, 0, 0
    )
end

function reset!(env::EvalEnv)
    empty!(env.decision_tree)
    empty!(env.chance_tree)
end

# ============================================================================
# UCT action selection
# ============================================================================

function select_uct_action(node::DecisionNode, cpuct::Float64, noise_ϵ::Float64=0.0, η=nothing)
    n_total = sum(node.N)
    sqrt_total = sqrt(max(n_total, 1))

    best_score = -Inf
    best_idx = 1

    for i in eachindex(node.actions)
        p = node.P[i]
        if noise_ϵ > 0 && η !== nothing
            p = (1 - noise_ϵ) * p + noise_ϵ * η[i]
        end

        q = node.N[i] > 0 ? node.W[i] / node.N[i] : Float64(node.Vest)
        u = cpuct * p * sqrt_total / (1 + node.N[i])
        score = q + u

        if score > best_score
            best_score = score
            best_idx = i
        end
    end
    return best_idx
end

# ============================================================================
# Simulation path tracking
# ============================================================================

struct PathEntry
    state::Any
    action_idx::Int        # -1 for chance nodes
    is_chance::Bool
    chance_outcome_idx::Int # which dice outcome was selected
    player_switch::Bool     # did the player switch after this action?
end

# ============================================================================
# Core: Single simulation with leaf collection
# ============================================================================

"""
Run one MCTS simulation from `game`. Instead of evaluating the leaf immediately,
return the leaf state for batched evaluation.

Returns: (path, leaf_game, leaf_state, is_terminal, terminal_value)
"""
function traverse_to_leaf(env::EvalEnv, game; η=nothing, noise_ϵ=0.0)
    path = PathEntry[]
    depth = 0
    max_depth = 200

    while depth < max_depth
        if GI.game_terminated(game)
            return (path=path, leaf_game=game, leaf_state=GI.current_state(game),
                    is_terminal=true, terminal_value=GI.white_reward(game))
        end

        if GI.is_chance_node(game)
            state = GI.current_state(game)

            # Bear-off table: return exact value if available
            if env.bearoff_evaluator !== nothing
                val = env.bearoff_evaluator(game)
                if val !== nothing
                    return (path=path, leaf_game=game, leaf_state=state,
                            is_terminal=true, terminal_value=val)
                end
            end

            cnode = get(env.chance_tree, state, nothing)

            if cnode === nothing
                # First visit to this chance node: expand all outcomes
                outcomes = GI.chance_outcomes(game)
                probs = [p for (_, p) in outcomes]
                outcome_ids = [o for (o, _) in outcomes]
                cnode = ChanceNode(probs, zeros(length(outcomes)),
                                   zeros(Int, length(outcomes)), outcome_ids, false)
                env.chance_tree[state] = cnode
            end

            # Select outcome with largest visit deficit (proportional to probability)
            total_visits = max(sum(cnode.N), 1)
            best_idx = 1
            best_deficit = -Inf
            for i in eachindex(cnode.probs)
                deficit = cnode.probs[i] - cnode.N[i] / total_visits
                if deficit > best_deficit
                    best_deficit = deficit
                    best_idx = i
                end
            end

            # Apply virtual loss to selected chance outcome
            # (Prevents all sims in a batch from taking the same path)
            cnode.W[best_idx] -= VIRTUAL_LOSS
            cnode.N[best_idx] += 1

            wp = GI.white_playing(game)
            GI.apply_chance!(game, cnode.outcomes[best_idx])
            pswitch = wp != GI.white_playing(game)

            push!(path, PathEntry(state, -1, true, best_idx, pswitch))
            depth += 1
            continue
        end

        # Decision node
        state = GI.current_state(game)
        dnode = get(env.decision_tree, state, nothing)

        if dnode === nothing
            # Bear-off table: return exact value at decision node leaves
            if env.bearoff_evaluator !== nothing
                val = env.bearoff_evaluator(game)
                if val !== nothing
                    return (path=path, leaf_game=game, leaf_state=state,
                            is_terminal=true, terminal_value=val)
                end
            end
            # New leaf — needs NN evaluation
            return (path=path, leaf_game=game, leaf_state=state,
                    is_terminal=false, terminal_value=0.0)
        end

        # Select action via UCT
        ϵ = isempty(path) ? noise_ϵ : 0.0
        action_idx = select_uct_action(dnode, env.cpuct, ϵ, η)
        action = dnode.actions[action_idx]

        # Apply virtual loss: discourage other sims in this batch from same path
        dnode.W[action_idx] -= VIRTUAL_LOSS
        dnode.N[action_idx] += 1

        wp = GI.white_playing(game)
        GI.play!(game, action)
        pswitch = wp != GI.white_playing(game)

        push!(path, PathEntry(state, action_idx, false, 0, pswitch))
        depth += 1
    end

    # Max depth reached
    return (path=path, leaf_game=game, leaf_state=GI.current_state(game),
            is_terminal=true, terminal_value=GI.white_reward(game))
end

"""
Backpropagate a value through the path, removing virtual losses.
"""
function backpropagate!(env::EvalEnv, path, leaf_value)
    v = Float64(leaf_value)

    for i in length(path):-1:1
        entry = path[i]

        # Negate BEFORE update: entry.player_switch means the player changed
        # after this step's action, so v (from child's perspective) must be
        # flipped to this node's player perspective before updating W.
        # (Matches batched_mcts.jl convention: q = pswitch ? -q : q before W += q)
        if entry.player_switch
            v = -v
        end

        if entry.is_chance
            # Remove virtual loss + update: W += VL + v, N unchanged
            cnode = env.chance_tree[entry.state]
            oidx = entry.chance_outcome_idx
            cnode.W[oidx] += VIRTUAL_LOSS + v
            # N already incremented by virtual loss apply, don't increment again
        else
            # Remove virtual loss + update: W += VL + v, N unchanged (VL added 1, we don't add again)
            dnode = env.decision_tree[entry.state]
            aidx = entry.action_idx
            dnode.W[aidx] += VIRTUAL_LOSS + v
            # N already incremented by virtual loss apply, don't increment again
        end
    end
end

"""
Insert a new decision node into the tree from oracle output.
"""
function insert_leaf!(env::EvalEnv, state, actions, P, V)
    dnode = DecisionNode(actions, P, zeros(Float64, length(actions)),
                         zeros(Int, length(actions)), V)
    env.decision_tree[state] = dnode
end

# ============================================================================
# Batched exploration
# ============================================================================

"""
Run `nsims` MCTS simulations, batching leaf evaluations.

At each batch step:
1. Traverse `batch_size` paths to leaves
2. Collect all new leaves needing NN eval
3. Batch evaluate them (GPU-friendly)
4. Insert into tree and backpropagate
"""
function explore!(env::EvalEnv, game, nsims; batch_size=64, noise_ϵ=0.0, noise_α=0.0)
    # Generate Dirichlet noise for root
    η = if noise_α > 0
        n_actions = length(GI.available_actions(game))
        rand(Dirichlet(n_actions, Float64(noise_α)))
    else
        nothing
    end

    sims_done = 0

    while sims_done < nsims
        current_batch = min(batch_size, nsims - sims_done)

        # Phase 1: Traverse to leaves (applies virtual loss along paths)
        traversals = []
        for _ in 1:current_batch
            game_clone = GI.clone(game)
            result = traverse_to_leaf(env, game_clone; η=η, noise_ϵ=noise_ϵ)
            push!(traversals, result)
            env.total_sims += 1
        end

        # Phases 2-4 wrapped in try/finally to guarantee virtual loss cleanup.
        # If batch eval throws, backprop still runs (with value 0.0 for failed leaves),
        # ensuring virtual losses are always removed from the tree.
        try
            # Phase 2: Collect leaves needing evaluation
            new_leaves = []  # (index, state, game)
            for (i, t) in enumerate(traversals)
                if !t.is_terminal && !haskey(env.decision_tree, t.leaf_state)
                    push!(new_leaves, (i, t.leaf_state, t.leaf_game))
                end
            end

            # Phase 3: Batch evaluate
            if !isempty(new_leaves)
                states = [leaf[2] for leaf in new_leaves]

                if env.batch_oracle !== nothing && length(states) > 1
                    # Batched evaluation (GPU-friendly)
                    results = env.batch_oracle(states)
                    env.total_batch_evals += 1
                    env.total_positions_evaluated += length(states)

                    for (j, (_, state, leaf_game)) in enumerate(new_leaves)
                        P, V = results[j]
                        actions = GI.available_actions(leaf_game)
                        insert_leaf!(env, state, actions, P, V)
                    end
                else
                    # Single evaluation fallback
                    for (_, state, leaf_game) in new_leaves
                        P, V = env.single_oracle(state)
                        actions = GI.available_actions(leaf_game)
                        insert_leaf!(env, state, actions, P, V)
                        env.total_positions_evaluated += 1
                    end
                end
            end
        finally
            # Phase 4: Backpropagate (ALWAYS runs — removes virtual losses)
            # Values must be player-relative (from the perspective of the player at the leaf).
            # - Vest from oracle: already player-relative
            # - white_reward / bear-off equity: white-relative → convert to player-relative
            for t in traversals
                if t.is_terminal
                    # terminal_value is white-relative; convert to player-relative
                    leaf_wp = GI.white_playing(t.leaf_game)
                    leaf_v = leaf_wp ? t.terminal_value : -t.terminal_value
                    backpropagate!(env, t.path, leaf_v)
                else
                    dnode = get(env.decision_tree, t.leaf_state, nothing)
                    # Vest is already player-relative from the oracle
                    leaf_v = dnode !== nothing ? Float64(dnode.Vest) : 0.0
                    backpropagate!(env, t.path, leaf_v)
                end
            end
        end

        sims_done += current_batch
    end
end

# ============================================================================
# Policy extraction
# ============================================================================

function policy(env::EvalEnv, game)
    state = GI.current_state(game)

    # If at a chance node, we can't return a policy — caller should handle dice first
    if GI.is_chance_node(game)
        error("Cannot get policy at a chance node. Apply dice first.")
    end

    dnode = get(env.decision_tree, state, nothing)
    if dnode === nothing
        # Unexplored — uniform policy
        actions = GI.available_actions(game)
        return actions, fill(1.0 / length(actions), length(actions))
    end

    total_n = sum(dnode.N)
    if total_n == 0
        return dnode.actions, fill(1.0 / length(dnode.actions), length(dnode.actions))
    end

    π = [n / total_n for n in dnode.N]
    return dnode.actions, π
end

"""Return expectimax value at a chance node (probability-weighted average)."""
function chance_value(env::EvalEnv, game)
    state = GI.current_state(game)
    cnode = get(env.chance_tree, state, nothing)
    cnode === nothing && return 0.0

    total_val = 0.0
    total_prob = 0.0
    for i in eachindex(cnode.probs)
        if cnode.N[i] > 0
            total_val += cnode.probs[i] * (cnode.W[i] / cnode.N[i])
            total_prob += cnode.probs[i]
        end
    end
    return total_prob > 0 ? total_val / total_prob : 0.0
end

# ============================================================================
# Player wrapper
# ============================================================================

struct EvalMctsPlayer
    env::EvalEnv
    niters::Int
    batch_size::Int
end

function EvalMctsPlayer(gspec, single_oracle; batch_oracle=nothing,
                        cpuct=1.5, niters=1600, batch_size=64, bearoff_evaluator=nothing)
    env = EvalEnv(gspec, single_oracle; batch_oracle=batch_oracle, cpuct=cpuct,
                  bearoff_evaluator=bearoff_evaluator)
    EvalMctsPlayer(env, niters, batch_size)
end

function eval_think!(player::EvalMctsPlayer, game)
    explore!(player.env, game, player.niters; batch_size=player.batch_size)
    return policy(player.env, game)
end

function eval_reset!(player::EvalMctsPlayer)
    reset!(player.env)
end

"""Return MCTS stats for diagnostics."""
function stats(player::EvalMctsPlayer)
    env = player.env
    (total_sims=env.total_sims,
     total_batch_evals=env.total_batch_evals,
     total_positions_evaluated=env.total_positions_evaluated,
     decision_nodes=length(env.decision_tree),
     chance_nodes=length(env.chance_tree),
     avg_batch_size=env.total_batch_evals > 0 ?
         env.total_positions_evaluated / env.total_batch_evals : 0.0)
end

end # module
