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
import Random
using ..AlphaZero: GI, Util, MCTS

export BatchedEnv, SearchMetrics, search_metrics, take_search_metrics!,
       batched_explore!, BatchedMctsPlayer, think, reset_player!, player_temperature

const EMPTY_INT_VEC = Int[]  # Shared empty vector for terminal/bearoff leaf_actions

"""
Low-overhead, cumulative search counters for one `BatchedEnv`.

These are deliberately plain integers rather than atomics: each self-play worker owns
its player. Counters are updated in the existing traversal/evaluation hot paths and
are only sampled once per completed game, so observability adds no per-simulation
logging or allocation.
"""
mutable struct SearchMetrics
    simulations::Int64
    tree_hits::Int64
    tree_misses::Int64
    nn_evaluations::Int64
    oracle_calls::Int64
    bearoff_hits::Int64
    bearoff_misses::Int64
    search_ns::Int64
    max_depth::Int64
end

SearchMetrics() = SearchMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)

function _reset!(m::SearchMetrics)
    m.simulations = 0
    m.tree_hits = 0
    m.tree_misses = 0
    m.nn_evaluations = 0
    m.oracle_calls = 0
    m.bearoff_hits = 0
    m.bearoff_misses = 0
    m.search_ns = 0
    m.max_depth = 0
    return m
end

_snapshot(m::SearchMetrics) = SearchMetrics(
    m.simulations, m.tree_hits, m.tree_misses, m.nn_evaluations,
    m.oracle_calls, m.bearoff_hits, m.bearoff_misses, m.search_ns, m.max_depth)

#####
##### Pending simulation state
#####

"""
Stores the state of a simulation that's waiting for neural network evaluation.

Chance-node handling (only active under `chance_mode == :exact_expectation`, an
EVAL-ONLY mode) reuses this struct via the `kind` field:

  - `:normal`        — decision leaf / terminal / bearoff leaf (the passthrough
                       and training path always uses this; behaviour unchanged).
  - `:chance_first`  — FIRST visit to a chance node: the chance node itself is the
                       leaf; its pre-dice NN value becomes `ChanceNodeInfo.Vest`.
  - `:chance_expand` — SECOND visit: expand ALL outcomes. Outcomes with a known
                       value (terminal/bearoff/transposition) are filled during
                       traversal; outcomes needing an NN eval are recorded in
                       `expand_states`/`expand_actions`/`expand_out_idx` and filled
                       in the batch. The chance node's expectation is the leaf value.

`is_chance` runs parallel to `path`: a `true` entry marks a chance OUTCOME edge
`(chance_state, outcome_idx)` (no reward, no gamma, no sign flip; the value
propagated up is the chance node's probability-weighted expectation). A `false`
entry marks an ordinary decision edge (existing reward/gamma/pswitch backup).
"""
mutable struct PendingSimulation{S}
    leaf_state::S           # State to be evaluated
    leaf_actions::Vector{Int}    # Available actions at leaf (for new node creation)
    path::Vector{Tuple{S, Int}}  # (state, action_id) pairs for backpropagation
    rewards::Vector{Float64}     # Rewards accumulated along path
    player_switches::Vector{Bool}  # Whether player switched at each step
    is_new_node::Bool       # Whether this is a new node (needs oracle) or terminal
    terminal_value::Float64 # Value if terminal (no oracle needed)
    # ── Chance-node fields (only used under :exact_expectation) ──────────────
    kind::Symbol                       # :normal, :chance_first, :chance_expand
    is_chance::Vector{Bool}            # parallel to path: true => chance outcome edge
    chance_state::S                    # the chance node state (first/expand)
    expand_states::Vector{S}           # outcome-child states needing an NN eval
    expand_actions::Vector{Vector{Int}}  # child actions for each expand state
    expand_out_idx::Vector{Int}        # outcome index (into chance_tree) for each
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
mutable struct BatchedEnv{S, O, R, BO, BOWA, BE, G}
    env::MCTS.Env{S, O, R}   # Underlying MCTS environment
    batch_size::Int          # Number of simulations to batch
    pending::Vector{PendingSimulation{S}}  # Simulations waiting for evaluation
    batch_oracle::BO         # Optional: (Vector{S}) -> Vector{(P,V)} for batched GPU eval
    batch_oracle_with_actions::BOWA  # Optional: (Vector{S}, Vector{Vector{Int}}) -> Vector{(P,V)}
    game_pool::Vector{G}     # Pre-allocated game clones (filled lazily on first use)
    sim_pool::Vector{PendingSimulation{S}}  # Pre-allocated sim objects (reuse vectors)
    # Pre-allocated buffers for batch_evaluate_pending! (avoid per-call allocation)
    _eval_states::Vector{S}
    _eval_actions::Vector{Vector{Int}}
    _eval_indices::Vector{Int}
    # Routing buffers for :exact_expectation eval dispatch (stay empty otherwise, so
    # the passthrough/training path never touches them). Parallel to _eval_states:
    #   _eval_route_sim[k] = index of the pending sim that owns eval state k
    #   _eval_route_out[k] = 0 => normal decision leaf; -1 => chance_first (Vest);
    #                        >0 => chance_expand outcome index to fill
    _eval_route_sim::Vector{Int}
    _eval_route_out::Vector{Int}
    # Bear-off evaluator: (game) -> Union{Float64, Nothing}
    bearoff_evaluator::BE
    # 1 / GI.reward_scale(gspec): rewards are multiplied by this so tree Q-values
    # stay on the same [-1,1] scale as NN value outputs (backgammon: 1/3)
    inv_reward_scale::Float64
    metrics::SearchMetrics
end

function BatchedEnv(env::MCTS.Env{S, O, R}, batch_size::Int;
                    batch_oracle::BO=nothing, batch_oracle_with_actions::BOWA=nothing,
                    bearoff_evaluator::BE=nothing) where {S, O, R, BO, BOWA, BE}
    game_type = typeof(GI.init(env.gspec))
    eval_states = S[]; sizehint!(eval_states, batch_size)
    eval_actions = Vector{Int}[]; sizehint!(eval_actions, batch_size)
    eval_indices = Int[]; sizehint!(eval_indices, batch_size)
    eval_route_sim = Int[]; sizehint!(eval_route_sim, batch_size)
    eval_route_out = Int[]; sizehint!(eval_route_out, batch_size)
    BatchedEnv{S, O, R, BO, BOWA, BE, game_type}(env, batch_size, PendingSimulation{S}[],
                              batch_oracle, batch_oracle_with_actions,
                              game_type[], PendingSimulation{S}[], eval_states,
                              eval_actions, eval_indices, eval_route_sim, eval_route_out,
                              bearoff_evaluator,
                              1.0 / GI.reward_scale(env.gspec), SearchMetrics())
end

"""Pre-allocate sim pool with sizehinted vectors (called once, reused forever)."""
function _init_sim_pool!(benv::BatchedEnv{S}, game) where S
    dummy_state = GI.current_state(game)
    for _ in 1:benv.batch_size
        path = Tuple{S, Int}[]; sizehint!(path, 12)
        rewards = Float64[]; sizehint!(rewards, 12)
        pswitches = Bool[]; sizehint!(pswitches, 12)
        is_chance = Bool[]; sizehint!(is_chance, 12)
        expand_states = S[]; sizehint!(expand_states, 24)
        expand_actions = Vector{Int}[]; sizehint!(expand_actions, 24)
        expand_out_idx = Int[]; sizehint!(expand_out_idx, 24)
        push!(benv.sim_pool, PendingSimulation{S}(
            dummy_state, Int[], path, rewards, pswitches, false, 0.0,
            :normal, is_chance, dummy_state, expand_states, expand_actions, expand_out_idx
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
##### Exact-expectation chance-node helpers (EVAL-ONLY :exact_expectation mode)
#####

"""
Probability-weighted expectation over a chance node's outcome children:

    Σ o.prob * (o.N > 0 ? o.W/o.N : Vest)

The `N==0 → Vest` fallback is load-bearing for batched-wave correctness: a
follower's backprop may read this before the expander has filled an outcome
(outcome still at N=0); using 0 there would silently drag the value toward 0 and
that bias would be baked into the parent decision node's cumulative W.
"""
function chance_expectation(cinfo::MCTS.ChanceNodeInfo)
    Vest = Float64(cinfo.Vest)
    e = 0.0
    @inbounds for o in cinfo.outcomes
        e += o.prob * (o.N > 0 ? o.W / o.N : Vest)
    end
    return e
end

"""
Resolve the player-relative value of a post-chance outcome child, or return
`nothing` if it needs an NN eval (which enqueues it on `sim` for the batch).

`apply_chance` does NOT switch the player-to-move, so the child's value is
already on the chance node's (mover-relative) scale — it is combined UNFLIPPED.

  - terminal            → 0.0  (dice never terminate backgammon; matches the
                                recursive `expand_chance_node!` convention)
  - bearoff-covered     → exact value (white-relative → player-relative)
  - transposition (in env.tree with real visits) → its running mean W/N
  - otherwise           → nothing; enqueued for an NN eval of the child
"""
function resolve_known_outcome_value!(benv::BatchedEnv, sim::PendingSimulation, child, out_idx::Int)
    env = benv.env
    if GI.game_terminated(child)
        return 0.0
    end
    if benv.bearoff_evaluator !== nothing
        val = benv.bearoff_evaluator(child)
        if val !== nothing
            benv.metrics.bearoff_hits += 1
            wp = GI.white_playing(child)
            return wp ? Float64(val) : -Float64(val)
        end
        benv.metrics.bearoff_misses += 1
    end
    cstate = GI.current_state(child)
    cinfo = get(env.tree, cstate, nothing)
    if cinfo !== nothing
        ntot = MCTS.Ntot(cinfo)
        if ntot > 0
            wsum = 0.0
            @inbounds for a in cinfo.stats; wsum += a.W; end
            return wsum / ntot
        else
            return Float64(cinfo.Vest)
        end
    end
    # Needs an NN eval: enqueue the child (a decision node) for the batch.
    cactions = GI.available_actions(GI.spec(child), cstate)
    push!(sim.expand_states, cstate)
    push!(sim.expand_actions, cactions)
    push!(sim.expand_out_idx, out_idx)
    return nothing
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
    # Chance-node bookkeeping reset (cheap; vectors stay empty on the passthrough path)
    empty!(sim.is_chance)
    empty!(sim.expand_states)
    empty!(sim.expand_actions)
    empty!(sim.expand_out_idx)
    sim.kind = :normal

    depth = 0
    max_depth = 500

    while depth < max_depth
        if GI.game_terminated(game)
            sim.leaf_state = GI.current_state(game)
            sim.leaf_actions = EMPTY_INT_VEC
            sim.is_new_node = false
            sim.terminal_value = 0.0
            return sim
        end

        if GI.is_chance_node(game)
            # Bear-off table: return exact value if available at chance node
            if benv.bearoff_evaluator !== nothing
                val = benv.bearoff_evaluator(game)
                if val !== nothing
                    benv.metrics.bearoff_hits += 1
                    # Bear-off returns white-relative equity; convert to player-relative
                    wp = GI.white_playing(game)
                    player_val = wp ? val : -val
                    sim.leaf_state = GI.current_state(game)
                    sim.leaf_actions = Int[]
                    sim.is_new_node = false
                    sim.terminal_value = player_val
                    return sim
                end
                benv.metrics.bearoff_misses += 1
            end

            # EVAL-ONLY exact-expectation chance node (first-class chance_tree entries).
            # Gated on :exact_expectation ONLY — :full / :passthrough / training stay
            # on the passthrough path below (byte-identical).
            if env.chance_mode == :exact_expectation
                cstate = GI.current_state(game)
                cinfo = get(env.chance_tree, cstate, nothing)
                if cinfo === nothing
                    # FIRST VISIT: claim the node (so concurrent same-wave sims do not
                    # both take the first-visit path). The chance node itself is the
                    # leaf; its pre-dice NN value (filled in the batch) becomes Vest.
                    outcomes = GI.chance_outcomes(game)
                    ostats = [MCTS.ChanceOutcomeStats(prob, 0.0, 0.0) for (_, prob) in outcomes]
                    env.chance_tree[cstate] = MCTS.ChanceNodeInfo(ostats, Float32(0), false)
                    sim.kind = :chance_first
                    sim.chance_state = cstate
                    sim.leaf_state = cstate
                    sim.leaf_actions = EMPTY_INT_VEC
                    sim.is_new_node = false
                    sim.terminal_value = 0.0
                    return sim
                elseif !cinfo.expanded
                    # SECOND VISIT: expand ALL outcomes. Mark expanded FIRST so
                    # same-wave followers take the expanded path. Known-value outcomes
                    # are filled now; NN-needed outcomes are enqueued for the batch.
                    cinfo.expanded = true
                    cinfo.num_expanded = length(cinfo.outcomes)
                    env.total_chance_nodes_expanded += 1
                    outcomes = GI.chance_outcomes(game)
                    wp_chance = GI.white_playing(game)
                    @inbounds for k in eachindex(outcomes)
                        child = GI.clone(game)
                        GI.apply_chance!(child, outcomes[k][1])
                        # apply_chance must not switch the player-to-move.
                        @assert GI.game_terminated(child) || GI.white_playing(child) == wp_chance
                        v = resolve_known_outcome_value!(benv, sim, child, k)
                        if v !== nothing
                            o = cinfo.outcomes[k]
                            cinfo.outcomes[k] = MCTS.ChanceOutcomeStats(o.prob, o.W + v, o.N + 1)
                        end
                    end
                    sim.kind = :chance_expand
                    sim.chance_state = cstate
                    sim.leaf_state = cstate
                    sim.leaf_actions = EMPTY_INT_VEC
                    sim.is_new_node = false
                    sim.terminal_value = 0.0
                    return sim
                else
                    # LATER VISIT: select an outcome by visit deficit, apply virtual
                    # loss to that outcome edge, record the chance edge, descend.
                    Ntot = 0.0
                    @inbounds for o in cinfo.outcomes; Ntot += o.N; end
                    best = 1; best_score = -Inf
                    @inbounds for k in eachindex(cinfo.outcomes)
                        o = cinfo.outcomes[k]
                        score = o.prob - o.N / max(Ntot, 1.0)
                        if score > best_score
                            best_score = score; best = k
                        end
                    end
                    o = cinfo.outcomes[best]
                    cinfo.outcomes[best] = MCTS.ChanceOutcomeStats(o.prob, o.W - VIRTUAL_LOSS, o.N + 1)
                    push!(sim.path, (cstate, best))
                    push!(sim.rewards, 0.0)          # no reward at the chance hop
                    push!(sim.player_switches, false)  # dice do not switch player-to-move
                    push!(sim.is_chance, true)
                    outcomes = GI.chance_outcomes(game)
                    GI.apply_chance!(game, outcomes[best][1])
                    depth += 1
                    env.total_nodes_traversed += 1
                    continue
                end
            end

            # Passthrough: sample one outcome, continue to decision node.
            # No tree entry, no backprop path entry. Equivalent to old deterministic wrapper.
            outcomes = GI.chance_outcomes(game)
            r_val = rand(env.rng)
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
            benv.metrics.tree_misses += 1
            # Derive leaf actions from the cloned state, not the pooled live
            # env, so the stored node contract matches the oracle input exactly.
            leaf_actions = GI.available_actions(GI.spec(game), state)

            # Single-option states (e.g., forced PASS): create trivial tree entry
            # and continue traversal. No oracle evaluation needed since P=[1.0]
            # and the value will come from the child via backpropagation.
            if length(leaf_actions) == 1
                stats = [MCTS.ActionStats(Float32(1.0), Float64(0), 0)]
                info = MCTS.StateInfo(stats, leaf_actions, Float32(0))
                env.tree[state] = info
                # Fall through to normal traversal below (info is now set)
            else
                # Bear-off table: use exact value instead of NN oracle
                if benv.bearoff_evaluator !== nothing
                    val = benv.bearoff_evaluator(game)
                    if val !== nothing
                        benv.metrics.bearoff_hits += 1
                        # Bear-off returns white-relative equity; convert to player-relative
                        # (matching NN oracle convention where Vest is player-relative)
                        wp = GI.white_playing(game)
                        player_val = wp ? val : -val
                        # Create tree entry with uniform policy + exact bear-off value
                        n = length(leaf_actions)
                        P_uniform = Float32(1.0 / n)
                        stats = [MCTS.ActionStats(P_uniform, Float64(0), 0) for _ in 1:n]
                        info = MCTS.StateInfo(stats, leaf_actions, Float32(player_val))
                        env.tree[state] = info
                        sim.leaf_state = state
                        sim.leaf_actions = Int[]
                        sim.is_new_node = false
                        sim.terminal_value = player_val
                        return sim
                    end
                    benv.metrics.bearoff_misses += 1
                end

                sim.leaf_state = state
                sim.leaf_actions = leaf_actions
                sim.is_new_node = true
                sim.terminal_value = 0.0
                return sim
            end
        else
            benv.metrics.tree_hits += 1
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
        push!(sim.is_chance, false)  # decision edge

        GI.play!(game, action)
        # Normalize by reward scale so terminal rewards (±1/±2/±3 in backgammon)
        # mix with NN values ([-1,1]) on the same scale in backprop Q totals
        wr = GI.white_reward(game) * benv.inv_reward_scale
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
function batch_evaluate(benv::BatchedEnv, states::Vector, actions_by_state=nothing)
    if isempty(states)
        return Tuple{Vector{Float32}, Float32}[]
    end

    # The batched inference oracle has a fixed-width buffer sized to benv.batch_size.
    # The passthrough/training path never enqueues more than batch_size states per wave
    # (one leaf per sim), so it always takes the single-call path below unchanged. The
    # EVAL-ONLY :exact_expectation path can enqueue up to ~21 chance-children per sim, so
    # chunk to batch_size to avoid overflowing the oracle buffer (behaviour identical to
    # calling it batch_size states at a time).
    maxw = benv.batch_size
    benv.metrics.nn_evaluations += length(states)
    if length(states) > maxw && (benv.batch_oracle_with_actions !== nothing || benv.batch_oracle !== nothing)
        benv.metrics.oracle_calls += cld(length(states), maxw)
        results = Tuple{Vector{Float32}, Float32}[]
        sizehint!(results, length(states))
        for i in 1:maxw:length(states)
            j = min(i + maxw - 1, length(states))
            chunk = if actions_by_state !== nothing && benv.batch_oracle_with_actions !== nothing
                benv.batch_oracle_with_actions(states[i:j], actions_by_state[i:j])
            else
                benv.batch_oracle(states[i:j])
            end
            append!(results, chunk)
        end
        return results
    end

    if actions_by_state !== nothing && benv.batch_oracle_with_actions !== nothing
        benv.metrics.oracle_calls += 1
        return benv.batch_oracle_with_actions(states, actions_by_state)
    elseif benv.batch_oracle !== nothing
        # Use batched oracle for GPU evaluation (all states in one call)
        benv.metrics.oracle_calls += 1
        return benv.batch_oracle(states)
    else
        # Fall back to sequential oracle calls
        benv.metrics.oracle_calls += length(states)
        return [benv.env.oracle(state) for state in states]
    end
end

"""Evaluate pending simulations using a batched oracle call.

Most sims contribute a single leaf state (`kind == :normal`, `is_new_node`). Under
the EVAL-ONLY `:exact_expectation` mode two extra routes exist:
  - `:chance_first`  contributes the pre-dice chance state (its V becomes Vest);
  - `:chance_expand` contributes 0..N outcome-child states (each fills one outcome).
The `_eval_route_*` buffers stay empty for the passthrough/training path, so its
behaviour is unchanged.
"""
function batch_evaluate_pending!(benv::BatchedEnv{S}) where S
    env = benv.env

    # Reuse pre-allocated vectors (avoid per-call allocation)
    states_to_eval = benv._eval_states
    actions_to_eval = benv._eval_actions
    route_sim = benv._eval_route_sim
    route_out = benv._eval_route_out
    empty!(states_to_eval)
    empty!(actions_to_eval)
    empty!(route_sim)
    empty!(route_out)

    @inbounds for (i, sim) in enumerate(benv.pending)
        if sim.kind === :normal
            if sim.is_new_node
                push!(states_to_eval, sim.leaf_state)
                push!(actions_to_eval, sim.leaf_actions)
                push!(route_sim, i)
                push!(route_out, 0)          # normal decision leaf
            end
        elseif sim.kind === :chance_first
            push!(states_to_eval, sim.chance_state)
            push!(actions_to_eval, EMPTY_INT_VEC)  # zero action-mask for chance states
            push!(route_sim, i)
            push!(route_out, -1)             # chance_first: result V -> Vest
        elseif sim.kind === :chance_expand
            for j in eachindex(sim.expand_states)
                push!(states_to_eval, sim.expand_states[j])
                push!(actions_to_eval, sim.expand_actions[j])
                push!(route_sim, i)
                push!(route_out, sim.expand_out_idx[j])  # >0: fill this outcome
            end
        end
    end

    if isempty(states_to_eval)
        return
    end

    # Batch evaluate
    results = batch_evaluate(benv, states_to_eval, actions_to_eval)
    length(results) == length(states_to_eval) ||
        error("Batched oracle result count mismatch: got $(length(results)) results for $(length(states_to_eval)) states")

    # Route results back to the owning sim / tree / chance-tree entry.
    @inbounds for k in eachindex(results)
        P, V = results[k]
        i = route_sim[k]
        out = route_out[k]
        sim = benv.pending[i]
        if out == 0
            # Normal decision leaf: create the tree node, store Vest for backprop.
            if length(P) != length(sim.leaf_actions)
                recomputed_actions = GI.available_actions(env.gspec, sim.leaf_state)
                error("Oracle policy/action mismatch: length(P)=$(length(P)) length(actions)=$(length(sim.leaf_actions)) length(recomputed)=$(length(recomputed_actions)) same_actions=$(sim.leaf_actions == recomputed_actions) state=$(sim.leaf_state)")
            end
            info = MCTS.init_state_info(P, V, env.prior_temperature, sim.leaf_actions)
            env.tree[sim.leaf_state] = info
            sim.terminal_value = V  # Store for backpropagation
        elseif out == -1
            # chance_first: V is the pre-dice NN value estimate for the chance node.
            cinfo = env.chance_tree[sim.chance_state]
            cinfo.Vest = Float32(V)
            sim.terminal_value = Float64(V)  # leaf value propagated up the path
        else
            # chance_expand outcome child: create its decision-node entry (so later
            # followers can descend), and fill the outcome edge additively (W += V,
            # N += 1) — mirrors the recursive expand_chance_node! per-outcome visit.
            cstate = states_to_eval[k]
            cactions = actions_to_eval[k]
            length(P) == length(cactions) ||
                error("Oracle policy/action mismatch at chance-expand child: length(P)=$(length(P)) length(actions)=$(length(cactions)) state=$(cstate)")
            env.tree[cstate] = MCTS.init_state_info(P, V, env.prior_temperature, cactions)
            cinfo = env.chance_tree[sim.chance_state]
            o = cinfo.outcomes[out]
            cinfo.outcomes[out] = MCTS.ChanceOutcomeStats(o.prob, o.W + Float64(V), o.N + 1)
        end
    end
end

#####
##### Backpropagation
#####

"""
Backpropagate value through the path, removing virtual losses.
Combined remove_virtual_loss + update_state_info into single Dict lookup per node
(was: 3 lookups per node = haskey + index for remove_VL + index for update).

Chance nodes: passthrough mode keeps no tree entries and adds nothing to the path.
Under :exact_expectation, chance OUTCOME edges appear in the path (`is_chance[i]`)
and back up the probability-weighted expectation with no reward/gamma/sign flip.

Value/sign convention:
- `sim.terminal_value` is player-relative at the leaf
- each stored `reward` is player-relative at the parent state
- `player_switches[i]` says whether the side to move changed across edge `i`

So reverse-time backup is:
1. flip the child value if control passed to the opponent
2. add the immediate parent-frame reward
3. discount
"""
function backpropagate!(benv::BatchedEnv, sim::PendingSimulation)
    env = benv.env

    # Get leaf value (single lookup via get instead of haskey + index)
    if sim.kind === :chance_expand
        # Leaf IS the chance node: its value is the probability-weighted expectation
        # over the (now-filled) outcome children. Passes to the parent decision node
        # exactly like a passthrough sample did.
        cinfo = env.chance_tree[sim.chance_state]
        q = chance_expectation(cinfo)
    elseif sim.kind === :chance_first
        # First visit: pre-dice NN value (stored in terminal_value from the batch).
        q = sim.terminal_value
    elseif sim.is_new_node
        leaf_info = get(env.tree, sim.leaf_state, nothing)
        q = leaf_info !== nothing ? Float64(leaf_info.Vest) : sim.terminal_value
    else
        q = sim.terminal_value
    end

    # Backpropagate through path in reverse
    @inbounds for i in length(sim.path):-1:1
        state, action_id = sim.path[i]

        if sim.is_chance[i]
            # Chance OUTCOME edge: NO reward, NO gamma, NO sign flip (dice do not
            # switch the player-to-move — the switch was captured at the preceding
            # decision edge's pswitch). Remove virtual loss and add q additively
            # (W += VL + q, N unchanged; net over descend+backup is W += q, N += 1),
            # then propagate the chance node's expectation to the parent.
            cinfo = get(env.chance_tree, state, nothing)
            if cinfo !== nothing
                o = cinfo.outcomes[action_id]
                cinfo.outcomes[action_id] = MCTS.ChanceOutcomeStats(o.prob, o.W + VIRTUAL_LOSS + q, o.N)
                q = chance_expectation(cinfo)
            end
        else
            reward = sim.rewards[i]
            pswitch = sim.player_switches[i]

            # Compute Q-value for this step
            q = pswitch ? -q : q
            q = reward + env.gamma * q

            # Decision node: remove virtual loss + update (1 lookup instead of 3)
            # VL removal: W += VL, N -= 1; Update: W += q, N += 1; Net: W += (VL + q), N unchanged
            info = get(env.tree, state, nothing)
            if info !== nothing
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
    simulations_before = env.total_simulations
    search_started = time_ns()

    try

    # Ensure root is in the tree before generating noise (need action count).
    # Without this, if batch_size >= nsims, ALL simulations hit an empty tree,
    # no actions are selected, visit counts stay at 0, and policy() returns NaN.
    state = GI.current_state(game)
    root_info = get(env.tree, state, nothing)
    if root_info === nothing
        benv.metrics.tree_misses += 1
    else
        benv.metrics.tree_hits += 1
    end
    if !GI.game_terminated(game) && !GI.is_chance_node(game) && root_info === nothing
        empty!(benv.pending)
        env.total_simulations += 1
        # Use pool sim for root init too
        if isempty(benv.sim_pool)
            _init_sim_pool!(benv, game)
        end
        traverse_to_leaf!(benv.sim_pool[1], benv, GI.clone(game), Float64[])
        benv.metrics.max_depth = max(benv.metrics.max_depth,
                                     length(benv.sim_pool[1].path))
        push!(benv.pending, benv.sim_pool[1])
        batch_evaluate_pending!(benv)
        for s in benv.pending
            backpropagate!(benv, s)
        end
        root_info = get(env.tree, state, nothing)
        nsims -= 1
    end

    # Generate Dirichlet noise using cached actions from tree (avoids GI.available_actions)
    η = if env.noise_α != 0 && root_info !== nothing
        n_actions = length(root_info.actions)
        rand(env.rng, Dirichlet(n_actions, Float64(env.noise_α)))
    elseif env.noise_α != 0
        n_actions = length(GI.available_actions(game))
        rand(env.rng, Dirichlet(n_actions, Float64(env.noise_α)))
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
        @inbounds for sim_idx in 1:current_batch_size
            env.total_simulations += 1
            if sim_idx <= length(benv.game_pool)
                game_clone = GI.clone_into!(benv.game_pool[sim_idx], game)
            else
                game_clone = GI.clone(game)
            end
            sim = benv.sim_pool[sim_idx]
            traverse_to_leaf!(sim, benv, game_clone, η)
            benv.metrics.max_depth = max(benv.metrics.max_depth, length(sim.path))
            push!(benv.pending, sim)
        end

        # Phase 2: Batch evaluate
        batch_evaluate_pending!(benv)

        # Phase 3: Backpropagate
        @inbounds for sim in benv.pending
            backpropagate!(benv, sim)
        end

        sims_done += current_batch_size
    end
    finally
        benv.metrics.simulations += env.total_simulations - simulations_before
        benv.metrics.search_ns += time_ns() - search_started
    end
end

#####
##### Convenience functions
#####

"""
Create a batched MCTS environment from a regular one.
"""
function make_batched(env::MCTS.Env, batch_size::Int; batch_oracle=nothing, batch_oracle_with_actions=nothing)
    return BatchedEnv(env, batch_size; batch_oracle=batch_oracle,
                      batch_oracle_with_actions=batch_oracle_with_actions)
end

"""
Get the underlying MCTS environment.
"""
get_env(benv::BatchedEnv) = benv.env

"""Return a copy of the cumulative metrics without resetting them."""
search_metrics(benv::BatchedEnv) = _snapshot(benv.metrics)

"""Return and reset the metrics accumulated since the previous take."""
function take_search_metrics!(benv::BatchedEnv)
    result = _snapshot(benv.metrics)
    _reset!(benv.metrics)
    return result
end

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
mutable struct BatchedMctsPlayer{B, T, F} <: Function
    benv::B
    niters::Int
    batch_size::Int
    τ::T  # Temperature schedule (AbstractSchedule{Float64})
    sim_budget_fn::F
    turn_count::Int
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
function BatchedMctsPlayer(game_spec, oracle, params; batch_size=32, batch_oracle=nothing,
                           batch_oracle_with_actions=nothing, bearoff_evaluator=nothing,
                           sim_budget_fn=nothing,
                           rng::Random.AbstractRNG=Random.Xoshiro(rand(UInt)))
    mcts = MCTS.Env(game_spec, oracle,
        gamma=params.gamma,
        cpuct=params.cpuct,
        noise_ϵ=params.dirichlet_noise_ϵ,
        noise_α=params.dirichlet_noise_α,
        prior_temperature=params.prior_temperature,
        chance_mode=params.chance_mode,
        progressive_widening_alpha=params.progressive_widening_alpha,
        prior_virtual_visits=params.prior_virtual_visits,
        rng=rng)
    benv = BatchedEnv(mcts, batch_size; batch_oracle=batch_oracle,
                      batch_oracle_with_actions=batch_oracle_with_actions,
                      bearoff_evaluator=bearoff_evaluator)
    return BatchedMctsPlayer(benv, params.num_iters_per_turn, batch_size, params.temperature,
                             sim_budget_fn, 0)
end

function think(p::BatchedMctsPlayer, game)
    niters = p.sim_budget_fn === nothing ? p.niters : p.sim_budget_fn(p.turn_count)
    batched_explore!(p.benv, game, niters)
    p.turn_count += 1
    return MCTS.policy(p.benv.env, game)
end

function player_temperature(p::BatchedMctsPlayer, game, turn)
    return p.τ[turn]
end

function reset_player!(p::BatchedMctsPlayer)
    reset!(p.benv)
    p.turn_count = 0
end

search_metrics(p::BatchedMctsPlayer) = search_metrics(p.benv)
take_search_metrics!(p::BatchedMctsPlayer) = take_search_metrics!(p.benv)

end  # module
