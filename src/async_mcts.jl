#####
##### Async MCTS with Continuous GPU Inference Pipeline
##### Decouples CPU tree traversal from GPU neural network inference
#####

"""
Async MCTS Architecture:

┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│   Workers   │────▶│  Request Queue  │────▶│  GPU Server │
│ (CPU bound) │     └─────────────────┘     │ (GPU bound) │
│             │                             │             │
│  Tree       │     ┌─────────────────┐     │  Batched    │
│  Traversal  │◀────│ Response Queue  │◀────│  Inference  │
└─────────────┘     └─────────────────┘     └─────────────┘

Workers continuously traverse trees and submit leaf states.
GPU server continuously batches and evaluates.
No synchronization barriers - both run at full speed.
"""
module AsyncMCTS

using Base.Threads
using Distributions: Dirichlet
using ..AlphaZero: GI, Util, MCTS, Network

export AsyncPipeline, AsyncWorker, start_pipeline!, stop_pipeline!
export async_explore!, async_self_play

#####
##### Request/Response Types
#####

struct InferenceRequest
    id::Int
    state::Any
    worker_id::Int
end

struct InferenceResponse
    id::Int
    policy::Vector{Float32}
    value::Float32
end

#####
##### Pending Simulation (in-flight, waiting for NN result)
#####

mutable struct PendingSimulation
    request_id::Int
    leaf_state::Any
    leaf_actions::Vector{Int}       # Available actions at leaf node
    path::Vector{Tuple{Any, Int}}  # (state, action_id) pairs
    rewards::Vector{Float64}
    player_switches::Vector{Bool}
end

#####
##### Async Pipeline
#####

mutable struct AsyncPipeline
    # Queues
    request_queue::Channel{InferenceRequest}
    response_queues::Dict{Int, Channel{InferenceResponse}}

    # GPU server
    network::Any
    batch_size::Int
    max_wait_ms::Float64

    # State
    running::Ref{Bool}
    server_task::Union{Task, Nothing}

    # Stats
    total_requests::Ref{Int}
    total_batches::Ref{Int}

    function AsyncPipeline(network; batch_size=64, max_wait_ms=5.0, num_workers=64)
        request_queue = Channel{InferenceRequest}(batch_size * 4)
        response_queues = Dict{Int, Channel{InferenceResponse}}()

        # Response queue must be large enough to hold all responses from a full batch
        # plus some buffer to avoid blocking
        response_queue_size = batch_size * 2
        for i in 1:num_workers
            response_queues[i] = Channel{InferenceResponse}(response_queue_size)
        end

        new(
            request_queue,
            response_queues,
            network,
            batch_size,
            max_wait_ms,
            Ref(false),
            nothing,
            Ref(0),
            Ref(0)
        )
    end
end

#####
##### GPU Inference Server
#####

function gpu_server_loop!(pipeline::AsyncPipeline)
    pending = InferenceRequest[]
    last_batch_time = time()
    debug_iter = 0

    while pipeline.running[] || !isempty(pending) || isready(pipeline.request_queue)
        debug_iter += 1
        # Collect requests (non-blocking drain)
        while isready(pipeline.request_queue) && length(pending) < pipeline.batch_size * 2
            try
                req = take!(pipeline.request_queue)
                push!(pending, req)
            catch e
                if !isa(e, InvalidStateException)
                    @warn "GPU server receive error" exception=e
                end
                break
            end
        end

        # Decide whether to process
        current_time = time()
        elapsed_ms = (current_time - last_batch_time) * 1000

        should_process = length(pending) >= pipeline.batch_size ||
                        (length(pending) > 0 && elapsed_ms >= pipeline.max_wait_ms) ||
                        (!pipeline.running[] && length(pending) > 0)

        if should_process && !isempty(pending)
            # Take a batch
            actual_batch_size = min(length(pending), pipeline.batch_size)
            batch_requests = splice!(pending, 1:actual_batch_size)

            # Build batch of states
            states = [req.state for req in batch_requests]

            # Run inference
            try
                results = Network.evaluate_batch(pipeline.network, states)

                # Distribute results to workers
                responses_sent = 0
                for (i, req) in enumerate(batch_requests)
                    P, V = results[i]
                    response = InferenceResponse(req.id, P, V)

                    if haskey(pipeline.response_queues, req.worker_id)
                        try
                            put!(pipeline.response_queues[req.worker_id], response)
                            responses_sent += 1
                        catch e
                            @warn "Failed to send response to worker $(req.worker_id)" exception=e
                        end
                    else
                        @warn "No response queue for worker $(req.worker_id)"
                    end
                end

                pipeline.total_batches[] += 1
                pipeline.total_requests[] += actual_batch_size
            catch e
                @error "GPU inference error" exception=e
            end

            last_batch_time = time()
        else
            # Yield to allow worker threads to submit requests
            yield()
        end
    end
end

function start_pipeline!(pipeline::AsyncPipeline)
    pipeline.running[] = true
    pipeline.server_task = Threads.@spawn gpu_server_loop!(pipeline)
    return pipeline
end

function stop_pipeline!(pipeline::AsyncPipeline)
    pipeline.running[] = false
    if pipeline.server_task !== nothing
        wait(pipeline.server_task)
    end
    close(pipeline.request_queue)
    for (_, ch) in pipeline.response_queues
        close(ch)
    end
end

#####
##### Async Worker
#####

mutable struct AsyncWorker
    id::Int
    pipeline::AsyncPipeline
    gspec::Any
    mcts_env::MCTS.Env

    # In-flight simulations
    pending_sims::Dict{Int, PendingSimulation}
    next_request_id::Int

    # MCTS params
    num_sims::Int
    noise_α::Float64
    noise_ϵ::Float64
    cpuct::Float64
    gamma::Float64
end

function AsyncWorker(id::Int, pipeline::AsyncPipeline, gspec, mcts_params)
    # Create a dummy oracle (we'll use async submission instead)
    dummy_oracle = state -> error("Should not be called directly")

    mcts_env = MCTS.Env(gspec, dummy_oracle,
        cpuct=mcts_params.cpuct,
        gamma=mcts_params.gamma,
        noise_ϵ=mcts_params.dirichlet_noise_ϵ,
        noise_α=mcts_params.dirichlet_noise_α,
        prior_temperature=mcts_params.prior_temperature)

    AsyncWorker(
        id, pipeline, gspec, mcts_env,
        Dict{Int, PendingSimulation}(),
        1,
        mcts_params.num_iters_per_turn,
        mcts_params.dirichlet_noise_α,
        mcts_params.dirichlet_noise_ϵ,
        mcts_params.cpuct,
        mcts_params.gamma
    )
end

#####
##### Virtual Loss for Concurrent Traversal
#####

const VIRTUAL_LOSS = 3.0

function apply_virtual_loss!(env::MCTS.Env, state, action_id)
    if haskey(env.tree, state)
        stats = env.tree[state].stats
        astats = stats[action_id]
        stats[action_id] = MCTS.ActionStats(astats.P, astats.W - VIRTUAL_LOSS, astats.N + 1)
    end
end

function remove_virtual_loss!(env::MCTS.Env, state, action_id)
    if haskey(env.tree, state)
        stats = env.tree[state].stats
        astats = stats[action_id]
        stats[action_id] = MCTS.ActionStats(astats.P, astats.W + VIRTUAL_LOSS, astats.N - 1)
    end
end

#####
##### Async Tree Traversal
#####

"""
Traverse tree to leaf, applying virtual loss along the way.
Returns (leaf_state, path, rewards, player_switches, is_terminal)
Does NOT call the oracle - just submits to queue.
"""
function traverse_to_leaf!(worker::AsyncWorker, game, η)
    env = worker.mcts_env
    path = Tuple{Any, Int}[]
    rewards = Float64[]
    player_switches = Bool[]

    depth = 0
    max_depth = 500

    while depth < max_depth
        if GI.game_terminated(game)
            return (nothing, Int[], path, rewards, player_switches, true, 0.0)
        end

        if GI.is_chance_node(game)
            outcomes = GI.chance_outcomes(game)
            probs = [p for (_, p) in outcomes]
            idx = Util.rand_categorical(probs)
            outcome, _ = outcomes[idx]
            GI.apply_chance!(game, outcome)
            continue
        end

        state = GI.current_state(game)

        if !haskey(env.tree, state)
            # New leaf node - compute actions for node creation
            actions = GI.available_actions(game)
            return (state, actions, path, rewards, player_switches, false, 0.0)
        end

        # Existing node - use cached actions (no allocation)
        info = env.tree[state]
        actions = info.actions
        ϵ = isempty(path) ? worker.noise_ϵ : 0.0
        scores = MCTS.uct_scores(info, worker.cpuct, ϵ, η)
        action_id = argmax(scores)
        action = actions[action_id]

        # Apply virtual loss
        apply_virtual_loss!(env, state, action_id)

        # Record path
        wp = GI.white_playing(game)
        push!(path, (state, action_id))

        # Take action
        GI.play!(game, action)
        wr = GI.white_reward(game)
        r = wp ? wr : -wr
        push!(rewards, r)
        push!(player_switches, wp != GI.white_playing(game))

        depth += 1
    end

    # Depth limit
    return (nothing, Int[], path, rewards, player_switches, true, 0.0)
end

"""
Submit a leaf state for async evaluation.
"""
function submit_for_evaluation!(worker::AsyncWorker, leaf_state, leaf_actions, path, rewards, player_switches)
    request_id = worker.next_request_id
    worker.next_request_id += 1

    # Store pending simulation
    worker.pending_sims[request_id] = PendingSimulation(
        request_id, leaf_state, leaf_actions, path, rewards, player_switches
    )

    # Submit to GPU server
    request = InferenceRequest(request_id, leaf_state, worker.id)
    put!(worker.pipeline.request_queue, request)

    return request_id
end

"""
Check for completed evaluations and backpropagate.
"""
function process_completed!(worker::AsyncWorker)
    env = worker.mcts_env
    response_queue = worker.pipeline.response_queues[worker.id]
    completed = 0

    while isready(response_queue)
        response = take!(response_queue)

        if !haskey(worker.pending_sims, response.id)
            continue  # Already processed or invalid
        end

        sim = worker.pending_sims[response.id]
        delete!(worker.pending_sims, response.id)

        # Initialize leaf node in tree
        info = MCTS.init_state_info(response.policy, response.value, 1.0, sim.leaf_actions)
        env.tree[sim.leaf_state] = info

        # Backpropagate
        q = Float64(response.value)

        for i in length(sim.path):-1:1
            state, action_id = sim.path[i]
            reward = sim.rewards[i]
            pswitch = sim.player_switches[i]

            # Remove virtual loss
            remove_virtual_loss!(env, state, action_id)

            # Update Q-value
            q = pswitch ? -q : q
            q = reward + worker.gamma * q

            # Update stats
            MCTS.update_state_info!(env, state, action_id, q)
        end

        completed += 1
    end

    return completed
end

"""
Backpropagate terminal state (no oracle needed).
"""
function backpropagate_terminal!(worker::AsyncWorker, path, rewards, player_switches, terminal_value)
    env = worker.mcts_env
    q = terminal_value

    for i in length(path):-1:1
        state, action_id = path[i]
        reward = rewards[i]
        pswitch = player_switches[i]

        # Remove virtual loss
        remove_virtual_loss!(env, state, action_id)

        # Update Q-value
        q = pswitch ? -q : q
        q = reward + worker.gamma * q

        # Update stats
        MCTS.update_state_info!(env, state, action_id, q)
    end
end

#####
##### Async Exploration
#####

"""
    async_explore!(worker, game, num_sims)

Run async MCTS exploration. Launches multiple simulations concurrently,
submits leaves to GPU server, and backpropagates as results arrive.
"""
function async_explore!(worker::AsyncWorker, game, num_sims=nothing)
    if isnothing(num_sims)
        num_sims = worker.num_sims
    end

    env = worker.mcts_env

    # Generate Dirichlet noise for root
    actions = GI.available_actions(game)
    η = worker.noise_α > 0 ? rand(Dirichlet(length(actions), Float64(worker.noise_α))) : Float64[]

    # Evaluate root state first (blocking, to initialize tree)
    root_state = GI.current_state(game)
    if !haskey(env.tree, root_state)
        request = InferenceRequest(0, root_state, worker.id)
        put!(worker.pipeline.request_queue, request)

        # Wait for root evaluation
        response_queue = worker.pipeline.response_queues[worker.id]
        while true
            if isready(response_queue)
                response = take!(response_queue)
                info = MCTS.init_state_info(response.policy, response.value, 1.0, actions)
                env.tree[root_state] = info
                break
            end
            sleep(0.0001)
        end
    end

    # Now run async simulations
    sims_launched = 0
    sims_completed = 0
    max_in_flight = min(32, num_sims)  # Limit concurrent simulations

    iter = 0

    while sims_completed < num_sims
        iter += 1

        # Launch new simulations if we have capacity
        while sims_launched < num_sims && length(worker.pending_sims) < max_in_flight
            game_clone = GI.clone(game)
            leaf_state, leaf_actions, path, rewards, player_switches, is_terminal, term_value =
                traverse_to_leaf!(worker, game_clone, η)

            if is_terminal
                # Terminal state - backpropagate immediately
                if !isempty(path)
                    backpropagate_terminal!(worker, path, rewards, player_switches, term_value)
                end
                sims_launched += 1
                sims_completed += 1
            elseif leaf_state !== nothing
                # Submit for async evaluation
                submit_for_evaluation!(worker, leaf_state, leaf_actions, path, rewards, player_switches)
                sims_launched += 1
            else
                # Empty path terminal
                sims_launched += 1
                sims_completed += 1
            end
        end

        # Process any completed evaluations
        completed = process_completed!(worker)
        sims_completed += completed

        # Wait for responses if we have pending simulations but no responses ready
        if !isempty(worker.pending_sims) && !isready(worker.pipeline.response_queues[worker.id])
            # Small sleep to allow GPU server thread to process and respond
            sleep(0.001)  # 1ms
            # Try to process again after sleep
            completed2 = process_completed!(worker)
            sims_completed += completed2
        end

        # Safety check to prevent infinite loop
        if iter > 100000
            @error "async_explore! appears stuck: launched=$sims_launched completed=$sims_completed pending=$(length(worker.pending_sims))"
            break
        end
    end

    # Drain any remaining pending simulations
    drain_iter = 0
    while !isempty(worker.pending_sims)
        completed = process_completed!(worker)
        if completed == 0
            sleep(0.001)  # Wait for GPU server to respond
        end
        drain_iter += 1
        if drain_iter > 1000
            @error "Drain appears stuck: pending=$(length(worker.pending_sims))"
            break
        end
    end
end

#####
##### Async Self-Play
#####

"""
Play a complete game using async MCTS.
"""
function async_play_game(worker::AsyncWorker, temperature_schedule)
    game = GI.init(worker.gspec)
    MCTS.reset!(worker.mcts_env)

    trace_states = []
    trace_policies = []
    turn = 0

    while !GI.game_terminated(game)
        turn += 1

        # Handle chance nodes
        if GI.is_chance_node(game)
            outcomes = GI.chance_outcomes(game)
            probs = [p for (_, p) in outcomes]
            idx = Util.rand_categorical(probs)
            outcome, _ = outcomes[idx]
            GI.apply_chance!(game, outcome)
            continue
        end

        state = GI.current_state(game)
        push!(trace_states, state)

        # Run async exploration
        async_explore!(worker, game)

        # Get policy
        actions, probs = MCTS.policy(worker.mcts_env, game)
        push!(trace_policies, (actions, probs))

        # Select action
        τ = temperature_schedule[turn]
        if τ > 0
            π_τ = probs .^ (1/τ)
            π_τ ./= sum(π_τ)
            action_idx = Util.rand_categorical(π_τ)
        else
            action_idx = argmax(probs)
        end
        action = actions[action_idx]

        GI.play!(game, action)
    end

    final_reward = GI.white_reward(game)
    return (states=trace_states, policies=trace_policies, reward=final_reward, turns=turn)
end

"""
Run async self-play with multiple workers sharing the GPU pipeline.
"""
function async_self_play(pipeline::AsyncPipeline, gspec, mcts_params, num_games, num_workers,
                         temperature_schedule)

    # Create workers
    workers = [AsyncWorker(i, pipeline, gspec, mcts_params) for i in 1:num_workers]

    # Distribute games across workers
    games_per_worker = cld(num_games, num_workers)

    all_traces = Vector{Any}(undef, num_games)
    game_idx = Threads.Atomic{Int}(1)

    # Run workers in parallel
    tasks = map(workers) do worker
        Threads.@spawn begin
            while true
                idx = Threads.atomic_add!(game_idx, 1)
                if idx > num_games
                    break
                end
                trace = async_play_game(worker, temperature_schedule)
                all_traces[idx] = trace
            end
        end
    end

    # Wait for all workers
    foreach(wait, tasks)

    return all_traces
end

end  # module
