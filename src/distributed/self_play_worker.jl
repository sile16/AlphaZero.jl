#####
##### Self-Play Worker
#####

"""
Self-play worker that runs MCTS games and submits samples to the replay buffer.

The worker supports two inference modes:
1. **Local inference** (recommended for workers with GPU): Loads network weights
   and runs inference locally. Receives periodic weight updates from coordinator.
2. **Remote inference** (for CPU-only workers): Sends states to centralized
   inference server for evaluation.

Local inference is preferred when the worker has a GPU, as it avoids network
latency for every MCTS node evaluation.
"""

using ZMQ
using Random

#####
##### Worker State
#####

"""
    SelfPlayWorker

Worker process for distributed self-play.

# Fields
- `config`: Worker configuration
- `gspec`: Game specification
- `network`: Local network for inference (if using local mode)
- `oracle`: Remote oracle for inference (if using remote mode)
- `replay_socket`: ZMQ PUSH socket for sample submission
- `weight_sub_socket`: ZMQ SUB socket for weight updates
- `context`: ZMQ context
- `running`: Worker running flag
- `stats`: Worker statistics
- `current_iteration`: Current network iteration
- `game_counter`: Total games played
- `use_local_inference`: Whether to use local GPU inference
"""
mutable struct SelfPlayWorker
    config::WorkerConfig
    gspec::AbstractGameSpec
    network::Union{Nothing, AbstractNetwork}
    oracle::Union{Nothing, ZMQRemoteOracle}
    replay_socket::ZMQ.Socket
    weight_sub_socket::ZMQ.Socket
    context::ZMQ.Context
    running::Bool
    stats::Dict{String, Any}
    current_iteration::Int
    game_counter::UInt64
    use_local_inference::Bool

    function SelfPlayWorker(
        config::WorkerConfig,
        gspec::AbstractGameSpec;
        network::Union{Nothing, AbstractNetwork}=nothing,
        use_local_inference::Bool=!isnothing(network)
    )
        # Create ZMQ context and sockets
        ctx = ZMQ.Context()

        # PUSH socket for sending samples to replay buffer
        replay_socket = ZMQ.Socket(ctx, ZMQ.PUSH)
        ZMQ.connect(replay_socket, endpoint_string(config.replay_endpoint, bind=false))

        # SUB socket for receiving weight updates from coordinator
        weight_sub_socket = ZMQ.Socket(ctx, ZMQ.SUB)
        ZMQ.connect(weight_sub_socket, endpoint_string(config.coordinator_endpoint, bind=false))
        ZMQ.subscribe(weight_sub_socket, "")  # Subscribe to all messages

        # Setup inference mode
        oracle = nothing
        if use_local_inference
            if isnothing(network)
                error("Network required for local inference mode")
            end
            # Prepare network for inference
            if config.use_gpu
                network = Network.to_gpu(network)
            end
            Network.set_test_mode!(network, true)
        else
            # Create remote oracle for centralized inference
            oracle = ZMQRemoteOracle(gspec, config.inference_endpoint, config.worker_id)
        end

        stats = Dict{String, Any}(
            "games_completed" => 0,
            "samples_generated" => 0,
            "total_moves" => 0,
            "weight_updates_received" => 0,
            "start_time" => time(),
        )

        return new(
            config, gspec, network, oracle, replay_socket, weight_sub_socket,
            ctx, false, stats, 0, UInt64(0), use_local_inference
        )
    end
end

#####
##### Weight updates
#####

"""
    check_weight_updates!(worker::SelfPlayWorker) -> Bool

Check for and apply any pending weight updates.
Returns true if weights were updated.
"""
function check_weight_updates!(worker::SelfPlayWorker)
    updated = false

    while ZMQ.isready(worker.weight_sub_socket)
        try
            msg_bytes = ZMQ.recv(worker.weight_sub_socket, ZMQ.DONTWAIT)
            if isempty(msg_bytes)
                break
            end

            # Try to parse as WeightUpdate or WorkerCommand containing WeightUpdate
            msg_data = Vector{UInt8}(msg_bytes)

            # First try as WorkerCommand
            try
                command = deserialize_message(msg_data, WorkerCommand)
                if command.command == :shutdown
                    worker.running = false
                    return updated
                elseif command.command == :update_weights && !isnothing(command.payload)
                    apply_weight_update!(worker, command.payload)
                    updated = true
                end
                continue
            catch
            end

            # Try as direct WeightUpdate
            try
                update = deserialize_message(msg_data, WeightUpdate)
                apply_weight_update!(worker, update)
                updated = true
            catch e
                @debug "Failed to parse weight update" exception=e
            end

        catch e
            if !(e isa ZMQ.StateError)
                @warn "Error receiving weight update" exception=e
            end
            break
        end
    end

    return updated
end

"""
    apply_weight_update!(worker::SelfPlayWorker, update::WeightUpdate)

Apply a weight update to the local network.
"""
function apply_weight_update!(worker::SelfPlayWorker, update::WeightUpdate)
    if !worker.use_local_inference || isnothing(worker.network)
        @debug "Ignoring weight update (not using local inference)"
        return
    end

    # Only apply if newer than current
    if update.iteration <= worker.current_iteration
        @debug "Ignoring old weight update: $(update.iteration) <= $(worker.current_iteration)"
        return
    end

    # Verify checksum if provided
    if update.checksum != 0
        computed = compute_checksum(update.weights_data)
        if computed != update.checksum
            @warn "Weight checksum mismatch, skipping update"
            return
        end
    end

    # Deserialize and load weights
    weight_arrays = deserialize_network_weights(update.weights_data)
    load_weights_into_network!(worker.network, weight_arrays)

    worker.current_iteration = update.iteration
    worker.stats["weight_updates_received"] += 1

    @info "Worker $(worker.config.worker_id) updated to iteration $(worker.current_iteration)"
end

"""
    request_initial_weights!(worker::SelfPlayWorker)

Request initial weights from coordinator on startup.
"""
function request_initial_weights!(worker::SelfPlayWorker)
    if !worker.use_local_inference
        return
    end

    # Create a REQ socket for initial weight request
    req_socket = ZMQ.Socket(worker.context, ZMQ.REQ)
    ZMQ.connect(req_socket, endpoint_string(worker.config.coordinator_endpoint, bind=false))

    request = WeightRequest(worker_id=worker.config.worker_id)
    msg_bytes = serialize_message(request)

    try
        ZMQ.send(req_socket, msg_bytes)

        # Wait for response with timeout
        # Note: ZMQ.jl doesn't have native timeout, so we use a simple poll
        start = time()
        timeout = 30.0  # 30 second timeout

        while time() - start < timeout
            if ZMQ.isready(req_socket)
                response_bytes = ZMQ.recv(req_socket)
                update = deserialize_message(Vector{UInt8}(response_bytes), WeightUpdate)
                apply_weight_update!(worker, update)
                @info "Received initial weights (iteration $(worker.current_iteration))"
                break
            end
            sleep(0.1)
        end
    catch e
        @warn "Failed to get initial weights, will use initialized network" exception=e
    finally
        ZMQ.close(req_socket)
    end
end

#####
##### MCTS Player creation
#####

"""
    create_mcts_player(worker::SelfPlayWorker) -> AbstractPlayer

Create an MCTS player using either local or remote inference.
"""
function create_mcts_player(worker::SelfPlayWorker)
    oracle = if worker.use_local_inference
        worker.network
    else
        MCTSOracleAdapter(worker.oracle)
    end

    if !isnothing(worker.config.gumbel_params)
        return GumbelMctsPlayer(worker.gspec, oracle, worker.config.gumbel_params)
    elseif !isnothing(worker.config.mcts_params)
        return MctsPlayer(worker.gspec, oracle, worker.config.mcts_params)
    else
        error("Worker config must specify either mcts_params or gumbel_params")
    end
end

#####
##### Game simulation
#####

"""
    play_self_play_game(worker::SelfPlayWorker) -> (Trace, Vector{SerializedSample})

Play a single self-play game and return the trace and serialized samples.
"""
function play_self_play_game(worker::SelfPlayWorker)
    player = create_mcts_player(worker)
    trace = play_game(worker.gspec, player)

    # Convert trace to serialized samples
    samples = trace_to_samples(worker, trace)

    # Reset player for next game
    reset_player!(player)

    return trace, samples
end

"""
    trace_to_samples(worker::SelfPlayWorker, trace::Trace) -> Vector{SerializedSample}

Convert a game trace to serialized training samples.
"""
function trace_to_samples(worker::SelfPlayWorker, trace::Trace)
    gamma = if !isnothing(worker.config.mcts_params)
        worker.config.mcts_params.gamma
    elseif !isnothing(worker.config.gumbel_params)
        worker.config.gumbel_params.gamma
    else
        1.0
    end

    n = length(trace)
    samples = SerializedSample[]
    sizehint!(samples, n)

    # Compute cumulative rewards
    wr = 0.0
    cumulative_rewards = zeros(n)
    for i in reverse(1:n)
        wr = gamma * wr + trace.rewards[i]
        cumulative_rewards[i] = wr
    end

    # Check for game outcome (multi-head training)
    has_outcome = !isnothing(trace.outcome)

    for i in 1:n
        state = trace.states[i]
        policy = trace.policies[i]
        is_chance = trace.is_chance[i]

        # Determine value from perspective of current player
        wp = GI.white_playing(worker.gspec, state)
        z = wp ? cumulative_rewards[i] : -cumulative_rewards[i]
        t = Float32(n - i + 1)

        # Vectorize state
        state_vec = Vector{Float32}(GI.vectorize_state(worker.gspec, state))

        # Policy (empty for chance nodes)
        policy_vec = Vector{Float32}(policy)

        # Equity targets if available
        equity = if has_outcome
            outcome = trace.outcome
            won = outcome.white_won == wp
            if won
                MultiHeadValue(
                    1.0f0,  # p_win
                    outcome.is_gammon ? 1.0f0 : 0.0f0,  # p_gammon_win
                    outcome.is_backgammon ? 1.0f0 : 0.0f0,  # p_bg_win
                    0.0f0,  # p_gammon_loss
                    0.0f0   # p_bg_loss
                )
            else
                MultiHeadValue(
                    0.0f0,  # p_win
                    0.0f0,  # p_gammon_win
                    0.0f0,  # p_bg_win
                    outcome.is_gammon ? 1.0f0 : 0.0f0,  # p_gammon_loss
                    outcome.is_backgammon ? 1.0f0 : 0.0f0   # p_bg_loss
                )
            end
        else
            nothing
        end

        push!(samples, SerializedSample(
            state=state_vec,
            policy=policy_vec,
            value=Float32(z),
            turn=t,
            is_chance=is_chance,
            equity=equity
        ))
    end

    return samples
end

#####
##### Sample submission
#####

"""
    submit_game_samples(worker::SelfPlayWorker, samples::Vector{SerializedSample}, num_games::Int)

Submit game samples to the replay buffer.
"""
function submit_game_samples(worker::SelfPlayWorker, samples::Vector{SerializedSample}, num_games::Int)
    worker.game_counter += 1

    game_samples = GameSamples(
        worker_id=worker.config.worker_id,
        game_id=worker.game_counter,
        samples=samples,
        metadata=Dict{String,Any}(
            "iteration" => worker.current_iteration,
            "num_games" => num_games,
            "game_length" => length(samples) ÷ max(num_games, 1),
        )
    )

    # Serialize and send
    msg_bytes = serialize_message(game_samples)
    ZMQ.send(worker.replay_socket, msg_bytes)

    # Update stats
    worker.stats["games_completed"] += num_games
    worker.stats["samples_generated"] += length(samples)
end

#####
##### Main worker loop
#####

"""
    run_worker(worker::SelfPlayWorker)

Run the self-play worker main loop.
"""
function run_worker(worker::SelfPlayWorker)
    worker.running = true

    mode = worker.use_local_inference ? "local GPU inference" : "remote inference"
    @info "Worker $(worker.config.worker_id) starting ($mode)"
    @info "  Replay:  $(endpoint_string(worker.config.replay_endpoint, bind=false))"
    @info "  Weights: $(endpoint_string(worker.config.coordinator_endpoint, bind=false))"
    if !worker.use_local_inference
        @info "  Inference: $(endpoint_string(worker.config.inference_endpoint, bind=false))"
    end

    # Request initial weights if using local inference
    if worker.use_local_inference
        request_initial_weights!(worker)
    end

    games_this_batch = 0
    pending_samples = SerializedSample[]
    last_weight_check = time()
    weight_check_interval = 5.0  # Check for weight updates every 5 seconds

    while worker.running
        try
            # Periodically check for weight updates
            if time() - last_weight_check > weight_check_interval
                check_weight_updates!(worker)
                last_weight_check = time()
            end

            # Play a game
            trace, samples = play_self_play_game(worker)
            append!(pending_samples, samples)
            games_this_batch += 1
            worker.stats["total_moves"] += length(trace)

            # Submit samples in batches
            if games_this_batch >= worker.config.games_per_batch
                submit_game_samples(worker, pending_samples, games_this_batch)

                games_this_batch = 0
                empty!(pending_samples)

                # Check for weight updates after each batch
                check_weight_updates!(worker)
            end

            # Log progress periodically
            if worker.stats["games_completed"] % 100 == 0 && worker.stats["games_completed"] > 0
                games_per_min = worker.stats["games_completed"] / ((time() - worker.stats["start_time"]) / 60)
                @info "Worker $(worker.config.worker_id): $(worker.stats["games_completed"]) games ($(round(games_per_min, digits=1))/min), iter $(worker.current_iteration)"
            end

        catch e
            if e isa InterruptException
                @info "Worker $(worker.config.worker_id) interrupted"
                break
            else
                @error "Worker error" exception=(e, catch_backtrace())
                sleep(1.0)  # Back off on error
            end
        end
    end

    # Submit any remaining samples
    if !isempty(pending_samples)
        try
            submit_game_samples(worker, pending_samples, games_this_batch)
        catch end
    end

    shutdown_worker(worker)
    @info "Worker $(worker.config.worker_id) stopped"

    return worker.stats
end

"""
    run_worker_async(worker::SelfPlayWorker) -> Task

Run the worker in a background task.
"""
function run_worker_async(worker::SelfPlayWorker)
    return @async run_worker(worker)
end

#####
##### Lifecycle management
#####

"""
    shutdown_worker(worker::SelfPlayWorker)

Shutdown the worker and clean up resources.
"""
function shutdown_worker(worker::SelfPlayWorker)
    worker.running = false

    # Close oracle connection if using remote inference
    if !isnothing(worker.oracle)
        try
            close(worker.oracle)
        catch end
    end

    # Close sockets
    try
        ZMQ.close(worker.replay_socket)
        ZMQ.close(worker.weight_sub_socket)
        ZMQ.close(worker.context)
    catch e
        @warn "Error during worker shutdown" exception=e
    end
end

#####
##### Statistics
#####

"""
    get_worker_stats(worker::SelfPlayWorker) -> Dict

Get current worker statistics.
"""
function get_worker_stats(worker::SelfPlayWorker)
    stats = copy(worker.stats)
    uptime = time() - stats["start_time"]
    stats["uptime_seconds"] = uptime

    if uptime > 0
        stats["games_per_minute"] = 60.0 * stats["games_completed"] / uptime
        stats["samples_per_second"] = stats["samples_generated"] / uptime
    end

    stats["current_iteration"] = worker.current_iteration
    stats["worker_id"] = worker.config.worker_id
    stats["inference_mode"] = worker.use_local_inference ? "local" : "remote"

    return stats
end

#####
##### Convenience functions
#####

"""
    create_worker(
        gspec::AbstractGameSpec,
        worker_id::String,
        coordinator_host::String;
        network::Union{Nothing, AbstractNetwork}=nothing,
        inference_port::Int=5555,
        replay_port::Int=5556,
        command_port::Int=5557,
        mcts_params::MctsParams,
        use_gpu::Bool=true,
        kwargs...
    ) -> SelfPlayWorker

Create a self-play worker with simple configuration.

If `network` is provided, the worker uses local GPU inference.
Otherwise, it connects to the centralized inference server.
"""
function create_worker(
    gspec::AbstractGameSpec,
    worker_id::String,
    coordinator_host::String;
    network::Union{Nothing, AbstractNetwork}=nothing,
    inference_port::Int=5555,
    replay_port::Int=5556,
    command_port::Int=5557,
    mcts_params::Union{Nothing, MctsParams}=nothing,
    gumbel_params::Union{Nothing, GumbelMctsParams}=nothing,
    games_per_batch::Int=10,
    use_gpu::Bool=true,
    kwargs...
)
    config = WorkerConfig(
        worker_id=worker_id,
        inference_endpoint=EndpointConfig(host=coordinator_host, port=inference_port),
        replay_endpoint=EndpointConfig(host=coordinator_host, port=replay_port),
        coordinator_endpoint=EndpointConfig(host=coordinator_host, port=command_port),
        mcts_params=mcts_params,
        gumbel_params=gumbel_params,
        games_per_batch=games_per_batch,
        use_gpu=use_gpu;
        kwargs...
    )

    return SelfPlayWorker(config, gspec; network=network)
end

"""
    create_local_worker(
        gspec::AbstractGameSpec,
        network::AbstractNetwork,
        worker_id::String,
        coordinator_host::String;
        kwargs...
    ) -> SelfPlayWorker

Create a worker with local GPU inference (recommended for workers with GPU).
"""
function create_local_worker(
    gspec::AbstractGameSpec,
    network::AbstractNetwork,
    worker_id::String,
    coordinator_host::String;
    kwargs...
)
    return create_worker(gspec, worker_id, coordinator_host; network=network, kwargs...)
end

"""
    create_remote_worker(
        gspec::AbstractGameSpec,
        worker_id::String,
        coordinator_host::String;
        kwargs...
    ) -> SelfPlayWorker

Create a worker that uses remote inference server (for CPU-only workers).
"""
function create_remote_worker(
    gspec::AbstractGameSpec,
    worker_id::String,
    coordinator_host::String;
    kwargs...
)
    return create_worker(gspec, worker_id, coordinator_host; network=nothing, kwargs...)
end
