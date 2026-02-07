#####
##### Replay Buffer Manager
#####

"""
Centralized replay buffer that receives samples from distributed workers.

The replay manager:
- Receives game samples from workers via ZMQ PULL socket
- Maintains a circular buffer of training samples
- Provides random batches to the training process
- Tracks sample statistics for monitoring
"""

using ZMQ
using Random
using DataStructures: CircularBuffer

#####
##### Replay Buffer State
#####

"""
    ReplayBufferManager

Thread-safe replay buffer for distributed training.

# Fields
- `config`: Buffer configuration
- `buffer`: Circular buffer of samples
- `context`: ZMQ context
- `socket`: ZMQ PULL socket for receiving samples
- `running`: Server running flag
- `stats`: Buffer statistics
- `lock`: Thread lock for buffer access
"""
mutable struct ReplayBufferManager
    config::ReplayBufferConfig
    buffer::CircularBuffer{SerializedSample}
    context::ZMQ.Context
    socket::ZMQ.Socket
    running::Bool
    stats::Dict{String, Any}
    lock::ReentrantLock

    function ReplayBufferManager(config::ReplayBufferConfig)
        ctx = ZMQ.Context()
        socket = ZMQ.Socket(ctx, ZMQ.PULL)
        ZMQ.bind(socket, endpoint_string(config.endpoint))

        buffer = CircularBuffer{SerializedSample}(config.capacity)

        stats = Dict{String, Any}(
            "total_samples_received" => 0,
            "total_games_received" => 0,
            "current_buffer_size" => 0,
            "samples_by_worker" => Dict{String, Int}(),
            "start_time" => time(),
        )

        return new(config, buffer, ctx, socket, false, stats, ReentrantLock())
    end
end

#####
##### Sample management
#####

"""
    add_samples!(manager::ReplayBufferManager, game_samples::GameSamples)

Add samples from a completed game to the buffer.
"""
function add_samples!(manager::ReplayBufferManager, game_samples::GameSamples)
    lock(manager.lock) do
        for sample in game_samples.samples
            push!(manager.buffer, sample)
        end

        # Update stats
        manager.stats["total_samples_received"] += length(game_samples.samples)
        manager.stats["total_games_received"] += 1
        manager.stats["current_buffer_size"] = length(manager.buffer)

        # Track per-worker stats
        worker_id = game_samples.worker_id
        if !haskey(manager.stats["samples_by_worker"], worker_id)
            manager.stats["samples_by_worker"][worker_id] = 0
        end
        manager.stats["samples_by_worker"][worker_id] += length(game_samples.samples)
    end
end

"""
    sample_batch(manager::ReplayBufferManager, batch_size::Int) -> Vector{SerializedSample}

Sample a random batch of samples from the buffer.
"""
function sample_batch(manager::ReplayBufferManager, batch_size::Int)
    lock(manager.lock) do
        n = length(manager.buffer)
        if n == 0
            return SerializedSample[]
        end

        # Sample with replacement if batch_size > buffer size
        actual_size = min(batch_size, n)
        indices = if manager.config.prioritized
            prioritized_sample_indices(manager, actual_size)
        else
            rand(1:n, actual_size)
        end

        return [manager.buffer[i] for i in indices]
    end
end

"""
    prioritized_sample_indices(manager, n) -> Vector{Int}

Sample indices using prioritized experience replay.
More recent samples have higher priority.
"""
function prioritized_sample_indices(manager::ReplayBufferManager, n::Int)
    buffer_size = length(manager.buffer)
    alpha = manager.config.priority_alpha

    # Priority based on recency (newer = higher priority)
    priorities = [(i / buffer_size)^alpha for i in 1:buffer_size]
    total = sum(priorities)
    probs = priorities ./ total

    # Sample according to priorities
    return [Util.rand_categorical(probs) for _ in 1:n]
end

"""
    get_batch_for_training(manager::ReplayBufferManager, batch_size::Int) -> TrainingSampleBatch

Get a batch of samples formatted for the training process.
"""
function get_batch_for_training(manager::ReplayBufferManager, batch_size::Int)
    samples = sample_batch(manager, batch_size)

    return TrainingSampleBatch(
        batch_id=UInt64(time() * 1e9),
        samples=samples,
        total_buffer_size=length(manager.buffer)
    )
end

"""
    is_ready_for_training(manager::ReplayBufferManager) -> Bool

Check if buffer has enough samples to start training.
"""
function is_ready_for_training(manager::ReplayBufferManager)
    return length(manager.buffer) >= manager.config.min_samples_for_training
end

#####
##### Server loop
#####

"""
    run_replay_manager(manager::ReplayBufferManager)

Run the replay buffer manager main loop.
"""
function run_replay_manager(manager::ReplayBufferManager)
    manager.running = true
    @info "Replay buffer manager starting on $(endpoint_string(manager.config.endpoint))"

    while manager.running
        try
            # Non-blocking receive with timeout
            if ZMQ.isready(manager.socket)
                msg_bytes = ZMQ.recv(manager.socket)
                msg_data = Vector{UInt8}(msg_bytes)

                # Try to deserialize as envelope first
                try
                    envelope = deserialize_message(msg_data, MessageEnvelope)
                    msg_type, msg = unwrap_message(envelope)

                    if msg_type == :game_samples
                        add_samples!(manager, msg)
                        @debug "Received $(length(msg.samples)) samples from $(msg.worker_id)"
                    else
                        @warn "Unexpected message type: $msg_type"
                    end
                catch
                    # Try direct deserialization
                    game_samples = deserialize_message(msg_data, GameSamples)
                    add_samples!(manager, game_samples)
                end
            else
                # Small sleep to avoid busy waiting
                sleep(0.001)
            end

        catch e
            if e isa InterruptException
                @info "Replay buffer manager interrupted"
                manager.running = false
            elseif e isa ZMQ.StateError
                # Socket closed, exit gracefully
                manager.running = false
            else
                @error "Replay buffer error" exception=(e, catch_backtrace())
            end
        end
    end

    shutdown_replay_manager(manager)
    @info "Replay buffer manager stopped"

    return manager.stats
end

"""
    run_replay_manager_async(manager::ReplayBufferManager) -> Task

Run the replay buffer manager in a background task.
"""
function run_replay_manager_async(manager::ReplayBufferManager)
    return @async run_replay_manager(manager)
end

#####
##### With training integration
#####

"""
    ReplayManagerWithREP

Replay manager that also responds to batch requests from training process.
Uses both PULL (for samples) and REP (for batch requests) sockets.
"""
mutable struct ReplayManagerWithREP
    manager::ReplayBufferManager
    rep_socket::ZMQ.Socket
    rep_endpoint::EndpointConfig

    function ReplayManagerWithREP(config::ReplayBufferConfig, rep_endpoint::EndpointConfig)
        manager = ReplayBufferManager(config)
        rep_socket = ZMQ.Socket(manager.context, ZMQ.REP)
        ZMQ.bind(rep_socket, endpoint_string(rep_endpoint))
        return new(manager, rep_socket, rep_endpoint)
    end
end

"""
    run_replay_manager_with_rep(manager::ReplayManagerWithREP)

Run replay manager with REP socket for training batch requests.
"""
function run_replay_manager_with_rep(manager::ReplayManagerWithREP)
    manager.manager.running = true
    @info "Replay buffer manager starting"
    @info "  PULL: $(endpoint_string(manager.manager.config.endpoint))"
    @info "  REP:  $(endpoint_string(manager.rep_endpoint))"

    # Create poll items
    pull_socket = manager.manager.socket
    rep_socket = manager.rep_socket

    while manager.manager.running
        try
            # Check PULL socket for incoming samples
            if ZMQ.isready(pull_socket)
                msg_bytes = ZMQ.recv(pull_socket)
                msg_data = Vector{UInt8}(msg_bytes)

                try
                    game_samples = deserialize_message(msg_data, GameSamples)
                    add_samples!(manager.manager, game_samples)
                catch
                    @warn "Failed to deserialize game samples"
                end
            end

            # Check REP socket for batch requests
            if ZMQ.isready(rep_socket)
                msg_bytes = ZMQ.recv(rep_socket)
                request = deserialize_message(Vector{UInt8}(msg_bytes), TrainingSampleBatch)

                # Send batch response
                batch_size = length(request.samples)  # Request uses samples field for batch size
                if batch_size == 0
                    batch_size = 2048  # Default batch size
                end

                response = get_batch_for_training(manager.manager, batch_size)
                response_bytes = serialize_message(response)
                ZMQ.send(rep_socket, response_bytes)
            end

            # Small sleep if nothing to do
            if !ZMQ.isready(pull_socket) && !ZMQ.isready(rep_socket)
                sleep(0.001)
            end

        catch e
            if e isa InterruptException
                manager.manager.running = false
            else
                @error "Replay manager error" exception=(e, catch_backtrace())
            end
        end
    end

    # Cleanup
    try
        ZMQ.close(rep_socket)
    catch end
    shutdown_replay_manager(manager.manager)
end

#####
##### Lifecycle management
#####

"""
    shutdown_replay_manager(manager::ReplayBufferManager)

Shutdown the replay buffer manager.
"""
function shutdown_replay_manager(manager::ReplayBufferManager)
    manager.running = false
    try
        ZMQ.close(manager.socket)
        ZMQ.close(manager.context)
    catch e
        @warn "Error during replay manager shutdown" exception=e
    end
end

#####
##### Statistics
#####

"""
    get_buffer_stats(manager::ReplayBufferManager) -> Dict

Get current buffer statistics.
"""
function get_buffer_stats(manager::ReplayBufferManager)
    lock(manager.lock) do
        stats = copy(manager.stats)
        stats["current_buffer_size"] = length(manager.buffer)
        stats["buffer_capacity"] = manager.config.capacity
        stats["fill_percentage"] = 100.0 * length(manager.buffer) / manager.config.capacity

        uptime = time() - stats["start_time"]
        stats["uptime_seconds"] = uptime
        if uptime > 0
            stats["samples_per_second"] = stats["total_samples_received"] / uptime
            stats["games_per_minute"] = 60.0 * stats["total_games_received"] / uptime
        end

        return stats
    end
end

#####
##### Sample age tracking
#####

"""
    compute_sample_age_distribution(manager::ReplayBufferManager) -> Dict

Compute the age distribution of samples in the buffer.
"""
function compute_sample_age_distribution(manager::ReplayBufferManager)
    lock(manager.lock) do
        if isempty(manager.buffer)
            return Dict{String, Float64}()
        end

        # Since samples are in order (circular buffer), oldest is at beginning
        n = length(manager.buffer)
        ages = collect(n:-1:1)

        return Dict{String, Float64}(
            "min_age" => minimum(ages),
            "max_age" => maximum(ages),
            "mean_age" => mean(ages),
            "median_age" => median(ages),
        )
    end
end
