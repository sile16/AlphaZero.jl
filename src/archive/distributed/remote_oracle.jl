#####
##### Remote Oracle - Client for ZMQ Inference Server
#####

"""
Remote oracle that sends inference requests to a centralized server.

The RemoteOracle acts as a neural network oracle for MCTS, but delegates
actual inference to a remote GPU server via ZMQ.
"""

using ZMQ

#####
##### Remote Oracle Implementation
#####

"""
    ZMQRemoteOracle

Oracle that delegates inference to a remote ZMQ server.

# Fields
- `gspec`: Game specification
- `endpoint`: Server endpoint string
- `worker_id`: Unique identifier for this client
- `context`: ZMQ context
- `socket`: ZMQ REQ socket
- `request_counter`: Counter for unique request IDs
- `is_multihead`: Whether server returns multi-head values
- `connected`: Connection status
"""
mutable struct ZMQRemoteOracle
    gspec::AbstractGameSpec
    endpoint::String
    worker_id::String
    context::ZMQ.Context
    socket::ZMQ.Socket
    request_counter::UInt64
    is_multihead::Bool
    connected::Bool

    function ZMQRemoteOracle(
        gspec::AbstractGameSpec,
        endpoint::String,
        worker_id::String;
        is_multihead::Bool=false
    )
        ctx = ZMQ.Context()
        socket = ZMQ.Socket(ctx, ZMQ.REQ)

        oracle = new(gspec, endpoint, worker_id, ctx, socket, UInt64(0), is_multihead, false)
        connect!(oracle)
        return oracle
    end
end

"""
    ZMQRemoteOracle(gspec, config::EndpointConfig, worker_id; kwargs...)

Create a remote oracle from endpoint configuration.
"""
function ZMQRemoteOracle(
    gspec::AbstractGameSpec,
    config::EndpointConfig,
    worker_id::String;
    kwargs...
)
    endpoint = endpoint_string(config, bind=false)
    return ZMQRemoteOracle(gspec, endpoint, worker_id; kwargs...)
end

#####
##### Connection management
#####

"""
    connect!(oracle::ZMQRemoteOracle)

Connect to the inference server.
"""
function connect!(oracle::ZMQRemoteOracle)
    if !oracle.connected
        ZMQ.connect(oracle.socket, oracle.endpoint)
        oracle.connected = true
        @debug "Connected to inference server at $(oracle.endpoint)"
    end
end

"""
    disconnect!(oracle::ZMQRemoteOracle)

Disconnect from the inference server.
"""
function disconnect!(oracle::ZMQRemoteOracle)
    if oracle.connected
        try
            ZMQ.close(oracle.socket)
            ZMQ.close(oracle.context)
        catch e
            @warn "Error disconnecting from inference server" exception=e
        end
        oracle.connected = false
    end
end

"""
    reconnect!(oracle::ZMQRemoteOracle)

Reconnect to the inference server after a failure.
"""
function reconnect!(oracle::ZMQRemoteOracle)
    disconnect!(oracle)
    oracle.context = ZMQ.Context()
    oracle.socket = ZMQ.Socket(oracle.context, ZMQ.REQ)
    connect!(oracle)
end

#####
##### Inference interface
#####

"""
    (oracle::ZMQRemoteOracle)(state)

Evaluate a single state using the remote inference server.
Returns (policy, value) tuple compatible with MCTS oracle interface.
"""
function (oracle::ZMQRemoteOracle)(state)
    # Single state inference
    policies, values = batch_inference(oracle, [state])
    return policies[1], values[1]
end

"""
    batch_inference(oracle::ZMQRemoteOracle, states) -> (policies, values)

Evaluate a batch of states using the remote inference server.
"""
function batch_inference(oracle::ZMQRemoteOracle, states)
    # Serialize states
    serialized_states = [Vector{Float32}(GI.vectorize_state(oracle.gspec, s)) for s in states]

    # Create request
    oracle.request_counter += 1
    request = InferenceRequest(
        worker_id=oracle.worker_id,
        request_id=oracle.request_counter,
        states=serialized_states
    )

    # Send request
    request_bytes = serialize_message(request)

    max_retries = 3
    for attempt in 1:max_retries
        try
            ZMQ.send(oracle.socket, request_bytes)

            # Receive response
            response_bytes = ZMQ.recv(oracle.socket)
            response = deserialize_message(Vector{UInt8}(response_bytes), InferenceResponse)

            # Verify response matches request
            if response.request_id != request.request_id
                @warn "Response ID mismatch: expected $(request.request_id), got $(response.request_id)"
            end

            # Update multihead flag from response
            oracle.is_multihead = response.is_multihead

            # Convert to oracle output format
            policies = response.policies
            values = if response.is_multihead
                # Convert MultiHeadValue to scalar equity for MCTS
                [compute_scalar_value(v) for v in response.values]
            else
                response.values
            end

            return policies, values

        catch e
            if attempt < max_retries
                @warn "Inference request failed (attempt $attempt/$max_retries), reconnecting..." exception=e
                reconnect!(oracle)
            else
                @error "Inference request failed after $max_retries attempts" exception=e
                rethrow(e)
            end
        end
    end
end

"""
    compute_scalar_value(v::MultiHeadValue) -> Float32

Compute scalar equity from multi-head value output.
Equity = P(win) * (1 + P(g|w) + P(bg|w)) - P(loss) * (1 + P(g|l) + P(bg|l))
"""
function compute_scalar_value(v::MultiHeadValue)
    p_win = v.p_win
    p_loss = 1.0f0 - p_win

    win_equity = p_win * (1.0f0 + v.p_gammon_win + v.p_bg_win)
    loss_equity = p_loss * (1.0f0 + v.p_gammon_loss + v.p_bg_loss)

    # Normalize to [-1, 1] range (max equity is 3, min is -3)
    equity = (win_equity - loss_equity) / 3.0f0
    return clamp(equity, -1.0f0, 1.0f0)
end

compute_scalar_value(v::Float32) = v
compute_scalar_value(v::Number) = Float32(v)

#####
##### MCTS Oracle adapter
#####

"""
    MCTSOracleAdapter

Wrapper that makes ZMQRemoteOracle compatible with the MCTS oracle interface.
"""
struct MCTSOracleAdapter{O<:ZMQRemoteOracle}
    oracle::O
    actions_mask_cache::Dict{Any, Vector{Bool}}

    function MCTSOracleAdapter(oracle::ZMQRemoteOracle)
        cache = Dict{Any, Vector{Bool}}()
        new{typeof(oracle)}(oracle, cache)
    end
end

"""
    (adapter::MCTSOracleAdapter)(state)

MCTS-compatible oracle call returning (policy, value).
The policy is masked to valid actions.
"""
function (adapter::MCTSOracleAdapter)(state)
    policy_full, value = adapter.oracle(state)

    # Get actions mask
    game = GI.init(adapter.oracle.gspec, state)
    mask = GI.actions_mask(game)

    # Mask and renormalize policy
    policy = policy_full[mask]
    policy_sum = sum(policy)
    if policy_sum > 0
        policy ./= policy_sum
    else
        # Uniform over valid actions if all zeroed
        policy .= 1.0f0 / length(policy)
    end

    return (policy, value)
end

#####
##### Batch oracle for multiple parallel games
#####

"""
    BatchingRemoteOracle

Oracle that batches multiple inference requests for efficiency.
Useful when running multiple parallel games.
"""
mutable struct BatchingRemoteOracle{O<:ZMQRemoteOracle}
    oracle::O
    pending_states::Vector{Any}
    pending_callbacks::Vector{Any}
    batch_size::Int
    auto_flush::Bool

    function BatchingRemoteOracle(
        oracle::ZMQRemoteOracle;
        batch_size::Int=32,
        auto_flush::Bool=true
    )
        new{typeof(oracle)}(oracle, [], [], batch_size, auto_flush)
    end
end

"""
    queue_inference!(batching_oracle, state, callback)

Queue a state for batch inference.
The callback will be called with (policy, value) when results are ready.
"""
function queue_inference!(oracle::BatchingRemoteOracle, state, callback)
    push!(oracle.pending_states, state)
    push!(oracle.pending_callbacks, callback)

    if oracle.auto_flush && length(oracle.pending_states) >= oracle.batch_size
        flush_batch!(oracle)
    end
end

"""
    flush_batch!(oracle::BatchingRemoteOracle)

Send all pending requests as a batch and call callbacks with results.
"""
function flush_batch!(oracle::BatchingRemoteOracle)
    if isempty(oracle.pending_states)
        return
    end

    # Get batch inference results
    policies, values = batch_inference(oracle.oracle, oracle.pending_states)

    # Call callbacks
    for (i, callback) in enumerate(oracle.pending_callbacks)
        callback(policies[i], values[i])
    end

    # Clear pending
    empty!(oracle.pending_states)
    empty!(oracle.pending_callbacks)
end

#####
##### Cleanup
#####

"""
    Base.close(oracle::ZMQRemoteOracle)

Close the oracle connection.
"""
function Base.close(oracle::ZMQRemoteOracle)
    disconnect!(oracle)
end

# Finalizer to ensure cleanup
function Base.finalize(oracle::ZMQRemoteOracle)
    close(oracle)
end
