#####
##### ZMQ-based Inference Server
#####

"""
GPU inference server for distributed training.

The inference server:
- Receives batched state evaluation requests from workers
- Batches requests for efficient GPU inference
- Returns policy and value predictions
"""

using ZMQ
import Flux

#####
##### Server state
#####

mutable struct ZMQInferenceServer
    network::AbstractNetwork
    gspec::AbstractGameSpec
    config::InferenceServerConfig
    context::ZMQ.Context
    socket::ZMQ.Socket
    running::Bool
    stats::Dict{String, Any}

    function ZMQInferenceServer(
        network::AbstractNetwork,
        gspec::AbstractGameSpec,
        config::InferenceServerConfig
    )
        ctx = ZMQ.Context()
        socket = ZMQ.Socket(ctx, ZMQ.REP)
        ZMQ.bind(socket, endpoint_string(config.endpoint))

        # Move network to GPU if configured
        if config.use_gpu
            network = Network.to_gpu(network)
        end
        Network.set_test_mode!(network, true)

        stats = Dict{String, Any}(
            "requests_processed" => 0,
            "states_processed" => 0,
            "batches_processed" => 0,
            "total_inference_time" => 0.0,
            "start_time" => time(),
        )

        return new(network, gspec, config, ctx, socket, false, stats)
    end
end

#####
##### Core inference
#####

"""
    evaluate_batch(server, states::Vector{Vector{Float32}}) -> (policies, values)

Run neural network inference on a batch of states.
"""
function evaluate_batch(server::ZMQInferenceServer, states::Vector{Vector{Float32}})
    # Stack states into batch tensor
    batch_size = length(states)
    state_dim = length(states[1])
    X = zeros(Float32, state_dim, batch_size)
    for (i, s) in enumerate(states)
        X[:, i] = s
    end

    # Move to GPU if needed
    X_device = Network.convert_input(server.network, X)

    # Run inference
    start_time = time()
    result = Network.forward(server.network, X_device)
    inference_time = time() - start_time

    # Update stats
    server.stats["total_inference_time"] += inference_time

    # Check if multi-head network
    is_multihead = server.network isa FluxLib.FCResNetMultiHead

    if is_multihead
        # Multi-head output: (policy, p_win, p_gw, p_bgw, p_gl, p_bgl)
        P, V_win, V_gw, V_bgw, V_gl, V_bgl = result
        P = Network.convert_output(server.network, P)
        V_win = Network.convert_output(server.network, V_win)
        V_gw = Network.convert_output(server.network, V_gw)
        V_bgw = Network.convert_output(server.network, V_bgw)
        V_gl = Network.convert_output(server.network, V_gl)
        V_bgl = Network.convert_output(server.network, V_bgl)

        policies = [Vector{Float32}(P[:, i]) for i in 1:batch_size]
        values = [MultiHeadValue(
            V_win[1, i], V_gw[1, i], V_bgw[1, i], V_gl[1, i], V_bgl[1, i]
        ) for i in 1:batch_size]

        return policies, values, true
    else
        # Standard two-head output: (policy, value)
        P, V = result
        P = Network.convert_output(server.network, P)
        V = Network.convert_output(server.network, V)

        policies = [Vector{Float32}(P[:, i]) for i in 1:batch_size]
        values = [Float32(V[1, i]) for i in 1:batch_size]

        return policies, values, false
    end
end

#####
##### Request handling
#####

"""
    handle_request(server, request::InferenceRequest) -> InferenceResponse

Process an inference request and return response.
"""
function handle_request(server::ZMQInferenceServer, request::InferenceRequest)
    states = request.states
    batch_size = length(states)

    # Run inference
    policies, values, is_multihead = evaluate_batch(server, states)

    # Update stats
    server.stats["requests_processed"] += 1
    server.stats["states_processed"] += batch_size
    server.stats["batches_processed"] += 1

    return InferenceResponse(
        request_id=request.request_id,
        policies=policies,
        values=values,
        is_multihead=is_multihead
    )
end

#####
##### Server main loop
#####

"""
    run_inference_server(server::ZMQInferenceServer)

Run the inference server main loop.
"""
function run_inference_server(server::ZMQInferenceServer)
    server.running = true
    @info "Inference server starting on $(endpoint_string(server.config.endpoint))"

    # Warmup
    if server.config.warmup_batches > 0
        @info "Running $(server.config.warmup_batches) warmup batches..."
        dummy_state = zeros(Float32, GI.state_dim(server.gspec))
        for _ in 1:server.config.warmup_batches
            dummy_batch = [dummy_state for _ in 1:server.config.batch_size]
            evaluate_batch(server, dummy_batch)
        end
        @info "Warmup complete"
    end

    while server.running
        try
            # Receive request (blocking)
            msg_bytes = ZMQ.recv(server.socket)
            msg_data = Vector{UInt8}(msg_bytes)

            # Deserialize request
            request = deserialize_message(msg_data, InferenceRequest)

            # Check for shutdown command
            if request.worker_id == "__shutdown__"
                @info "Received shutdown command"
                server.running = false
                # Send acknowledgment
                response = InferenceResponse(
                    request_id=request.request_id,
                    policies=[],
                    values=[],
                    is_multihead=false
                )
                response_bytes = serialize_message(response)
                ZMQ.send(server.socket, response_bytes)
                break
            end

            # Handle inference request
            response = handle_request(server, request)

            # Send response
            response_bytes = serialize_message(response)
            ZMQ.send(server.socket, response_bytes)

        catch e
            if e isa InterruptException
                @info "Inference server interrupted"
                server.running = false
            else
                @error "Inference server error" exception=(e, catch_backtrace())
            end
        end
    end

    # Cleanup
    shutdown_inference_server(server)
    @info "Inference server stopped"

    return server.stats
end

"""
    run_inference_server_async(server::ZMQInferenceServer) -> Task

Run the inference server in a background task.
"""
function run_inference_server_async(server::ZMQInferenceServer)
    return @async run_inference_server(server)
end

#####
##### Lifecycle management
#####

"""
    shutdown_inference_server(server::ZMQInferenceServer)

Shutdown the inference server and clean up resources.
"""
function shutdown_inference_server(server::ZMQInferenceServer)
    server.running = false
    try
        ZMQ.close(server.socket)
        ZMQ.close(server.context)
    catch e
        @warn "Error during inference server shutdown" exception=e
    end
end

"""
    update_network!(server::ZMQInferenceServer, network::AbstractNetwork)

Update the server's network with new weights.
Thread-safe network update.
"""
function update_network!(server::ZMQInferenceServer, network::AbstractNetwork)
    # Move to GPU if needed
    if server.config.use_gpu
        network = Network.to_gpu(network)
    end
    Network.set_test_mode!(network, true)

    # Atomic swap
    server.network = network
end

"""
    update_network_weights!(server::ZMQInferenceServer, weights::Vector{UInt8})

Update network weights from serialized data.
"""
function update_network_weights!(server::ZMQInferenceServer, weights::Vector{UInt8})
    weight_arrays = deserialize_network_weights(weights)
    load_weights_into_network!(server.network, weight_arrays)
end

#####
##### Statistics
#####

"""
    get_server_stats(server::ZMQInferenceServer) -> Dict

Get current server statistics.
"""
function get_server_stats(server::ZMQInferenceServer)
    stats = copy(server.stats)
    uptime = time() - stats["start_time"]
    stats["uptime_seconds"] = uptime

    if stats["batches_processed"] > 0
        stats["avg_inference_time"] = stats["total_inference_time"] / stats["batches_processed"]
        stats["states_per_second"] = stats["states_processed"] / uptime
    else
        stats["avg_inference_time"] = 0.0
        stats["states_per_second"] = 0.0
    end

    return stats
end

#####
##### Convenience constructor
#####

"""
    create_inference_server(
        network_path::String,
        gspec::AbstractGameSpec;
        port::Int=5555,
        batch_size::Int=64,
        use_gpu::Bool=true
    ) -> ZMQInferenceServer

Create an inference server from a saved network file.
"""
function create_inference_server(
    network_path::String,
    gspec::AbstractGameSpec;
    port::Int=5555,
    batch_size::Int=64,
    use_gpu::Bool=true
)
    # Load network
    network = Network.load(network_path)

    config = InferenceServerConfig(
        endpoint=EndpointConfig(port=port),
        batch_size=batch_size,
        use_gpu=use_gpu
    )

    return ZMQInferenceServer(network, gspec, config)
end

"""
    create_inference_server(
        network::AbstractNetwork,
        gspec::AbstractGameSpec,
        config::InferenceServerConfig
    ) -> ZMQInferenceServer

Create an inference server from an existing network.
"""
function create_inference_server(
    network::AbstractNetwork,
    gspec::AbstractGameSpec,
    config::InferenceServerConfig
)
    return ZMQInferenceServer(network, gspec, config)
end
