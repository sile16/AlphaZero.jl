#####
##### Cluster Worker
#####
##### Self-play worker types and helper functions.
##### The actual worker loop is in Cluster.jl using threads.
#####

"""
    ClusterWorker

Self-play worker state (for reference, actual implementation uses threads).
"""
mutable struct ClusterWorker
    worker_id::Int
    gspec::AbstractGameSpec
    network::Union{Nothing, AbstractNetwork}
    mcts_params::MctsParams
    games_played::Int
    samples_generated::Int
    current_iteration::Int
    running::Bool
end

function ClusterWorker(worker_id::Int, gspec::AbstractGameSpec, mcts_params::MctsParams)
    return ClusterWorker(
        worker_id,
        gspec,
        nothing,  # Network loaded later
        mcts_params,
        0,
        0,
        0,
        true
    )
end

"""
Load network weights into worker's local network.
"""
function update_worker_weights!(worker::ClusterWorker, weights::Vector{UInt8}, network_constructor)
    if isnothing(worker.network)
        # First time - create network
        worker.network = network_constructor(worker.gspec)
    end

    # Deserialize and load weights
    weight_data = deserialize_weights(weights)
    load_weights!(worker.network, weight_data)
    Network.set_test_mode!(worker.network, true)
end

"""
Stop the worker.
"""
function stop_worker!(worker::ClusterWorker)
    worker.running = false
end
