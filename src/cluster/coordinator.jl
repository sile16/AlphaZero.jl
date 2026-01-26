#####
##### Cluster Coordinator
#####
##### Central coordinator that manages training, replay buffer, and weight distribution.
#####

using Distributed
import Flux

"""
    ClusterCoordinator

Central coordinator for distributed AlphaZero training.
Manages:
- Replay buffer
- Training loop
- Weight distribution to workers
- Evaluation
"""
mutable struct ClusterCoordinator
    gspec::AbstractGameSpec
    network::AbstractNetwork
    learning_params::LearningParams
    mcts_params::MctsParams

    # Replay buffer
    buffer::Vector{ClusterSample}
    buffer_capacity::Int

    # Training state
    iteration::Int
    total_games::Int
    total_samples::Int
    opt_state::Any  # Flux optimizer state

    # Configuration
    use_gpu::Bool
    checkpoint_dir::String

    # Channels (set when starting)
    sample_channel::Union{Nothing, RemoteChannel}
    weight_channels::Vector{RemoteChannel}  # One per worker for broadcasting
end

function ClusterCoordinator(
    gspec::AbstractGameSpec,
    network::AbstractNetwork,
    learning_params::LearningParams,
    mcts_params::MctsParams;
    buffer_capacity::Int = 100000,
    use_gpu::Bool = true,
    checkpoint_dir::String = "checkpoints"
)
    mkpath(checkpoint_dir)

    # Move network to GPU if needed
    if use_gpu
        network = Network.to_gpu(network)
    end

    # Setup optimizer
    opt_state = Flux.setup(Flux.Adam(learning_params.optimiser.lr), network)

    return ClusterCoordinator(
        gspec,
        network,
        learning_params,
        mcts_params,
        ClusterSample[],
        buffer_capacity,
        0,
        0,
        0,
        opt_state,
        use_gpu,
        checkpoint_dir,
        nothing,
        RemoteChannel[]
    )
end

"""
Add samples from a game batch to the replay buffer.
"""
function add_samples!(coord::ClusterCoordinator, batch::GameBatch)
    append!(coord.buffer, batch.samples)
    coord.total_games += 1
    coord.total_samples += length(batch.samples)

    # Trim to capacity (remove oldest)
    if length(coord.buffer) > coord.buffer_capacity
        deleteat!(coord.buffer, 1:(length(coord.buffer) - coord.buffer_capacity))
    end
end

"""
Sample a random batch from the replay buffer.
"""
function sample_batch(coord::ClusterCoordinator, batch_size::Int)
    n = length(coord.buffer)
    if n < batch_size
        return nothing
    end
    indices = rand(1:n, batch_size)
    return [coord.buffer[i] for i in indices]
end

"""
Prepare training batch from ClusterSamples.
"""
function prepare_training_batch(coord::ClusterCoordinator, samples::Vector{ClusterSample})
    n = length(samples)
    state_shape = GI.state_dim(coord.gspec)
    policy_dim = GI.num_actions(coord.gspec)

    # Prepare arrays
    xs = Vector{Array{Float32}}(undef, n)
    ws = Vector{Vector{Float32}}(undef, n)
    ps = Vector{Vector{Float32}}(undef, n)
    vs = Vector{Vector{Float32}}(undef, n)
    as = Vector{Vector{Float32}}(undef, n)
    is_chances = Vector{Vector{Float32}}(undef, n)
    eq_wins = Vector{Vector{Float32}}(undef, n)
    eq_gws = Vector{Vector{Float32}}(undef, n)
    eq_bgws = Vector{Vector{Float32}}(undef, n)
    eq_gls = Vector{Vector{Float32}}(undef, n)
    eq_bgls = Vector{Vector{Float32}}(undef, n)
    has_eqs = Vector{Vector{Float32}}(undef, n)

    for (i, s) in enumerate(samples)
        xs[i] = reshape(s.state, state_shape)
        ws[i] = Float32[1]
        ps[i] = s.policy
        vs[i] = Float32[s.value]

        # Action mask from policy (non-zero = valid)
        if s.is_chance
            as[i] = ones(Float32, policy_dim)
        else
            as[i] = Float32.(s.policy .> 0)
        end

        is_chances[i] = Float32[s.is_chance ? 1.0f0 : 0.0f0]
        eq_wins[i] = Float32[s.equity_p_win]
        eq_gws[i] = Float32[s.equity_p_gw]
        eq_bgws[i] = Float32[s.equity_p_bgw]
        eq_gls[i] = Float32[s.equity_p_gl]
        eq_bgls[i] = Float32[s.equity_p_bgl]
        has_eqs[i] = Float32[s.has_equity ? 1.0f0 : 0.0f0]
    end

    # Batch using Flux
    W = Flux.batch(ws)
    X = Flux.batch(xs)
    A = Flux.batch(as)
    P = Flux.batch(ps)
    V = Flux.batch(vs)
    IsChance = Flux.batch(is_chances)
    EqWin = Flux.batch(eq_wins)
    EqGW = Flux.batch(eq_gws)
    EqBGW = Flux.batch(eq_bgws)
    EqGL = Flux.batch(eq_gls)
    EqBGL = Flux.batch(eq_bgls)
    HasEquity = Flux.batch(has_eqs)

    f32(arr) = convert(AbstractArray{Float32}, arr)
    batch = map(f32, (; W, X, A, P, V, IsChance, EqWin, EqGW, EqBGW, EqGL, EqBGL, HasEquity))

    if coord.use_gpu
        batch = Network.convert_input_tuple(coord.network, batch)
    end

    return batch
end

"""
Run one training step.
"""
function training_step!(coord::ClusterCoordinator, batch_size::Int)
    samples = sample_batch(coord, batch_size)
    if isnothing(samples)
        return nothing
    end

    batch_data = prepare_training_batch(coord, samples)

    Wmean = mean(batch_data.W)
    Hp = 0.0f0

    loss_fn(nn) = losses(nn, coord.learning_params, Wmean, Hp, batch_data)[1]

    l, grads = Flux.withgradient(loss_fn, coord.network)
    Flux.update!(coord.opt_state, coord.network, grads[1])

    return Float64(l)
end

"""
Serialize current network weights for distribution.
"""
function get_network_weights(coord::ClusterCoordinator)
    cpu_network = Network.to_cpu(coord.network)
    return serialize_weights(cpu_network)
end

"""
Broadcast weights to all workers.
"""
function broadcast_weights!(coord::ClusterCoordinator)
    weights = get_network_weights(coord)
    update = WeightUpdate(coord.iteration, weights, time())

    for ch in coord.weight_channels
        # Non-blocking put - if channel is full, skip (worker will catch up)
        if isready(ch)
            try
                take!(ch)  # Remove old weights
            catch end
        end
        try
            put!(ch, update)
        catch e
            @warn "Failed to send weights to worker" exception=e
        end
    end
end

"""
Save checkpoint.
"""
function save_checkpoint!(coord::ClusterCoordinator)
    cpu_network = Network.to_cpu(coord.network)
    save_weights(
        joinpath(coord.checkpoint_dir, "network_iter$(coord.iteration).data"),
        cpu_network
    )
    save_weights(
        joinpath(coord.checkpoint_dir, "latest.data"),
        cpu_network
    )
end

"""
Collect samples from workers (non-blocking).
"""
function collect_samples!(coord::ClusterCoordinator; max_batches::Int = 100)
    collected = 0
    while isready(coord.sample_channel) && collected < max_batches
        try
            batch = take!(coord.sample_channel)
            add_samples!(coord, batch)
            collected += 1
        catch e
            break
        end
    end
    return collected
end

"""
Get current statistics.
"""
function get_stats(coord::ClusterCoordinator)
    return TrainingMetrics(
        coord.iteration,
        0.0,  # Loss filled in by caller
        length(coord.buffer),
        coord.total_games,
        coord.total_samples,
        0.0   # Games/min filled in by caller
    )
end
