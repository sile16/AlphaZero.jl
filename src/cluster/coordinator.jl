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
Sample a random batch from the replay buffer (uniform sampling).
Returns (samples, indices, weights) where weights are all 1.0.
"""
function sample_batch(coord::ClusterCoordinator, batch_size::Int)
    n = length(coord.buffer)
    if n < batch_size
        return nothing, nothing, nothing
    end
    indices = rand(1:n, batch_size)
    samples = [coord.buffer[i] for i in indices]
    weights = ones(Float32, batch_size)
    return samples, indices, weights
end

"""
Sample a batch using Prioritized Experience Replay (PER).

Uses proportional prioritization: P(i) = p_i^α / Σ p_k^α
where p_i = |δ_i| + ε (TD-error + small constant).

Returns (samples, indices, importance_weights) for bias correction.
Importance weights: w_i = (N · P(i))^(-β) / max_j(w_j)

Based on Schaul et al. 2016 "Prioritized Experience Replay".
"""
function sample_batch_prioritized(
    coord::ClusterCoordinator,
    batch_size::Int,
    config::PrioritizedSamplingConfig,
    current_step::Int
)
    n = length(coord.buffer)
    if n < batch_size
        return nothing, nothing, nothing
    end

    # Compute priorities: p_i = priority + epsilon
    # Then raise to power alpha
    priorities = Float64[]
    for sample in coord.buffer
        p = Float64(max(sample.priority, 0.0f0) + config.epsilon)
        push!(priorities, p ^ config.alpha)
    end

    # Compute sampling probabilities
    total_priority = sum(priorities)
    probs = priorities ./ total_priority

    # Sample according to probabilities (with replacement)
    indices = Int[]
    for iter in 1:batch_size
        r = rand()
        cumsum_p = 0.0
        selected = false
        for (j, p) in enumerate(probs)
            cumsum_p += p
            if r <= cumsum_p
                push!(indices, j)
                selected = true
                break
            end
        end
        # Fallback if numerical issues (r > sum of all probs due to floating point)
        if !selected
            push!(indices, rand(1:n))
        end
    end

    # Ensure we got the right number of samples (shouldn't happen but just in case)
    while length(indices) < batch_size
        push!(indices, rand(1:n))
    end

    # Compute importance sampling weights
    # w_i = (N * P(i))^(-beta)
    beta = get_beta(config, current_step)
    weights = Float32[]
    for idx in indices
        w = (n * probs[idx]) ^ (-beta)
        push!(weights, Float32(w))
    end

    # Normalize weights so max = 1
    max_weight = maximum(weights)
    if max_weight > 0
        weights ./= max_weight
    end

    samples = [coord.buffer[i] for i in indices]
    return samples, indices, weights
end

"""
Update priorities for sampled indices based on TD-errors.
Called after training step with computed TD-errors.
"""
function update_priorities!(
    coord::ClusterCoordinator,
    indices::Vector{Int},
    td_errors::Vector{Float32}
)
    for (idx, td_error) in zip(indices, td_errors)
        if idx <= length(coord.buffer)
            old_sample = coord.buffer[idx]
            # Create new sample with updated priority
            coord.buffer[idx] = ClusterSample(
                old_sample.state,
                old_sample.policy,
                old_sample.value,
                old_sample.turn,
                old_sample.is_chance,
                old_sample.equity_p_win,
                old_sample.equity_p_gw,
                old_sample.equity_p_bgw,
                old_sample.equity_p_gl,
                old_sample.equity_p_bgl,
                old_sample.has_equity,
                abs(td_error),  # New priority = |TD-error|
                old_sample.added_step,
                old_sample.last_reanalyze_step,
                old_sample.reanalyze_count,
                old_sample.model_iter_reanalyzed
            )
        end
    end
end

"""
Prepare training batch from ClusterSamples.

If importance_weights is provided (from PER), they are incorporated into the
sample weights W for proper bias correction.
"""
function prepare_training_batch(
    coord::ClusterCoordinator,
    samples::Vector{ClusterSample};
    importance_weights::Union{Nothing, Vector{Float32}} = nothing
)
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
        # Apply importance sampling weight if provided (for PER bias correction)
        if !isnothing(importance_weights)
            ws[i] = Float32[importance_weights[i]]
        else
            ws[i] = Float32[1]
        end
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
Run one training step with uniform sampling.
Returns (total_loss, policy_loss, value_loss, reg_loss) or nothing.
"""
function training_step!(coord::ClusterCoordinator, batch_size::Int)
    samples, indices, weights = sample_batch(coord, batch_size)
    if isnothing(samples)
        return nothing
    end

    batch_data = prepare_training_batch(coord, samples)

    Wmean = mean(batch_data.W)
    Hp = 0.0f0

    loss_fn(nn) = losses(nn, coord.learning_params, Wmean, Hp, batch_data)[1]

    l, grads = Flux.withgradient(loss_fn, coord.network)
    Flux.update!(coord.opt_state, coord.network, grads[1])

    # Get individual loss components for logging (no gradient needed)
    L, Lp, Lv, Lreg, Linv = losses(coord.network, coord.learning_params, Wmean, Hp, batch_data)

    return (Float64(L), Float64(Lp), Float64(Lv), Float64(Lreg))
end

"""
Run one training step with Prioritized Experience Replay (PER).

Returns (total_loss, policy_loss, value_loss, reg_loss) or nothing.
Also updates sample priorities based on TD-errors.

Based on Schaul et al. 2016 "Prioritized Experience Replay".
"""
function training_step_prioritized!(
    coord::ClusterCoordinator,
    batch_size::Int,
    per_config::PrioritizedSamplingConfig,
    current_step::Int
)
    samples, indices, importance_weights = sample_batch_prioritized(
        coord, batch_size, per_config, current_step
    )
    if isnothing(samples)
        return nothing
    end

    # Prepare batch with importance sampling weights for bias correction
    batch_data = prepare_training_batch(coord, samples; importance_weights=importance_weights)

    Wmean = mean(batch_data.W)
    Hp = 0.0f0

    loss_fn(nn) = losses(nn, coord.learning_params, Wmean, Hp, batch_data)[1]

    l, grads = Flux.withgradient(loss_fn, coord.network)
    Flux.update!(coord.opt_state, coord.network, grads[1])

    # Get individual loss components for logging (no gradient needed)
    L, Lp, Lv, Lreg, Linv = losses(coord.network, coord.learning_params, Wmean, Hp, batch_data)

    # Compute TD-errors and update priorities
    # TD-error = |predicted_value - target_value|
    td_errors = compute_td_errors(coord, samples, batch_data)
    update_priorities!(coord, indices, td_errors)

    return (Float64(L), Float64(Lp), Float64(Lv), Float64(Lreg))
end

"""
Compute TD-errors for priority updates.
TD-error = |network_value - target_value|
"""
function compute_td_errors(
    coord::ClusterCoordinator,
    samples::Vector{ClusterSample},
    batch_data
)
    # Get network predictions
    # Network.forward returns (P, V) tuple
    P, V = Network.forward(coord.network, batch_data.X)

    # Move values to CPU if on GPU
    cpu_fn = coord.use_gpu ? Flux.cpu : identity
    predicted_values = cpu_fn(V)[1, :]  # V has shape (1, batch_size)

    # Target values from samples
    target_values = [s.value for s in samples]

    # TD-error = |predicted - target|
    td_errors = Float32[abs(predicted_values[i] - target_values[i]) for i in 1:length(samples)]

    return td_errors
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
