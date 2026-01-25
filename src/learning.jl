#####
##### Converting samples
#####

# A samples collection is represented on the learning side as a (W, X, A, P, V)
# named-tuple. Each component is a `Float32` tensor whose last dimension corresponds
# to the sample index. Writing `n` the number of samples and `a` the total
# number of actions:
# - W (size 1×n) contains the samples weights
# - X (size …×n) contains the board representations
# - A (size a×n) contains the action masks (values are either 0 or 1)
# - P (size a×n) contains the recorded MCTS policies
# - V (size 1×n) contains the recorded values
# Note that the weight of a sample is computed as an increasing
# function of its `n` field.

function convert_sample(
    gspec::AbstractGameSpec,
    wp::SamplesWeighingPolicy,
    e::TrainingSample)

  if wp == CONSTANT_WEIGHT
    w = Float32[1]
  elseif wp == LOG_WEIGHT
    w = Float32[log2(e.n) + 1]
  else
    @assert wp == LINEAR_WEIGHT
    w = Float32[e.n]
  end
  x = GI.vectorize_state(gspec, e.s)
  a = GI.actions_mask(GI.init(gspec, e.s))
  p = zeros(size(a))
  # For chance nodes, policy should be empty (no targets)
  if !e.is_chance && !isempty(e.π)
    p[a] = e.π
  end
  v = [e.z]
  is_chance = [e.is_chance ? 1f0 : 0f0]  # Convert to float for batching

  # Multi-head equity targets (if available)
  if !isnothing(e.equity)
    eq = e.equity
    eq_win = [eq.p_win]
    eq_gw = [eq.p_gammon_win]
    eq_bgw = [eq.p_bg_win]
    eq_gl = [eq.p_gammon_loss]
    eq_bgl = [eq.p_bg_loss]
    has_equity = [1f0]  # Flag indicating equity targets are present
  else
    eq_win = [0f0]
    eq_gw = [0f0]
    eq_bgw = [0f0]
    eq_gl = [0f0]
    eq_bgl = [0f0]
    has_equity = [0f0]  # No equity targets
  end

  return (; w, x, a, p, v, is_chance, eq_win, eq_gw, eq_bgw, eq_gl, eq_bgl, has_equity)
end

function convert_samples(
    gspec::AbstractGameSpec,
    wp::SamplesWeighingPolicy,
    es::AbstractVector{<:TrainingSample})

  ces = [convert_sample(gspec, wp, e) for e in es]
  W = Flux.batch([e.w for e in ces])
  X = Flux.batch([e.x for e in ces])
  A = Flux.batch([e.a for e in ces])
  P = Flux.batch([e.p for e in ces])
  V = Flux.batch([e.v for e in ces])
  IsChance = Flux.batch([e.is_chance for e in ces])

  # Multi-head equity targets
  EqWin = Flux.batch([e.eq_win for e in ces])
  EqGW = Flux.batch([e.eq_gw for e in ces])
  EqBGW = Flux.batch([e.eq_bgw for e in ces])
  EqGL = Flux.batch([e.eq_gl for e in ces])
  EqBGL = Flux.batch([e.eq_bgl for e in ces])
  HasEquity = Flux.batch([e.has_equity for e in ces])

  f32(arr) = convert(AbstractArray{Float32}, arr)
  return map(f32, (; W, X, A, P, V, IsChance, EqWin, EqGW, EqBGW, EqGL, EqBGL, HasEquity))
end

#####
##### Loss Function
#####

# Surprisingly, Flux does not like the following code (scalar operations):
# mse_wmean(ŷ, y, w) = sum((ŷ .- y).^2 .* w) / sum(w)
mse_wmean(ŷ, y, w) = sum((ŷ .- y) .* (ŷ .- y) .* w) / sum(w)

klloss_wmean(π̂, π, w) = -sum(π .* log.(π̂ .+ eps(eltype(π))) .* w) / sum(w)

entropy_wmean(π, w) = -sum(π .* log.(π .+ eps(eltype(π))) .* w) / sum(w)

wmean(x, w) = sum(x .* w) / sum(w)

# Binary cross-entropy loss (for multi-head outputs)
bce_wmean(ŷ, y, w) = -sum((y .* log.(ŷ .+ eps(eltype(y))) .+
                           (1f0 .- y) .* log.(1f0 .- ŷ .+ eps(eltype(y)))) .* w) / sum(w)

"""
    losses(nn, params, Wmean, Hp, batch)

Compute training losses for a neural network.
Supports both single-head and multi-head (equity) networks.
"""
function losses(nn, params, Wmean, Hp, batch)
  W, X, A, P, V, IsChance = batch.W, batch.X, batch.A, batch.P, batch.V, batch.IsChance

  # Ideally, we would only apply the L2 penalty to weight parameters and not
  # bias parameters. However, Flux currently cannot differentiate through
  # `Flux.modules`, which is used in the implementation of
  # `Network.regularized_params`. Thus, we regularize with respect to ALL
  # parameters, which does not make a big difference in practice anyway.
  # regws = Network.regularized_params(nn)
  regws = Network.params(nn)
  creg = params.l2_regularization
  cinv = params.nonvalidity_penalty

  # Mask for decision nodes only (policy loss should not apply to chance nodes)
  decision_mask = 1f0 .- IsChance
  W_decision = W .* decision_mask

  # Check if this is a multi-head network
  is_multihead = nn isa FluxLib.FCResNetMultiHead

  if is_multihead
    # Multi-head forward pass
    P̂, V̂_win, V̂_gw, V̂_bgw, V̂_gl, V̂_bgl, p_invalid =
      FluxLib.forward_normalized_multihead(nn, X, A)

    # Get equity targets
    EqWin, EqGW, EqBGW, EqGL, EqBGL, HasEquity =
      batch.EqWin, batch.EqGW, batch.EqBGW, batch.EqGL, batch.EqBGL, batch.HasEquity

    # Weight for samples that have equity targets
    W_equity = W .* HasEquity

    # Multi-head value losses (binary cross-entropy for each head)
    if sum(W_equity) > 0
      Lv_win = bce_wmean(V̂_win, EqWin, W_equity)
      Lv_gw = bce_wmean(V̂_gw, EqGW, W_equity)
      Lv_bgw = bce_wmean(V̂_bgw, EqBGW, W_equity)
      Lv_gl = bce_wmean(V̂_gl, EqGL, W_equity)
      Lv_bgl = bce_wmean(V̂_bgl, EqBGL, W_equity)
      Lv = Lv_win + Lv_gw + Lv_bgw + Lv_gl + Lv_bgl
    else
      # Fallback: use standard value loss if no equity targets
      V_normalized = V ./ params.rewards_renormalization
      equity = FluxLib.compute_equity(V̂_win, V̂_gw, V̂_bgw, V̂_gl, V̂_bgl)
      V̂_combined = equity ./ 3f0  # Scale to [-1, 1]
      Lv = mse_wmean(V̂_combined, V_normalized, W)
    end
  else
    # Standard single-head forward pass
    P̂, V̂, p_invalid = Network.forward_normalized(nn, X, A)
    V_normalized = V ./ params.rewards_renormalization
    V̂ = V̂ ./ params.rewards_renormalization

    # Value loss: For ALL nodes (including chance nodes)
    Lv = mse_wmean(V̂, V_normalized, W)
  end

  # Policy loss: ONLY for decision nodes
  # If there are no decision nodes in the batch, skip policy loss
  Lp = if sum(W_decision) > 0
    klloss_wmean(P̂, P, W_decision) - Hp
  else
    zero(Float32)
  end

  Lreg = iszero(creg) ?
    zero(Lv) :
    creg * sum(sum(w .* w) for w in regws)

  # Invalid action penalty: ONLY for decision nodes
  Linv = iszero(cinv) ?
    zero(Lv) :
    cinv * wmean(p_invalid, W_decision)

  L = (mean(W) / Wmean) * (Lp + Lv + Lreg + Linv)
  return (L, Lp, Lv, Lreg, Linv)
end

#####
##### Trainer Utility
#####

struct Trainer
  network :: AbstractNetwork
  samples :: AbstractVector{<:TrainingSample}
  params :: LearningParams
  data :: NamedTuple # (W, X, A, P, V, IsChance) tuple obtained after converting `samples`
  Wmean :: Float32
  Hp :: Float32
  batches_stream # infinite stateful iterator of training batches
  function Trainer(gspec, network, samples, params; test_mode=false)
    if params.use_position_averaging
      samples = merge_by_state(samples)
    end
    data = convert_samples(gspec, params.samples_weighing_policy, samples)
    network = Network.copy(network, on_gpu=params.use_gpu, test_mode=test_mode)
    W, X, A, P, V, IsChance = data.W, data.X, data.A, data.P, data.V, data.IsChance
    Wmean = mean(W)
    # Compute entropy only for decision nodes
    decision_mask = 1f0 .- IsChance
    W_decision = W .* decision_mask
    Hp = if sum(W_decision) > 0
      entropy_wmean(P, W_decision)
    else
      zero(Float32)
    end
    # Create a batches stream
    batchsize = min(params.batch_size, length(W))
    batches = Flux.DataLoader(data; batchsize, partial=false, shuffle=true)
    batches_stream = map(batches) do b
      Network.convert_input_tuple(network, b)
    end |> Util.cycle_iterator |> Iterators.Stateful
    return new(network, samples, params, data, Wmean, Hp, batches_stream)
  end
end

data_weights(tr::Trainer) = tr.data.W

num_samples(tr::Trainer) = length(data_weights(tr))

num_batches_total(tr::Trainer) = num_samples(tr) ÷ tr.params.batch_size

function get_trained_network(tr::Trainer)
  return Network.copy(tr.network, on_gpu=false, test_mode=true)
end

function batch_updates!(tr::Trainer, n)
  L(net, batch) = losses(net, tr.params, tr.Wmean, tr.Hp, batch)[1]
  data = Iterators.take(tr.batches_stream, n)
  ls = Vector{Float32}()
  Network.train!(tr.network, tr.params.optimiser, L, data, n) do i, l
    push!(ls, l)
  end
  Network.gc(tr.network)
  return ls
end

#####
##### Generating debugging reports
#####

function mean_learning_status(reports, ws)
  L     = wmean([r.loss.L     for r in reports], ws)
  Lp    = wmean([r.loss.Lp    for r in reports], ws)
  Lv    = wmean([r.loss.Lv    for r in reports], ws)
  Lreg  = wmean([r.loss.Lreg  for r in reports], ws)
  Linv  = wmean([r.loss.Linv  for r in reports], ws)
  Hpnet = wmean([r.Hpnet      for r in reports], ws)
  Hp    = wmean([r.Hp         for r in reports], ws)
  return Report.LearningStatus(Report.Loss(L, Lp, Lv, Lreg, Linv), Hp, Hpnet)
end

function learning_status(tr::Trainer, samples)
  # As done now, this is slighly inefficient as we solve the
  # same neural network inference problem twice
  W, X, A, P, V, IsChance = samples.W, samples.X, samples.A, samples.P, samples.V, samples.IsChance
  Ls = losses(tr.network, tr.params, tr.Wmean, tr.Hp, samples)
  Ls = Network.convert_output_tuple(tr.network, Ls)
  Pnet, _ = Network.forward_normalized(tr.network, X, A)
  # Compute entropy only for decision nodes
  decision_mask = 1f0 .- IsChance
  W_decision = W .* decision_mask
  Hpnet = if sum(W_decision) > 0
    entropy_wmean(Pnet, W_decision)
  else
    zero(Float32)
  end
  Hpnet = Network.convert_output(tr.network, Hpnet)
  return Report.LearningStatus(Report.Loss(Ls...), tr.Hp, Hpnet)
end

function learning_status(tr::Trainer)
  batchsize = min(tr.params.loss_computation_batch_size, num_samples(tr))
  batches = Flux.DataLoader(tr.data; batchsize, partial=true)
  reports = map(batches) do batch
    batch = Network.convert_input_tuple(tr.network, batch)
    return learning_status(tr, batch)
  end
  ws = [sum(batch.W) for batch in batches]
  return mean_learning_status(reports, ws)
end

function samples_report(tr::Trainer)
  status = learning_status(tr)
  # Samples in `tr.samples` can be merged by board or not
  num_samples = sum(e.n for e in tr.samples)
  num_boards = length(merge_by_state(tr.samples))
  Wtot = sum(data_weights(tr))
  return Report.Samples(num_samples, num_boards, Wtot, status)
end

function memory_report(
    mem::MemoryBuffer,
    nn::AbstractNetwork,
    learning_params::LearningParams,
    params::MemAnalysisParams
  )
  # It is important to load the neural network in test mode so as to not
  # overwrite the batch norm statistics based on biased data.
  Tr(samples) = Trainer(mem.gspec, nn, samples, learning_params, test_mode=true)
  all_samples = samples_report(Tr(get_experience(mem)))
  latest_batch = isempty(last_batch(mem)) ?
    all_samples :
    samples_report(Tr(last_batch(mem)))
  per_game_stage = begin
    es = get_experience(mem)
    sort!(es, by=(e->e.t))
    csize = ceil(Int, length(es) / params.num_game_stages)
    stages = collect(Iterators.partition(es, csize))
    map(stages) do es
      ts = [e.t for e in es]
      stats = samples_report(Tr(es))
      Report.StageSamples(minimum(ts), maximum(ts), stats)
    end
  end
  return Report.Memory(latest_batch, all_samples, per_game_stage)
end
