#####
##### Fully-Connected ResNet with Multi-Head Equity Output
##### Based on TD-Gammon/gnubg approach for backgammon
#####

using Flux: LayerNorm

"""
    EquityOutput

Container for multi-head equity output from the network.
Contains 5 joint cumulative probabilities that determine the expected game value.

| Field | Description |
|:------|:------------|
| `p_win` | P(win) |
| `p_gammon_win` | P(win ∧ gammon+) — joint cumulative, includes backgammon |
| `p_bg_win` | P(win ∧ backgammon) |
| `p_gammon_loss` | P(lose ∧ gammon+) — joint cumulative, includes backgammon |
| `p_bg_loss` | P(lose ∧ backgammon) |

Equity formula (joint): `(2*p_win - 1) + (p_wg - p_lg) + (p_wbg - p_lbg)`
"""
struct EquityOutput
  p_win::Float32
  p_gammon_win::Float32
  p_bg_win::Float32
  p_gammon_loss::Float32
  p_bg_loss::Float32
end

"""
    compute_equity(e::EquityOutput) -> Float32

Compute the expected game value from joint cumulative equity output.

The heads are joint cumulative probabilities:
- `p_win` = P(win)
- `p_gammon_win` = P(win ∧ gammon+) — includes backgammon
- `p_bg_win` = P(win ∧ backgammon)
- `p_gammon_loss` = P(lose ∧ gammon+) — includes backgammon
- `p_bg_loss` = P(lose ∧ backgammon)

Joint equity formula (GnuBG-style):
```
E = (2*p_win - 1) + (p_wg - p_lg) + (p_wbg - p_lbg)
```

Returns a value in approximately [-3, +3] range.
"""
function compute_equity(e::EquityOutput)
  return (2f0 * e.p_win - 1f0) +
         (e.p_gammon_win - e.p_gammon_loss) +
         (e.p_bg_win - e.p_bg_loss)
end

"""
    compute_equity(p_win, p_gammon_win, p_bg_win, p_gammon_loss, p_bg_loss)

Compute equity from individual probability values (for batched computation).
Uses joint formula: `(2*pw - 1) + (wg - lg) + (wbg - lbg)`
"""
function compute_equity(p_win, p_gammon_win, p_bg_win, p_gammon_loss, p_bg_loss)
  return (2f0 .* p_win .- 1f0) .+
         (p_gammon_win .- p_gammon_loss) .+
         (p_bg_win .- p_bg_loss)
end

"""
    FCResNetMultiHeadHP

Hyperparameters for the multi-head FCResNet architecture.

| Parameter                     | Description                                  |
|:------------------------------|:---------------------------------------------|
| `width :: Int`                | Width of hidden layers (default 256)         |
| `num_blocks :: Int`           | Number of residual blocks (default 10)       |
| `depth_phead :: Int = 2`      | Number of layers in policy head              |
| `depth_vhead :: Int = 2`      | Number of layers in each value head          |
| `share_value_trunk :: Bool`   | Share trunk between value heads (default true) |
"""
@kwdef struct FCResNetMultiHeadHP
  width :: Int = 256
  num_blocks :: Int = 10
  depth_phead :: Int = 2
  depth_vhead :: Int = 2
  share_value_trunk :: Bool = true
end

"""
    FCResNetMultiHead <: FluxNetwork

A fully-connected ResNet with multi-head value output for backgammon-style games.

Instead of a single value output, this network outputs 5 raw logits
(sigmoid applied at inference only, BCEWithLogits used for training):
- p_win: P(win)
- p_gammon_win: P(win ∧ gammon+) — joint cumulative
- p_bg_win: P(win ∧ backgammon) — joint
- p_gammon_loss: P(lose ∧ gammon+) — joint cumulative
- p_bg_loss: P(lose ∧ backgammon) — joint

All 5 heads are trained on all samples (no masking needed).
Equity formula: `(2*pw - 1) + (wg - lg) + (wbg - lbg)`

This matches the GnuBG/TD-Gammon/BGBlitz/Wildbg joint cumulative convention.
"""
mutable struct FCResNetMultiHead <: FluxNetwork
  gspec
  hyper
  common        # Shared trunk (input -> hidden)
  vhead_trunk   # Shared value trunk (if share_value_trunk=true)
  vhead_win     # P(win) head
  vhead_gw      # P(gammon | win) head
  vhead_bgw     # P(backgammon | win) head
  vhead_gl      # P(gammon | loss) head
  vhead_bgl     # P(backgammon | loss) head
  phead         # Policy head
end

function FCResNetMultiHead(gspec::AbstractGameSpec, hyper::FCResNetMultiHeadHP)
  indim = prod(GI.state_dim(gspec))
  outdim = GI.num_actions(gspec)
  width = hyper.width

  # Input projection with LayerNorm
  input_layer = Chain(
    flatten,
    Dense(indim, width),
    LayerNorm(width),
    x -> relu.(x)
  )

  # Residual tower
  res_blocks = [PreActResBlock(width) for _ in 1:hyper.num_blocks]

  # Post-tower normalization
  common = Chain(
    input_layer,
    res_blocks...,
    LayerNorm(width),
    x -> relu.(x)
  )

  # Shared value trunk (optional)
  if hyper.share_value_trunk && hyper.depth_vhead > 1
    vhead_trunk_layers = []
    for i in 1:(hyper.depth_vhead - 1)
      push!(vhead_trunk_layers, Dense(width, width))
      push!(vhead_trunk_layers, LayerNorm(width))
      push!(vhead_trunk_layers, x -> relu.(x))
    end
    vhead_trunk = Chain(vhead_trunk_layers...)
  else
    vhead_trunk = identity
  end

  # Individual value heads (each outputs a single raw logit — no sigmoid)
  # Sigmoid is applied at inference time only; training uses BCEWithLogits.
  function make_value_head(depth)
    layers = []
    if !hyper.share_value_trunk
      # Each head has its own full depth
      for i in 1:depth
        push!(layers, Dense(width, width))
        push!(layers, LayerNorm(width))
        push!(layers, x -> relu.(x))
      end
    end
    # Final layer: raw logit output (no sigmoid)
    push!(layers, Dense(width, 1))
    return Chain(layers...)
  end

  vhead_depth = hyper.share_value_trunk ? 1 : hyper.depth_vhead
  vhead_win = make_value_head(vhead_depth)
  vhead_gw = make_value_head(vhead_depth)
  vhead_bgw = make_value_head(vhead_depth)
  vhead_gl = make_value_head(vhead_depth)
  vhead_bgl = make_value_head(vhead_depth)

  # Policy head
  phead_layers = []
  for i in 1:hyper.depth_phead
    push!(phead_layers, Dense(width, width))
    push!(phead_layers, LayerNorm(width))
    push!(phead_layers, x -> relu.(x))
  end
  push!(phead_layers, Dense(width, outdim))
  push!(phead_layers, softmax)
  phead = Chain(phead_layers...)

  FCResNetMultiHead(gspec, hyper, common, vhead_trunk,
                    vhead_win, vhead_gw, vhead_bgw, vhead_gl, vhead_bgl, phead)
end

Network.HyperParams(::Type{FCResNetMultiHead}) = FCResNetMultiHeadHP

"""
    forward(nn::FCResNetMultiHead, state)

Compute forward pass returning (policy, value_heads).

Returns:
- `P`: Policy logits of shape (num_actions, batch_size)
- `V`: Combined equity value of shape (1, batch_size) for compatibility
- Additionally stores individual heads accessible via `forward_multihead`
"""
function Network.forward(nn::FCResNetMultiHead, state)
  c = nn.common(state)

  # Value heads (raw logits → sigmoid for probabilities)
  v_trunk = nn.vhead_trunk(c)
  p_win = Flux.sigmoid.(nn.vhead_win(v_trunk))
  p_gw = Flux.sigmoid.(nn.vhead_gw(v_trunk))
  p_bgw = Flux.sigmoid.(nn.vhead_bgw(v_trunk))
  p_gl = Flux.sigmoid.(nn.vhead_gl(v_trunk))
  p_bgl = Flux.sigmoid.(nn.vhead_bgl(v_trunk))

  # Compute combined equity for MCTS compatibility
  # Scale from [-3, 3] range to [-1, 1] for tanh-like behavior
  equity = compute_equity(p_win, p_gw, p_bgw, p_gl, p_bgl)
  v = equity ./ 3f0  # Normalize to [-1, 1]

  # Policy
  p = nn.phead(c)

  return (p, v)
end

"""
    forward_multihead(nn::FCResNetMultiHead, state)

Compute forward pass returning all value heads as **raw logits**.

Returns:
- `P`: Policy of shape (num_actions, batch_size)
- `L_win`: logit for P(win) of shape (1, batch_size)
- `L_gw`: logit for P(win∧gammon+) of shape (1, batch_size)
- `L_bgw`: logit for P(win∧bg) of shape (1, batch_size)
- `L_gl`: logit for P(lose∧gammon+) of shape (1, batch_size)
- `L_bgl`: logit for P(lose∧bg) of shape (1, batch_size)

Callers that need probabilities should apply sigmoid. The loss function
uses BCEWithLogits directly on the logits.
"""
function forward_multihead(nn::FCResNetMultiHead, state)
  c = nn.common(state)

  v_trunk = nn.vhead_trunk(c)
  l_win = nn.vhead_win(v_trunk)
  l_gw = nn.vhead_gw(v_trunk)
  l_bgw = nn.vhead_bgw(v_trunk)
  l_gl = nn.vhead_gl(v_trunk)
  l_bgl = nn.vhead_bgl(v_trunk)

  p = nn.phead(c)

  return (p, l_win, l_gw, l_bgw, l_gl, l_bgl)
end

"""
    forward_normalized_multihead(nn::FCResNetMultiHead, state, actions_mask)

Like forward_normalized but returns all value heads as **raw logits**.
The loss function uses BCEWithLogits directly on these logits.
"""
function forward_normalized_multihead(nn::FCResNetMultiHead, state, actions_mask)
  p, l_win, l_gw, l_bgw, l_gl, l_bgl = forward_multihead(nn, state)
  p = p .* actions_mask
  sp = sum(p, dims=1)
  p = p ./ (sp .+ eps(eltype(p)))
  p_invalid = 1f0 .- sp
  return (p, l_win, l_gw, l_bgw, l_gl, l_bgl, p_invalid)
end

Network.hyperparams(nn::FCResNetMultiHead) = nn.hyper
Network.game_spec(nn::FCResNetMultiHead) = nn.gspec
Network.on_gpu(nn::FCResNetMultiHead) = array_on_gpu(nn.vhead_win[end].bias)

function Base.copy(nn::FCResNetMultiHead)
  return FCResNetMultiHead(
    nn.gspec,
    nn.hyper,
    deepcopy(nn.common),
    deepcopy(nn.vhead_trunk),
    deepcopy(nn.vhead_win),
    deepcopy(nn.vhead_gw),
    deepcopy(nn.vhead_bgw),
    deepcopy(nn.vhead_gl),
    deepcopy(nn.vhead_bgl),
    deepcopy(nn.phead)
  )
end

# Make Flux recognize all components for training
Flux.@layer FCResNetMultiHead
