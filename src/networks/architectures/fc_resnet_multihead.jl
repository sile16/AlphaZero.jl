#####
##### Fully-Connected ResNet with Multi-Head Equity Output
##### Based on TD-Gammon/gnubg approach for backgammon
#####

using Flux: LayerNorm

"""
    EquityOutput

Container for multi-head equity output from the network.
Contains 5 probabilities that together determine the expected game value.

| Field | Description |
|:------|:------------|
| `p_win` | Probability of winning the game |
| `p_gammon_win` | P(gammon | win) - probability of gammon given we win |
| `p_bg_win` | P(backgammon | win) - probability of backgammon given we win |
| `p_gammon_loss` | P(gammon | loss) - probability opponent gammons us given we lose |
| `p_bg_loss` | P(backgammon | loss) - probability opponent backgammons us given we lose |
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

Compute the expected game value from equity output components.

The formula is:
```
E = P(win) * (1 + P(gammon|win) + P(bg|win))
  - P(loss) * (1 + P(gammon|loss) + P(bg|loss))
```

This accounts for backgammon scoring where:
- Single game: 1 point
- Gammon: 2 points (adds 1 to base)
- Backgammon: 3 points (adds 1 more beyond gammon)

Note: P(bg|win) is additive on top of P(gammon|win), so:
- p_gammon=0, p_bg=0 → 1 point (single)
- p_gammon=1, p_bg=0 → 2 points (gammon)
- p_gammon=1, p_bg=1 → 3 points (backgammon)

Returns a value in approximately [-3, +3] range.
"""
function compute_equity(e::EquityOutput)
  p_loss = 1f0 - e.p_win

  # Expected points from winning (1 base + gammon bonus + backgammon bonus)
  win_value = e.p_win * (1f0 + e.p_gammon_win + e.p_bg_win)

  # Expected points lost from losing
  loss_value = p_loss * (1f0 + e.p_gammon_loss + e.p_bg_loss)

  return win_value - loss_value
end

"""
    compute_equity(p_win, p_gammon_win, p_bg_win, p_gammon_loss, p_bg_loss)

Compute equity from individual probability values (for batched computation).
"""
function compute_equity(p_win, p_gammon_win, p_bg_win, p_gammon_loss, p_bg_loss)
  p_loss = 1f0 .- p_win
  win_value = p_win .* (1f0 .+ p_gammon_win .+ p_bg_win)
  loss_value = p_loss .* (1f0 .+ p_gammon_loss .+ p_bg_loss)
  return win_value .- loss_value
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

Instead of a single value output, this network outputs 5 probabilities:
- p_win: P(win)
- p_gammon_win: P(gammon | win)
- p_bg_win: P(backgammon | win)
- p_gammon_loss: P(gammon | loss)
- p_bg_loss: P(backgammon | loss)

These are combined to compute the expected game equity.

Based on the TD-Gammon/gnubg approach where conditional probabilities
are more stable training targets than raw outcome probabilities.
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

  # Individual value heads (each outputs a single sigmoid probability)
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
    # Final layer: sigmoid output
    push!(layers, Dense(width, 1))
    push!(layers, x -> Flux.sigmoid.(x))
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

  # Value heads
  v_trunk = nn.vhead_trunk(c)
  p_win = nn.vhead_win(v_trunk)
  p_gw = nn.vhead_gw(v_trunk)
  p_bgw = nn.vhead_bgw(v_trunk)
  p_gl = nn.vhead_gl(v_trunk)
  p_bgl = nn.vhead_bgl(v_trunk)

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

Compute forward pass returning all value heads separately.

Returns:
- `P`: Policy of shape (num_actions, batch_size)
- `V_win`: P(win) of shape (1, batch_size)
- `V_gw`: P(gammon|win) of shape (1, batch_size)
- `V_bgw`: P(bg|win) of shape (1, batch_size)
- `V_gl`: P(gammon|loss) of shape (1, batch_size)
- `V_bgl`: P(bg|loss) of shape (1, batch_size)
"""
function forward_multihead(nn::FCResNetMultiHead, state)
  c = nn.common(state)

  v_trunk = nn.vhead_trunk(c)
  p_win = nn.vhead_win(v_trunk)
  p_gw = nn.vhead_gw(v_trunk)
  p_bgw = nn.vhead_bgw(v_trunk)
  p_gl = nn.vhead_gl(v_trunk)
  p_bgl = nn.vhead_bgl(v_trunk)

  p = nn.phead(c)

  return (p, p_win, p_gw, p_bgw, p_gl, p_bgl)
end

"""
    forward_normalized_multihead(nn::FCResNetMultiHead, state, actions_mask)

Like forward_normalized but returns all value heads.
"""
function forward_normalized_multihead(nn::FCResNetMultiHead, state, actions_mask)
  p, p_win, p_gw, p_bgw, p_gl, p_bgl = forward_multihead(nn, state)
  p = p .* actions_mask
  sp = sum(p, dims=1)
  p = p ./ (sp .+ eps(eltype(p)))
  p_invalid = 1f0 .- sp
  return (p, p_win, p_gw, p_bgw, p_gl, p_bgl, p_invalid)
end

Network.hyperparams(nn::FCResNetMultiHead) = nn.hyper
Network.game_spec(nn::FCResNetMultiHead) = nn.gspec
Network.on_gpu(nn::FCResNetMultiHead) = array_on_gpu(nn.vhead_win[end-1].bias)

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
