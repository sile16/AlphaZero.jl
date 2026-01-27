#####
##### Fully-Connected ResNet v2 with Layer Normalization
##### Pre-activation residual blocks for non-convolutional inputs
#####

using Flux: LayerNorm

"""
    FCResNetHP

Hyperparameters for the fully-connected ResNet v2 architecture.

| Parameter                     | Description                                  |
|:------------------------------|:---------------------------------------------|
| `width :: Int`                | Width of hidden layers (default 256)         |
| `num_blocks :: Int`           | Number of residual blocks (default 10)       |
| `depth_phead :: Int = 2`      | Number of layers in policy head              |
| `depth_vhead :: Int = 2`      | Number of layers in value head               |
"""
@kwdef struct FCResNetHP
  width :: Int = 256
  num_blocks :: Int = 10
  depth_phead :: Int = 2
  depth_vhead :: Int = 2
end

"""
Pre-activation residual block (ResNet v2 style):
  LayerNorm -> ReLU -> Dense -> LayerNorm -> ReLU -> Dense + skip connection
"""
struct PreActResBlock
  ln1
  dense1
  ln2
  dense2
end

Flux.@layer PreActResBlock

function PreActResBlock(width::Int)
  PreActResBlock(
    LayerNorm(width),
    Dense(width, width),
    LayerNorm(width),
    Dense(width, width)
  )
end

function (block::PreActResBlock)(x)
  # Pre-activation: norm -> relu -> linear
  h = block.dense1(relu.(block.ln1(x)))
  h = block.dense2(relu.(block.ln2(h)))
  return x .+ h  # Skip connection
end

"""
    FCResNet <: TwoHeadNetwork

A fully-connected ResNet v2 architecture with pre-activation residual blocks
and Layer Normalization. Designed for non-image inputs like backgammon.

This follows the architecture described in:
- He et al., 2016 "Identity Mappings in Deep Residual Networks" (pre-activation)
- Ba et al., 2016 "Layer Normalization"
"""
mutable struct FCResNet <: TwoHeadNetwork
  gspec
  hyper
  common
  vhead
  phead
end

function FCResNet(gspec::AbstractGameSpec, hyper::FCResNetHP)
  indim = prod(GI.state_dim(gspec))
  outdim = GI.num_actions(gspec)
  width = hyper.width

  # Input projection
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

  # Value head: multiple dense layers -> single output
  vhead_layers = []
  for i in 1:hyper.depth_vhead
    push!(vhead_layers, Dense(width, width))
    push!(vhead_layers, LayerNorm(width))
    push!(vhead_layers, x -> relu.(x))
  end
  push!(vhead_layers, Dense(width, 1, tanh))
  vhead = Chain(vhead_layers...)

  # Policy head: multiple dense layers -> action logits -> softmax
  phead_layers = []
  for i in 1:hyper.depth_phead
    push!(phead_layers, Dense(width, width))
    push!(phead_layers, LayerNorm(width))
    push!(phead_layers, x -> relu.(x))
  end
  push!(phead_layers, Dense(width, outdim))
  push!(phead_layers, softmax)
  phead = Chain(phead_layers...)

  FCResNet(gspec, hyper, common, vhead, phead)
end

Network.HyperParams(::Type{FCResNet}) = FCResNetHP

function Base.copy(nn::FCResNet)
  return FCResNet(
    nn.gspec,
    nn.hyper,
    deepcopy(nn.common),
    deepcopy(nn.vhead),
    deepcopy(nn.phead)
  )
end
