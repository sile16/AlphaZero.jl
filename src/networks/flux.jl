"""
This module provides utilities to build neural networks with Flux,
along with a library of standard architectures.
"""
module FluxLib

export SimpleNet, SimpleNetHP, ResNet, ResNetHP, FCResNet, FCResNetHP
export FCResNetMultiHead, FCResNetMultiHeadHP
export EquityOutput, compute_equity, forward_multihead, forward_normalized_multihead

using ..AlphaZero

using CUDA
using Base: @kwdef

import Flux

CUDA.allowscalar(false)
array_on_gpu(::Array) = false
array_on_gpu(::CuArray) = true
array_on_gpu(arr) = error("Usupported array type: ", typeof(arr))

using Flux: relu, softmax, flatten
using Flux: Chain, Dense, Conv, BatchNorm, SkipConnection

#####
##### Flux Networks
#####

"""
    FluxNetwork <: AbstractNetwork

Abstract type for neural networks implemented using the _Flux_ framework.

The `regularized_params_` function must be overrided for all layers containing
parameters that are subject to regularization.

Provided that the above holds, `FluxNetwork` implements the full
network interface with the following exceptions:
[`Network.HyperParams`](@ref), [`Network.hyperparams`](@ref),
[`Network.forward`](@ref) and [`Network.on_gpu`](@ref).
"""
abstract type FluxNetwork <: AbstractNetwork end

function Base.copy(nn::Net) where Net <: FluxNetwork
  #new = Net(Network.hyperparams(nn))
  #Flux.loadparams!(new, Flux.params(nn))
  #return new
  return Base.deepcopy(nn)
end

Network.to_cpu(nn::FluxNetwork) = Flux.cpu(nn)

function Network.to_gpu(nn::FluxNetwork)
  CUDA.allowscalar(false)
  return Flux.gpu(nn)
end

function Network.set_test_mode!(nn::FluxNetwork, mode)
  Flux.testmode!(nn, mode)
end

Network.convert_input(nn::FluxNetwork, x) =
  Network.on_gpu(nn) ? Flux.gpu(x) : x

Network.convert_output(nn::FluxNetwork, x) = Flux.cpu(x)

Network.params(nn::FluxNetwork) = Flux.trainables(nn)

function Network.train!(callback, nn::FluxNetwork, opt::Adam, loss, data, n)
  opt_state = Flux.setup(Flux.Adam(opt.lr), nn)
  for (i, d) in enumerate(data)
    l, grads = Flux.withgradient(nn -> loss(nn, d), nn)
    Flux.update!(opt_state, nn, grads[1])
    callback(i, l)
  end
end

function Network.train!(
    callback, nn::FluxNetwork, opt::CyclicNesterov, loss, data, n)
  lr = CyclicSchedule(
    opt.lr_base,
    opt.lr_high,
    opt.lr_low, n=n)
  momentum = CyclicSchedule(
    opt.momentum_high,
    opt.momentum_low,
    opt.momentum_high, n=n)
  opt_state = Flux.setup(Flux.Nesterov(opt.lr_low, opt.momentum_high), nn)
  for (i, d) in enumerate(data)
    l, grads = Flux.withgradient(nn -> loss(nn, d), nn)
    Flux.update!(opt_state, nn, grads[1])
    Flux.adjust!(opt_state; eta=lr[i], rho=momentum[i])
    callback(i, l)
  end
end

regularized_params_(l) = []
regularized_params_(l::Flux.Dense) = [l.weight]
regularized_params_(l::Flux.Conv) = [l.weight]

function Network.regularized_params(net::FluxNetwork)
  return (w for l in Flux.modules(net) for w in regularized_params_(l))
end

function Network.gc(::FluxNetwork)
  GC.gc(true)
  # CUDA.reclaim()
end

#####
##### Common functions between two-head neural networks
#####

"""
    TwoHeadNetwork <: FluxNetwork

An abstract type for two-head neural networks implemented with Flux.

Subtypes are assumed to have fields
`hyper`, `gspec`, `common`, `vhead` and `phead`. Based on those, an implementation
is provided for [`Network.hyperparams`](@ref), [`Network.game_spec`](@ref),
[`Network.forward`](@ref) and [`Network.on_gpu`](@ref), leaving only
[`Network.HyperParams`](@ref) to be implemented.
"""
abstract type TwoHeadNetwork <: FluxNetwork end

function Network.forward(nn::TwoHeadNetwork, state)
  c = nn.common(state)
  v = nn.vhead(c)
  p = nn.phead(c)
  return (p, v)
end

Network.hyperparams(nn::TwoHeadNetwork) = nn.hyper

Network.game_spec(nn::TwoHeadNetwork) = nn.gspec

Network.on_gpu(nn::TwoHeadNetwork) = array_on_gpu(nn.vhead[end].bias)

#####
##### Include networks library
#####

include("architectures/simplenet.jl")
include("architectures/resnet.jl")
include("architectures/fc_resnet.jl")
include("architectures/fc_resnet_multihead.jl")

#####
##### Serialization helpers for distributed training
#####

"""
    serialize_weights(nn::FluxNetwork) -> Vector{UInt8}

Serialize network weights to bytes for distributed training.
"""
function serialize_weights(nn::FluxNetwork)
    io = IOBuffer()
    params = Network.params(nn)
    # Write number of parameter arrays
    write(io, Int32(length(params)))
    for p in params
        arr = Array(p)  # Ensure on CPU
        # Write dimensions
        ndims = length(size(arr))
        write(io, Int32(ndims))
        for d in size(arr)
            write(io, Int32(d))
        end
        # Write element type indicator (1=Float32, 2=Float64)
        if eltype(arr) == Float32
            write(io, UInt8(1))
        elseif eltype(arr) == Float64
            write(io, UInt8(2))
        else
            error("Unsupported element type: $(eltype(arr))")
        end
        # Write data
        write(io, arr)
    end
    return take!(io)
end

"""
    deserialize_weights(bytes::Vector{UInt8}) -> Vector{Array}

Deserialize network weights from bytes.
"""
function deserialize_weights(bytes::Vector{UInt8})
    io = IOBuffer(bytes)
    num_params = read(io, Int32)
    params = Vector{Array}()
    sizehint!(params, num_params)

    for _ in 1:num_params
        # Read dimensions
        ndims = read(io, Int32)
        shape = Tuple(read(io, Int32) for _ in 1:ndims)
        # Read element type
        type_indicator = read(io, UInt8)
        T = type_indicator == 1 ? Float32 : Float64
        # Read data
        arr = Array{T}(undef, shape)
        read!(io, arr)
        push!(params, arr)
    end

    return params
end

"""
    load_weights!(nn::FluxNetwork, weights::Vector{Array})

Load weights into a network.
"""
function load_weights!(nn::FluxNetwork, weights::Vector)
    params = Network.params(nn)
    @assert length(params) == length(weights) "Parameter count mismatch: got $(length(weights)), expected $(length(params))"
    for (p, w) in zip(params, weights)
        copyto!(p, w)
    end
    return nn
end

"""
    save_weights(path::String, nn::FluxNetwork)

Save network weights to a file.
"""
function save_weights(path::String, nn::FluxNetwork)
    bytes = serialize_weights(nn)
    write(path, bytes)
end

"""
    load_weights(path::String, nn::FluxNetwork)

Load network weights from a file into an existing network.
"""
function load_weights(path::String, nn::FluxNetwork)
    bytes = read(path)
    weights = deserialize_weights(bytes)
    load_weights!(nn, weights)
    return nn
end

export serialize_weights, deserialize_weights, load_weights!, save_weights, load_weights

end
