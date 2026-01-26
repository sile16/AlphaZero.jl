#####
##### Serialization utilities for distributed training
#####

"""
Efficient serialization for ZMQ message passing.

Uses MsgPack for compact binary serialization with fallback to Julia Serialization
for complex types.
"""

using MsgPack
using Serialization

#####
##### Core serialization functions
#####

"""
    serialize_message(msg) -> Vector{UInt8}

Serialize a message to bytes using MsgPack.
Falls back to Julia Serialization for complex types.
"""
function serialize_message(msg)
    try
        return MsgPack.pack(to_dict(msg))
    catch
        # Fallback to Julia serialization for complex types
        io = IOBuffer()
        Serialization.serialize(io, msg)
        return take!(io)
    end
end

"""
    deserialize_message(bytes::Vector{UInt8}, ::Type{T}) -> T

Deserialize bytes to a message of type T.
"""
function deserialize_message(bytes::Vector{UInt8}, ::Type{T}) where T
    try
        dict = MsgPack.unpack(bytes)
        return from_dict(T, dict)
    catch
        # Fallback to Julia deserialization
        io = IOBuffer(bytes)
        return Serialization.deserialize(io)
    end
end

#####
##### Conversion to/from Dict for MsgPack
#####

"""
    to_dict(obj) -> Dict

Convert a struct to a Dict for MsgPack serialization.
"""
function to_dict(obj)
    T = typeof(obj)
    dict = Dict{String,Any}()
    for field in fieldnames(T)
        val = getfield(obj, field)
        dict[String(field)] = to_dict_value(val)
    end
    return dict
end

to_dict_value(x::Number) = x
to_dict_value(x::String) = x
to_dict_value(x::Symbol) = String(x)
to_dict_value(x::Bool) = x
to_dict_value(x::Nothing) = nothing
to_dict_value(x::Vector) = [to_dict_value(v) for v in x]
to_dict_value(x::Dict) = Dict(String(k) => to_dict_value(v) for (k, v) in x)
to_dict_value(x) = to_dict(x)  # Nested struct

"""
    from_dict(::Type{T}, dict::Dict) -> T

Reconstruct a struct of type T from a Dict.
"""
function from_dict(::Type{T}, dict::Dict) where T
    args = []
    for field in fieldnames(T)
        field_type = fieldtype(T, field)
        key = String(field)
        if haskey(dict, key)
            push!(args, from_dict_value(field_type, dict[key]))
        else
            # Try to use default value from @kwdef
            error("Missing required field: $field for type $T")
        end
    end
    return T(args...)
end

from_dict_value(::Type{T}, x) where T<:Number = convert(T, x)
from_dict_value(::Type{String}, x) = String(x)
from_dict_value(::Type{Symbol}, x) = Symbol(x)
from_dict_value(::Type{Bool}, x) = Bool(x)
from_dict_value(::Type{Nothing}, x) = nothing
from_dict_value(::Type{Union{Nothing,T}}, x) where T = x === nothing ? nothing : from_dict_value(T, x)
from_dict_value(::Type{Vector{T}}, x) where T = T[from_dict_value(T, v) for v in x]
from_dict_value(::Type{Dict{K,V}}, x) where {K,V} = Dict{K,V}(from_dict_value(K, k) => from_dict_value(V, v) for (k, v) in x)
from_dict_value(::Type{Any}, x) = x

# Handle nested structs
function from_dict_value(::Type{T}, dict::Dict) where T
    if T <: Dict
        return Dict(k => v for (k, v) in dict)
    else
        return from_dict(T, dict)
    end
end

#####
##### Network weight serialization
#####

"""
    serialize_network_weights(nn) -> Vector{UInt8}

Serialize neural network weights to bytes for distribution.
"""
function serialize_network_weights(nn)
    io = IOBuffer()
    # Get all parameters
    params = Network.params(nn)
    # Serialize parameter count
    write(io, Int32(length(params)))
    # Serialize each parameter array
    for p in params
        arr = Array(p)  # Ensure on CPU
        # Write shape
        ndims = length(size(arr))
        write(io, Int32(ndims))
        for d in size(arr)
            write(io, Int32(d))
        end
        # Write data type
        write(io, Int32(sizeof(eltype(arr))))
        # Write data
        write(io, arr)
    end
    return take!(io)
end

"""
    deserialize_network_weights(bytes::Vector{UInt8}) -> Vector{Array}

Deserialize network weights from bytes.
Returns a vector of arrays matching the network parameter structure.
"""
function deserialize_network_weights(bytes::Vector{UInt8})
    io = IOBuffer(bytes)
    # Read parameter count
    num_params = read(io, Int32)
    params = Vector{Array{Float32}}()
    sizehint!(params, num_params)

    for _ in 1:num_params
        # Read shape
        ndims = read(io, Int32)
        shape = Tuple(read(io, Int32) for _ in 1:ndims)
        # Read data type size
        elem_size = read(io, Int32)
        # Read data
        arr = Array{Float32}(undef, shape)
        read!(io, arr)
        push!(params, arr)
    end

    return params
end

"""
    load_weights_into_network!(nn, weights::Vector{Array})

Load deserialized weights into a network.
"""
function load_weights_into_network!(nn, weights::Vector)
    params = Network.params(nn)
    @assert length(params) == length(weights) "Parameter count mismatch"
    for (p, w) in zip(params, weights)
        copyto!(p, w)
    end
    return nn
end

#####
##### State serialization
#####

"""
    serialize_state(gspec, state) -> Vector{Float32}

Serialize a game state to a vector for network transmission.
Uses the game's vectorize_state function.
"""
function serialize_state(gspec, state)
    return GI.vectorize_state(gspec, state)
end

"""
    serialize_states_batch(gspec, states) -> Vector{Vector{Float32}}

Serialize multiple states for batch transmission.
"""
function serialize_states_batch(gspec, states)
    return [serialize_state(gspec, s) for s in states]
end

#####
##### Sample serialization
#####

"""
    serialize_training_sample(sample::TrainingSample, gspec) -> SerializedSample

Convert a TrainingSample to serialized form for transmission.
"""
function serialize_training_sample(sample, gspec)
    state_vec = Vector{Float32}(serialize_state(gspec, sample.s))
    policy_vec = Vector{Float32}(sample.π)

    # Convert equity targets if present
    equity = if !isnothing(sample.equity)
        MultiHeadValue(
            Float32(sample.equity.p_win),
            Float32(sample.equity.p_gammon_win),
            Float32(sample.equity.p_bg_win),
            Float32(sample.equity.p_gammon_loss),
            Float32(sample.equity.p_bg_loss)
        )
    else
        nothing
    end

    return SerializedSample(
        state=state_vec,
        policy=policy_vec,
        value=Float32(sample.z),
        turn=Float32(sample.t),
        is_chance=sample.is_chance,
        equity=equity
    )
end

"""
    deserialize_training_sample(ss::SerializedSample, gspec) -> TrainingSample

Convert a SerializedSample back to TrainingSample.
Note: This reconstructs a simplified state representation.
"""
function deserialize_training_sample(ss::SerializedSample, gspec)
    # The state is stored as a vector - the training code works directly with vectors
    # So we store it as-is and convert during training batch construction

    equity = if !isnothing(ss.equity)
        EquityTargets(
            Float64(ss.equity.p_win),
            Float64(ss.equity.p_gammon_win),
            Float64(ss.equity.p_bg_win),
            Float64(ss.equity.p_gammon_loss),
            Float64(ss.equity.p_bg_loss)
        )
    else
        nothing
    end

    # Return a compatible sample structure
    # Note: We store the vectorized state directly as the "state"
    return (
        x=ss.state,
        π=Vector{Float64}(ss.policy),
        z=Float64(ss.value),
        t=Float64(ss.turn),
        is_chance=ss.is_chance,
        equity=equity
    )
end

#####
##### Message envelope helpers
#####

"""
    wrap_message(msg, sender_id::String) -> MessageEnvelope

Wrap a message in an envelope for routing.
"""
function wrap_message(msg, sender_id::String)
    T = typeof(msg)
    msg_type = nothing
    for (sym, typ) in MESSAGE_TYPES
        if typ == T
            msg_type = sym
            break
        end
    end
    if isnothing(msg_type)
        error("Unknown message type: $T")
    end

    return MessageEnvelope(
        msg_type=msg_type,
        payload=serialize_message(msg),
        sender_id=sender_id,
        timestamp=time()
    )
end

"""
    unwrap_message(envelope::MessageEnvelope) -> (Symbol, Any)

Unwrap a message envelope, returning (type, message).
"""
function unwrap_message(envelope::MessageEnvelope)
    T = MESSAGE_TYPES[envelope.msg_type]
    msg = deserialize_message(envelope.payload, T)
    return (envelope.msg_type, msg)
end

#####
##### ZMQ helpers
#####

"""
    send_zmq_message(socket, msg, sender_id::String)

Serialize and send a message over ZMQ.
"""
function send_zmq_message(socket, msg, sender_id::String)
    envelope = wrap_message(msg, sender_id)
    envelope_bytes = serialize_message(envelope)
    ZMQ.send(socket, envelope_bytes)
end

"""
    recv_zmq_message(socket) -> (Symbol, Any, String, Float64)

Receive and deserialize a message from ZMQ.
Returns (msg_type, message, sender_id, timestamp).
"""
function recv_zmq_message(socket)
    bytes = ZMQ.recv(socket)
    envelope = deserialize_message(Vector{UInt8}(bytes), MessageEnvelope)
    msg_type, msg = unwrap_message(envelope)
    return (msg_type, msg, envelope.sender_id, envelope.timestamp)
end

#####
##### Checksum utilities
#####

"""
    compute_checksum(data::Vector{UInt8}) -> UInt64

Compute a simple checksum for data verification.
"""
function compute_checksum(data::Vector{UInt8})
    # Simple FNV-1a hash
    hash = UInt64(14695981039346656037)
    for byte in data
        hash = xor(hash, UInt64(byte))
        hash *= UInt64(1099511628211)
    end
    return hash
end
