"""
Wire protocol for distributed training.

Defines the sample format and serialization for communication between
self-play clients and the training server.

Supports MsgPack (Julia clients) and JSON (web clients) via content-type negotiation.
"""

using MsgPack
using JSON

"""Columnar batch of self-play samples for efficient serialization."""
struct SampleBatch
    n::Int32                           # Number of samples
    states::Matrix{Float32}            # (state_dim, n)
    policies::Matrix{Float32}          # (num_actions, n)
    values::Vector{Float32}            # (n,)
    equities::Matrix{Float32}          # (5, n)
    has_equity::Vector{Bool}           # (n,)
    is_contact::Vector{Bool}           # (n,)
end

"""Convert a vector of NamedTuple samples to a SampleBatch."""
function samples_to_batch(samples::Vector)::SampleBatch
    n = length(samples)
    n == 0 && error("Cannot create batch from empty samples")

    state_dim = length(samples[1].state)
    num_actions = length(samples[1].policy)

    states = Matrix{Float32}(undef, state_dim, n)
    policies = Matrix{Float32}(undef, num_actions, n)
    values = Vector{Float32}(undef, n)
    equities = zeros(Float32, 5, n)
    has_equity = Vector{Bool}(undef, n)
    is_contact = Vector{Bool}(undef, n)

    for i in 1:n
        s = samples[i]
        states[:, i] .= s.state
        policies[:, i] .= s.policy
        values[i] = s.value
        has_equity[i] = s.has_equity
        is_contact[i] = s.is_contact
        if s.has_equity
            eq = s.equity
            for j in 1:5
                equities[j, i] = eq[j]
            end
        end
    end

    SampleBatch(Int32(n), states, policies, values, equities, has_equity, is_contact)
end

"""Convert a SampleBatch back to a vector of NamedTuples."""
function batch_to_samples(batch::SampleBatch)::Vector
    samples = Vector{Any}(undef, batch.n)
    for i in 1:batch.n
        eq = batch.has_equity[i] ? Float32[batch.equities[j, i] for j in 1:5] : zeros(Float32, 5)
        samples[i] = (
            state = batch.states[:, i],
            policy = batch.policies[:, i],
            value = batch.values[i],
            equity = eq,
            has_equity = batch.has_equity[i],
            is_chance = false,  # Chance nodes not traced in self-play
            bearoff_probs = nothing,
            wp = true,  # White perspective (already flipped during self-play)
            is_contact = batch.is_contact[i],
        )
    end
    samples
end

"""Serialize a SampleBatch to MsgPack bytes."""
function pack_samples(batch::SampleBatch)::Vector{UInt8}
    d = Dict{String, Any}(
        "n" => Int(batch.n),
        "states" => vec(batch.states),          # Column-major flat
        "state_dim" => size(batch.states, 1),
        "policies" => vec(batch.policies),      # Column-major flat
        "num_actions" => size(batch.policies, 1),
        "values" => batch.values,
        "equities" => vec(batch.equities),      # Column-major flat
        "has_equity" => batch.has_equity,
        "is_contact" => batch.is_contact,
    )
    MsgPack.pack(d)
end

"""Deserialize MsgPack bytes to a SampleBatch."""
function unpack_samples(bytes::Vector{UInt8})::SampleBatch
    d = MsgPack.unpack(bytes)
    n = Int32(d["n"])
    state_dim = Int(d["state_dim"])
    num_actions = Int(d["num_actions"])

    states = reshape(Float32.(d["states"]), state_dim, Int(n))
    policies = reshape(Float32.(d["policies"]), num_actions, Int(n))
    values = Float32.(d["values"])
    equities = reshape(Float32.(d["equities"]), 5, Int(n))
    has_equity = Bool.(d["has_equity"])
    is_contact = Bool.(d["is_contact"])

    SampleBatch(n, states, policies, values, equities, has_equity, is_contact)
end

"""Serialize a SampleBatch to JSON string (for web clients)."""
function pack_samples_json(batch::SampleBatch)::String
    d = Dict{String, Any}(
        "n" => Int(batch.n),
        "states" => vec(batch.states),
        "state_dim" => size(batch.states, 1),
        "policies" => vec(batch.policies),
        "num_actions" => size(batch.policies, 1),
        "values" => batch.values,
        "equities" => vec(batch.equities),
        "has_equity" => batch.has_equity,
        "is_contact" => batch.is_contact,
    )
    JSON.json(d)
end

"""Deserialize JSON string to a SampleBatch."""
function unpack_samples_json(json_str::String)::SampleBatch
    d = JSON.parse(json_str)
    n = Int32(d["n"])
    state_dim = Int(d["state_dim"])
    num_actions = Int(d["num_actions"])

    states = reshape(Float32.(d["states"]), state_dim, Int(n))
    policies = reshape(Float32.(d["policies"]), num_actions, Int(n))
    values = Float32.(d["values"])
    equities = reshape(Float32.(d["equities"]), 5, Int(n))
    has_equity = Bool.(d["has_equity"])
    is_contact = Bool.(d["is_contact"])

    SampleBatch(n, states, policies, values, equities, has_equity, is_contact)
end

# Weight serialization with metadata header

const WEIGHT_MAGIC = UInt8['A', 'Z', '0', '1']

"""Metadata header for weight files."""
struct WeightHeader
    model_type::UInt8      # 1=contact, 2=race
    iteration::Int32
    width::Int32
    blocks::Int32
    checksum::UInt64
end

"""Serialize network weights with metadata header.
Uses existing FluxLib.serialize_weights format with a prepended header."""
function serialize_weights_with_header(nn, header::WeightHeader)::Vector{UInt8}
    weight_bytes = FluxLib.serialize_weights(nn)

    buf = IOBuffer()
    write(buf, WEIGHT_MAGIC)
    write(buf, header.model_type)
    write(buf, header.iteration)
    write(buf, header.width)
    write(buf, header.blocks)
    write(buf, header.checksum)
    write(buf, weight_bytes)
    take!(buf)
end

"""Deserialize weights with metadata header.
Returns (header, weight_bytes)."""
function deserialize_weights_with_header(bytes::Vector{UInt8})
    io = IOBuffer(bytes)
    magic = read(io, 4)
    magic == WEIGHT_MAGIC || error("Invalid weight file magic: $magic")

    header = WeightHeader(
        read(io, UInt8),
        read(io, Int32),
        read(io, Int32),
        read(io, Int32),
        read(io, UInt64),
    )

    weight_bytes = read(io)  # Remaining bytes
    weights = FluxLib.deserialize_weights(weight_bytes)
    return (header, weights)
end
