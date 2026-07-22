"""
Wire protocol for distributed training.

Defines the sample format and serialization for communication between
self-play clients and the training server.

Supports MsgPack (Julia clients) and JSON (web clients) via content-type negotiation.
"""

using MsgPack
using JSON
using SHA

const DISTRIBUTED_PROTOCOL_VERSION = 4

"""Canonical SHA-256 fingerprint for a distributed ML contract dictionary."""
function contract_fingerprint(contract::AbstractDict)::String
    function canonical(x)
        if x isa AbstractDict
            keys_sorted = sort!(String.(collect(keys(x))))
            return "{" * join((repr(k) * ":" * canonical(
                haskey(x, k) ? x[k] : x[Symbol(k)]) for k in keys_sorted), ",") * "}"
        elseif x isa AbstractVector || x isa Tuple
            return "[" * join(canonical.(collect(x)), ",") * "]"
        elseif x isa Symbol
            return repr(String(x))
        elseif x isa AbstractString
            return repr(String(x))
        elseif x isa Nothing
            return "null"
        else
            return repr(x)
        end
    end
    return bytes2hex(sha256(codeunits(canonical(contract))))
end

"""Fail with field-level diagnostics when two distributed ML contracts differ."""
function validate_contract!(expected::AbstractDict, actual::AbstractDict;
                            label::AbstractString="distributed ML contract")
    expected_keys = Set(String.(collect(keys(expected))))
    actual_keys = Set(String.(collect(keys(actual))))
    problems = String[]
    for key in sort!(collect(union(expected_keys, actual_keys)))
        key in expected_keys || (push!(problems, "unexpected $key"); continue)
        key in actual_keys || (push!(problems, "missing $key"); continue)
        ev = haskey(expected, key) ? expected[key] : expected[Symbol(key)]
        av = haskey(actual, key) ? actual[key] : actual[Symbol(key)]
        contract_fingerprint(Dict("value" => ev)) ==
            contract_fingerprint(Dict("value" => av)) ||
            push!(problems, "$key expected=$(repr(ev)) actual=$(repr(av))")
    end
    isempty(problems) || throw(ArgumentError(
        "$label mismatch: " * join(problems, "; ")))
    return contract_fingerprint(expected)
end

"""Columnar batch of self-play samples for efficient serialization."""
struct SampleBatch
    n::Int32                           # Number of samples
    states::Matrix{Float32}            # (state_dim, n)
    policies::Matrix{Float32}          # (num_actions, n)
    values::Vector{Float32}            # (n,)
    equities::Matrix{Float32}          # (5, n)
    has_equity::Vector{Bool}           # (n,)
    is_chance::Vector{Bool}            # (n,) chance node, no policy loss
    is_contact::Vector{Bool}           # (n,)
    is_bearoff::Vector{Bool}           # (n,) exact table value, skip reanalyze
end

"""
Compact per-upload self-play telemetry.

Only cumulative counters are sent; no positions, per-move events, or histograms are
added to the wire format. This keeps uploads cheap and gives the server enough data
to compute the small set of search-health ratios shown in TensorBoard.
"""
struct SelfPlayMetrics
    games::Int64
    mcts_simulations::Int64
    tree_hits::Int64
    tree_misses::Int64
    nn_evaluations::Int64
    oracle_calls::Int64
    bearoff_hits::Int64
    bearoff_misses::Int64
    search_ns::Int64
    max_depth::Int64
end

SelfPlayMetrics() = SelfPlayMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

"""Wire metadata plus one decoded sample batch."""
struct SampleEnvelope
    protocol_version::Int
    contract_fingerprint::String
    batch_id::String
    batch::SampleBatch
    metrics::SelfPlayMetrics
    source_iteration::Int
end

SampleEnvelope(protocol_version::Int, contract_fingerprint::String,
               batch_id::String, batch::SampleBatch) =
    SampleEnvelope(protocol_version, contract_fingerprint, batch_id, batch,
                   SelfPlayMetrics(), -1)

SampleEnvelope(protocol_version::Int, contract_fingerprint::String,
               batch_id::String, batch::SampleBatch, metrics::SelfPlayMetrics) =
    SampleEnvelope(protocol_version, contract_fingerprint, batch_id, batch,
                   metrics, -1)

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
    is_chance = Vector{Bool}(undef, n)
    is_contact = Vector{Bool}(undef, n)
    is_bearoff = Vector{Bool}(undef, n)

    for i in 1:n
        s = samples[i]
        states[:, i] .= s.state
        policies[:, i] .= s.policy
        values[i] = s.value
        has_equity[i] = s.has_equity
        is_chance[i] = s.is_chance
        is_contact[i] = s.is_contact
        is_bearoff[i] = s.is_bearoff
        if s.has_equity
            eq = s.equity
            for j in 1:5
                equities[j, i] = eq[j]
            end
        end
    end

    SampleBatch(Int32(n), states, policies, values, equities, has_equity,
                is_chance, is_contact, is_bearoff)
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
            is_chance = batch.is_chance[i],
            is_contact = batch.is_contact[i],
            is_bearoff = batch.is_bearoff[i],
        )
    end
    samples
end

function _packed_samples_dict(batch::SampleBatch; protocol_version::Integer,
                              contract_fingerprint::AbstractString,
                              batch_id::AbstractString,
                              metrics::SelfPlayMetrics,
                              source_iteration::Integer)
    return Dict{String, Any}(
        "protocol_version" => Int(protocol_version),
        "contract_fingerprint" => String(contract_fingerprint),
        "batch_id" => String(batch_id),
        "source_iteration" => Int(source_iteration),
        "metrics" => Dict{String,Any}(
            "games" => metrics.games,
            "mcts_simulations" => metrics.mcts_simulations,
            "tree_hits" => metrics.tree_hits,
            "tree_misses" => metrics.tree_misses,
            "nn_evaluations" => metrics.nn_evaluations,
            "oracle_calls" => metrics.oracle_calls,
            "bearoff_hits" => metrics.bearoff_hits,
            "bearoff_misses" => metrics.bearoff_misses,
            "search_ns" => metrics.search_ns,
            "max_depth" => metrics.max_depth,
        ),
        "n" => Int(batch.n),
        "states" => vec(batch.states),          # Column-major flat
        "state_dim" => size(batch.states, 1),
        "policies" => vec(batch.policies),      # Column-major flat
        "num_actions" => size(batch.policies, 1),
        "values" => batch.values,
        "equities" => vec(batch.equities),      # Column-major flat
        "has_equity" => batch.has_equity,
        "is_chance" => batch.is_chance,
        "is_contact" => batch.is_contact,
        "is_bearoff" => batch.is_bearoff,
    )
end

function _unpack_samples_dict(d::AbstractDict)::SampleEnvelope
    protocol_version = Int(get(d, "protocol_version", 0))
    fingerprint = String(get(d, "contract_fingerprint", ""))
    batch_id = String(get(d, "batch_id", ""))
    source_iteration = Int(get(d, "source_iteration", -1))
    md = get(d, "metrics", Dict{String,Any}())
    metric(name) = Int64(get(md, name, 0))
    metrics = SelfPlayMetrics(
        metric("games"), metric("mcts_simulations"), metric("tree_hits"),
        metric("tree_misses"), metric("nn_evaluations"), metric("oracle_calls"),
        metric("bearoff_hits"), metric("bearoff_misses"), metric("search_ns"),
        metric("max_depth"))
    n = Int32(d["n"])
    state_dim = Int(d["state_dim"])
    num_actions = Int(d["num_actions"])

    n > 0 || throw(ArgumentError("sample batch n must be positive (got $n)"))
    state_dim > 0 || throw(ArgumentError("sample state_dim must be positive"))
    num_actions > 0 || throw(ArgumentError("sample num_actions must be positive"))
    expected_lengths = (
        "states" => state_dim * Int(n), "policies" => num_actions * Int(n),
        "values" => Int(n), "equities" => 5 * Int(n),
        "has_equity" => Int(n), "is_chance" => Int(n),
        "is_contact" => Int(n), "is_bearoff" => Int(n))
    for (name, expected) in expected_lengths
        length(d[name]) == expected || throw(ArgumentError(
            "sample field $name has length $(length(d[name])); expected $expected"))
    end

    states = reshape(Float32.(d["states"]), state_dim, Int(n))
    policies = reshape(Float32.(d["policies"]), num_actions, Int(n))
    values = Float32.(d["values"])
    equities = reshape(Float32.(d["equities"]), 5, Int(n))
    has_equity = Bool.(d["has_equity"])
    is_chance = Bool.(d["is_chance"])
    is_contact = Bool.(d["is_contact"])
    is_bearoff = Bool.(d["is_bearoff"])

    batch = SampleBatch(n, states, policies, values, equities, has_equity,
                        is_chance, is_contact, is_bearoff)
    return SampleEnvelope(protocol_version, fingerprint, batch_id, batch, metrics,
                          source_iteration)
end

"""Serialize a SampleBatch and its contract envelope to MsgPack bytes."""
function pack_samples(batch::SampleBatch;
                      protocol_version::Integer=DISTRIBUTED_PROTOCOL_VERSION,
                      contract_fingerprint::AbstractString="",
                      batch_id::AbstractString="",
                      metrics::SelfPlayMetrics=SelfPlayMetrics(),
                      source_iteration::Integer=-1)::Vector{UInt8}
    return MsgPack.pack(_packed_samples_dict(batch; protocol_version,
        contract_fingerprint, batch_id, metrics, source_iteration))
end

unpack_samples_envelope(bytes::Vector{UInt8}) = _unpack_samples_dict(MsgPack.unpack(bytes))

"""Deserialize MsgPack bytes to a SampleBatch (metadata available via `unpack_samples_envelope`)."""
unpack_samples(bytes::Vector{UInt8})::SampleBatch = unpack_samples_envelope(bytes).batch

function validate_sample_envelope!(envelope::SampleEnvelope,
                                   expected_fingerprint::AbstractString)
    envelope.protocol_version == DISTRIBUTED_PROTOCOL_VERSION || throw(ArgumentError(
        "sample protocol version mismatch: expected $DISTRIBUTED_PROTOCOL_VERSION, " *
        "got $(envelope.protocol_version)"))
    isempty(envelope.batch_id) && throw(ArgumentError("sample batch_id is required"))
    envelope.contract_fingerprint == expected_fingerprint || throw(ArgumentError(
        "sample contract fingerprint mismatch: expected $expected_fingerprint, " *
        "got $(envelope.contract_fingerprint)"))
    metric_values = (
        envelope.metrics.games, envelope.metrics.mcts_simulations,
        envelope.metrics.tree_hits, envelope.metrics.tree_misses,
        envelope.metrics.nn_evaluations, envelope.metrics.oracle_calls,
        envelope.metrics.bearoff_hits, envelope.metrics.bearoff_misses,
        envelope.metrics.search_ns, envelope.metrics.max_depth)
    all(>=(0), metric_values) || throw(ArgumentError(
        "sample metrics must be non-negative"))
    -1 <= envelope.source_iteration <= typemax(Int32) || throw(ArgumentError(
        "sample source_iteration must be -1 or a non-negative Int32 value"))
    return envelope
end

"""Serialize a SampleBatch to JSON string (for web clients)."""
function pack_samples_json(batch::SampleBatch;
                           protocol_version::Integer=DISTRIBUTED_PROTOCOL_VERSION,
                           contract_fingerprint::AbstractString="",
                           batch_id::AbstractString="",
                           metrics::SelfPlayMetrics=SelfPlayMetrics(),
                           source_iteration::Integer=-1)::String
    return JSON.json(_packed_samples_dict(batch; protocol_version,
        contract_fingerprint, batch_id, metrics, source_iteration))
end

unpack_samples_json_envelope(json_str::String) = _unpack_samples_dict(JSON.parse(json_str))
unpack_samples_json(json_str::String)::SampleBatch = unpack_samples_json_envelope(json_str).batch

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

function _weights_checksum(weight_bytes::Vector{UInt8})::UInt64
    digest = sha256(weight_bytes)
    checksum = UInt64(0)
    @inbounds for i in 1:8
        checksum = (checksum << 8) | UInt64(digest[i])
    end
    return checksum
end

"""Serialize network weights with metadata header.
Uses existing FluxLib.serialize_weights format with a prepended header."""
function serialize_weights_with_header(nn, header::WeightHeader)::Vector{UInt8}
    weight_bytes = FluxLib.serialize_weights(nn)
    checksum = _weights_checksum(weight_bytes)

    buf = IOBuffer()
    write(buf, WEIGHT_MAGIC)
    write(buf, header.model_type)
    write(buf, header.iteration)
    write(buf, header.width)
    write(buf, header.blocks)
    write(buf, checksum)
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
    actual_checksum = _weights_checksum(weight_bytes)
    if header.checksum != actual_checksum
        error("Weight checksum mismatch: expected $(header.checksum), got $actual_checksum")
    end
    weights = FluxLib.deserialize_weights(weight_bytes)
    return (header, weights)
end
