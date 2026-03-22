using Test
using MsgPack
using JSON

include("../src/distributed/protocol.jl")

@testset "Protocol Round-Trip" begin

# Helper to compare two SampleBatch structs
function batches_equal(a::SampleBatch, b::SampleBatch)
    a.n == b.n &&
    a.states ≈ b.states &&
    a.policies ≈ b.policies &&
    a.values ≈ b.values &&
    a.equities ≈ b.equities &&
    a.has_equity == b.has_equity &&
    a.is_contact == b.is_contact &&
    a.is_bearoff == b.is_bearoff
end

function make_batch(n; state_dim=344, num_actions=676)
    SampleBatch(
        Int32(n),
        rand(Float32, state_dim, n),
        rand(Float32, num_actions, n),
        rand(Float32, n),
        rand(Float32, 5, n),
        rand(Bool, n),
        rand(Bool, n),
        rand(Bool, n),
    )
end

@testset "MsgPack round-trip basic" begin
    batch = make_batch(10)
    bytes = pack_samples(batch)
    recovered = unpack_samples(bytes)
    @test batches_equal(batch, recovered)
    @test size(recovered.states) == (344, 10)
    @test size(recovered.policies) == (676, 10)
    @test size(recovered.equities) == (5, 10)
    @test length(recovered.values) == 10
    @test length(recovered.has_equity) == 10
    @test length(recovered.is_contact) == 10
    @test length(recovered.is_bearoff) == 10
end

@testset "JSON round-trip basic" begin
    batch = make_batch(10)
    json_str = pack_samples_json(batch)
    recovered = unpack_samples_json(json_str)
    @test batches_equal(batch, recovered)
end

@testset "Equity matrix reshaping (column-major)" begin
    n = 4
    eq = Float32[
        1 5  9 13;
        2 6 10 14;
        3 7 11 15;
        4 8 12 16;
        0 0  0  0
    ]
    batch = SampleBatch(
        Int32(n),
        zeros(Float32, 2, n),
        zeros(Float32, 3, n),
        zeros(Float32, n),
        eq,
        fill(true, n),
        fill(true, n),
        fill(false, n),
    )
    bytes = pack_samples(batch)
    recovered = unpack_samples(bytes)
    # Column-major: vec flattens column by column, reshape restores
    @test recovered.equities == eq
    # Verify specific elements survived the flatten/reshape
    @test recovered.equities[1, 1] == 1f0
    @test recovered.equities[2, 1] == 2f0
    @test recovered.equities[1, 2] == 5f0
    @test recovered.equities[5, 4] == 0f0
end

@testset "Edge case: single sample (n=1)" begin
    batch = make_batch(1)
    bytes = pack_samples(batch)
    recovered = unpack_samples(bytes)
    @test batches_equal(batch, recovered)
    @test recovered.n == Int32(1)
    @test size(recovered.states) == (344, 1)
    @test size(recovered.equities) == (5, 1)
end

@testset "Edge case: large batch (n=256)" begin
    batch = make_batch(256)
    bytes = pack_samples(batch)
    recovered = unpack_samples(bytes)
    @test batches_equal(batch, recovered)
    @test recovered.n == Int32(256)
end

@testset "Edge case: all zeros" begin
    n = 5
    batch = SampleBatch(
        Int32(n),
        zeros(Float32, 344, n),
        zeros(Float32, 676, n),
        zeros(Float32, n),
        zeros(Float32, 5, n),
        fill(false, n),
        fill(false, n),
        fill(false, n),
    )
    bytes = pack_samples(batch)
    recovered = unpack_samples(bytes)
    @test batches_equal(batch, recovered)
    @test all(recovered.states .== 0f0)
    @test all(recovered.values .== 0f0)
end

@testset "Edge case: all ones" begin
    n = 5
    batch = SampleBatch(
        Int32(n),
        ones(Float32, 344, n),
        ones(Float32, 676, n),
        ones(Float32, n),
        ones(Float32, 5, n),
        fill(true, n),
        fill(true, n),
        fill(true, n),
    )
    bytes = pack_samples(batch)
    recovered = unpack_samples(bytes)
    @test batches_equal(batch, recovered)
    @test all(recovered.states .== 1f0)
    @test all(recovered.has_equity)
end

@testset "Edge case: mixed equity flags" begin
    n = 6
    eq = zeros(Float32, 5, n)
    eq[:, 1] .= Float32[0.6, 0.1, 0.0, 0.05, 0.0]
    eq[:, 3] .= Float32[0.8, 0.2, 0.1, 0.0, 0.0]
    eq[:, 5] .= Float32[0.5, 0.0, 0.0, 0.3, 0.1]
    has_eq = Bool[true, false, true, false, true, false]

    batch = SampleBatch(
        Int32(n),
        rand(Float32, 344, n),
        rand(Float32, 676, n),
        rand(Float32, n),
        eq,
        has_eq,
        rand(Bool, n),
        rand(Bool, n),
    )
    bytes = pack_samples(batch)
    recovered = unpack_samples(bytes)
    @test batches_equal(batch, recovered)
    @test recovered.has_equity == has_eq
    # Equity values preserved for has_equity=true samples
    @test recovered.equities[:, 1] ≈ Float32[0.6, 0.1, 0.0, 0.05, 0.0]
    @test recovered.equities[:, 3] ≈ Float32[0.8, 0.2, 0.1, 0.0, 0.0]
    # Equity values also preserved for has_equity=false (pack/unpack is lossless)
    @test all(recovered.equities[:, 2] .== 0f0)
end

@testset "samples_to_batch conversion" begin
    samples = [
        (
            state = rand(Float32, 344),
            policy = rand(Float32, 676),
            value = 0.5f0,
            equity = Float32[0.6, 0.1, 0.0, 0.05, 0.0],
            has_equity = true,
            is_chance = false,
            is_contact = true,
            is_bearoff = false,
        ),
        (
            state = rand(Float32, 344),
            policy = rand(Float32, 676),
            value = -0.3f0,
            equity = Float32[0.0, 0.0, 0.0, 0.0, 0.0],
            has_equity = false,
            is_chance = false,
            is_contact = false,
            is_bearoff = true,
        ),
        (
            state = rand(Float32, 344),
            policy = rand(Float32, 676),
            value = 1.0f0,
            equity = Float32[0.9, 0.3, 0.05, 0.0, 0.0],
            has_equity = true,
            is_chance = true,
            is_contact = true,
            is_bearoff = false,
        ),
    ]

    batch = samples_to_batch(samples)
    @test batch.n == Int32(3)
    @test size(batch.states) == (344, 3)
    @test size(batch.policies) == (676, 3)
    @test batch.values == Float32[0.5, -0.3, 1.0]
    @test batch.has_equity == Bool[true, false, true]
    @test batch.is_contact == Bool[true, false, true]
    @test batch.is_bearoff == Bool[false, true, false]
    # State/policy columns match input
    @test batch.states[:, 1] == samples[1].state
    @test batch.states[:, 2] == samples[2].state
    @test batch.policies[:, 3] == samples[3].policy
    # Equity filled for has_equity=true
    @test batch.equities[:, 1] ≈ Float32[0.6, 0.1, 0.0, 0.05, 0.0]
    @test batch.equities[:, 3] ≈ Float32[0.9, 0.3, 0.05, 0.0, 0.0]
    # Equity zeroed for has_equity=false
    @test all(batch.equities[:, 2] .== 0f0)
end

@testset "samples_to_batch without is_bearoff field" begin
    # Old samples may not have is_bearoff
    samples = [
        (
            state = rand(Float32, 10),
            policy = rand(Float32, 5),
            value = 0.0f0,
            equity = zeros(Float32, 5),
            has_equity = false,
            is_chance = false,
            is_contact = true,
        ),
    ]
    batch = samples_to_batch(samples)
    @test batch.is_bearoff == Bool[false]
end

@testset "samples_to_batch -> pack -> unpack round-trip" begin
    samples = [
        (
            state = rand(Float32, 344),
            policy = rand(Float32, 676),
            value = Float32(i / 10),
            equity = rand(Float32, 5),
            has_equity = isodd(i),
            is_chance = false,
            is_contact = i <= 5,
            is_bearoff = i > 8,
        )
        for i in 1:10
    ]
    batch = samples_to_batch(samples)
    bytes = pack_samples(batch)
    recovered = unpack_samples(bytes)
    @test batches_equal(batch, recovered)
end

@testset "Field name consistency (MsgPack)" begin
    batch = make_batch(2; state_dim=10, num_actions=5)
    bytes = pack_samples(batch)
    d = MsgPack.unpack(bytes)
    # Verify all expected keys are present
    expected_keys = Set(["n", "states", "state_dim", "policies", "num_actions",
                         "values", "equities", "has_equity", "is_contact", "is_bearoff"])
    @test Set(keys(d)) == expected_keys
    # Verify dimension metadata
    @test d["n"] == 2
    @test d["state_dim"] == 10
    @test d["num_actions"] == 5
    @test length(d["states"]) == 10 * 2
    @test length(d["policies"]) == 5 * 2
    @test length(d["values"]) == 2
    @test length(d["equities"]) == 5 * 2
    @test length(d["has_equity"]) == 2
    @test length(d["is_contact"]) == 2
    @test length(d["is_bearoff"]) == 2
end

@testset "Field name consistency (JSON)" begin
    batch = make_batch(2; state_dim=10, num_actions=5)
    json_str = pack_samples_json(batch)
    d = JSON.parse(json_str)
    expected_keys = Set(["n", "states", "state_dim", "policies", "num_actions",
                         "values", "equities", "has_equity", "is_contact", "is_bearoff"])
    @test Set(keys(d)) == expected_keys
end

@testset "Empty samples error" begin
    @test_throws ErrorException samples_to_batch(Any[])
end

end  # top-level testset
