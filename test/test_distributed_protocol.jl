using Test
using AlphaZero

if !isdefined(Main, :BackgammonDeterministic)
    include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "main.jl"))
end

module DistributedProtocolTest
using AlphaZero
using SHA
const FluxLib = AlphaZero.FluxLib
include("../src/distributed/protocol.jl")
end

@testset "Distributed Protocol" begin
    gspec = Main.BackgammonDeterministic.GameSpec()
    nn = AlphaZero.FluxLib.SimpleNet(
        gspec,
        AlphaZero.FluxLib.SimpleNetHP(width=16, depth_common=2))
    header = DistributedProtocolTest.WeightHeader(0x01, Int32(7), Int32(16), Int32(2), UInt64(0))

    bytes = DistributedProtocolTest.serialize_weights_with_header(nn, header)
    parsed_header, weights = DistributedProtocolTest.deserialize_weights_with_header(bytes)

    @test parsed_header.iteration == 7
    @test parsed_header.checksum != 0
    @test !isempty(weights)

    corrupted = copy(bytes)
    corrupted[end] ⊻= 0x01
    @test_throws Exception DistributedProtocolTest.deserialize_weights_with_header(corrupted)
end
