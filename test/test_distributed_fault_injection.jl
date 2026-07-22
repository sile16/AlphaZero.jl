using Test
using AlphaZero
using HTTP
using JSON
using Logging

module DistributedFaultHarness
using AlphaZero
using HTTP
using JSON
using MsgPack
using SHA
using Dates
using Logging
const FluxLib = AlphaZero.FluxLib
const TB_LOGGER = NullLogger()
include(joinpath(@__DIR__, "..", "src", "distributed", "buffer.jl"))
include(joinpath(@__DIR__, "..", "src", "distributed", "protocol.jl"))
include(joinpath(@__DIR__, "..", "src", "distributed", "server.jl"))
end

const DFH = DistributedFaultHarness

function fault_request(target, body=UInt8[]; method="POST", client="fault-client")
    HTTP.Request(method, target,
        ["Authorization" => "Bearer key",
         "Content-Type" => "application/msgpack",
         "X-Client-Id" => client], body)
end

@testset "Distributed fault injection" begin
    fingerprint = DFH.contract_fingerprint(Dict("state_dim" => 4, "num_actions" => 6))
    config = Dict{String,Any}("seed" => 1, "contract_fingerprint" => fingerprint)
    state = DFH.ServerState(api_key="key", config=config)
    buffer = DFH.PERBuffer(64, 4, 6)
    batch = DFH.SampleBatch(Int32(2), rand(Float32, 4, 2), rand(Float32, 6, 2),
        rand(Float32, 2), rand(Float32, 5, 2), trues(2), falses(2), trues(2), falses(2))
    bytes = DFH.pack_samples(batch; contract_fingerprint=fingerprint,
                             batch_id="lost-ack", source_iteration=3)
    request = fault_request("/api/samples", bytes)

    # Simulate a response lost after commit, followed by the client's identical retry.
    DFH.handle_samples(request, state, buffer)
    retry = DFH.handle_samples(request, state, buffer)
    @test retry.status == 200
    @test JSON.parse(String(retry.body))["duplicate"] == true
    @test DFH.buf_length(buffer) == 2
    @test state.total_samples[] == 2

    corrupt = with_logger(NullLogger()) do
        DFH.handle_samples(fault_request("/api/samples", UInt8[0xff, 0x00]), state, buffer)
    end
    @test corrupt.status == 400
    @test DFH.buf_length(buffer) == 2

    # Pinned eval versions remain downloadable while the current version advances.
    state.weight_history[7] = (UInt8[1, 2, 3], UInt8[4, 5, 6])
    pinned = DFH.handle_weights(fault_request("/api/weights/contact?version=7";
        method="GET"), state, :contact)
    @test pinned.status == 200
    @test pinned.body == UInt8[1, 2, 3]

    @test DFH.handle_health(state).status == 200
    @test DFH.handle_ready(state).status == 503
    state.contact_weight_bytes = UInt8[1]
    state.race_weight_bytes = UInt8[2]
    @test DFH.handle_ready(state).status == 200

    drained = DFH.handle_drain(fault_request("/api/drain"), state)
    @test drained.status == 202
    @test state.shutdown_requested[]
    @test DFH.handle_ready(state).status == 503
    @test DFH.handle_samples(request, state, buffer).status == 503

    # A bounded upload queue applies backpressure instead of dropping a batch.
    queue = Channel{Int}(1)
    put!(queue, 1)
    blocked_put = @async put!(queue, 2)
    yield()
    @test !istaskdone(blocked_put)
    @test take!(queue) == 1
    wait(blocked_put)
    @test take!(queue) == 2
end
