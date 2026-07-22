using Test
using AlphaZero
using HTTP
using JSON
using Logging

module DistributedServerContractHarness
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
include(joinpath(@__DIR__, "..", "src", "distributed", "client.jl"))
include(joinpath(@__DIR__, "..", "src", "distributed", "server.jl"))
end

const DSCH = DistributedServerContractHarness

function contract_request(target::String, body; content_type="application/json",
                          client_id="neo")
    return HTTP.Request("POST", target,
        ["Authorization" => "Bearer test-key",
         "Content-Type" => content_type,
         "X-Client-Id" => client_id], body)
end

@testset "Distributed server contract and idempotency" begin
    contract = Dict{String,Any}(
        "state_dim" => 352,
        "num_actions" => 676,
        "chance_outcomes" => 21,
    )
    fingerprint = DSCH.contract_fingerprint(contract)
    config = Dict{String,Any}(
        "seed" => 42,
        "contract_fingerprint" => fingerprint,
    )
    state = DSCH.ServerState(api_key="test-key", config=config)
    buffer = DSCH.PERBuffer(32, 352, 676)

    client = DSCH.SelfPlayClient("http://localhost:9090", "test-key";
                                 client_id="neo")
    client.contact_iteration = 4
    client.race_iteration = 6
    first_batch_id = DSCH.next_batch_id!(client)
    second_batch_id = DSCH.next_batch_id!(client)
    @test first_batch_id != second_batch_id
    @test startswith(first_batch_id, "neo-")
    @test DSCH.stable_client_seed(42, "neo") == DSCH.stable_client_seed(42, "neo")
    @test DSCH.stable_client_seed(42, "neo") != DSCH.stable_client_seed(42, "jarvis")

    sample(i) = (
        state=fill(Float32(i), 2), policy=Float32[0.25, 0.75],
        value=Float32(i), equity=zeros(Float32, 5), has_equity=false,
        is_chance=false, is_contact=true, is_bearoff=false)
    append!(client.upload_buffer, [sample(1), sample(2)])
    pending = DSCH.prepare_pending_upload!(client)
    @test DSCH.unpack_samples_envelope(pending.bytes).source_iteration == 4
    push!(client.upload_buffer, sample(3))
    same_pending = DSCH.prepare_pending_upload!(client)
    @test same_pending.batch_id == pending.batch_id
    @test same_pending.bytes == pending.bytes
    @test same_pending.n == 2
    @test DSCH.acknowledge_pending_upload!(client, pending.batch_id, 2) == 2
    @test length(client.upload_buffer) == 1
    @test client.upload_buffer[1].value == 3.0f0

    bad_registration = contract_request("/api/register", JSON.json(Dict(
        "client_id" => "neo",
        "protocol_version" => DSCH.DISTRIBUTED_PROTOCOL_VERSION - 1,
    )))
    @test DSCH.handle_register(bad_registration, state).status == 409
    @test isempty(state.clients)

    registration_body = JSON.json(Dict(
        "client_id" => "neo",
        "client_type" => "julia",
        "name" => "neo",
        "protocol_version" => DSCH.DISTRIBUTED_PROTOCOL_VERSION,
    ))
    first_registration = DSCH.handle_register(
        contract_request("/api/register", registration_body), state)
    @test first_registration.status == 200
    first_result = JSON.parse(String(first_registration.body))
    @test first_result["contract_fingerprint"] == fingerprint

    second_registration = DSCH.handle_register(
        contract_request("/api/register", registration_body), state)
    second_result = JSON.parse(String(second_registration.body))
    @test second_registration.status == 200
    @test second_result["assigned_seed"] == first_result["assigned_seed"]
    @test length(state.clients) == 1

    n = 3
    batch = DSCH.SampleBatch(
        Int32(n),
        rand(Float32, 352, n),
        rand(Float32, 676, n),
        rand(Float32, n),
        rand(Float32, 5, n),
        fill(true, n), fill(false, n), fill(true, n), fill(false, n),
    )
    search_metrics = DSCH.SelfPlayMetrics(
        2, 120, 90, 30, 40, 5, 8, 12, 2_000_000, 11)
    bytes = DSCH.pack_samples(batch; contract_fingerprint=fingerprint,
                              batch_id="neo-process-1", metrics=search_metrics,
                              source_iteration=7)
    upload = contract_request("/api/samples", bytes;
                              content_type="application/msgpack")
    response = DSCH.handle_samples(upload, state, buffer)
    @test response.status == 200
    result = JSON.parse(String(response.body))
    @test result["accepted"] == n
    @test result["duplicate"] == false
    @test DSCH.buf_length(buffer) == n
    @test buffer.source_iterations[1:n] == fill(Int32(7), n)
    @test state.total_samples[] == n
    @test state.total_games[] == 2
    @test state.clients["neo"].games_contributed == 2

    retry = DSCH.handle_samples(upload, state, buffer)
    @test retry.status == 200
    retry_result = JSON.parse(String(retry.body))
    @test retry_result["accepted"] == n
    @test retry_result["duplicate"] == true
    @test DSCH.buf_length(buffer) == n
    @test state.total_samples[] == n
    @test state.total_games[] == 2

    metrics = DSCH.server_metrics_snapshot(state)
    @test metrics.upload_requests == 2
    @test metrics.accepted_batches == 1
    @test metrics.duplicate_batches == 1
    @test metrics.mcts_simulations == 120
    @test metrics.tree_hits == 90
    @test metrics.tree_misses == 30
    @test metrics.nn_evaluations == 40
    @test metrics.oracle_calls == 5
    @test metrics.bearoff_hits == 8
    @test metrics.bearoff_misses == 12
    @test metrics.max_depth == 11

    wrong_contract = DSCH.pack_samples(batch; contract_fingerprint="wrong",
                                       batch_id="neo-process-2")
    rejected = with_logger(NullLogger()) do
        DSCH.handle_samples(
            contract_request("/api/samples", wrong_contract;
                             content_type="application/msgpack"), state, buffer)
    end
    @test rejected.status == 400
    @test DSCH.buf_length(buffer) == n
    @test DSCH.server_metrics_snapshot(state).rejected_batches == 1

    status = DSCH.handle_status(HTTP.Request("GET", "/api/status"), state, buffer)
    status_body = JSON.parse(String(status.body))
    @test status_body["total_games"] == 2
    @test status_body["observability"]["tree_hits"] == 90
    @test status_body["observability"]["duplicate_batches"] == 1
end
