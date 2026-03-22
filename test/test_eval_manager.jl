using Test

include("../src/distributed/eval_manager.jl")
using .EvalManager

@testset "EvalManager" begin

@testset "create_eval_job" begin
    job = create_eval_job(1, 2000, 42; chunk_size=50)
    @test job.iter == 1
    @test job.weights_version == 42
    # 2000 positions / 50 per chunk = 40 chunks per side × 2 sides = 80
    @test length(job.chunks) == 80

    # First 40 chunks are white, last 40 are black
    for i in 1:40
        @test job.chunks[i].az_is_white == true
    end
    for i in 41:80
        @test job.chunks[i].az_is_white == false
    end

    # Check position ranges cover 1:2000 for each side
    white_ranges = [job.chunks[i].position_range for i in 1:40]
    @test first(white_ranges[1]) == 1
    @test last(white_ranges[end]) == 2000
    # Ranges are contiguous
    for i in 2:length(white_ranges)
        @test first(white_ranges[i]) == last(white_ranges[i-1]) + 1
    end

    black_ranges = [job.chunks[i].position_range for i in 41:80]
    @test white_ranges == black_ranges

    # All chunks start uncompleted and unchecked out
    for chunk in job.chunks
        @test chunk.checked_out_by === nothing
        @test chunk.completed == false
    end

    # Chunk IDs are unique and sequential
    ids = [c.chunk_id for c in job.chunks]
    @test ids == 1:80
end

@testset "create_eval_job non-divisible" begin
    # 110 positions with chunk_size=50 → 3 chunks per side (50, 50, 10)
    job = create_eval_job(1, 110, 1; chunk_size=50)
    @test length(job.chunks) == 6
    @test length(job.chunks[1].position_range) == 50
    @test length(job.chunks[2].position_range) == 50
    @test length(job.chunks[3].position_range) == 10
    @test job.chunks[3].position_range == 101:110
end

@testset "checkout_chunk!" begin
    job = create_eval_job(1, 100, 1; chunk_size=50)
    # 4 chunks total (2 white, 2 black)

    c1 = checkout_chunk!(job, "neo")
    @test c1 !== nothing
    @test c1.chunk_id == 1
    @test c1.checked_out_by == "neo"

    c2 = checkout_chunk!(job, "jarvis")
    @test c2 !== nothing
    @test c2.chunk_id == 2
    @test c2.checked_out_by == "jarvis"

    c3 = checkout_chunk!(job, "neo")
    @test c3 !== nothing
    @test c3.chunk_id == 3

    c4 = checkout_chunk!(job, "neo")
    @test c4 !== nothing
    @test c4.chunk_id == 4

    # All checked out, should return nothing
    c5 = checkout_chunk!(job, "neo")
    @test c5 === nothing
end

@testset "submit_chunk!" begin
    job = create_eval_job(1, 100, 1; chunk_size=50)
    chunk = checkout_chunk!(job, "neo")

    result = EvalChunkResult(
        chunk.chunk_id, true,
        [1.0, -1.0, 2.0],
        [0.5, -0.3, 0.8],
        [0.4, -0.2, 0.7],
        [true, false, true]
    )
    @test submit_chunk!(job, result) == true
    @test job.chunks[1].completed == true
    @test job.chunks[1].checked_out_by === nothing
    @test haskey(job.results, chunk.chunk_id)

    # Duplicate completion should be rejected and must not overwrite
    replacement = EvalChunkResult(
        chunk.chunk_id, true,
        [99.0],
        [99.0],
        [99.0],
        [false]
    )
    @test submit_chunk!(job, replacement) == false
    @test job.results[chunk.chunk_id].rewards == [1.0, -1.0, 2.0]

    # Invalid chunk_id
    bad_result = EvalChunkResult(999, true, [], [], [], Bool[])
    @test submit_chunk!(job, bad_result) == false
end

@testset "extend_lease!" begin
    job = create_eval_job(1, 100, 1; chunk_size=50)
    chunk = checkout_chunk!(job, "neo")
    old_time = chunk.checkout_time

    sleep(0.01)
    @test extend_lease!(job, chunk.chunk_id, "neo") == true
    @test job.chunks[1].checkout_time > old_time

    # Wrong client
    @test extend_lease!(job, chunk.chunk_id, "other") == false

    # Invalid chunk_id
    @test extend_lease!(job, 999, "neo") == false
end

@testset "expire_stale_checkouts!" begin
    job = create_eval_job(1, 100, 1; chunk_size=50)

    c1 = checkout_chunk!(job, "neo")
    # Fake an old checkout time
    job.chunks[1].checkout_time = time() - 400.0

    c2 = checkout_chunk!(job, "jarvis")
    # c2 is fresh

    expired = expire_stale_checkouts!(job; lease_seconds=300.0)
    @test expired == 1
    @test job.chunks[1].checked_out_by === nothing
    @test job.chunks[2].checked_out_by == "jarvis"

    # c1's slot is now available again
    c3 = checkout_chunk!(job, "neo")
    @test c3 !== nothing
    @test c3.chunk_id == c1.chunk_id
end

@testset "is_complete" begin
    job = create_eval_job(1, 100, 1; chunk_size=100)
    # 2 chunks: one white, one black
    @test is_complete(job) == false

    for i in 1:length(job.chunks)
        chunk = checkout_chunk!(job, "neo")
        result = EvalChunkResult(chunk.chunk_id, chunk.az_is_white,
            [0.0], [0.0], [0.0], [false])
        submit_chunk!(job, result)
    end

    @test is_complete(job) == true
end

@testset "status" begin
    job = create_eval_job(1, 100, 1; chunk_size=50)
    s = status(job)
    @test s.eval_iter == 1
    @test s.total_chunks == 4
    @test s.completed == 0
    @test s.available == 4
    @test s.checked_out == 0

    checkout_chunk!(job, "neo")
    s = status(job)
    @test s.checked_out == 1
    @test s.available == 3

    # Complete one chunk
    chunk = job.chunks[1]
    result = EvalChunkResult(chunk.chunk_id, chunk.az_is_white,
        [0.0], [0.0], [0.0], [false])
    submit_chunk!(job, result)
    s = status(job)
    @test s.completed == 1
    @test s.checked_out == 0
    @test s.available == 3
end

@testset "finalize_eval" begin
    job = create_eval_job(1, 2, 1; chunk_size=2)
    # 2 chunks: chunk 1 (white, positions 1:2), chunk 2 (black, positions 1:2)

    # White side: AZ wins both (+1 each, backgammon scale)
    r1 = EvalChunkResult(1, true,
        [1.0, 2.0],      # rewards
        [0.8, 1.5],      # value_nn
        [0.9, 1.6],      # value_opp
        [true, false])
    submit_chunk!(job, r1)

    # Black side: AZ loses both (-1 each)
    r2 = EvalChunkResult(2, false,
        [-1.0, -2.0],
        [-0.7, -1.3],
        [-0.8, -1.4],
        [true, true])
    submit_chunk!(job, r2)

    stats = finalize_eval(job)

    @test stats.num_games == 4
    @test stats.white_equity == 1.5       # mean([1.0, 2.0])
    @test stats.black_equity == -1.5      # mean([-1.0, -2.0])
    @test stats.equity == 0.0             # mean([1.0, 2.0, -1.0, -2.0])
    @test stats.win_pct == 0.5            # 2 wins out of 4

    # Value MSE: diffs are [0.8-0.9, 1.5-1.6, -0.7-(-0.8), -1.3-(-1.4)]
    #          = [-0.1, -0.1, 0.1, 0.1], squared = [0.01, 0.01, 0.01, 0.01]
    @test stats.value_mse ≈ 0.01 atol=1e-10

    # Value correlation: nn and opp are perfectly linearly related (nn = opp + 0.1)
    @test stats.value_corr ≈ 1.0 atol=1e-3
end

@testset "finalize_eval empty" begin
    job = create_eval_job(1, 100, 1; chunk_size=50)
    stats = finalize_eval(job)
    @test stats.num_games == 0
    @test stats.equity == 0.0
    @test stats.win_pct == 0.0
end

end # testset
