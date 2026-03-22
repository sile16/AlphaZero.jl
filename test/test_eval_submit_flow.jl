using Test

if !isdefined(Main, :EvalManager)
    include(joinpath(@__DIR__, "..", "src", "distributed", "eval_manager.jl"))
end
using .EvalManager

@testset "Eval Submit Flow" begin

    @testset "Full eval flow: create → checkout → submit → finalize" begin
        # 6 positions, chunk_size=3 → 2 white chunks + 2 black chunks = 4 chunks
        job = create_eval_job(1, 6, 1; chunk_size=3)
        @test length(job.chunks) == 4
        @test !is_complete(job)

        # Checkout and submit all chunks
        for i in 1:4
            chunk = checkout_chunk!(job, "client-1")
            @test chunk !== nothing
            result = EvalChunkResult(
                chunk.chunk_id, chunk.az_is_white,
                fill(0.5, length(chunk.position_range)),  # rewards
                Float64[], Float64[], Bool[]
            )
            @test submit_chunk!(job, result)
        end

        # No more chunks available
        @test checkout_chunk!(job, "client-1") === nothing
        @test is_complete(job)

        stats = finalize_eval(job)
        @test stats.num_games == 12  # 6 positions × 2 sides
        @test stats.equity ≈ 0.5
        @test stats.win_pct ≈ 1.0  # all rewards > 0
        @test stats.white_equity ≈ 0.5
        @test stats.black_equity ≈ 0.5
    end

    @testset "White/black split correctness" begin
        # 4 positions, chunk_size=4 → 1 white chunk + 1 black chunk
        job = create_eval_job(1, 4, 1; chunk_size=4)
        @test length(job.chunks) == 2

        # White chunk: all wins
        c1 = checkout_chunk!(job, "c")
        @test c1.az_is_white == true
        submit_chunk!(job, EvalChunkResult(c1.chunk_id, true,
            [1.0, 1.0, 1.0, 1.0], Float64[], Float64[], Bool[]))

        # Black chunk: all losses
        c2 = checkout_chunk!(job, "c")
        @test c2.az_is_white == false
        submit_chunk!(job, EvalChunkResult(c2.chunk_id, false,
            [-1.0, -1.0, -1.0, -1.0], Float64[], Float64[], Bool[]))

        stats = finalize_eval(job)
        @test stats.white_equity ≈ 1.0
        @test stats.black_equity ≈ -1.0
        @test stats.equity ≈ 0.0
        @test stats.win_pct ≈ 0.5  # 4 wins out of 8
        @test stats.num_games == 8
    end

    @testset "Ownership: checked_out_by is set correctly" begin
        job = create_eval_job(1, 10, 1; chunk_size=5)

        c1 = checkout_chunk!(job, "alice")
        @test c1.checked_out_by == "alice"

        c2 = checkout_chunk!(job, "bob")
        @test c2.checked_out_by == "bob"
        @test c2.chunk_id != c1.chunk_id

        # submit_chunk! does not validate client — it accepts from anyone
        result = EvalChunkResult(c1.chunk_id, c1.az_is_white,
            fill(0.0, length(c1.position_range)), Float64[], Float64[], Bool[])
        @test submit_chunk!(job, result) == true

        # After submit, checked_out_by is cleared
        idx = findfirst(c -> c.chunk_id == c1.chunk_id, job.chunks)
        @test job.chunks[idx].checked_out_by === nothing
        @test job.chunks[idx].completed == true
    end

    @testset "Weights version tracking" begin
        job = create_eval_job(42, 10, 5; chunk_size=5)
        @test job.weights_version == 5

        chunk = checkout_chunk!(job, "client")
        @test chunk !== nothing
        # The chunk itself doesn't carry weights_version; the job does
        @test job.weights_version == 5

        s = status(job)
        # status doesn't include weights_version directly, but includes eval_iter
        @test s.eval_iter == 42
        @test s.total_chunks == 4  # 2 white + 2 black
        @test s.completed == 0
        @test s.checked_out == 1
        @test s.available == 3
    end

    @testset "Finalize with known reward data" begin
        # 3 positions, chunk_size=3 → 1 white chunk + 1 black chunk
        job = create_eval_job(1, 3, 1; chunk_size=3)

        c1 = checkout_chunk!(job, "c")
        @test c1.az_is_white == true
        submit_chunk!(job, EvalChunkResult(c1.chunk_id, true,
            [1.0, -1.0, 1.0], Float64[], Float64[], Bool[]))

        c2 = checkout_chunk!(job, "c")
        @test c2.az_is_white == false
        submit_chunk!(job, EvalChunkResult(c2.chunk_id, false,
            [-1.0, 1.0, -1.0], Float64[], Float64[], Bool[]))

        stats = finalize_eval(job)
        all_rewards = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        @test stats.equity ≈ 0.0
        @test stats.win_pct ≈ 0.5  # 3 out of 6 > 0
        @test stats.white_equity ≈ 1/3  # mean([1, -1, 1])
        @test stats.black_equity ≈ -1/3  # mean([-1, 1, -1])
        @test stats.num_games == 6
    end

    @testset "Value metrics in finalize" begin
        job = create_eval_job(1, 4, 1; chunk_size=4)

        # White chunk with value data
        c1 = checkout_chunk!(job, "c")
        submit_chunk!(job, EvalChunkResult(c1.chunk_id, true,
            fill(0.0, 4),
            [0.1, 0.3, 0.5, 0.7],   # value_nn
            [0.2, 0.4, 0.4, 0.8],   # value_opp
            [true, true, false, false]))

        # Black chunk with value data
        c2 = checkout_chunk!(job, "c")
        submit_chunk!(job, EvalChunkResult(c2.chunk_id, false,
            fill(0.0, 4),
            [0.0, 0.6],              # value_nn (fewer samples is fine)
            [0.1, 0.5],              # value_opp
            [true, false]))

        stats = finalize_eval(job)

        # All value_nn and value_opp concatenated
        nn  = [0.1, 0.3, 0.5, 0.7, 0.0, 0.6]
        opp = [0.2, 0.4, 0.4, 0.8, 0.1, 0.5]
        diffs = nn .- opp
        expected_mse = sum(diffs .^ 2) / length(diffs)
        @test stats.value_mse ≈ expected_mse

        # Correlation should be positive and well-defined (6 samples)
        using Statistics: cor
        expected_corr = cor(nn, opp)
        @test stats.value_corr ≈ expected_corr
        @test stats.value_corr > 0
    end

    @testset "Edge: empty value data" begin
        job = create_eval_job(1, 2, 1; chunk_size=2)

        c1 = checkout_chunk!(job, "c")
        submit_chunk!(job, EvalChunkResult(c1.chunk_id, true,
            [1.0, -1.0], Float64[], Float64[], Bool[]))

        c2 = checkout_chunk!(job, "c")
        submit_chunk!(job, EvalChunkResult(c2.chunk_id, false,
            [0.5, -0.5], Float64[], Float64[], Bool[]))

        stats = finalize_eval(job)
        @test stats.value_mse == 0.0
        @test stats.value_corr == 0.0
        @test stats.num_games == 4
    end

    @testset "Edge: single value sample (< 2 needed for cor)" begin
        job = create_eval_job(1, 1, 1; chunk_size=1)

        c1 = checkout_chunk!(job, "c")
        submit_chunk!(job, EvalChunkResult(c1.chunk_id, true,
            [1.0], [0.5], [0.3], [true]))

        c2 = checkout_chunk!(job, "c")
        submit_chunk!(job, EvalChunkResult(c2.chunk_id, false,
            [-1.0], Float64[], Float64[], Bool[]))

        stats = finalize_eval(job)
        # Only 1 value sample — below the 2-sample threshold
        @test stats.value_mse == 0.0
        @test stats.value_corr == 0.0
    end

    @testset "Edge: single chunk job" begin
        job = create_eval_job(1, 1, 1; chunk_size=100)
        # 1 position → 1 white chunk + 1 black chunk (chunk_size > num_positions)
        @test length(job.chunks) == 2

        for _ in 1:2
            c = checkout_chunk!(job, "c")
            submit_chunk!(job, EvalChunkResult(c.chunk_id, c.az_is_white,
                [0.0], Float64[], Float64[], Bool[]))
        end

        @test is_complete(job)
        stats = finalize_eval(job)
        @test stats.num_games == 2
        @test stats.equity ≈ 0.0
    end

    @testset "Edge: all chunks same side (white)" begin
        # We can't make create_eval_job produce only white chunks, but we can
        # test finalize behavior with only white results submitted.
        job = create_eval_job(1, 2, 1; chunk_size=2)

        # Only submit white chunk, skip black
        c1 = checkout_chunk!(job, "c")
        @test c1.az_is_white == true
        submit_chunk!(job, EvalChunkResult(c1.chunk_id, true,
            [1.0, -1.0], Float64[], Float64[], Bool[]))

        # Don't complete the black chunk — finalize with partial results
        stats = finalize_eval(job)
        @test stats.white_equity ≈ 0.0
        @test stats.black_equity ≈ 0.0  # no black rewards → default 0
        @test stats.num_games == 2  # only white games counted
    end

    @testset "Lease expiry and re-checkout" begin
        job = create_eval_job(1, 4, 1; chunk_size=4)

        c1 = checkout_chunk!(job, "slow-client")
        @test c1 !== nothing
        # Simulate stale checkout by backdating
        c1.checkout_time = time() - 400.0

        expired = expire_stale_checkouts!(job; lease_seconds=300.0)
        @test expired == 1
        @test c1.checked_out_by === nothing

        # Can re-checkout the same chunk
        c1_again = checkout_chunk!(job, "fast-client")
        @test c1_again.chunk_id == c1.chunk_id
        @test c1_again.checked_out_by == "fast-client"
    end

    @testset "Extend lease" begin
        job = create_eval_job(1, 4, 1; chunk_size=4)

        c = checkout_chunk!(job, "client-a")
        old_time = c.checkout_time

        # Extending from wrong client fails
        @test extend_lease!(job, c.chunk_id, "client-b") == false

        # Extending from correct client succeeds
        @test extend_lease!(job, c.chunk_id, "client-a") == true
        @test c.checkout_time >= old_time

        # Extending nonexistent chunk fails
        @test extend_lease!(job, 9999, "client-a") == false
    end

    @testset "Submit invalid chunk_id" begin
        job = create_eval_job(1, 2, 1; chunk_size=2)
        result = EvalChunkResult(9999, true, [1.0], Float64[], Float64[], Bool[])
        @test submit_chunk!(job, result) == false
    end

    @testset "Chunk position ranges are correct" begin
        # 10 positions, chunk_size=3 → 4 white chunks (3+3+3+1) + 4 black = 8
        job = create_eval_job(1, 10, 1; chunk_size=3)
        @test length(job.chunks) == 8

        white_chunks = filter(c -> c.az_is_white, job.chunks)
        black_chunks = filter(c -> !c.az_is_white, job.chunks)
        @test length(white_chunks) == 4
        @test length(black_chunks) == 4

        # White chunks cover 1:3, 4:6, 7:9, 10:10
        @test white_chunks[1].position_range == 1:3
        @test white_chunks[2].position_range == 4:6
        @test white_chunks[3].position_range == 7:9
        @test white_chunks[4].position_range == 10:10

        # Black chunks mirror the same ranges
        @test black_chunks[1].position_range == 1:3
        @test black_chunks[4].position_range == 10:10
    end

    @testset "Status reflects progress" begin
        job = create_eval_job(1, 4, 1; chunk_size=4)
        # 2 chunks: 1 white + 1 black

        s0 = status(job)
        @test s0.total_chunks == 2
        @test s0.completed == 0
        @test s0.checked_out == 0
        @test s0.available == 2

        c1 = checkout_chunk!(job, "c")
        s1 = status(job)
        @test s1.checked_out == 1
        @test s1.available == 1

        submit_chunk!(job, EvalChunkResult(c1.chunk_id, c1.az_is_white,
            fill(0.0, 4), Float64[], Float64[], Bool[]))
        s2 = status(job)
        @test s2.completed == 1
        @test s2.checked_out == 0
        @test s2.available == 1
    end

end
