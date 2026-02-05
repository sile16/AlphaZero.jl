using Test
using AlphaZero
using AlphaZero: ReanalyzeConfig, ReanalyzeStats, get_reanalyze_metrics
using AlphaZero: sample_for_reanalysis, sample_for_smart_reanalysis, count_stale_samples
using AlphaZero.Cluster: ClusterSample

@testset "Reanalyze Module" begin
    @testset "ReanalyzeConfig defaults" begin
        config = ReanalyzeConfig()
        @test config.enabled == true
        @test config.batch_size == 256
        @test config.update_interval == 1
        @test config.reanalyze_alpha == 0.5f0
        @test config.max_reanalyze_count == 5
        @test config.prioritize_high_td == true
        @test config.log_interval == 10
    end

    @testset "ReanalyzeConfig custom" begin
        config = ReanalyzeConfig(
            enabled=false,
            batch_size=128,
            reanalyze_alpha=0.3f0
        )
        @test config.enabled == false
        @test config.batch_size == 128
        @test config.reanalyze_alpha == 0.3f0
    end

    @testset "ReanalyzeStats" begin
        stats = ReanalyzeStats()
        @test stats.total_reanalyzed == 0
        @test stats.total_steps == 0
        @test stats.avg_td_error == 0.0
        @test stats.max_td_error == 0.0
        @test stats.avg_value_change == 0.0

        # Update stats
        stats.total_reanalyzed = 100
        stats.total_steps = 5
        stats.avg_td_error = 0.15
        stats.max_td_error = 0.45
        stats.avg_value_change = 0.08

        @test stats.total_reanalyzed == 100
        @test stats.total_steps == 5
    end

    @testset "get_reanalyze_metrics" begin
        stats = ReanalyzeStats()
        stats.total_reanalyzed = 256
        stats.total_steps = 10
        stats.avg_td_error = 0.2
        stats.max_td_error = 0.5
        stats.avg_value_change = 0.1

        metrics = get_reanalyze_metrics(stats)

        @test metrics["reanalyze/total_reanalyzed"] == 256.0
        @test metrics["reanalyze/total_steps"] == 10.0
        @test metrics["reanalyze/avg_td_error"] == 0.2
        @test metrics["reanalyze/max_td_error"] == 0.5
        @test metrics["reanalyze/avg_value_change"] == 0.1
    end

    @testset "sample_for_reanalysis - uniform" begin
        # Create mock samples with reanalyze fields
        samples = [
            ClusterSample(
                randn(Float32, 10), randn(Float32, 5), rand(Float32), Float32(i),
                false, 0.5f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, true,
                0.1f0,  # priority
                1,      # added_step
                1,      # last_reanalyze_step
                0,      # reanalyze_count
                0       # model_iter_reanalyzed
            )
            for i in 1:100
        ]

        # Uniform sampling (prioritize_high_td=false)
        indices = sample_for_reanalysis(samples, 20, false, 10, 5)
        @test length(indices) == 20
        @test all(1 <= i <= 100 for i in indices)
    end

    @testset "sample_for_reanalysis - priority" begin
        # Create samples with varying priorities
        samples = [
            ClusterSample(
                randn(Float32, 10), randn(Float32, 5), rand(Float32), Float32(i),
                false, 0.5f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, true,
                Float32(i) / 100.0f0,  # priority increases with index
                1,      # added_step
                1,      # last_reanalyze_step
                0,      # reanalyze_count
                0       # model_iter_reanalyzed
            )
            for i in 1:100
        ]

        # Priority sampling should work
        indices = sample_for_reanalysis(samples, 20, true, 10, 5)
        @test length(indices) == 20
        @test all(1 <= i <= 100 for i in indices)
    end

    @testset "sample_for_reanalysis - max count filtering" begin
        # Create samples where some are at max reanalyze count
        samples = [
            ClusterSample(
                randn(Float32, 10), randn(Float32, 5), rand(Float32), Float32(i),
                false, 0.5f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, true,
                0.1f0,           # priority
                1,               # added_step
                1,               # last_reanalyze_step
                i <= 50 ? 5 : 0, # first 50 at max count
                0                # model_iter_reanalyzed
            )
            for i in 1:100
        ]

        # With priority sampling, should only select from indices 51-100
        indices = sample_for_reanalysis(samples, 20, true, 10, 5)
        @test length(indices) == 20
        # All selected should be from the eligible ones (51-100)
        @test all(i > 50 for i in indices)
    end

    @testset "ClusterSample with reanalyze fields" begin
        sample = ClusterSample(
            randn(Float32, 10), randn(Float32, 5), 0.5f0, 5.0f0,
            false, 0.6f0, 0.1f0, 0.02f0, 0.15f0, 0.03f0, true,
            0.25f0,  # priority
            10,      # added_step
            15,      # last_reanalyze_step
            2,       # reanalyze_count
            5        # model_iter_reanalyzed
        )

        @test sample.priority == 0.25f0
        @test sample.added_step == 10
        @test sample.last_reanalyze_step == 15
        @test sample.reanalyze_count == 2
        @test sample.model_iter_reanalyzed == 5
    end

    @testset "ClusterSample keyword constructor" begin
        sample = ClusterSample(
            randn(Float32, 10), randn(Float32, 5), 0.5f0, 5.0f0,
            false, 0.6f0, 0.1f0, 0.02f0, 0.15f0, 0.03f0, true;
            priority=0.3f0,
            added_step=5
        )

        @test sample.priority == 0.3f0
        @test sample.added_step == 5
        @test sample.last_reanalyze_step == 0  # default
        @test sample.reanalyze_count == 0       # default
        @test sample.model_iter_reanalyzed == 0  # default
    end

    @testset "sample_for_smart_reanalysis" begin
        # Create samples with varying model iterations
        samples = [
            ClusterSample(
                randn(Float32, 10), randn(Float32, 5), rand(Float32), Float32(i),
                false, 0.5f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, true,
                0.1f0,  # priority
                1,      # added_step
                1,      # last_reanalyze_step
                0,      # reanalyze_count
                i <= 50 ? 5 : 10  # first 50 at iter 5, rest at iter 10
            )
            for i in 1:100
        ]

        # Smart sampling for model iter 15 - all samples are stale
        indices = sample_for_smart_reanalysis(samples, 20, 15)
        @test length(indices) == 20
        # Should prioritize samples with older model iter (the first 50)
        @test all(i <= 50 for i in indices)

        # Smart sampling for model iter 10 - only first 50 are stale
        indices = sample_for_smart_reanalysis(samples, 20, 10)
        @test length(indices) == 20
        @test all(i <= 50 for i in indices)

        # Smart sampling for model iter 5 - no samples are stale
        indices = sample_for_smart_reanalysis(samples, 20, 5)
        @test isempty(indices)
    end

    @testset "count_stale_samples" begin
        samples = [
            ClusterSample(
                randn(Float32, 10), randn(Float32, 5), rand(Float32), Float32(i),
                false, 0.5f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, true,
                0.1f0, 1, 1, 0,
                i <= 30 ? 5 : 10  # first 30 at iter 5, rest at iter 10
            )
            for i in 1:100
        ]

        @test count_stale_samples(samples, 15) == 100  # all stale
        @test count_stale_samples(samples, 10) == 30   # only first 30 stale
        @test count_stale_samples(samples, 5) == 0     # none stale
    end
end

println("All reanalyze tests passed!")
