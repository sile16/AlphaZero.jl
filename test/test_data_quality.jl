using Test

include(joinpath(@__DIR__, "..", "src", "distributed", "data_quality.jl"))
using .DataQuality

@testset "Corpus data quality" begin
    accumulator = QualityAccumulator()
    add_artifact!(accumulator; path="train-a", role="train",
        fingerprints=["a", "b", "b"], strata=Dict("contact" => 2, "race" => 1))
    add_artifact!(accumulator; path="eval-a", role="eval",
        fingerprints=["c"], strata=Dict("race" => 1))
    report = quality_report(accumulator)
    @test report["ok"]
    @test report["duplicate_training_positions"] == 1
    @test report["train_eval_leakage"] == 0

    add_artifact!(accumulator; path="eval-b", role="eval",
        fingerprints=["a"], strata=Dict("contact" => 1), illegal_policy_entries=1)
    failed = quality_report(accumulator)
    @test !failed["ok"]
    @test failed["train_eval_leakage"] == 1
    @test failed["illegal_policy_entries"] == 1

    mktempdir() do dir
        paths = write_quality_report(joinpath(dir, "quality"), failed)
        @test isfile(paths.json)
        @test isfile(paths.markdown)
    end
end
