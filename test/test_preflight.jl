using Test

include(joinpath(@__DIR__, "..", "src", "distributed", "preflight.jl"))
using .Preflight

@testset "Preflight report" begin
    mktempdir() do dir
        path = joinpath(dir, "report.json")
        report = run_preflight!([
            "contract" => (() -> Dict("fingerprint" => "abc")),
            "inference" => (() -> "finite"),
        ], path; metadata=Dict("commit" => "deadbeef"))
        @test preflight_ok(report)
        @test isfile(path)
        @test length(report["checks"]) == 2

        @test_throws ErrorException run_preflight!([
            "broken" => (() -> error("deliberate failure")),
        ], joinpath(dir, "failed.json"))
    end
end
