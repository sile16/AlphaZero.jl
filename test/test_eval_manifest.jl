using Test
using Serialization

include(joinpath(@__DIR__, "..", "src", "distributed", "eval_manifest.jl"))
using .EvalManifest

@testset "Evaluation manifests" begin
    mktempdir() do dir
        artifact = joinpath(dir, "positions.jls")
        positions = [(id=i, class=isodd(i) ? "contact" : "race") for i in 1:6]
        Serialization.serialize(artifact, positions)
        fp(position) = string(position.id)
        manifest = build_eval_manifest(artifact, positions;
            suite="fixed-v1", contract_fingerprint="contract",
            classify=position -> position.class, fingerprint=fp)
        path = write_eval_manifest(joinpath(dir, "positions.manifest.json"), manifest)
        validated = validate_eval_manifest(path, artifact, positions;
            contract_fingerprint="contract", fingerprint=fp)
        @test validated["strata"] == Dict("contact" => 3, "race" => 3)
        @test validated["unique_positions"] == 6
        @test evaluation_leakage(["1", "9"], ["1", "2"]) == Set(["1"])
        summary = grouped_eval_summary([1.0, -1.0, 2.0], ["race", "race", "contact"])
        @test summary["race"]["equity"] == 0.0
        @test summary["contact"]["win_rate"] == 1.0

        open(artifact, "a") do io
            write(io, UInt8(0))
        end
        @test_throws ArgumentError validate_eval_manifest(path, artifact, positions;
            contract_fingerprint="contract", fingerprint=fp)
    end
end
