using Test
using Flux
using Random
using Serialization

include(joinpath(@__DIR__, "..", "src", "distributed", "checkpoint_manager.jl"))
using .CheckpointManager

@testset "Transactional checkpoint bundles" begin
    mktempdir() do root
        writer(text) = path -> write(path, text)
        first_bundle = write_checkpoint_bundle!(root, 10, Dict(
            "contact.data" => writer("contact-10"),
            "race.data" => writer("race-10"),
        ); metadata=Dict("contract_fingerprint" => "abc"))
        manifest = validate_checkpoint_bundle(first_bundle;
            required_files=["contact.data", "race.data"])
        @test manifest["iteration"] == 10
        @test manifest["metadata"]["contract_fingerprint"] == "abc"
        @test latest_valid_checkpoint(root) == first_bundle

        second_bundle = write_checkpoint_bundle!(root, 20, Dict(
            "contact.data" => writer("contact-20"),
            "race.data" => writer("race-20"),
        ))
        @test latest_valid_checkpoint(root) == second_bundle

        # Corruption is detected and discovery falls back to iteration 10.
        open(joinpath(second_bundle, "race.data"), "a") do io
            write(io, "corrupt")
        end
        @test_throws ArgumentError validate_checkpoint_bundle(second_bundle)
        @test latest_valid_checkpoint(root) == first_bundle

        # Temporary/incomplete directories are never candidates.
        mkdir(joinpath(root, ".tmp-bundle_iter_00000030-dead"))
        @test latest_valid_checkpoint(root) == first_bundle
    end
end


@testset "Optimizer and RNG checkpoint payloads" begin
    model = Chain(Dense(2 => 3, tanh), Dense(3 => 1))
    optimizer = Flux.setup(Flux.AdamW(1f-3), model)
    _, gradients = Flux.withgradient(model) do current
        sum(abs2, current(ones(Float32, 2, 4)))
    end
    Flux.update!(optimizer, model, gradients[1])
    rng = MersenneTwister(99)
    rand(rng, UInt64, 5)

    mktempdir() do root
        bundle = write_checkpoint_bundle!(root, 4, Dict{String,Function}(
            "optimizer.jls" => path -> Serialization.serialize(path, Flux.cpu(optimizer)),
            "rng.jls" => path -> Serialization.serialize(path, rng),
        ))
        restored_optimizer = Serialization.deserialize(joinpath(bundle, "optimizer.jls"))
        restored_rng = Serialization.deserialize(joinpath(bundle, "rng.jls"))
        @test restored_optimizer == Flux.cpu(optimizer)
        @test rand(restored_rng, UInt64, 10) == rand(rng, UInt64, 10)
    end
end
