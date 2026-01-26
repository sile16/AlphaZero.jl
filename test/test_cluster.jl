"""
Unit tests for the Cluster module (Julia Distributed-based training).
"""

using Test
using AlphaZero
using AlphaZero.Cluster
using AlphaZero: GI, Network, FluxLib
using AlphaZero.NetLib: serialize_weights, deserialize_weights, load_weights!
using Statistics: mean

# Load a simple game for testing
const GAMES_DIR = joinpath(@__DIR__, "..", "games")
include(joinpath(GAMES_DIR, "backgammon-deterministic", "main.jl"))

@testset "Cluster Module" begin

    @testset "ClusterSample" begin
        # Create a sample
        state = randn(Float32, 198)
        policy = randn(Float32, 6786)
        policy = abs.(policy) ./ sum(abs.(policy))  # Normalize
        value = 0.5f0
        turn = 10.0f0
        is_chance = false

        sample = ClusterSample(
            state, policy, value, turn, is_chance,
            0.6f0, 0.1f0, 0.02f0, 0.15f0, 0.03f0, true
        )

        @test length(sample.state) == 198
        @test length(sample.policy) == 6786
        @test sample.value == 0.5f0
        @test sample.turn == 10.0f0
        @test !sample.is_chance
        @test sample.equity_p_win == 0.6f0
        @test sample.has_equity
    end

    @testset "GameBatch" begin
        # Create some samples
        samples = [
            ClusterSample(
                randn(Float32, 10), randn(Float32, 5), 0.5f0, 1.0f0, false,
                0.5f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, true
            )
            for _ in 1:10
        ]

        batch = GameBatch(1, samples, 10, 1.0f0)

        @test batch.worker_id == 1
        @test length(batch.samples) == 10
        @test batch.game_length == 10
        @test batch.outcome == 1.0f0
    end

    @testset "WeightUpdate" begin
        weights = rand(UInt8, 1000)
        update = WeightUpdate(5, weights, time())

        @test update.iteration == 5
        @test length(update.weights) == 1000
        @test update.timestamp > 0
    end

    @testset "ClusterWorker" begin
        gspec = BackgammonDeterministic.GameSpec()
        mcts_params = MctsParams(
            num_iters_per_turn=10,  # Low for testing
            cpuct=2.0,
            temperature=ConstSchedule(1.0),
            gamma=1.0,
            dirichlet_noise_ϵ=0.25,
            dirichlet_noise_α=1.0
        )

        worker = ClusterWorker(1, gspec, mcts_params)

        @test worker.worker_id == 1
        @test isnothing(worker.network)
        @test worker.games_played == 0
        @test worker.running
    end

    @testset "ClusterCoordinator" begin
        gspec = BackgammonDeterministic.GameSpec()

        # Create a simple network
        hp = NetLib.SimpleNetHP(width=32, depth_common=2)
        network = NetLib.SimpleNet(gspec, hp)

        learning_params = LearningParams(
            batch_size=32,
            loss_computation_batch_size=32,
            optimiser=Adam(lr=0.001),
            l2_regularization=Float32(1e-4),
            use_gpu=false,
            samples_weighing_policy=CONSTANT_WEIGHT,
            min_checkpoints_per_epoch=1,
            max_batches_per_checkpoint=10,
            num_checkpoints=1
        )

        mcts_params = MctsParams(
            num_iters_per_turn=10,
            cpuct=2.0,
            temperature=ConstSchedule(1.0),
            gamma=1.0,
            dirichlet_noise_ϵ=0.25,
            dirichlet_noise_α=1.0
        )

        coord = ClusterCoordinator(
            gspec, network, learning_params, mcts_params;
            buffer_capacity=1000,
            use_gpu=false
        )

        @test coord.iteration == 0
        @test coord.total_games == 0
        @test isempty(coord.buffer)
        @test coord.buffer_capacity == 1000
    end

    @testset "add_samples!" begin
        gspec = BackgammonDeterministic.GameSpec()
        hp = NetLib.SimpleNetHP(width=32, depth_common=2)
        network = NetLib.SimpleNet(gspec, hp)

        learning_params = LearningParams(
            batch_size=32,
            loss_computation_batch_size=32,
            optimiser=Adam(lr=0.001),
            l2_regularization=Float32(1e-4),
            use_gpu=false,
            samples_weighing_policy=CONSTANT_WEIGHT,
            min_checkpoints_per_epoch=1,
            max_batches_per_checkpoint=10,
            num_checkpoints=1
        )

        mcts_params = MctsParams(num_iters_per_turn=10, cpuct=2.0, dirichlet_noise_ϵ=0.25, dirichlet_noise_α=1.0)

        coord = ClusterCoordinator(
            gspec, network, learning_params, mcts_params;
            buffer_capacity=100,
            use_gpu=false
        )

        # Create a batch of samples
        state_dim = prod(GI.state_dim(gspec))
        policy_dim = GI.num_actions(gspec)

        samples = [
            ClusterSample(
                randn(Float32, state_dim), randn(Float32, policy_dim),
                0.5f0, Float32(i), false,
                0.5f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, true
            )
            for i in 1:10
        ]

        batch = GameBatch(1, samples, 10, 1.0f0)
        add_samples!(coord, batch)

        @test length(coord.buffer) == 10
        @test coord.total_games == 1
        @test coord.total_samples == 10

        # Add more and test capacity limit
        for _ in 1:15
            add_samples!(coord, batch)
        end

        @test length(coord.buffer) == 100  # Should be at capacity
        @test coord.total_games == 16
    end

    @testset "sample_batch" begin
        gspec = BackgammonDeterministic.GameSpec()
        hp = NetLib.SimpleNetHP(width=32, depth_common=2)
        network = NetLib.SimpleNet(gspec, hp)

        learning_params = LearningParams(
            batch_size=32,
            loss_computation_batch_size=32,
            optimiser=Adam(lr=0.001),
            l2_regularization=Float32(1e-4),
            use_gpu=false,
            samples_weighing_policy=CONSTANT_WEIGHT,
            min_checkpoints_per_epoch=1,
            max_batches_per_checkpoint=10,
            num_checkpoints=1
        )

        mcts_params = MctsParams(num_iters_per_turn=10, cpuct=2.0, dirichlet_noise_ϵ=0.25, dirichlet_noise_α=1.0)

        coord = ClusterCoordinator(
            gspec, network, learning_params, mcts_params;
            buffer_capacity=1000,
            use_gpu=false
        )

        # Empty buffer should return nothing
        @test isnothing(sample_batch(coord, 10))

        # Add samples
        state_dim = prod(GI.state_dim(gspec))
        policy_dim = GI.num_actions(gspec)

        samples = [
            ClusterSample(
                randn(Float32, state_dim), randn(Float32, policy_dim),
                0.5f0, Float32(i), false,
                0.5f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, true
            )
            for i in 1:50
        ]

        batch = GameBatch(1, samples, 50, 1.0f0)
        add_samples!(coord, batch)

        # Now sampling should work
        sampled = sample_batch(coord, 20)
        @test !isnothing(sampled)
        @test length(sampled) == 20

        # Request more than buffer size should return nothing
        @test isnothing(sample_batch(coord, 100))
    end

    @testset "Weight serialization" begin
        gspec = BackgammonDeterministic.GameSpec()

        # Create network
        hp = NetLib.SimpleNetHP(width=32, depth_common=2)
        network = NetLib.SimpleNet(gspec, hp)

        # Serialize
        bytes = serialize_weights(network)
        @test length(bytes) > 0

        # Deserialize
        weights = deserialize_weights(bytes)
        @test length(weights) > 0

        # Create new network and load weights
        network2 = NetLib.SimpleNet(gspec, hp)
        load_weights!(network2, weights)

        # Compare parameters
        params1 = Network.params(network)
        params2 = Network.params(network2)

        @test length(params1) == length(params2)
        for (p1, p2) in zip(params1, params2)
            @test p1 ≈ p2
        end
    end

    @testset "prepare_training_batch" begin
        gspec = BackgammonDeterministic.GameSpec()
        hp = NetLib.SimpleNetHP(width=32, depth_common=2)
        network = NetLib.SimpleNet(gspec, hp)

        learning_params = LearningParams(
            batch_size=32,
            loss_computation_batch_size=32,
            optimiser=Adam(lr=0.001),
            l2_regularization=Float32(1e-4),
            use_gpu=false,
            samples_weighing_policy=CONSTANT_WEIGHT,
            min_checkpoints_per_epoch=1,
            max_batches_per_checkpoint=10,
            num_checkpoints=1
        )

        mcts_params = MctsParams(num_iters_per_turn=10, cpuct=2.0, dirichlet_noise_ϵ=0.25, dirichlet_noise_α=1.0)

        coord = ClusterCoordinator(
            gspec, network, learning_params, mcts_params;
            buffer_capacity=1000,
            use_gpu=false
        )

        state_dim = prod(GI.state_dim(gspec))
        policy_dim = GI.num_actions(gspec)

        # Create samples
        samples = [
            ClusterSample(
                randn(Float32, state_dim),
                abs.(randn(Float32, policy_dim)),  # Non-negative for action mask
                rand(Float32) * 2 - 1,  # Value in [-1, 1]
                Float32(i),
                false,
                rand(Float32), 0.0f0, 0.0f0, 0.0f0, 0.0f0, true
            )
            for i in 1:16
        ]

        batch = prepare_training_batch(coord, samples)

        @test haskey(batch, :W)
        @test haskey(batch, :X)
        @test haskey(batch, :A)
        @test haskey(batch, :P)
        @test haskey(batch, :V)
        @test haskey(batch, :IsChance)
        @test haskey(batch, :HasEquity)

        @test size(batch.W, 2) == 16  # Batch dimension
        @test size(batch.V, 2) == 16
    end

    @testset "training_step!" begin
        gspec = BackgammonDeterministic.GameSpec()
        hp = NetLib.SimpleNetHP(width=32, depth_common=2)
        network = NetLib.SimpleNet(gspec, hp)

        learning_params = LearningParams(
            batch_size=16,
            loss_computation_batch_size=16,
            optimiser=Adam(lr=0.001),
            l2_regularization=Float32(1e-4),
            use_gpu=false,
            samples_weighing_policy=CONSTANT_WEIGHT,
            min_checkpoints_per_epoch=1,
            max_batches_per_checkpoint=10,
            num_checkpoints=1
        )

        mcts_params = MctsParams(num_iters_per_turn=10, cpuct=2.0, dirichlet_noise_ϵ=0.25, dirichlet_noise_α=1.0)

        coord = ClusterCoordinator(
            gspec, network, learning_params, mcts_params;
            buffer_capacity=1000,
            use_gpu=false
        )

        # Empty buffer - no training
        @test isnothing(training_step!(coord, 16))

        # Add samples
        state_dim = prod(GI.state_dim(gspec))
        policy_dim = GI.num_actions(gspec)

        samples = [
            ClusterSample(
                randn(Float32, state_dim),
                abs.(randn(Float32, policy_dim)) .+ 0.01f0,  # Ensure non-zero for action mask
                rand(Float32) * 2 - 1,
                Float32(i),
                false,
                rand(Float32), 0.0f0, 0.0f0, 0.0f0, 0.0f0, true
            )
            for i in 1:50
        ]

        batch = GameBatch(1, samples, 50, 1.0f0)
        add_samples!(coord, batch)

        # Now training should work
        loss = training_step!(coord, 16)
        @test !isnothing(loss)
        @test loss > 0  # Loss should be positive
    end

    @testset "TrainingMetrics and EvalResults" begin
        metrics = TrainingMetrics(10, 0.5, 5000, 100, 50000, 15.5)

        @test metrics.iteration == 10
        @test metrics.loss == 0.5
        @test metrics.buffer_size == 5000
        @test metrics.total_games == 100
        @test metrics.games_per_minute == 15.5

        eval_results = EvalResults(10, 0.8, 0.7, 0.75, 100, 60.0)

        @test eval_results.iteration == 10
        @test eval_results.vs_random_white == 0.8
        @test eval_results.vs_random_black == 0.7
        @test eval_results.vs_random_combined == 0.75
        @test eval_results.num_games == 100
    end

end

println("All cluster tests passed!")
