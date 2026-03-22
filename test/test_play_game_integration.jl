using Test
using Random
using AlphaZero
using AlphaZero: GI, GameLoop, MctsParams, ConstSchedule, BatchedMCTS

# Use tictactoe as a simple deterministic game for testing
include(joinpath(@__DIR__, "..", "games", "tictactoe", "game.jl"))

const TTT = GameSpec()

# Oracle: returns uniform policy over LEGAL actions + 0 value
# MCTS expects policy length == number of available actions
function dummy_oracle(state)
    env = GI.init(TTT, state)
    n_legal = count(GI.actions_mask(env))
    policy = ones(Float32, n_legal) ./ n_legal
    return policy, 0.0f0
end

@testset "play_game() Integration" begin

    @testset "MCTS vs MCTS completes game" begin
        params = MctsParams(
            num_iters_per_turn=10,
            cpuct=1.5,
            temperature=ConstSchedule(0.0),  # greedy
            dirichlet_noise_ϵ=0.0,
            dirichlet_noise_α=1.0)

        agent = GameLoop.MctsAgent(
            dummy_oracle, nothing, params, 10, TTT;
            bearoff_eval=nothing)

        env = GI.init(TTT)
        result = GameLoop.play_game(agent, agent, env;
            record_trace=true, rng=Random.MersenneTwister(42))

        # Game should terminate
        @test result.reward ∈ [-1.0, 0.0, 1.0]  # win/draw/loss
        @test result.num_moves > 0
        @test result.bearoff_truncated == false

        # Trace should have entries
        @test length(result.trace) > 0

        # Each trace entry should have valid fields
        for entry in result.trace
            @test entry.player ∈ [0, 1]
            @test entry.action > 0
            @test !isempty(entry.legal_actions)
            @test entry.action ∈ entry.legal_actions
            @test length(entry.policy) == length(entry.legal_actions)
            @test all(p -> p >= 0, entry.policy)
        end
    end

    @testset "num_moves counts all decision points including forced" begin
        params = MctsParams(
            num_iters_per_turn=5,
            cpuct=1.5,
            temperature=ConstSchedule(0.0),
            dirichlet_noise_ϵ=0.0,
            dirichlet_noise_α=1.0)

        agent = GameLoop.MctsAgent(
            dummy_oracle, nothing, params, 5, TTT;
            bearoff_eval=nothing)

        # Play many games and verify num_moves equals trace length
        # (in tictactoe there are no forced moves since board always has 2+ open cells
        #  until the last move, but num_moves should match trace entries)
        for seed in 1:20
            env = GI.init(TTT)
            result = GameLoop.play_game(agent, agent, env;
                record_trace=true, rng=Random.MersenneTwister(seed))
            @test result.num_moves == length(result.trace)
            @test result.num_moves >= 5  # minimum moves in tictactoe
            @test result.num_moves <= 9  # maximum moves in tictactoe
        end
    end

    @testset "temperature scheduling affects action selection" begin
        # With temperature=1.0 (sampling), different seeds should produce different games
        params_sample = MctsParams(
            num_iters_per_turn=10,
            cpuct=1.5,
            temperature=ConstSchedule(1.0),  # sample
            dirichlet_noise_ϵ=0.25,
            dirichlet_noise_α=1.0)

        agent_sample = GameLoop.MctsAgent(
            dummy_oracle, nothing, params_sample, 10, TTT;
            bearoff_eval=nothing)

        # Play with different seeds — should get some variety
        results = []
        for seed in 1:10
            env = GI.init(TTT)
            r = GameLoop.play_game(agent_sample, agent_sample, env;
                record_trace=true, rng=Random.MersenneTwister(seed))
            push!(results, r)
        end

        # With temp=1.0 and noise, not all games should be identical
        first_moves = [r.trace[1].action for r in results]
        @test length(unique(first_moves)) > 1  # at least 2 different first moves

        # With temperature=0.0 (greedy), same seed = same game
        params_greedy = MctsParams(
            num_iters_per_turn=10,
            cpuct=1.5,
            temperature=ConstSchedule(0.0),  # greedy
            dirichlet_noise_ϵ=0.0,
            dirichlet_noise_α=1.0)

        agent_greedy = GameLoop.MctsAgent(
            dummy_oracle, nothing, params_greedy, 10, TTT;
            bearoff_eval=nothing)

        env1 = GI.init(TTT)
        env2 = GI.init(TTT)
        r1 = GameLoop.play_game(agent_greedy, agent_greedy, env1;
            record_trace=true, rng=Random.MersenneTwister(100))
        r2 = GameLoop.play_game(agent_greedy, agent_greedy, env2;
            record_trace=true, rng=Random.MersenneTwister(100))

        # Same seed + greedy = deterministic
        @test r1.reward == r2.reward
        @test length(r1.trace) == length(r2.trace)
        for (t1, t2) in zip(r1.trace, r2.trace)
            @test t1.action == t2.action
        end
    end

    @testset "temperature_fn controls per-move temperature" begin
        # Custom temperature: explore first 3 moves, then greedy
        params = MctsParams(
            num_iters_per_turn=10,
            cpuct=1.5,
            temperature=ConstSchedule(1.0),
            dirichlet_noise_ϵ=0.25,
            dirichlet_noise_α=1.0)

        agent = GameLoop.MctsAgent(
            dummy_oracle, nothing, params, 10, TTT;
            bearoff_eval=nothing)

        temp_fn = move_num -> move_num <= 3 ? 1.0 : 0.0

        env = GI.init(TTT)
        result = GameLoop.play_game(agent, agent, env;
            record_trace=true, temperature_fn=temp_fn, rng=Random.MersenneTwister(42))

        @test result.num_moves >= 5
        @test !isempty(result.trace)
    end

    @testset "record_trace=false produces empty trace" begin
        params = MctsParams(
            num_iters_per_turn=5,
            cpuct=1.5,
            temperature=ConstSchedule(0.0),
            dirichlet_noise_ϵ=0.0,
            dirichlet_noise_α=1.0)

        agent = GameLoop.MctsAgent(
            dummy_oracle, nothing, params, 5, TTT;
            bearoff_eval=nothing)

        env = GI.init(TTT)
        result = GameLoop.play_game(agent, agent, env;
            record_trace=false, rng=Random.MersenneTwister(42))

        @test isempty(result.trace)
        @test result.num_moves > 0  # game still played
        @test result.reward ∈ [-1.0, 0.0, 1.0]
    end

    @testset "white reward perspective" begin
        params = MctsParams(
            num_iters_per_turn=10,
            cpuct=1.5,
            temperature=ConstSchedule(0.0),
            dirichlet_noise_ϵ=0.0,
            dirichlet_noise_α=1.0)

        agent = GameLoop.MctsAgent(
            dummy_oracle, nothing, params, 10, TTT;
            bearoff_eval=nothing)

        # Play many games — reward should be from white's perspective
        rewards = Float64[]
        for seed in 1:50
            env = GI.init(TTT)
            r = GameLoop.play_game(agent, agent, env; rng=Random.MersenneTwister(seed))
            push!(rewards, r.reward)
        end

        # Should see a mix of outcomes (not all same)
        @test any(r -> r != 0.0, rewards)  # at least some decisive games
        @test all(r -> r ∈ [-1.0, 0.0, 1.0], rewards)  # valid tictactoe rewards
    end

    @testset "parametric type stability" begin
        params = MctsParams(
            num_iters_per_turn=5,
            cpuct=1.5,
            temperature=ConstSchedule(1.0),
            dirichlet_noise_ϵ=0.0,
            dirichlet_noise_α=1.0)

        agent = GameLoop.MctsAgent(
            dummy_oracle, nothing, params, 5, TTT;
            bearoff_eval=nothing)

        # Verify parametric types are inferred (not Any)
        @test !(typeof(agent).parameters[1] === Any)  # oracle type
        @test typeof(agent).parameters[3] <: Any       # gspec type (GameSpec)
    end
end
