using Test
using AlphaZero
using AlphaZero: GameLoop, MctsParams, ConstSchedule

@testset "GameLoop Types" begin
    @testset "MctsAgent construction" begin
        # MctsAgent with all fields
        params = MctsParams(
            num_iters_per_turn=10,
            cpuct=1.5,
            temperature=ConstSchedule(1.0),
            dirichlet_noise_ϵ=0.0,
            dirichlet_noise_α=1.0)

        agent = GameLoop.MctsAgent(
            identity,       # dummy oracle
            identity,       # dummy batch oracle
            params,
            32,             # batch_size
            nothing;        # gspec
            bearoff_eval=nothing)

        @test agent isa GameLoop.MctsAgent
        @test agent isa GameLoop.GameAgent
        @test agent.mcts_params === params
        @test agent.batch_size == 32
        @test agent.bearoff_eval === nothing
    end

    @testset "ExternalAgent construction" begin
        agent = GameLoop.ExternalAgent(:dummy_backend)
        @test agent isa GameLoop.ExternalAgent
        @test agent isa GameLoop.GameAgent
        @test agent.backend === :dummy_backend
    end

    @testset "TraceEntry construction" begin
        entry = GameLoop.TraceEntry(
            nothing,        # state
            0,              # player
            1,              # action
            [1, 2, 3],      # legal_actions
            Float32[0.5, 0.3, 0.2],  # policy
            false,          # is_chance
            false,          # is_bearoff
            true)           # is_contact

        @test entry.player == 0
        @test entry.action == 1
        @test length(entry.legal_actions) == 3
        @test sum(entry.policy) ≈ 1.0
        @test entry.is_contact == true
        @test entry.is_bearoff == false
    end

    @testset "PositionValueSample construction" begin
        sample = GameLoop.PositionValueSample(0.5, 0.3, true)
        @test sample.nn_val == 0.5
        @test sample.opponent_val == 0.3
        @test sample.is_contact == true
    end

    @testset "GameResult construction" begin
        result = GameLoop.GameResult(
            1.0,                                    # reward
            GameLoop.TraceEntry[],                  # trace
            GameLoop.PositionValueSample[],         # value_samples
            42,                                     # num_moves
            false,                                  # bearoff_truncated
            nothing,                                # first_bearoff_result
            nothing)                                # first_bearoff_white_playing

        @test result.reward == 1.0
        @test isempty(result.trace)
        @test isempty(result.value_samples)
        @test result.num_moves == 42
        @test result.bearoff_truncated == false
        @test result.first_bearoff_result === nothing
        @test result.first_bearoff_white_playing === nothing
    end

    @testset "GameResult with bearoff data" begin
        bo_result = (value=0.75, equity=Float32[0.8, 0.1, 0.0, 0.05, 0.0])
        result = GameLoop.GameResult(
            0.5, GameLoop.TraceEntry[], GameLoop.PositionValueSample[],
            10, true, bo_result, true)

        @test result.bearoff_truncated == true
        @test result.first_bearoff_result.value == 0.75
        @test length(result.first_bearoff_result.equity) == 5
        @test result.first_bearoff_white_playing == true
    end

    @testset "create_player dispatches correctly" begin
        # ExternalAgent returns nothing
        ext = GameLoop.ExternalAgent(:dummy)
        @test GameLoop.create_player(ext) === nothing
    end
end
