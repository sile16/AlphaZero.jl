# Tests for stochastic MCTS with chance nodes

using Test
using AlphaZero
using AlphaZero: GI, MCTS, Trace, TrainingSample, play_game, TwoPlayers, MctsPlayer

# Include the Pig game
include("../games/pig/main.jl")
using .Pig

@testset "Stochastic MCTS" begin

  @testset "Pig Game Interface" begin
    gspec = Pig.GameSpec()
    game = GI.init(gspec)

    # Test initial state
    @test !GI.game_terminated(game)
    @test !GI.is_chance_node(game)
    @test GI.white_playing(game)  # Player 1 starts

    # Test actions mask at decision node
    mask = GI.actions_mask(game)
    @test mask == [true, true]  # ROLL and HOLD available

    # Test ROLL action transitions to chance node
    GI.play!(game, Pig.ROLL)
    @test GI.is_chance_node(game)
    @test GI.actions_mask(game) == [false, false]  # No actions at chance node

    # Test chance outcomes
    outcomes = GI.chance_outcomes(game)
    @test length(outcomes) == 6
    @test all(prob ≈ 1/6 for (_, prob) in outcomes)
    @test sum(prob for (_, prob) in outcomes) ≈ 1.0

    # Test apply_chance! with non-1 roll
    GI.apply_chance!(game, 4)
    @test !GI.is_chance_node(game)
    @test game.state.turn_total == 4

    # Test apply_chance! with roll of 1 (lose turn)
    game2 = GI.init(gspec)
    GI.play!(game2, Pig.ROLL)
    GI.apply_chance!(game2, 1)
    @test !GI.is_chance_node(game2)
    @test game2.state.turn_total == 0
    @test !GI.white_playing(game2)  # Turn switched

    # Test HOLD action
    game3 = GI.init(gspec)
    GI.play!(game3, Pig.ROLL)
    GI.apply_chance!(game3, 5)
    GI.play!(game3, Pig.HOLD)
    @test game3.state.p1_score == 5
    @test !GI.white_playing(game3)  # Turn switched
  end

  @testset "Game Termination" begin
    gspec = Pig.GameSpec()
    game = GI.init(gspec)

    # Manually set state close to winning
    winning_state = (p1_score=99, p2_score=50, turn_total=0, curplayer=true, awaiting_dice=false)
    GI.set_state!(game, winning_state)

    # Roll and get 2+
    GI.play!(game, Pig.ROLL)
    GI.apply_chance!(game, 2)
    GI.play!(game, Pig.HOLD)

    @test GI.game_terminated(game)
    @test GI.white_reward(game) == 1.0  # Player 1 wins
  end

  @testset "MCTS with Chance Nodes" begin
    gspec = Pig.GameSpec()
    game = GI.init(gspec)

    # Create MCTS environment with random oracle
    oracle = MCTS.RolloutOracle(gspec)
    mcts_env = MCTS.Env(gspec, oracle, cpuct=1.0, noise_ϵ=0.0)

    # Run a few simulations
    MCTS.explore!(mcts_env, game, 10)

    # Check that both trees have entries
    @test length(mcts_env.tree) > 0
    @test length(mcts_env.chance_tree) > 0

    # Check policy is valid
    actions, policy = MCTS.policy(mcts_env, game)
    @test length(policy) == 2
    @test sum(policy) ≈ 1.0
    @test all(p >= 0 for p in policy)

    # Reset should clear both trees
    MCTS.reset!(mcts_env)
    @test length(mcts_env.tree) == 0
    @test length(mcts_env.chance_tree) == 0
  end

  @testset "Trace with Chance Nodes" begin
    gspec = Pig.GameSpec()

    # Create a trace manually
    init_state = GI.current_state(GI.init(gspec))
    trace = Trace(init_state, is_chance_node=false)

    # Simulate: decision -> chance -> decision
    push!(trace, [0.5, 0.5], 0.0, init_state, is_chance=true)  # After ROLL
    push!(trace, Float64[], 0.0, init_state, is_chance=false)  # After dice roll

    @test length(trace.is_chance) == 3
    @test trace.is_chance[1] == false  # Initial state
    @test trace.is_chance[2] == true   # Chance node
    @test trace.is_chance[3] == false  # Decision node
  end

  @testset "TrainingSample with is_chance" begin
    # Test backward compatibility
    sample1 = TrainingSample((1,2,3), [0.5, 0.5], 1.0, 5.0, 1)
    @test sample1.is_chance == false

    # Test with explicit is_chance
    sample2 = TrainingSample((1,2,3), Float64[], 1.0, 5.0, 1, true)
    @test sample2.is_chance == true
  end

  @testset "Play Game with Chance Nodes" begin
    gspec = Pig.GameSpec()

    # Create simple MCTS player
    oracle = MCTS.RolloutOracle(gspec)
    mcts_env = MCTS.Env(gspec, oracle, cpuct=1.0, noise_ϵ=0.0)
    player = MctsPlayer(mcts_env, niters=10, τ=ConstSchedule(1.0))

    # Play a game
    trace = play_game(gspec, TwoPlayers(player, player))

    # Verify trace has both chance and decision nodes
    @test length(trace) > 0
    num_chance = sum(trace.is_chance)
    num_decision = length(trace.is_chance) - num_chance
    @test num_chance > 0  # Should have some chance nodes
    @test num_decision > 0  # Should have some decision nodes

    # Verify game terminated
    final_game = GI.init(gspec, trace.states[end])
    @test GI.game_terminated(final_game)
  end

  @testset "Default Implementations for Non-Stochastic Games" begin
    # Test that non-stochastic games work with default implementations
    # Using tictactoe as example
    ttt_gspec = Examples.games["tictactoe"]
    ttt_game = GI.init(ttt_gspec)

    @test GI.is_chance_node(ttt_game) == false
    @test GI.num_chance_outcomes(ttt_gspec) == 0
  end

end
