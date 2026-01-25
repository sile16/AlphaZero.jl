#####
##### Unit tests for multi-head equity network
#####

using Test
using AlphaZero
using AlphaZero.GI
import AlphaZero.Network
import AlphaZero.FluxLib

# For direct module access
using AlphaZero: EquityTargets, equity_targets_from_outcome, TrainingSample
using AlphaZero: Trace, set_outcome!, push_trace!, MemoryBuffer

# Define a simple test game inline for testing
module TestGame
  import AlphaZero.GI
  using StaticArrays

  struct GameSpec <: GI.AbstractGameSpec end

  const NUM_ACTIONS = 9
  const State = SVector{9, Int8}  # Simple 3x3 board
  const INITIAL_STATE = State(zeros(Int8, 9))

  mutable struct GameEnv <: GI.AbstractGameEnv
    state :: State
    current_player :: Int8
    terminated :: Bool
    reward :: Float32
  end

  GI.init(::GameSpec, state=INITIAL_STATE) = GameEnv(state, Int8(0), false, 0f0)
  GI.spec(::GameEnv) = GameSpec()
  GI.two_players(::GameSpec) = true
  GI.actions(::GameSpec) = collect(1:NUM_ACTIONS)
  GI.supports_equity_targets(::GameSpec) = true

  function GI.set_state!(g::GameEnv, state)
    g.state = state
    g.current_player = Int8(0)
  end

  GI.current_state(g::GameEnv) = g.state
  GI.game_terminated(g::GameEnv) = g.terminated
  GI.white_playing(g::GameEnv) = g.current_player == 0
  GI.white_playing(::GameSpec, state) = true  # Always white's perspective for simplicity
  GI.white_reward(g::GameEnv) = g.reward

  function GI.actions_mask(g::GameEnv)
    mask = trues(NUM_ACTIONS)
    for i in 1:NUM_ACTIONS
      if g.state[i] != 0
        mask[i] = false
      end
    end
    return mask
  end

  function GI.play!(g::GameEnv, action)
    new_state = MVector{9, Int8}(g.state)
    new_state[action] = g.current_player == 0 ? Int8(1) : Int8(-1)
    g.state = State(new_state)
    g.current_player = Int8(1 - g.current_player)

    # Simple termination: if all filled or dummy win condition
    if all(g.state .!= 0)
      g.terminated = true
      g.reward = 1.0f0  # White wins
    end
  end

  function GI.vectorize_state(::GameSpec, state)
    return Float32.(state)
  end

  GI.render(g::GameEnv) = println(g.state)
  GI.action_string(::GameSpec, a) = string(a)
  GI.parse_action(::GameSpec, s) = tryparse(Int, s)
end

@testset "Multi-head Equity Tests" begin

  @testset "EquityOutput and compute_equity" begin
    # Test equity computation
    eq = FluxLib.EquityOutput(1.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)  # Simple win
    @test FluxLib.compute_equity(eq) ≈ 1.0f0

    eq = FluxLib.EquityOutput(0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)  # Simple loss
    @test FluxLib.compute_equity(eq) ≈ -1.0f0

    eq = FluxLib.EquityOutput(1.0f0, 1.0f0, 0.0f0, 0.0f0, 0.0f0)  # Gammon win
    @test FluxLib.compute_equity(eq) ≈ 2.0f0

    eq = FluxLib.EquityOutput(0.0f0, 0.0f0, 0.0f0, 1.0f0, 0.0f0)  # Gammon loss
    @test FluxLib.compute_equity(eq) ≈ -2.0f0

    eq = FluxLib.EquityOutput(1.0f0, 1.0f0, 1.0f0, 0.0f0, 0.0f0)  # Backgammon win
    @test FluxLib.compute_equity(eq) ≈ 3.0f0

    eq = FluxLib.EquityOutput(0.0f0, 0.0f0, 0.0f0, 1.0f0, 1.0f0)  # Backgammon loss
    @test FluxLib.compute_equity(eq) ≈ -3.0f0

    # Test 50/50 game
    eq = FluxLib.EquityOutput(0.5f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
    @test FluxLib.compute_equity(eq) ≈ 0.0f0

    # Test mixed probabilities
    eq = FluxLib.EquityOutput(0.6f0, 0.2f0, 0.0f0, 0.3f0, 0.0f0)
    expected = 0.6f0 * (1f0 + 0.2f0) - 0.4f0 * (1f0 + 0.3f0)
    @test FluxLib.compute_equity(eq) ≈ expected
  end

  @testset "EquityTargets from GameOutcome" begin
    # White wins single game
    outcome = GI.GameOutcome(true, false, false)
    targets = equity_targets_from_outcome(outcome, true)  # White's perspective
    @test targets.p_win ≈ 1.0
    @test targets.p_gammon_win ≈ 0.0
    @test targets.p_bg_win ≈ 0.0
    @test targets.p_gammon_loss ≈ 0.0
    @test targets.p_bg_loss ≈ 0.0

    targets = equity_targets_from_outcome(outcome, false)  # Black's perspective
    @test targets.p_win ≈ 0.0
    @test targets.p_gammon_loss ≈ 0.0  # White didn't win by gammon

    # White wins gammon
    outcome = GI.GameOutcome(true, true, false)
    targets = equity_targets_from_outcome(outcome, true)
    @test targets.p_win ≈ 1.0
    @test targets.p_gammon_win ≈ 1.0
    @test targets.p_bg_win ≈ 0.0

    targets = equity_targets_from_outcome(outcome, false)  # Black's perspective
    @test targets.p_win ≈ 0.0
    @test targets.p_gammon_loss ≈ 1.0  # Black lost by gammon

    # Black wins backgammon
    outcome = GI.GameOutcome(false, true, true)
    targets = equity_targets_from_outcome(outcome, false)  # Black's perspective
    @test targets.p_win ≈ 1.0
    @test targets.p_gammon_win ≈ 1.0
    @test targets.p_bg_win ≈ 1.0

    targets = equity_targets_from_outcome(outcome, true)  # White's perspective
    @test targets.p_win ≈ 0.0
    @test targets.p_gammon_loss ≈ 1.0
    @test targets.p_bg_loss ≈ 1.0
  end

  @testset "TrainingSample with equity" begin
    # Create sample without equity
    s = "dummy_state"
    π = [0.5, 0.5]
    sample = TrainingSample(s, π, 1.0, 5.0, 1)
    @test isnothing(sample.equity)
    @test sample.is_chance == false

    # Create sample with equity
    equity = EquityTargets(1.0, 0.5, 0.0, 0.0, 0.0)
    sample = TrainingSample(s, π, 1.0, 5.0, 1, false, equity)
    @test !isnothing(sample.equity)
    @test sample.equity.p_win ≈ 1.0
    @test sample.equity.p_gammon_win ≈ 0.5
  end

  @testset "Trace with outcome" begin
    state = "initial"
    trace = Trace(state)
    @test isnothing(trace.outcome)

    outcome = GI.GameOutcome(true, true, false)
    set_outcome!(trace, outcome)
    @test !isnothing(trace.outcome)
    @test trace.outcome.white_won == true
    @test trace.outcome.is_gammon == true
    @test trace.outcome.is_backgammon == false
  end

  @testset "FCResNetMultiHead architecture" begin
    gspec = TestGame.GameSpec()

    # Create multi-head network
    hyper = FluxLib.FCResNetMultiHeadHP(
      width = 32,
      num_blocks = 2,
      depth_phead = 1,
      depth_vhead = 1
    )

    nn = FluxLib.FCResNetMultiHead(gspec, hyper)

    # Test forward pass
    game = GI.init(gspec)
    state = GI.current_state(game)
    x = GI.vectorize_state(gspec, state)
    x_batch = reshape(Float32.(x), size(x)..., 1)  # Add batch dimension

    # Standard forward (for MCTS compatibility)
    p, v = Network.forward(nn, x_batch)
    @test size(p) == (9, 1)  # 9 actions for test game
    @test size(v) == (1, 1)
    @test all(-1 .<= v .<= 1)  # Value in [-1, 1] range

    # Multi-head forward
    p, v_win, v_gw, v_bgw, v_gl, v_bgl = FluxLib.forward_multihead(nn, x_batch)
    @test size(p) == (9, 1)
    @test size(v_win) == (1, 1)
    @test size(v_gw) == (1, 1)
    @test all(0 .<= v_win .<= 1)  # Probabilities in [0, 1]
    @test all(0 .<= v_gw .<= 1)

    # Test evaluation interface
    p_eval, v_eval = Network.evaluate(nn, state)
    @test length(p_eval) == sum(GI.actions_mask(game))  # Only valid actions
    @test -1 <= v_eval <= 1

    println("Network parameters: ", Network.num_parameters(nn))
  end

  @testset "Learning with multi-head network" begin
    gspec = TestGame.GameSpec()

    # Create a simple multi-head network
    hyper = FluxLib.FCResNetMultiHeadHP(
      width = 16,
      num_blocks = 1,
      depth_phead = 1,
      depth_vhead = 1
    )
    nn = FluxLib.FCResNetMultiHead(gspec, hyper)

    # Create dummy samples with equity targets
    game = GI.init(gspec)
    state = GI.current_state(game)

    samples = TrainingSample[]
    for _ in 1:10
      π = rand(sum(GI.actions_mask(game)))
      π ./= sum(π)
      z = rand([-1.0, 1.0])

      # Create equity targets (simple win/loss)
      eq = if z > 0
        EquityTargets(1.0, 0.0, 0.0, 0.0, 0.0)  # Win
      else
        EquityTargets(0.0, 0.0, 0.0, 0.0, 0.0)  # Loss
      end

      push!(samples, TrainingSample(state, π, z, 5.0, 1, false, eq))
    end

    # Test convert_samples
    data = AlphaZero.convert_samples(gspec, AlphaZero.CONSTANT_WEIGHT, samples)
    @test haskey(data, :W)
    @test haskey(data, :EqWin)
    @test haskey(data, :HasEquity)
    @test all(data.HasEquity .== 1.0f0)  # All samples have equity

    # Test that learning params work (with all required fields)
    lparams = LearningParams(
      use_gpu = false,
      use_position_averaging = false,
      samples_weighing_policy = AlphaZero.CONSTANT_WEIGHT,
      optimiser = Adam(lr=1f-3),
      l2_regularization = 1f-4,
      batch_size = 5,
      loss_computation_batch_size = 5,
      min_checkpoints_per_epoch = 1,
      max_batches_per_checkpoint = 10,
      num_checkpoints = 1
    )

    trainer = AlphaZero.Trainer(gspec, nn, samples, lparams)
    @test trainer.Wmean > 0

    # Run a few training steps
    losses = AlphaZero.batch_updates!(trainer, 2)
    @test length(losses) == 2
    @test all(isfinite.(losses))

    println("Training losses: ", losses)
  end

  @testset "Backward compatibility (single-head networks)" begin
    gspec = TestGame.GameSpec()

    # Create a standard single-head network
    hyper = FluxLib.SimpleNetHP(width = 16, depth_common = 2)
    nn = FluxLib.SimpleNet(gspec, hyper)

    # Create samples WITHOUT equity targets
    game = GI.init(gspec)
    state = GI.current_state(game)

    samples = TrainingSample[]
    for _ in 1:10
      π = rand(sum(GI.actions_mask(game)))
      π ./= sum(π)
      z = rand([-1.0, 1.0])
      # No equity targets (backward compatible)
      push!(samples, TrainingSample(state, π, z, 5.0, 1, false, nothing))
    end

    # Test convert_samples still works
    data = AlphaZero.convert_samples(gspec, AlphaZero.CONSTANT_WEIGHT, samples)
    @test all(data.HasEquity .== 0.0f0)  # No equity targets

    # Training should still work
    lparams = LearningParams(
      use_gpu = false,
      use_position_averaging = false,
      samples_weighing_policy = AlphaZero.CONSTANT_WEIGHT,
      optimiser = Adam(lr=1f-3),
      l2_regularization = 1f-4,
      batch_size = 5,
      loss_computation_batch_size = 5,
      min_checkpoints_per_epoch = 1,
      max_batches_per_checkpoint = 10,
      num_checkpoints = 1
    )

    trainer = AlphaZero.Trainer(gspec, nn, samples, lparams)
    losses = AlphaZero.batch_updates!(trainer, 2)
    @test length(losses) == 2
    @test all(isfinite.(losses))

    println("Single-head backward compat losses: ", losses)
  end

end

println("All multi-head tests passed!")
