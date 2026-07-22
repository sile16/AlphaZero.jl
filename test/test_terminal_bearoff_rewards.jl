# Regression tests for terminal bearoff reward semantics and MCTS reward scaling.
#
# Bug (fixed 2026-07-03): selfplay_client.jl hard-coded terminal bearoff moves as
# simple wins (value 1.0, equity [1,0,0,0,0]), discarding gammon/backgammon value.
# BackgammonNet's `reward` at termination carries the win multiplier (±1/±2/±3),
# so the terminal branch must derive value and 5-head target from `reward`.
#
# Also: MCTS mixes NN values (equity/3 ∈ [-1,1]) and game rewards in the same
# Q totals, so `GI.reward_scale` must be 3.0 for backgammon and 1.0 by default.

using Test
using AlphaZero
using BackgammonNet
using StaticArrays
using Random

const _BG_GAMES_DIR_TR = joinpath(@__DIR__, "..", "games")
if !isdefined(Main, :BackgammonDeterministic)
  include(joinpath(_BG_GAMES_DIR_TR, "backgammon-deterministic", "main.jl"))
end
const BGD_TR = Main.BackgammonDeterministic

@testset "Terminal bearoff reward carries gammon multiplier" begin
  # White (P0): 14 borne off, 1 checker on point 24 — any die bears off.
  p0 = (UInt128(14) << (25 * 4)) | (UInt128(1) << (24 * 4))
  # Black (P1): 15 checkers in home (points 1-3), ZERO off — white win is a gammon.
  p1_gammon = (UInt128(5) << (1 * 4)) | (UInt128(5) << (2 * 4)) | (UInt128(5) << (3 * 4))
  # Control: black has 1 checker off — simple win.
  p1_simple = (UInt128(5) << (1 * 4)) | (UInt128(5) << (2 * 4)) | (UInt128(4) << (3 * 4)) |
              (UInt128(1) << 0)  # IDX_P1_OFF = 0

  for (p1, expected_reward) in [(p1_gammon, 2.0f0), (p1_simple, 1.0f0)]
    g = BGD_TR.backgammon_game(p0, p1, SVector{2, Int8}(0, 0), Int8(0), Int8(0), false, 0.0f0;
                                observation_type=:minimal_flat)
    @test is_chance_node(g)
    sample_chance!(g, MersenneTwister(7))
    found_terminal = false
    for a in legal_actions(g)
      g2 = BackgammonNet.clone(g)
      apply_action!(g2, a)
      if g2.terminated
        found_terminal = true
        @test g2.reward == expected_reward
      end
    end
    @test found_terminal
  end
end

@testset "Terminal 5-head target derives from board outcome" begin
  p0_off = UInt128(15) << (25 * 4)
  p1_single = ((UInt128(14) << (1 * 4)) | UInt128(1))
  p1_gammon = (UInt128(5) << (1 * 4)) | (UInt128(5) << (2 * 4)) | (UInt128(5) << (3 * 4))
  p1_bg = (UInt128(14) << (7 * 4)) | (UInt128(1) << (27 * 4))

  for (p1, reward, expected_eq, expected_points) in [
      (p1_single, 1.0f0, Float32[1, 0, 0, 0, 0], 1.0f0),
      (p1_gammon, 2.0f0, Float32[1, 1, 0, 0, 0], 2.0f0),
      (p1_bg, 3.0f0, Float32[1, 1, 1, 0, 0], 3.0f0)]
    g = BGD_TR.backgammon_game(p0_off, p1, SVector{2, Int8}(0, 0), Int8(0), Int8(0), true, reward;
                                observation_type=:minimal_flat)
    heads = BackgammonNet.terminal_heads_target(g, 0)
    eq = Float32[heads.p_win, heads.p_gammon_win, heads.p_bg_win, heads.p_gammon_loss, heads.p_bg_loss]
    @test eq == expected_eq
    @test BackgammonNet.compute_equity_joint(heads) == expected_points
  end

  cubed_single = BGD_TR.backgammon_game(p0_off, p1_single, SVector{2, Int8}(0, 0), Int8(0), Int8(0), true, 4.0f0;
                                         observation_type=:minimal_flat)
  cubed_single.cube_enabled = true
  cubed_single.cube_value = Int16(4)
  cubed_heads = BackgammonNet.terminal_heads_target(cubed_single, 0)
  @test Float32[cubed_heads.p_win, cubed_heads.p_gammon_win, cubed_heads.p_bg_win,
                cubed_heads.p_gammon_loss, cubed_heads.p_bg_loss] == Float32[1, 0, 0, 0, 0]
end

@testset "GI.reward_scale" begin
  # Default is 1.0 (games with rewards already in [-1,1])
  struct _DummySpec <: AlphaZero.GI.AbstractGameSpec end
  @test AlphaZero.GI.reward_scale(_DummySpec()) == 1.0
  # Backgammon overrides to 3.0 (win/gammon/backgammon = ±1/±2/±3)
  @test AlphaZero.GI.reward_scale(Main.BackgammonDeterministic.GameSpec()) == 3.0
end

@testset "Backgammon GameOutcome ignores cube multiplier" begin
  p0_off = UInt128(15) << (25 * 4)
  p1_single = ((UInt128(14) << (1 * 4)) | UInt128(1))
  env = Main.BackgammonDeterministic.GameEnv(
    BGD_TR.backgammon_game(p0_off, p1_single, SVector{2, Int8}(0, 0), Int8(0), Int8(0), true, 4.0f0;
                            observation_type=:minimal_flat),
    MersenneTwister(1),
  )
  env.game.cube_enabled = true
  env.game.cube_value = Int16(4)

  outcome = AlphaZero.GI.game_outcome(env)
  @test outcome !== nothing
  @test outcome.white_won
  @test !outcome.is_gammon
  @test !outcome.is_backgammon

  drop_env = AlphaZero.GI.init(Main.BackgammonDeterministic.GameSpec())
  drop_env.game.cube_enabled = true
  drop_env.game.cube_owner = Int8(-1)
  drop_env.game.phase = BackgammonNet.PHASE_CUBE_DECISION
  apply_action!(drop_env.game, BackgammonNet.ACTION_CUBE_DOUBLE)
  apply_action!(drop_env.game, BackgammonNet.ACTION_CUBE_PASS)
  drop_outcome = AlphaZero.GI.game_outcome(drop_env)
  @test drop_outcome !== nothing
  @test drop_outcome.white_won == (drop_env.game.reward > 0)
  @test !drop_outcome.is_gammon
  @test !drop_outcome.is_backgammon
end
