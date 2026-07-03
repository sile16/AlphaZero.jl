# Regression tests for terminal bearoff reward semantics and MCTS reward scaling.
#
# Bug (fixed 2026-07-03): selfplay_client.jl hard-coded terminal bearoff moves as
# simple wins (value 1.0, equity [1,0,0,0,0]), discarding gammon/backgammon value.
# BackgammonNet's `reward` at termination carries the win multiplier (±1/±2/±3),
# so the terminal branch must derive value and 5-head target from `reward`.
#
# Also: MCTS mixes NN values (equity/3 ∈ [-1,1]) and game rewards in the same
# Q totals, so `GI.reward_scale` must be 3.0 for backgammon and 1.0 by default.

using BackgammonNet
using StaticArrays
using Random

@testset "Terminal bearoff reward carries gammon multiplier" begin
  # White (P0): 14 borne off, 1 checker on point 24 — any die bears off.
  p0 = (UInt128(14) << (25 * 4)) | (UInt128(1) << (24 * 4))
  # Black (P1): 15 checkers in home (points 1-3), ZERO off — white win is a gammon.
  p1_gammon = (UInt128(5) << (1 * 4)) | (UInt128(5) << (2 * 4)) | (UInt128(5) << (3 * 4))
  # Control: black has 1 checker off — simple win.
  p1_simple = (UInt128(5) << (1 * 4)) | (UInt128(5) << (2 * 4)) | (UInt128(4) << (3 * 4)) |
              (UInt128(1) << 0)  # IDX_P1_OFF = 0

  for (p1, expected_reward) in [(p1_gammon, 2.0f0), (p1_simple, 1.0f0)]
    g = BackgammonGame(p0, p1, SVector{2, Int8}(0, 0), Int8(0), Int8(0), false, 0.0f0;
                       obs_type=:minimal_flat)
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

@testset "Terminal 5-head target consistent with reward scalar" begin
  # Mirrors the fixed terminal branch in selfplay_client.jl:
  # reward → joint cumulative vector, checked against the joint equity formula.
  for (white_r, expected_eq) in [
      (1.0f0, Float32[1, 0, 0, 0, 0]),
      (2.0f0, Float32[1, 1, 0, 0, 0]),
      (3.0f0, Float32[1, 1, 1, 0, 0])]
    mover_val = white_r  # mover == white
    is_g = mover_val >= 2.0f0
    is_bg = mover_val >= 3.0f0
    eq = Float32[1.0, is_g ? 1.0 : 0.0, is_bg ? 1.0 : 0.0, 0.0, 0.0]
    @test eq == expected_eq
    # Joint equity formula must reproduce the scalar
    eq_scalar = (2eq[1] - 1) + (eq[2] - eq[4]) + (eq[3] - eq[5])
    @test eq_scalar == mover_val
  end
end

@testset "GI.reward_scale" begin
  # Default is 1.0 (games with rewards already in [-1,1])
  struct _DummySpec <: AlphaZero.GI.AbstractGameSpec end
  @test AlphaZero.GI.reward_scale(_DummySpec()) == 1.0
  # Backgammon overrides to 3.0 (win/gammon/backgammon = ±1/±2/±3)
  games_dir = joinpath(@__DIR__, "..", "games")
  if !isdefined(Main, :BackgammonDeterministic)
    include(joinpath(games_dir, "backgammon-deterministic", "main.jl"))
  end
  @test AlphaZero.GI.reward_scale(Main.BackgammonDeterministic.GameSpec()) == 3.0
end
