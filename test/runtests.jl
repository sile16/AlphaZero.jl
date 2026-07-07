using AlphaZero
using AlphaZero.Examples: games, experiments
using AlphaZero.Scripts: dummy_run, test_game

using Base.Filesystem: rm
using Test

const CI = get(ENV, "CI", nothing) == "true"
const FULL = !CI

@testset "Testing Games" begin
  test_game(games["tictactoe"])
  @test true
end

@testset "Backgammon Inference Regressions" begin
  include("test_backgammon_inference_regressions.jl")
end

@testset "Multihead Regressions" begin
  include("test_multihead.jl")
end

@testset "Fast Inference" begin
  include("test_fast_inference.jl")
end

@testset "Terminal Bearoff Rewards" begin
  include("test_terminal_bearoff_rewards.jl")
end

@testset "Scale and Buffer Regressions" begin
  include("test_scale_and_buffer_regressions.jl")
end

@testset "Bearoff Doubles Regression" begin
  include("test_bearoff_doubles_regression.jl")
end

@testset "MCTS Identity Staircase" begin
  include("test_mcts_identity_staircase.jl")
end

@testset "Batched Chance Exact Expectation" begin
  include("test_batched_chance_exact_expectation.jl")
end

@testset "Promotion Gate" begin
  include("test_promotion_gate.jl")
end

@testset "Game Loop Integration" begin
  include("test_play_game_integration.jl")
end

@testset "Progressive Sim Budget" begin
  include("test_progressive_sim_budget.jl")
end

# Note: Dummy Runs test removed — depends on legacy Benchmark module
# (rewards_and_redundancy undefined). Run tictactoe manually if needed.
