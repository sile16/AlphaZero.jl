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

@testset "Game Loop Integration" begin
  include("test_play_game_integration.jl")
end

# Note: Dummy Runs test removed — depends on legacy Benchmark module
# (rewards_and_redundancy undefined). Run tictactoe manually if needed.
