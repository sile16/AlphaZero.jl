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

@testset "Cluster Module" begin
  include("test_cluster.jl")
end

@testset "Reanalyze Module" begin
  include("test_reanalyze.jl")
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

@testset "Distributed Protocol" begin
  include("test_distributed_protocol.jl")
end

@testset "Dummy Runs" begin
  @test dummy_run(experiments["tictactoe"], nostdout=false) == nothing
end
