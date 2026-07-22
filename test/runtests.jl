using AlphaZero

using Base.Filesystem: rm
using Test

const CI = get(ENV, "CI", nothing) == "true"
const FULL = !CI

@testset "Backgammon Inference Regressions" begin
  include("test_backgammon_inference_regressions.jl")
end

@testset "Backgammon Training Artifact Integration" begin
  include("test_backgammon_training_artifact.jl")
end

@testset "Backgammon Runtime Contracts" begin
  include("test_backgammon_runtime_contracts.jl")
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

@testset "Numerical Training Safety" begin
  include("test_numerical_safety.jl")
end

@testset "Checkpoint Manager" begin
  include("test_checkpoint_manager.jl")
end

@testset "Preflight" begin
  include("test_preflight.jl")
end

@testset "Evaluation Manifest" begin
  include("test_eval_manifest.jl")
end

@testset "Data Quality" begin
  include("test_data_quality.jl")
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

@testset "Distributed Protocol" begin
  include("test_distributed_protocol.jl")
end

include("test_protocol_roundtrip.jl")
include("test_eval_manager.jl")
include("test_eval_submit_flow.jl")
include("test_distributed_server_contract.jl")
include("test_distributed_fault_injection.jl")
include("test_tensorboard_dashboard.jl")
include("test_game_loop.jl")

# Note: Dummy Runs test removed — depends on legacy Benchmark module
# (rewards_and_redundancy undefined). Run tictactoe manually if needed.
