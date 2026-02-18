#!/usr/bin/env julia
using Pkg
Pkg.activate(dirname(@__DIR__))

using AlphaZero
using AlphaZero.FluxLib

# Load backgammon
const GAMES_DIR = joinpath(@__DIR__, "..", "games")
include(joinpath(GAMES_DIR, "backgammon-deterministic", "main.jl"))
using .BackgammonDeterministic

gspec = BackgammonDeterministic.GameSpec()

# Load our distributed network
println("Loading distributed training network...")
nn_path = joinpath("/homeshare/projects/AlphaZero.jl", "sessions/single_server_20260125_222323/checkpoints/latest.data")
nn_bytes = read(nn_path)
weights = FluxLib.deserialize_weights(nn_bytes)

# Create network and load weights
hp = FluxLib.FCResNetMultiHeadHP(width=128, num_blocks=3)
nn = FluxLib.FCResNetMultiHead(gspec, hp)
FluxLib.load_weights!(nn, weights)
nn = Network.to_gpu(nn)
Network.set_test_mode!(nn, true)
println("Network loaded: $(Network.num_parameters(nn)) parameters")

# Create players
mcts = MctsParams(
    num_iters_per_turn=100,
    cpuct=1.5,
    temperature=ConstSchedule(0.2),
    dirichlet_noise_ϵ=0.0,
    dirichlet_noise_α=0.3
)
az_player = MctsPlayer(gspec, nn, mcts)
random_player = AlphaZero.RandomPlayer()

# Play games
println("\nPlaying 50 games (AZ as white)...")
rewards_white = Float64[]
for i in 1:50
    trace = play_game(gspec, TwoPlayers(az_player, random_player))
    push!(rewards_white, total_reward(trace))
    i % 10 == 0 && print(".")
end
println()

println("Playing 50 games (AZ as black)...")
rewards_black = Float64[]
for i in 1:50
    trace = play_game(gspec, TwoPlayers(random_player, az_player))
    push!(rewards_black, -total_reward(trace))
    i % 10 == 0 && print(".")
end
println()

avg_white = sum(rewards_white) / length(rewards_white)
avg_black = sum(rewards_black) / length(rewards_black)
combined = (avg_white + avg_black) / 2

println("\n" * "="^50)
println("EVALUATION RESULTS (Distributed Training)")
println("="^50)
println("AZ as White: $(round(avg_white, digits=3))")
println("AZ as Black: $(round(avg_black, digits=3))")
println("Combined:    $(round(combined, digits=3))")
println("\nBaseline reference: +1.23 (at 69 iterations)")
