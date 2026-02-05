#!/usr/bin/env julia
# Parallel evaluation against GnuBG using multiple processes
#
# Usage:
#   julia --project scripts/eval_vs_gnubg_parallel.jl <checkpoint_path> [obs_type] [num_games] [width] [blocks] [num_workers] [mcts_iters]

using Distributed
using Printf
using Statistics

# Parse args before adding workers
args = filter(a -> !isempty(strip(a)), ARGS)

if length(args) < 1
    println("Usage: julia --project scripts/eval_vs_gnubg_parallel.jl <checkpoint> [obs_type] [num_games] [width] [blocks] [num_workers] [mcts_iters]")
    exit(1)
end

checkpoint_path_main = abspath(args[1])
obs_type_main = length(args) >= 2 ? args[2] : "minimal"
num_games_main = length(args) >= 3 ? parse(Int, strip(args[3])) : 500
net_width_main = length(args) >= 4 ? parse(Int, strip(args[4])) : 256
net_blocks_main = length(args) >= 5 ? parse(Int, strip(args[5])) : 6
num_workers_main = length(args) >= 6 ? parse(Int, strip(args[6])) : 4
mcts_iters_main = length(args) >= 7 ? parse(Int, strip(args[7])) : 100
script_dir_main = @__DIR__

println("=" ^ 60)
println("Parallel GnuBG Evaluation (Multi-Process)")
println("=" ^ 60)
println("Checkpoint: $checkpoint_path_main")
println("Observation type: $obs_type_main")
println("Games per matchup: $num_games_main")
println("Network: width=$net_width_main, blocks=$net_blocks_main")
println("Workers: $num_workers_main processes")
println("MCTS iterations: $mcts_iters_main")
println("=" ^ 60)
flush(stdout)

# Add worker processes with project
println("\nSpawning $num_workers_main worker processes...")
flush(stdout)
addprocs(num_workers_main; exeflags=`--project=$(Base.active_project())`)

# Send config to all workers (use global without const)
@everywhere global CHECKPOINT_PATH = $checkpoint_path_main
@everywhere global OBS_TYPE = $obs_type_main
@everywhere global NET_WIDTH = $net_width_main
@everywhere global NET_BLOCKS = $net_blocks_main
@everywhere global MCTS_ITERS = $mcts_iters_main
@everywhere global SCRIPT_DIR = $script_dir_main

# Set env on all workers
@everywhere ENV["BACKGAMMON_OBS_TYPE"] = OBS_TYPE

# Load packages on all workers
@everywhere using Random
@everywhere using AlphaZero
@everywhere using AlphaZero: GI, Network, FluxLib, MctsParams, MctsPlayer, ConstSchedule
@everywhere using BackgammonNet
@everywhere using Flux

# Include game module on all workers
@everywhere include(joinpath(SCRIPT_DIR, "..", "games", "backgammon-deterministic", "game.jl"))

# Include GnubgPlayer on all workers
@everywhere include(joinpath(SCRIPT_DIR, "GnubgPlayer.jl"))

# Initialize each worker
@everywhere begin
    global NetLib = FluxLib
    global gspec = GameSpec()
    
    # Initialize GnuBG
    println("Worker $(myid()): Initializing GnuBG...")
    let
        dummy_game = GI.init(gspec)
        gnubg_test = GnubgPlayer.GnubgBaseline(ply=0)
        AlphaZero.think(gnubg_test, dummy_game)
    end
    println("Worker $(myid()): GnuBG initialized")
    
    # Load network
    println("Worker $(myid()): Loading network...")
    global network = NetLib.FCResNetMultiHead(gspec, NetLib.FCResNetMultiHeadHP(width=NET_WIDTH, num_blocks=NET_BLOCKS))
    NetLib.load_weights(CHECKPOINT_PATH, network)
    println("Worker $(myid()): Network loaded ($(sum(length, Flux.params(network))) params)")
    
    # Create MCTS player
    global mcts_params = MctsParams(
        num_iters_per_turn=MCTS_ITERS,
        cpuct=2.0,
        temperature=ConstSchedule(0.0),
        dirichlet_noise_ϵ=0.0,
        dirichlet_noise_α=1.0
    )
    global az_player = MctsPlayer(gspec, network, mcts_params)
    global gnubg_0ply = GnubgPlayer.GnubgBaseline(ply=0)
    global gnubg_1ply = GnubgPlayer.GnubgBaseline(ply=1)
end

# Define game function on all workers
@everywhere function play_game_worker(gnubg_ply::Int, az_is_white::Bool, seed::Int)
    rng = MersenneTwister(seed)
    env = GI.init(gspec)
    env.rng = rng
    
    gnubg_player = gnubg_ply == 0 ? gnubg_0ply : gnubg_1ply
    
    while !GI.game_terminated(env)
        is_white = GI.white_playing(env)
        use_az = (is_white && az_is_white) || (!is_white && !az_is_white)
        
        if use_az
            actions, π = AlphaZero.think(az_player, env)
            action = actions[argmax(π)]
        else
            actions, π = AlphaZero.think(gnubg_player, env)
            action = actions[argmax(π)]
        end
        GI.play!(env, action)
    end
    
    AlphaZero.reset_player!(az_player)
    
    reward = GI.white_reward(env)
    return az_is_white ? reward : -reward
end

println("\nAll workers initialized")
flush(stdout)

# Store for use in functions
const NUM_WORKERS = num_workers_main
const NUM_GAMES = num_games_main
const OBS_TYPE_LOCAL = obs_type_main
const MCTS_ITERS_LOCAL = mcts_iters_main

# Parallel game execution
function run_games_parallel(gnubg_ply::Int, az_is_white::Bool, n_games::Int, base_seed::Int)
    game_args = [(gnubg_ply, az_is_white, base_seed + i * 104729) for i in 1:n_games]
    
    results = pmap(game_args) do (ply, is_white, seed)
        play_game_worker(ply, is_white, seed)
    end
    
    return results
end

# Evaluate matchup
function evaluate_matchup(gnubg_ply::Int, az_is_white::Bool, n_games::Int; base_seed::Int=1000)
    matchup_name = az_is_white ? "AZ(white) vs GnuBG-$(gnubg_ply)ply" : "GnuBG-$(gnubg_ply)ply vs AZ(black)"
    println("\n$matchup_name ($n_games games, $NUM_WORKERS workers)...")
    flush(stdout)
    
    start_time = time()
    rewards = run_games_parallel(gnubg_ply, az_is_white, n_games, base_seed)
    elapsed = time() - start_time
    
    avg_reward = mean(rewards)
    std_reward = std(rewards)
    se = std_reward / sqrt(n_games)
    wins = count(r -> r > 0, rewards)
    win_rate = wins / n_games
    
    println("  Result: avg=$(round(avg_reward, digits=3)) ± $(round(1.96*se, digits=3)) (95% CI)")
    println("  Win rate: $(round(100*win_rate, digits=1))%")
    println("  Time: $(round(elapsed, digits=1))s ($(round(n_games/elapsed, digits=1)) games/sec)")
    flush(stdout)
    
    return (avg=avg_reward, win_rate=win_rate, se=se, time=elapsed)
end

# Main
println("\n" * "=" ^ 60)
println("Starting evaluation...")
println("=" ^ 60)
flush(stdout)

results = Dict{String, Any}()
total_start = time()

r = evaluate_matchup(0, true, NUM_GAMES, base_seed=1000)
results["az_vs_gnubg0_white"] = r

r = evaluate_matchup(0, false, NUM_GAMES, base_seed=2000)
results["az_vs_gnubg0_black"] = r

r = evaluate_matchup(1, true, NUM_GAMES, base_seed=3000)
results["az_vs_gnubg1_white"] = r

r = evaluate_matchup(1, false, NUM_GAMES, base_seed=4000)
results["az_vs_gnubg1_black"] = r

total_time = time() - total_start

# Summary
println("\n" * "=" ^ 60)
println("SUMMARY")
println("=" ^ 60)
println("Observation type: $OBS_TYPE_LOCAL")
println("Games per matchup: $NUM_GAMES ($(4*NUM_GAMES) total)")
println("Workers: $NUM_WORKERS, MCTS iters: $MCTS_ITERS_LOCAL")
println("Total time: $(round(total_time, digits=1))s ($(round(total_time/60, digits=1)) min)")
println()

gnubg0_combined = (results["az_vs_gnubg0_white"].avg + results["az_vs_gnubg0_black"].avg) / 2
gnubg1_combined = (results["az_vs_gnubg1_white"].avg + results["az_vs_gnubg1_black"].avg) / 2
gnubg0_wr = (results["az_vs_gnubg0_white"].win_rate + results["az_vs_gnubg0_black"].win_rate) / 2
gnubg1_wr = (results["az_vs_gnubg1_white"].win_rate + results["az_vs_gnubg1_black"].win_rate) / 2

@printf("vs GnuBG 0-ply: Combined reward = %.3f, Win rate = %.1f%%\n", gnubg0_combined, 100*gnubg0_wr)
@printf("  As white: %.3f (%.1f%% wins)\n", results["az_vs_gnubg0_white"].avg, 100*results["az_vs_gnubg0_white"].win_rate)
@printf("  As black: %.3f (%.1f%% wins)\n", results["az_vs_gnubg0_black"].avg, 100*results["az_vs_gnubg0_black"].win_rate)
println()
@printf("vs GnuBG 1-ply: Combined reward = %.3f, Win rate = %.1f%%\n", gnubg1_combined, 100*gnubg1_wr)
@printf("  As white: %.3f (%.1f%% wins)\n", results["az_vs_gnubg1_white"].avg, 100*results["az_vs_gnubg1_white"].win_rate)
@printf("  As black: %.3f (%.1f%% wins)\n", results["az_vs_gnubg1_black"].avg, 100*results["az_vs_gnubg1_black"].win_rate)

println("\n" * "=" ^ 60)

rmprocs(workers())
