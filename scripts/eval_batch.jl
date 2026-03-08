#!/usr/bin/env julia
# Fast batch evaluation against GnuBG
# Evaluates multiple checkpoints against GnuBG 0-ply with configurable settings.
#
# Usage:
#   julia --threads 16 --project scripts/eval_batch.jl <checkpoint1> [checkpoint2] ...
#
# Options (via environment variables):
#   EVAL_PLY=0          GnuBG ply depth (default: 0)
#   EVAL_GAMES=500      Games per matchup (default: 500)
#   EVAL_WORKERS=8      Number of parallel worker processes (default: 8)
#   EVAL_MCTS=100       MCTS iterations for AZ player (default: 100)
#   EVAL_WIDTH=256      Network width (default: 256)
#   EVAL_BLOCKS=5       Network blocks (default: 5)

using Distributed
using Printf
using Statistics

# Parse args
args = filter(a -> !isempty(strip(a)), ARGS)
if isempty(args)
    println("Usage: julia --threads 16 --project scripts/eval_batch.jl <checkpoint1> [checkpoint2] ...")
    println("\nEnvironment variables:")
    println("  EVAL_PLY=0|1|2    GnuBG ply (default: 0)")
    println("  EVAL_GAMES=500    Games per matchup (default: 500)")
    println("  EVAL_WORKERS=8    Worker processes (default: 8)")
    println("  EVAL_MCTS=100     MCTS iterations (default: 100)")
    println("  EVAL_WIDTH=256    Network width (default: 256)")
    println("  EVAL_BLOCKS=5     Network blocks (default: 5)")
    exit(1)
end

checkpoints = [abspath(a) for a in args]
gnubg_ply = parse(Int, get(ENV, "EVAL_PLY", "0"))
num_games = parse(Int, get(ENV, "EVAL_GAMES", "500"))
num_workers = parse(Int, get(ENV, "EVAL_WORKERS", "8"))
mcts_iters = parse(Int, get(ENV, "EVAL_MCTS", "100"))
net_width = parse(Int, get(ENV, "EVAL_WIDTH", "256"))
net_blocks = parse(Int, get(ENV, "EVAL_BLOCKS", "5"))
script_dir = @__DIR__

println("=" ^ 60)
println("Batch GnuBG Evaluation")
println("=" ^ 60)
println("Checkpoints: $(length(checkpoints))")
println("GnuBG ply: $gnubg_ply")
println("Games per matchup: $num_games (×2 sides = $(2*num_games) total)")
println("Network: $(net_width)w×$(net_blocks)b")
println("Workers: $num_workers, MCTS iters: $mcts_iters")
println("=" ^ 60)
flush(stdout)

# Spawn workers
println("\nSpawning $num_workers workers...")
flush(stdout)
addprocs(num_workers; exeflags=`--project=$(Base.active_project())`)

@everywhere global GNUBG_PLY = $gnubg_ply
@everywhere global NET_WIDTH = $net_width
@everywhere global NET_BLOCKS = $net_blocks
@everywhere global MCTS_ITERS = $mcts_iters
@everywhere global SCRIPT_DIR = $script_dir

@everywhere ENV["BACKGAMMON_OBS_TYPE"] = "minimal"

@everywhere using Random
@everywhere using AlphaZero
@everywhere using AlphaZero: GI, Network, FluxLib, MctsParams, MctsPlayer, ConstSchedule
@everywhere using BackgammonNet
@everywhere using Flux

@everywhere include(joinpath(SCRIPT_DIR, "..", "games", "backgammon-deterministic", "game.jl"))
@everywhere include(joinpath(SCRIPT_DIR, "GnubgPlayer.jl"))

# Initialize gnubg on all workers
@everywhere begin
    global NetLib = FluxLib
    global gspec = GameSpec()
    let
        dummy_game = GI.init(gspec)
        gnubg_test = GnubgPlayer.GnubgBaseline(ply=0)
        AlphaZero.think(gnubg_test, dummy_game)
    end
    global gnubg_player = GnubgPlayer.GnubgBaseline(ply=GNUBG_PLY)
end

# Function to load weights on all workers
function load_checkpoint_everywhere(checkpoint_path)
    @everywhere begin
        global network = NetLib.FCResNetMultiHead(gspec, NetLib.FCResNetMultiHeadHP(width=NET_WIDTH, num_blocks=NET_BLOCKS))
        NetLib.load_weights($checkpoint_path, network)
        global mcts_params = MctsParams(
            num_iters_per_turn=MCTS_ITERS,
            cpuct=2.0,
            temperature=ConstSchedule(0.0),
            dirichlet_noise_ϵ=0.0,
            dirichlet_noise_α=1.0,
            chance_mode=:passthrough
        )
        global az_player = MctsPlayer(gspec, network, mcts_params)
    end
end

@everywhere function _sample_chance(rng, outcomes)
    r = rand(rng)
    acc = 0.0
    @inbounds for i in eachindex(outcomes)
        acc += outcomes[i][2]
        if r <= acc
            return i
        end
    end
    return length(outcomes)
end

@everywhere function play_game_worker(az_is_white::Bool, seed::Int)
    rng = MersenneTwister(seed)
    env = GI.init(gspec)
    env.rng = rng

    while !GI.game_terminated(env)
        if GI.is_chance_node(env)
            outcomes = GI.chance_outcomes(env)
            idx = _sample_chance(rng, outcomes)
            GI.apply_chance!(env, outcomes[idx][1])
            continue
        end

        actions = GI.available_actions(env)
        if length(actions) == 1
            GI.play!(env, actions[1])
            continue
        end

        is_white = GI.white_playing(env)
        use_az = (is_white && az_is_white) || (!is_white && !az_is_white)

        if use_az
            actions_az, π = AlphaZero.think(az_player, env)
            action = actions_az[argmax(π)]
        else
            actions_gnubg, π = AlphaZero.think(gnubg_player, env)
            action = actions_gnubg[argmax(π)]
        end
        GI.play!(env, action)
    end

    AlphaZero.reset_player!(az_player)
    reward = GI.white_reward(env)
    return az_is_white ? reward : -reward
end

function run_matchup(az_is_white::Bool, n_games::Int, base_seed::Int)
    game_args = [(az_is_white, base_seed + i * 104729) for i in 1:n_games]
    results = pmap(game_args) do (is_white, seed)
        play_game_worker(is_white, seed)
    end
    return results
end

function evaluate_checkpoint(checkpoint_path, n_games)
    name = basename(dirname(dirname(checkpoint_path)))
    println("\n--- Evaluating: $name ---")
    flush(stdout)

    load_checkpoint_everywhere(checkpoint_path)

    total_start = time()

    # AZ as white
    t1 = time()
    rewards_w = run_matchup(true, n_games, 3000)
    time_w = time() - t1

    # AZ as black
    t2 = time()
    rewards_b = run_matchup(false, n_games, 4000)
    time_b = time() - t2

    total_time = time() - total_start

    # Stats
    avg_w = mean(rewards_w)
    avg_b = mean(rewards_b)
    combined = (avg_w + avg_b) / 2
    wr_w = count(r -> r > 0, rewards_w) / n_games
    wr_b = count(r -> r > 0, rewards_b) / n_games
    wr = (wr_w + wr_b) / 2
    se_w = std(rewards_w) / sqrt(n_games)
    se_b = std(rewards_b) / sqrt(n_games)

    @printf("  vs GnuBG %d-ply: Combined=%.3f  Win=%.1f%%\n", gnubg_ply, combined, 100*wr)
    @printf("    White: %.3f ± %.3f (%.1f%% wins) [%.0fs, %.1f g/s]\n",
            avg_w, 1.96*se_w, 100*wr_w, time_w, n_games/time_w)
    @printf("    Black: %.3f ± %.3f (%.1f%% wins) [%.0fs, %.1f g/s]\n",
            avg_b, 1.96*se_b, 100*wr_b, time_b, n_games/time_b)
    @printf("    Total: %.1fs (%.1f g/s)\n", total_time, 2*n_games/total_time)
    flush(stdout)

    return (name=name, combined=combined, wr=wr,
            avg_w=avg_w, avg_b=avg_b, wr_w=wr_w, wr_b=wr_b,
            time=total_time)
end

println("\nWorkers ready. Starting evaluations...")
flush(stdout)

results = []
for ckpt in checkpoints
    r = evaluate_checkpoint(ckpt, num_games)
    push!(results, r)
end

# Summary table
println("\n" * "=" ^ 60)
println("SUMMARY (vs GnuBG $(gnubg_ply)-ply, $(num_games) games/side)")
println("=" ^ 60)
@printf("%-45s %8s %6s\n", "Session", "Equity", "Win%")
println("-" ^ 60)
for r in sort(results, by=x -> -x.combined)
    @printf("%-45s %+8.3f %5.1f%%\n", r.name, r.combined, 100*r.wr)
end
println("=" ^ 60)

rmprocs(workers())
