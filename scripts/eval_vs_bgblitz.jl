#!/usr/bin/env julia
# Fast parallel BGBlitz 0-ply evaluation using threads + TCP server.
# Single JVM, no process spawning, no focus stealing.
#
# Usage:
#   julia --threads 16 --project scripts/eval_vs_bgblitz.jl <checkpoint> [obs_type] [num_games] [width] [blocks] [num_workers] [mcts_iters]
#
# Example:
#   julia --threads 16 --project scripts/eval_vs_bgblitz.jl sessions/.../checkpoints/latest.data minimal 200 128 3 8 100

using Printf
using Statistics
using Random
using Sockets
using LinearAlgebra

# Minimize BLAS contention across parallel MCTS workers
BLAS.set_num_threads(1)

# Parse args
args = filter(a -> !isempty(strip(a)), ARGS)

if length(args) < 1
    println("Usage: julia --threads 16 --project scripts/eval_vs_bgblitz.jl <checkpoint> [obs_type] [num_games] [width] [blocks] [num_workers] [mcts_iters]")
    exit(1)
end

checkpoint_path = abspath(args[1])
obs_type = length(args) >= 2 ? args[2] : "minimal"
num_games = length(args) >= 3 ? parse(Int, strip(args[3])) : 200
net_width = length(args) >= 4 ? parse(Int, strip(args[4])) : 128
net_blocks = length(args) >= 5 ? parse(Int, strip(args[5])) : 3
num_workers = length(args) >= 6 ? parse(Int, strip(args[6])) : 8
mcts_iters = length(args) >= 7 ? parse(Int, strip(args[7])) : 100
script_dir = @__DIR__

println("=" ^ 60)
println("BGBlitz 0-ply Evaluation (Threaded + TCP Server)")
println("=" ^ 60)
println("Checkpoint: $checkpoint_path")
println("Network: $(net_width)w x $(net_blocks)b")
println("Games per side: $num_games ($(2 * num_games) total)")
println("Workers: $num_workers ($(Threads.nthreads()) Julia threads)")
println("MCTS: $mcts_iters iters, BLAS threads: $(BLAS.get_num_threads())")
println("=" ^ 60)
flush(stdout)

ENV["BACKGAMMON_OBS_TYPE"] = obs_type

using AlphaZero
using AlphaZero: GI, Network, FluxLib, MctsParams, MctsPlayer, ConstSchedule
using BackgammonNet
using Flux

include(joinpath(script_dir, "..", "games", "backgammon-deterministic", "game.jl"))
include(joinpath(script_dir, "BgblitzPlayer.jl"))

const NetLib = FluxLib
const gspec = GameSpec()

# ─── Start BGBlitz server ───────────────────────────────────────────────
println("\nStarting BGBlitz server ($num_workers evaluator slots)...")
flush(stdout)
BgblitzPlayer.start_server(slots=num_workers, ply=0)
println("BGBlitz server ready")
flush(stdout)

# ─── Load network ───────────────────────────────────────────────────────
println("Loading network...")
flush(stdout)
network = NetLib.FCResNetMultiHead(gspec, NetLib.FCResNetMultiHeadHP(width=net_width, num_blocks=net_blocks))
NetLib.load_weights(checkpoint_path, network)
n_params = sum(length, Flux.params(network))
println("Network loaded: $(n_params) params")
flush(stdout)

# ─── Create worker pool ────────────────────────────────────────────────
const mcts_params = MctsParams(
    num_iters_per_turn=mcts_iters,
    cpuct=2.0,
    temperature=ConstSchedule(0.0),
    dirichlet_noise_ϵ=0.0,
    dirichlet_noise_α=1.0
)

struct EvalWorker
    az_player::MctsPlayer
    bgblitz_conn::TCPSocket
end

const worker_pool = Channel{EvalWorker}(num_workers)

println("Creating $num_workers workers...")
flush(stdout)
for i in 1:num_workers
    player = MctsPlayer(gspec, network, mcts_params)
    conn = BgblitzPlayer.take_connection()
    put!(worker_pool, EvalWorker(player, conn))
end
println("All workers ready\n")
flush(stdout)

# ─── Game function ──────────────────────────────────────────────────────

function play_game(worker::EvalWorker, az_is_white::Bool, seed::Int)::Float64
    env = GI.init(gspec)
    env.rng = MersenneTwister(seed)

    while !GI.game_terminated(env)
        is_white = GI.white_playing(env)
        use_az = (is_white && az_is_white) || (!is_white && !az_is_white)

        if use_az
            actions, π = AlphaZero.think(worker.az_player, env)
            action = actions[argmax(π)]
        else
            action = BgblitzPlayer.best_move(worker.bgblitz_conn, env)
        end
        GI.play!(env, action)
    end

    AlphaZero.reset_player!(worker.az_player)
    reward = GI.white_reward(env)
    return az_is_white ? reward : -reward
end

# ─── Matchup runner ─────────────────────────────────────────────────────

function run_matchup(az_is_white::Bool, n_games::Int, base_seed::Int)
    label = az_is_white ? "AZ(white) vs BGBlitz-0ply" : "BGBlitz-0ply vs AZ(black)"
    println("  $label ($n_games games)...")
    flush(stdout)

    results = Vector{Float64}(undef, n_games)
    completed = Threads.Atomic{Int}(0)
    t0 = time()

    @sync for i in 1:n_games
        Threads.@spawn begin
            w = take!(worker_pool)
            try
                results[i] = play_game(w, az_is_white, base_seed + i * 104729)
            finally
                put!(worker_pool, w)
            end
            n = Threads.atomic_add!(completed, 1)
            if (n + 1) % 25 == 0
                elapsed = time() - t0
                @printf("    %d/%d (%.1f games/sec)\n", n + 1, n_games, (n + 1) / elapsed)
                flush(stdout)
            end
        end
    end

    elapsed = time() - t0
    avg = mean(results)
    se = std(results) / sqrt(n_games)
    wr = count(r -> r > 0, results) / n_games

    @printf("  => avg=%.3f ± %.3f, win=%.1f%%, %.1fs (%.1f g/s)\n",
            avg, 1.96 * se, 100 * wr, elapsed, n_games / elapsed)
    flush(stdout)

    return (results=results, avg=avg, se=se, wr=wr, time=elapsed)
end

# ─── Run evaluation ─────────────────────────────────────────────────────
println("=" ^ 60)
println("Starting evaluation...")
println("=" ^ 60)
flush(stdout)

total_start = time()

r_white = run_matchup(true, num_games, 1000)
r_black = run_matchup(false, num_games, 2000)

total_time = time() - total_start

# ─── Summary ────────────────────────────────────────────────────────────
combined_avg = (r_white.avg + r_black.avg) / 2
combined_wr = (r_white.wr + r_black.wr) / 2
combined_se = sqrt(r_white.se^2 + r_black.se^2) / 2

println("\n" * "=" ^ 60)
println("SUMMARY")
println("=" ^ 60)
@printf("vs BGBlitz 0-ply: reward = %.3f ± %.3f, win rate = %.1f%%\n",
        combined_avg, 1.96 * combined_se, 100 * combined_wr)
@printf("  As white: %.3f (%.1f%% wins)\n", r_white.avg, 100 * r_white.wr)
@printf("  As black: %.3f (%.1f%% wins)\n", r_black.avg, 100 * r_black.wr)
println()
@printf("Total: %d games in %.1fs (%.1f games/sec)\n",
        2 * num_games, total_time, 2 * num_games / total_time)
println("=" ^ 60)
flush(stdout)

# ─── Cleanup ────────────────────────────────────────────────────────────
println("\nShutting down...")
flush(stdout)
for _ in 1:num_workers
    w = take!(worker_pool)
    try
        println(w.bgblitz_conn, "QUIT")
        flush(w.bgblitz_conn)
        close(w.bgblitz_conn)
    catch
    end
end
BgblitzPlayer.stop_server()
println("Done.")
