#!/usr/bin/env julia
"""
Evaluate a race model on fixed race positions vs wildbg.

Loads the 2000 fixed race positions, plays each to completion using AZ MCTS
vs wildbg, tracks equity and value error.

Usage:
    julia --threads 30 --project scripts/eval_race.jl <checkpoint> [options...]

    # Race-only model:
    julia --threads 30 --project scripts/eval_race.jl /path/to/race_latest.data \\
        --width=128 --blocks=3 --num-workers=22 --mcts-iters=600

    # Defaults to wildbg small nets, 600 MCTS iters
"""

using ArgParse

function parse_args_eval()
    s = ArgParseSettings(description="Evaluate race model on fixed positions", autofix_names=true)
    @add_arg_table! s begin
        "checkpoint"
            help = "Race model checkpoint file"
            arg_type = String
            required = true
        "--obs-type"
            arg_type = String
            default = "minimal_flat"
        "--num-games"
            help = "How many of the 2000 positions to play (0=all)"
            arg_type = Int
            default = 0
        "--width"
            arg_type = Int
            default = 128
        "--blocks"
            arg_type = Int
            default = 3
        "--num-workers"
            arg_type = Int
            default = 22
        "--mcts-iters"
            arg_type = Int
            default = 600
        "--wildbg-lib"
            arg_type = String
            default = ""
        "--positions-file"
            arg_type = String
            default = "/homeshare/projects/AlphaZero.jl/eval_data/race_eval_2000.jls"
        "--inference-batch-size"
            arg_type = Int
            default = 50
        "--inference-backend"
            help = "CPU inference backend: auto, fast, or flux"
            arg_type = String
            default = "auto"
    end
    return ArgParse.parse_args(s)
end

const ARGS = parse_args_eval()

# Load packages
using AlphaZero
using AlphaZero: GI, MCTS, Network, FluxLib, MctsParams, ConstSchedule, BatchedMCTS, GameLoop
using AlphaZero.NetLib
import Flux
using Random
using Statistics
using Dates
using Printf
using Serialization
using StaticArrays

# BackgammonNet
using BackgammonNet

# Set up game
ENV["BACKGAMMON_OBS_TYPE"] = ARGS["obs_type"]
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
const gspec = GameSpec()
const NUM_ACTIONS = GI.num_actions(gspec)
const _state_dim = GI.state_dim(gspec)[1]
const ORACLE_CFG = AlphaZero.BackgammonInference.OracleConfig(
    _state_dim, NUM_ACTIONS, gspec; vectorize_state! = vectorize_state_into!)

# ── Wildbg Library ──────────────────────────────────────────────────────

function find_wildbg_lib()
    if !isempty(ARGS["wildbg_lib"])
        return ARGS["wildbg_lib"]
    end
    candidates = [
        joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg_main.dylib"),
        joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg_main.so"),
        joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg.dylib"),
        joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg.so"),
    ]
    for c in candidates
        isfile(c) && return c
    end
    error("libwildbg not found. Pass --wildbg-lib=/path/to/libwildbg")
end

# ── Value Stats ─────────────────────────────────────────────────────────

struct PositionValueSample
    nn_val::Float64
    wb_val::Float64
end

function compute_value_stats(samples::Vector{PositionValueSample})
    isempty(samples) && return nothing
    nn = [s.nn_val for s in samples]
    wb = [s.wb_val for s in samples]
    mse = mean((nn .- wb) .^ 2)
    mae = mean(abs.(nn .- wb))
    bias = mean(nn) - mean(wb)
    corr = length(nn) >= 3 ? cor(nn, wb) : NaN
    return (n=length(samples), mse=mse, mae=mae, bias=bias, corr=corr,
            nn_mean=mean(nn), wb_mean=mean(wb), nn_std=std(nn), wb_std=std(wb))
end

# ── Game Play from Position ─────────────────────────────────────────────

"""Play a game starting from a fixed race position. Returns (reward, value_samples).

`az_is_white`: if true, AZ plays as P0 (white); if false, AZ plays as P1 (black).

Uses GameLoop.play_game() for the game loop."""
function eval_race_game(single_oracle, batch_oracle, wildbg_backend, position_data::Tuple,
                        value_batch_oracle; seed::Int=1, az_is_white::Bool=true,
                        gspec=nothing, mcts_params=nothing, batch_size::Int=50)
    rng = MersenneTwister(seed)
    p0, p1, cp = position_data

    # Create game at chance node (needs dice roll first)
    game = BackgammonGame(p0, p1, SVector{2,Int8}(0, 0), Int8(0), cp, false, 0.0f0;
                          obs_type=:minimal_flat)
    env = GameEnv(game, rng)

    az = GameLoop.MctsAgent(single_oracle, batch_oracle, mcts_params, batch_size, gspec)
    wb = GameLoop.ExternalAgent(wildbg_backend)

    # Value comparison functions
    value_oracle_fn = env -> begin
        Float64(value_batch_oracle([env.game])[1][2])
    end
    wildbg_value_fn = env -> Float64(BackgammonNet.evaluate(wildbg_backend, env.game))

    w, b = az_is_white ? (az, wb) : (wb, az)
    result = GameLoop.play_game(w, b, env;
        record_value_comparison=true,
        value_oracle=value_oracle_fn,
        opponent_value_fn=wildbg_value_fn,
        rng=rng,
        temperature_fn=_ -> 0.0)

    az_reward = az_is_white ? result.reward : -result.reward
    value_samples = [PositionValueSample(s.nn_val, s.opponent_val) for s in result.value_samples]
    return (reward=az_reward, value_samples=value_samples)
end

# ── Main ────────────────────────────────────────────────────────────────

function main()
    # Load positions
    positions_file = ARGS["positions_file"]
    if !isfile(positions_file)
        error("Positions file not found: $positions_file\nRun scripts/generate_race_positions.jl first")
    end
    all_positions = deserialize(positions_file)
    println("Loaded $(length(all_positions)) race positions from $positions_file")

    # Subset if requested
    n_games = ARGS["num_games"]
    if n_games > 0 && n_games < length(all_positions)
        positions = all_positions[1:n_games]
    else
        positions = all_positions
        n_games = length(positions)
    end

    checkpoint = ARGS["checkpoint"]
    width = ARGS["width"]
    blocks = ARGS["blocks"]
    mcts_iters = ARGS["mcts_iters"]
    batch_size = ARGS["inference_batch_size"]
    num_workers = ARGS["num_workers"]
    backend = AlphaZero.BackgammonInference.resolve_cpu_backend(ARGS["inference_backend"])

    # Total games = 2 * n_positions (each position played from both sides)
    n_total = 2 * n_games

    println("Checkpoint: $checkpoint")
    println("Architecture: $(width)w×$(blocks)b")
    println("Positions: $n_games | Games: $n_total (both sides)")
    println("MCTS: $mcts_iters iterations")
    println("Workers: $num_workers CPU")
    println("CPU inference: $(AlphaZero.BackgammonInference.cpu_backend_summary(backend))")

    # Load network
    network = FluxLib.FCResNetMultiHead(
        gspec, FluxLib.FCResNetMultiHeadHP(width=width, num_blocks=blocks))
    FluxLib.load_weights(checkpoint, network)
    network = Flux.cpu(network)

    mcts_params = MctsParams(
        num_iters_per_turn=mcts_iters,
        cpuct=1.5,
        temperature=ConstSchedule(0.0),
        dirichlet_noise_ϵ=0.0,
        dirichlet_noise_α=1.0)

    single_oracle, batch_oracle = AlphaZero.BackgammonInference.make_cpu_oracles(
        backend, network, ORACLE_CFG; batch_size=batch_size)
    _, value_batch_oracle = AlphaZero.BackgammonInference.make_cpu_oracles(
        :flux, network, ORACLE_CFG; batch_size=1)
    # Wildbg setup
    wildbg_lib = find_wildbg_lib()
    lib_size = filesize(wildbg_lib)
    nets_variant = lib_size > 10_000_000 ? :large : :small
    if nets_variant == :large
        BackgammonNet.wildbg_set_lib_path!(large=wildbg_lib)
    else
        BackgammonNet.wildbg_set_lib_path!(small=wildbg_lib)
    end
    println("wildbg: $nets_variant ($(round(lib_size/1e6, digits=1))MB)")

    wildbg_backends = [begin
        wb = BackgammonNet.WildbgBackend(nets=nets_variant)
        BackgammonNet.open!(wb)
        wb
    end for _ in 1:num_workers]

    println("=" ^ 70)
    flush(stdout)

    # Play each position from both sides:
    #   job 1..n_games:         AZ as white (P0)
    #   job n_games+1..2*n_games: AZ as black (P1)
    rewards = zeros(Float64, n_total)
    vsamples = [PositionValueSample[] for _ in 1:n_total]

    t_start = time()
    claimed = Threads.Atomic{Int}(0)
    done = Threads.Atomic{Int}(0)
    errors = Threads.Atomic{Int}(0)

    Threads.@threads for tid in 1:num_workers
        wb = wildbg_backends[tid]
        while true
            job = Threads.atomic_add!(claimed, 1) + 1
            job > n_total && break
            if job <= n_games
                pos_idx = job
                az_white = true
                seed = job
            else
                pos_idx = job - n_games
                az_white = false
                seed = job  # different seed than white game
            end
            try
                result = eval_race_game(single_oracle, batch_oracle, wb, positions[pos_idx],
                                        value_batch_oracle; seed=seed, az_is_white=az_white,
                                        gspec=gspec, mcts_params=mcts_params, batch_size=batch_size)
                rewards[job] = result.reward
                vsamples[job] = result.value_samples
            catch e
                n_err = Threads.atomic_add!(errors, 1) + 1
                @warn "Eval game $job failed (pos=$pos_idx, white=$az_white, seed=$seed): $e"
                # Skip this game — reward stays 0.0, no value samples
            end
            d = Threads.atomic_add!(done, 1) + 1
            if d % 100 == 0
                elapsed = time() - t_start
                rate = d / elapsed
                eta = (n_total - d) / rate
                @printf("  %d/%d games (%.1f g/min, ETA %.0fs)\n", d, n_total, rate*60, eta)
                flush(stdout)
            end
        end
    end

    elapsed = time() - t_start

    for wb in wildbg_backends
        BackgammonNet.close(wb)
    end

    # Split results by side
    white_rewards = rewards[1:n_games]
    black_rewards = rewards[n_games+1:end]

    avg_reward = mean(rewards)
    white_avg = mean(white_rewards)
    black_avg = mean(black_rewards)
    win_count = count(r -> r > 0, rewards)
    win_pct = 100 * win_count / n_total

    all_vs = PositionValueSample[]
    for vs in vsamples; append!(all_vs, vs); end
    vstats = compute_value_stats(all_vs)

    println("=" ^ 70)
    println("Results (vs wildbg, race positions, both sides):")
    @printf("  Equity:    %.3f  (as white: %+.3f, as black: %+.3f)\n",
            avg_reward, white_avg, black_avg)
    @printf("  Win%%:      %.1f%%\n", win_pct)
    n_errors = errors[]
    @printf("  Games:     %d (%d positions × 2 sides, %d errors skipped)\n", n_total, n_games, n_errors)
    @printf("  Time:      %.2f s (%.3f min)\n", elapsed, elapsed / 60)
    @printf("  Rate:      %.1f games/min\n", n_total / (elapsed / 60))

    if vstats !== nothing
        println("\nValue Error (NN vs Wildbg):")
        @printf("  n=%d | MSE=%.4f | MAE=%.4f | bias=%+.4f | corr=%.4f\n",
                vstats.n, vstats.mse, vstats.mae, vstats.bias, vstats.corr)
        @printf("  NN=%.3f±%.3f | WB=%.3f±%.3f\n",
                vstats.nn_mean, vstats.nn_std, vstats.wb_mean, vstats.wb_std)
    end
    println("=" ^ 70)

    return (equity=avg_reward, win_pct=win_pct, vstats=vstats)
end

main()
