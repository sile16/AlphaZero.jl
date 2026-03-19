#!/usr/bin/env julia
"""
Benchmark CPU inference backends through the shared oracle layer.

This script compares `auto`, `fast`, and `flux` using the same code path
that eval and self-play now use.

Usage:
    julia --threads 30 --project scripts/bench_eval_backends.jl <checkpoint> [options...]
"""

using ArgParse

function parse_args()
    s = ArgParseSettings(description="Benchmark CPU inference backends", autofix_names=true)
    @add_arg_table! s begin
        "checkpoint"
            help = "Contact/main checkpoint"
            arg_type = String
            required = true
        "--race-checkpoint"
            help = "Optional race checkpoint for dual-model benchmarking"
            arg_type = String
            default = ""
        "--width"
            arg_type = Int
            default = 256
        "--blocks"
            arg_type = Int
            default = 5
        "--race-width"
            arg_type = Int
            default = 128
        "--race-blocks"
            arg_type = Int
            default = 3
        "--num-workers"
            arg_type = Int
            default = min(Threads.nthreads(), 24)
        "--num-games"
            arg_type = Int
            default = 50
        "--mcts-iters"
            arg_type = Int
            default = 100
        "--batch-size"
            arg_type = Int
            default = 50
        "--backends"
            help = "Comma-separated CPU backends: auto, fast, flux"
            arg_type = String
            default = "auto,fast,flux"
        "--obs-type"
            arg_type = String
            default = "minimal_flat"
        "--wildbg-lib"
            arg_type = String
            default = ""
        "--raw-positions"
            help = "Random positions used for raw oracle throughput"
            arg_type = Int
            default = 200
    end
    return ArgParse.parse_args(s)
end

const ARGS = parse_args()

using AlphaZero
using AlphaZero: GI, FluxLib
using AlphaZero.NetLib
import Flux
using Random
using Printf
using BackgammonNet

ENV["BACKGAMMON_OBS_TYPE"] = ARGS["obs_type"]
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))

const gspec = GameSpec()
const NUM_ACTIONS = GI.num_actions(gspec)
const STATE_DIM = GI.state_dim(gspec)[1]
const ORACLE_CFG = AlphaZero.BackgammonInference.OracleConfig(
    STATE_DIM, NUM_ACTIONS, gspec;
    vectorize_state! = vectorize_state_into!,
    route_state = s -> (s isa BackgammonNet.BackgammonGame && !BackgammonNet.is_contact_position(s) ? 2 : 1))

function parse_backend_list(s::String)
    seen = Symbol[]
    for part in split(s, ',')
        isempty(strip(part)) && continue
        backend = AlphaZero.BackgammonInference.normalize_cpu_backend(strip(part))
        backend in seen || push!(seen, backend)
    end
    isempty(seen) && error("No valid backends requested")
    return seen
end

function detect_race_checkpoint(contact_checkpoint::String, explicit_race::Union{Nothing, String})
    explicit_race !== nothing && return explicit_race
    dir = dirname(contact_checkpoint)
    sibling = joinpath(dir, "race_latest.data")
    return isfile(sibling) ? sibling : nothing
end

function generate_random_positions(n; rng=MersenneTwister(42))
    positions = Any[]
    while length(positions) < n
        env = GI.init(gspec)
        for _ in 1:rand(rng, 5:15)
            GI.game_terminated(env) && break
            if GI.is_chance_node(env)
                outcomes = GI.chance_outcomes(env)
                GI.apply_chance!(env, outcomes[rand(rng, 1:length(outcomes))][1])
            else
                actions = GI.available_actions(env)
                GI.play!(env, actions[rand(rng, 1:length(actions))])
            end
        end
        while !GI.game_terminated(env) && GI.is_chance_node(env)
            outcomes = GI.chance_outcomes(env)
            GI.apply_chance!(env, outcomes[1][1])
        end
        GI.game_terminated(env) || push!(positions, GI.current_state(env))
    end
    return positions
end

function bench_raw_oracle(batch_oracle, positions, max_batch::Int)
    println("  Raw oracle throughput:")
    for bs in (1, 10, 25, 50)
        n = min(bs, length(positions))
        n > max_batch && continue
        warm = batch_oracle(positions[1:n])
        @assert length(warm) == n
        iters = max(5, div(200, n))
        elapsed = @elapsed for _ in 1:iters
            batch_oracle(positions[1:n])
        end
        rate = n * iters / elapsed
        @printf("    batch=%3d | %9.0f st/s | %.3f ms/call\n", n, rate, elapsed / iters * 1000)
    end
end

clone_position(state) = state isa BackgammonNet.BackgammonGame ? BackgammonNet.clone(state) : deepcopy(state)

function bench_raw_oracle_mt(batch_oracle, positions, batch_size::Int, n_workers::Int)
    n = min(batch_size, length(positions))
    n == 0 && return
    base = positions[1:n]
    slices = [[clone_position(state) for state in base] for _ in 1:n_workers]
    iters_per_worker = max(20, div(800, n))
    results = zeros(Float64, n_workers)
    t0 = time()
    Threads.@threads for worker in 1:n_workers
        local_elapsed = @elapsed for _ in 1:iters_per_worker
            batch_oracle(slices[worker])
        end
        results[worker] = n * iters_per_worker / local_elapsed
    end
    elapsed = time() - t0
    total = sum(results)
    @printf("  Multi-thread batch=%d | total=%9.0f st/s | per-worker=%9.0f st/s | wall=%.3fs\n",
            n, total, total / n_workers, elapsed)
end

function main()
    backends = parse_backend_list(ARGS["backends"])
    checkpoint = ARGS["checkpoint"]
    race_checkpoint = detect_race_checkpoint(
        checkpoint,
        isempty(ARGS["race_checkpoint"]) ? nothing : ARGS["race_checkpoint"])

    println("Loading networks...")
    contact_net = FluxLib.FCResNetMultiHead(
        gspec, FluxLib.FCResNetMultiHeadHP(width=ARGS["width"], num_blocks=ARGS["blocks"]))
    FluxLib.load_weights(checkpoint, contact_net)
    contact_net = Flux.cpu(contact_net)

    race_net = nothing
    if race_checkpoint !== nothing
        race_net = FluxLib.FCResNetMultiHead(
            gspec, FluxLib.FCResNetMultiHeadHP(width=ARGS["race_width"], num_blocks=ARGS["race_blocks"]))
        FluxLib.load_weights(race_checkpoint, race_net)
        race_net = Flux.cpu(race_net)
        println("  Race checkpoint: $race_checkpoint")
    end

    println("Generating random positions...")
    test_positions = generate_random_positions(ARGS["raw_positions"])
    println("  Positions: $(length(test_positions))")

    println()
    println("=== CPU Backend Benchmark ===")
    println("Threads: $(Threads.nthreads())")
    println("Workers: $(ARGS["num_workers"])")
    println("Batch size: $(ARGS["batch_size"])")
    println("Backends: $(join(string.(backends), ", "))")
    println("Dual-model: $(race_net !== nothing)")
    println("Note: use eval_vs_wildbg.jl or eval_race.jl for end-to-end game throughput")
    println()

    for backend_name in backends
        resolved = AlphaZero.BackgammonInference.resolve_cpu_backend(backend_name)
        label = String(backend_name)
        println("--- $(uppercase(label)) -> $(AlphaZero.BackgammonInference.cpu_backend_summary(backend_name)) ---")

        single_oracle, batch_oracle = AlphaZero.BackgammonInference.make_cpu_oracles(
            resolved, contact_net, ORACLE_CFG;
            secondary_net=race_net,
            batch_size=ARGS["batch_size"],
            nslots=ARGS["num_workers"])
        single_oracle(test_positions[1])

        bench_raw_oracle(batch_oracle, test_positions, ARGS["batch_size"])
        bench_raw_oracle_mt(batch_oracle, test_positions, ARGS["batch_size"], ARGS["num_workers"])
        println()
        flush(stdout)
    end
end

main()
