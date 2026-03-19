#!/usr/bin/env julia
"""
Benchmark shared CPU inference backends on self-play-like BatchedMCTS games.

This uses the same shared oracle layer as eval/self-play production code:
`src/inference/backgammon_oracles.jl`.

Usage:
    julia --threads 30 --project scripts/bench_selfplay_backends.jl \
        /path/to/contact_latest.data --width=256 --blocks=5 \
        --race-width=128 --race-blocks=3 --num-workers=22 \
        --mcts-iters=400 --batch-size=50 --num-games=40 --backends=fast,flux
"""

using ArgParse

function parse_args()
    s = ArgParseSettings(description="Benchmark self-play backends", autofix_names=true)
    @add_arg_table! s begin
        "checkpoint"
            help = "Path to contact/latest checkpoint .data file"
            arg_type = String
            required = true
        "--race-checkpoint"
            help = "Path to race checkpoint .data file (auto-detected when omitted)"
            arg_type = String
            default = ""
        "--width"
            help = "Contact network width"
            arg_type = Int
            default = 256
        "--blocks"
            help = "Contact network residual blocks"
            arg_type = Int
            default = 5
        "--race-width"
            help = "Race network width"
            arg_type = Int
            default = 128
        "--race-blocks"
            help = "Race network residual blocks"
            arg_type = Int
            default = 3
        "--num-workers"
            help = "Parallel CPU workers"
            arg_type = Int
            default = min(Threads.nthreads(), 24)
        "--mcts-iters"
            help = "MCTS iterations per move"
            arg_type = Int
            default = 400
        "--batch-size"
            help = "Inference batch size"
            arg_type = Int
            default = 50
        "--num-games"
            help = "Total self-play games per backend"
            arg_type = Int
            default = 40
        "--obs-type"
            help = "Observation type"
            arg_type = String
            default = "minimal_flat"
        "--backends"
            help = "Comma-separated CPU backends: auto,fast,flux"
            arg_type = String
            default = "fast,flux"
    end
    return ArgParse.parse_args(s)
end

const ARGS = parse_args()

using AlphaZero
using AlphaZero: GI, FluxLib, MctsParams, ConstSchedule, BatchedMCTS
using AlphaZero: Network
import Flux
import LinearAlgebra
using Random
using Printf
using Statistics
using BackgammonNet

LinearAlgebra.BLAS.set_num_threads(1)

ENV["BACKGAMMON_OBS_TYPE"] = ARGS["obs_type"]
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))

const gspec = GameSpec()
const NUM_ACTIONS = GI.num_actions(gspec)
const STATE_DIM = let env = GI.init(gspec); length(vec(GI.vectorize_state(gspec, GI.current_state(env)))); end
const ORACLE_CFG = AlphaZero.BackgammonInference.OracleConfig(
    STATE_DIM, NUM_ACTIONS, gspec;
    vectorize_state! = vectorize_state_into!,
    route_state = s -> (s isa BackgammonNet.BackgammonGame && !BackgammonNet.is_contact_position(s) ? 2 : 1))

function detect_race_checkpoint(contact_ckpt::String)
    ckpt_dir = dirname(contact_ckpt)
    race_candidate = joinpath(ckpt_dir, "race_latest.data")
    return isfile(race_candidate) ? race_candidate : nothing
end

function load_network(checkpoint::String, width::Int, blocks::Int)
    net = FluxLib.FCResNetMultiHead(
        gspec, FluxLib.FCResNetMultiHeadHP(width=width, num_blocks=blocks))
    FluxLib.load_weights(checkpoint, net)
    return Flux.cpu(net)
end

function backend_list(spec::String)
    return [AlphaZero.BackgammonInference.normalize_cpu_backend(Symbol(strip(s)))
            for s in split(spec, ',') if !isempty(strip(s))]
end

function choose_action(rng::AbstractRNG, actions, pi)
    z = sum(pi)
    if !(z > 0)
        return actions[argmax(pi)]
    end
    r = rand(rng)
    acc = 0.0
    @inbounds for i in eachindex(pi)
        acc += pi[i] / z
        if r <= acc
            return actions[i]
        end
    end
    return actions[end]
end

function play_one_game(single_oracle, batch_oracle, mcts_params, batch_size, rng)
    player = BatchedMCTS.BatchedMctsPlayer(
        gspec, single_oracle, mcts_params;
        batch_size=batch_size, batch_oracle=batch_oracle)

    env = GI.init(gspec)
    turns = 0
    while !GI.game_terminated(env)
        if GI.is_chance_node(env)
            outcomes = GI.chance_outcomes(env)
            probs = [p for (_, p) in outcomes]
            r = rand(rng)
            acc = 0.0
            idx = lastindex(outcomes)
            for i in eachindex(outcomes)
                acc += probs[i]
                if r <= acc
                    idx = i
                    break
                end
            end
            GI.apply_chance!(env, outcomes[idx][1])
            continue
        end
        actions, pi = BatchedMCTS.think(player, env)
        GI.play!(env, choose_action(rng, actions, pi))
        BatchedMCTS.reset_player!(player)
        turns += 1
    end
    return turns
end

function bench_backend(backend, contact_net, race_net, mcts_params)
    resolved = AlphaZero.BackgammonInference.resolve_cpu_backend(backend)
    use_fast = resolved == :fast
    primary_fw = use_fast ? AlphaZero.FastInference.extract_fast_weights(contact_net) : nothing
    secondary_fw = use_fast && race_net !== nothing ? AlphaZero.FastInference.extract_fast_weights(race_net) : nothing
    single_oracle, batch_oracle = AlphaZero.BackgammonInference.make_cpu_oracles(
        resolved, contact_net, ORACLE_CFG;
        secondary_net=race_net,
        batch_size=ARGS["batch_size"],
        primary_fw=primary_fw,
        secondary_fw=secondary_fw,
        nslots=Threads.nthreads())

    warmup_rng = MersenneTwister(1234)
    play_one_game(single_oracle, batch_oracle, mcts_params, ARGS["batch_size"], warmup_rng)

    games_done = Threads.Atomic{Int}(0)
    total_turns = Threads.Atomic{Int}(0)
    elapsed = @elapsed begin
        tasks = Task[]
        for worker in 1:ARGS["num_workers"]
            rng = MersenneTwister(10_000 + worker * 7_919)
            t = Threads.@spawn begin
                local_turns = 0
                while true
                    game_idx = Threads.atomic_add!(games_done, 1)
                    game_idx >= ARGS["num_games"] && break
                    local_turns += play_one_game(single_oracle, batch_oracle,
                                                 mcts_params, ARGS["batch_size"], rng)
                end
                Threads.atomic_add!(total_turns, local_turns)
            end
            push!(tasks, t)
        end
        foreach(fetch, tasks)
    end

    games = min(games_done[], ARGS["num_games"])
    turns = total_turns[]
    gpm = games * 60 / elapsed
    tps = turns / elapsed
    return (backend=resolved, games=games, seconds=elapsed, games_per_min=gpm, turns=tps)
end

function main()
    contact_ckpt = ARGS["checkpoint"]
    race_ckpt = isempty(ARGS["race_checkpoint"]) ? detect_race_checkpoint(contact_ckpt) : ARGS["race_checkpoint"]

    println("Self-play backend benchmark")
    println("  Checkpoint: $contact_ckpt")
    println("  Race checkpoint: $(race_ckpt === nothing ? "none" : race_ckpt)")
    println("  Architecture: contact=$(ARGS["width"])w×$(ARGS["blocks"])b + race=$(ARGS["race_width"])w×$(ARGS["race_blocks"])b")
    println("  Workers: $(ARGS["num_workers"])")
    println("  MCTS: $(ARGS["mcts_iters"])")
    println("  Batch size: $(ARGS["batch_size"])")
    println("  Games/backend: $(ARGS["num_games"])")
    println("  Threads: $(Threads.nthreads())")
    println()

    contact_net = load_network(contact_ckpt, ARGS["width"], ARGS["blocks"])
    race_net = race_ckpt === nothing ? nothing : load_network(race_ckpt, ARGS["race_width"], ARGS["race_blocks"])
    mcts_params = MctsParams(
        num_iters_per_turn=ARGS["mcts_iters"],
        cpuct=2.0,
        temperature=ConstSchedule(1.0),
        dirichlet_noise_ϵ=0.25,
        dirichlet_noise_α=0.3)

    results = []
    for backend in backend_list(ARGS["backends"])
        println("=" ^ 70)
        println("Backend: $(AlphaZero.BackgammonInference.cpu_backend_summary(backend))")
        flush(stdout)
        result = bench_backend(backend, contact_net, race_net, mcts_params)
        push!(results, result)
        @printf("  Games:     %d\n", result.games)
        @printf("  Time:      %.2f s\n", result.seconds)
        @printf("  Rate:      %.1f games/min\n", result.games_per_min)
        @printf("  Turns/s:   %.1f\n", result.turns)
        flush(stdout)
    end

    println("\n" * "=" ^ 70)
    println("Summary")
    println("=" ^ 70)
    for result in results
        @printf("  %-12s | %7.2f s | %8.1f games/min | %8.1f turns/s\n",
                string(result.backend), result.seconds, result.games_per_min, result.turns)
    end
end

main()
