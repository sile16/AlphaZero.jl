#!/usr/bin/env julia
"""
Evaluate AlphaZero checkpoints against wildbg (Rust ONNX backgammon engine).

Uses BackgammonNet.jl's WildbgBackend + BackendAgent for opponent play,
and the AlphaZero network for our agent's play via batched MCTS.

Usage:
    julia --threads 16 --project scripts/eval_vs_wildbg.jl <checkpoint> [options...]

    # Single checkpoint:
    julia --threads 16 --project scripts/eval_vs_wildbg.jl /path/to/latest.data

    # Batch eval (multiple checkpoints):
    julia --threads 16 --project scripts/eval_vs_wildbg.jl --batch /path/to/sessions/

Options:
    --obs-type=minimal     Observation type (default: minimal)
    --num-games=500        Games per side (total = 2x this)
    --width=128            Network width
    --blocks=3             Network blocks
    --num-workers=8        Worker threads
    --mcts-iters=100       MCTS iterations per move
    --wildbg-lib=PATH      Path to libwildbg.so/.dylib
    --batch                Batch mode: eval all latest.data in session dir
"""

using ArgParse

function parse_eval_args()
    s = ArgParseSettings(description="Evaluate against wildbg", autofix_names=true)

    @add_arg_table! s begin
        "checkpoint"
            help = "Checkpoint file or session directory (with --batch)"
            arg_type = String
            required = true
        "--obs-type"
            help = "Observation type"
            arg_type = String
            default = "minimal"
        "--num-games"
            help = "Games per side (total = 2x)"
            arg_type = Int
            default = 500
        "--width"
            help = "Network width"
            arg_type = Int
            default = 128
        "--blocks"
            help = "Network blocks"
            arg_type = Int
            default = 3
        "--num-workers"
            help = "Worker threads"
            arg_type = Int
            default = 8
        "--mcts-iters"
            help = "MCTS iterations per move"
            arg_type = Int
            default = 100
        "--wildbg-lib"
            help = "Path to libwildbg shared library"
            arg_type = String
            default = ""
        "--batch"
            help = "Batch mode: eval all checkpoints in session directory"
            action = :store_true
        "--inference-batch-size"
            help = "Inference batch size for MCTS"
            arg_type = Int
            default = 50
    end

    return ArgParse.parse_args(s)
end

const ARGS = parse_eval_args()

# Load packages
using AlphaZero
using AlphaZero: GI, MCTS, Network, FluxLib, MctsParams, ConstSchedule, BatchedMCTS
using AlphaZero.NetLib
import Flux
using Random
using Statistics
using Dates

# BackgammonNet provides game + wildbg backend
using BackgammonNet
using BackgammonNet: WildbgBackend, BackendAgent, wildbg_set_lib_path!,
                     play_game, play_match, AbstractAgent, RandomAgent

# Set up game
ENV["BACKGAMMON_OBS_TYPE"] = ARGS["obs_type"]
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
const gspec = GameSpec()
const NUM_ACTIONS = GI.num_actions(gspec)
const _state_dim = GI.state_dim(gspec)[1]

# Detect wildbg library
function find_wildbg_lib()
    if !isempty(ARGS["wildbg_lib"])
        return ARGS["wildbg_lib"]
    end
    # Common paths
    candidates = [
        joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg.so"),
        joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg.dylib"),
        "/usr/local/lib/libwildbg.so",
        "/usr/local/lib/libwildbg.dylib",
    ]
    for c in candidates
        isfile(c) && return c
    end
    error("libwildbg not found. Pass --wildbg-lib=/path/to/libwildbg.so")
end

"""
AlphaZero MCTS agent that uses a loaded network for evaluation.
Wraps the network in a batch oracle compatible with BackgammonNet's play_game.
"""
struct AlphaZeroAgent <: AbstractAgent
    network::Any
    mcts_params::MctsParams
    batch_size::Int
    gspec::Any
end

function BackgammonNet.agent_move(agent::AlphaZeroAgent, g::BackgammonGame)
    # Create environment wrapper for MCTS
    env = GI.init(agent.gspec)
    # Copy game state into env
    env.game = BackgammonNet.clone(g)

    # Single oracle for MCTS
    function single_oracle(state)
        X = zeros(Float32, _state_dim, 1)
        GI.vectorize_state!((@view X[:, 1]), agent.gspec, state)
        A = zeros(Float32, NUM_ACTIONS, 1)
        if !BackgammonNet.game_terminated(state)
            for action in BackgammonNet.legal_actions(state)
                if 1 <= action <= NUM_ACTIONS
                    A[action, 1] = 1.0f0
                end
            end
        end
        P_raw, V, _ = Network.convert_output_tuple(
            agent.network, Network.forward_normalized(agent.network, X, A))
        legal = @view(A[:, 1]) .> 0
        return (P_raw[legal, 1], V[1, 1])
    end

    function batch_oracle(states::Vector)
        n = length(states)
        n == 0 && return Tuple{Vector{Float32}, Float32}[]
        X = zeros(Float32, _state_dim, n)
        A = zeros(Float32, NUM_ACTIONS, n)
        for (i, s) in enumerate(states)
            GI.vectorize_state!((@view X[:, i]), agent.gspec, s)
            if !BackgammonNet.game_terminated(s)
                for action in BackgammonNet.legal_actions(s)
                    if 1 <= action <= NUM_ACTIONS
                        A[action, i] = 1.0f0
                    end
                end
            end
        end
        P_raw, V, _ = Network.convert_output_tuple(
            agent.network, Network.forward_normalized(agent.network, X, A))
        results = Vector{Tuple{Vector{Float32}, Float32}}(undef, n)
        for i in 1:n
            legal = @view(A[:, i]) .> 0
            results[i] = (P_raw[legal, i], V[1, i])
        end
        return results
    end

    player = BatchedMCTS.BatchedMctsPlayer(
        agent.gspec, single_oracle, agent.mcts_params;
        batch_size=agent.batch_size, batch_oracle=batch_oracle)

    actions, policy = BatchedMCTS.think(player, env)
    BatchedMCTS.reset_player!(player)

    # Pick best action (greedy for eval)
    return actions[argmax(policy)]
end

"""Play a single eval game: AZ agent vs wildbg opponent."""
function eval_game(az_agent::AlphaZeroAgent, wildbg_agent::BackendAgent,
                   az_is_white::Bool; seed::Int=1)
    rng = MersenneTwister(seed)
    g = BackgammonNet.initial_state()
    num_moves = 0

    while !BackgammonNet.game_terminated(g)
        if BackgammonNet.is_chance_node(g)
            BackgammonNet.sample_chance!(g, rng)
        else
            cp = Int(g.current_player)
            is_az_turn = (cp == 0) == az_is_white
            agent = is_az_turn ? az_agent : wildbg_agent

            action = BackgammonNet.agent_move(agent, g)
            BackgammonNet.apply_action!(g, action)
            num_moves += 1
        end
    end

    # Reward from white's perspective
    white_reward = g.reward > 0 ? Float64(g.reward) : Float64(g.reward)
    az_reward = az_is_white ? white_reward : -white_reward
    return az_reward
end

"""Evaluate a single checkpoint against wildbg."""
function evaluate_checkpoint(checkpoint_path::String, wildbg_lib::String;
                             width::Int, blocks::Int, num_games::Int,
                             num_workers::Int, mcts_iters::Int, batch_size::Int)
    # Load network
    network = FluxLib.FCResNetMultiHead(
        gspec, FluxLib.FCResNetMultiHeadHP(width=width, num_blocks=blocks))
    FluxLib.load_weights(checkpoint_path, network)
    network = Flux.cpu(network)

    mcts_params = MctsParams(
        num_iters_per_turn=mcts_iters,
        cpuct=1.5,
        temperature=ConstSchedule(0.0),
        dirichlet_noise_ϵ=0.0,
        dirichlet_noise_α=1.0)

    az_agent = AlphaZeroAgent(network, mcts_params, batch_size, gspec)

    # Initialize wildbg
    wildbg_set_lib_path!(wildbg_lib)
    wb = WildbgBackend()
    BackgammonNet.open!(wb)
    wildbg_agent = BackendAgent(wb)

    games_per_side = num_games

    # Play as white
    println("  Playing $games_per_side games as white...")
    flush(stdout)
    white_rewards = Vector{Float64}(undef, games_per_side)
    white_claimed = Threads.Atomic{Int}(0)
    Threads.@threads for _ in 1:num_workers
        while true
            i = Threads.atomic_add!(white_claimed, 1) + 1
            i > games_per_side && break
            white_rewards[i] = eval_game(az_agent, wildbg_agent, true; seed=i)
        end
    end

    # Play as black
    println("  Playing $games_per_side games as black...")
    flush(stdout)
    black_rewards = Vector{Float64}(undef, games_per_side)
    black_claimed = Threads.Atomic{Int}(0)
    Threads.@threads for _ in 1:num_workers
        while true
            i = Threads.atomic_add!(black_claimed, 1) + 1
            i > games_per_side && break
            black_rewards[i] = eval_game(az_agent, wildbg_agent, false; seed=i + games_per_side)
        end
    end

    BackgammonNet.close(wb)

    white_avg = mean(white_rewards)
    black_avg = mean(black_rewards)
    combined = (white_avg + black_avg) / 2
    total_games = 2 * games_per_side
    win_count = count(r -> r > 0, white_rewards) + count(r -> r > 0, black_rewards)
    win_pct = 100 * win_count / total_games

    return (white_avg=white_avg, black_avg=black_avg, combined=combined,
            total_games=total_games, win_pct=win_pct)
end

"""Find all interesting checkpoints in a sessions directory."""
function find_checkpoints(sessions_dir::String)
    checkpoints = Tuple{String, String, Int}[]  # (session_name, checkpoint_path, iters)

    for entry in readdir(sessions_dir)
        session_path = joinpath(sessions_dir, entry)
        isdir(session_path) || continue

        ckpt_dir = joinpath(session_path, "checkpoints")
        isdir(ckpt_dir) || continue

        # Check for latest.data
        latest = joinpath(ckpt_dir, "latest.data")
        isfile(latest) || continue

        # Check iteration count
        iter_file = joinpath(ckpt_dir, "iter.txt")
        iters = if isfile(iter_file)
            parse(Int, strip(read(iter_file, String)))
        else
            # Count iter_*.data files
            iter_files = filter(f -> startswith(f, "iter_") && endswith(f, ".data"), readdir(ckpt_dir))
            isempty(iter_files) ? 0 : maximum(parse(Int, match(r"iter_(\d+)", f).captures[1]) for f in iter_files)
        end

        # Skip incomplete runs
        iters < 10 && continue

        push!(checkpoints, (entry, latest, iters))
    end

    sort!(checkpoints, by=x -> x[3], rev=true)
    return checkpoints
end

"""Detect network architecture from checkpoint."""
function detect_architecture(checkpoint_path::String)
    # Try common architectures
    for (w, b) in [(256, 10), (256, 5), (128, 3), (64, 2)]
        try
            net = FluxLib.FCResNetMultiHead(
                gspec, FluxLib.FCResNetMultiHeadHP(width=w, num_blocks=b))
            FluxLib.load_weights(checkpoint_path, net)
            return (width=w, blocks=b)
        catch
            continue
        end
    end
    return nothing
end

# Main
function main()
    wildbg_lib = find_wildbg_lib()
    println("wildbg library: $wildbg_lib")

    if ARGS["batch"]
        sessions_dir = ARGS["checkpoint"]
        println("Batch evaluation of sessions in: $sessions_dir")
        println("=" ^ 70)

        checkpoints = find_checkpoints(sessions_dir)
        println("Found $(length(checkpoints)) checkpoints with 10+ iterations:\n")
        for (name, path, iters) in checkpoints
            println("  [$iters iter] $name")
        end
        println()

        results = []
        for (name, ckpt_path, iters) in checkpoints
            println("=" ^ 70)
            println("Evaluating: $name ($iters iterations)")
            println("  Checkpoint: $ckpt_path")

            # Auto-detect architecture
            arch = detect_architecture(ckpt_path)
            if arch === nothing
                println("  SKIP: Could not detect architecture")
                continue
            end
            println("  Architecture: $(arch.width)w×$(arch.blocks)b")
            flush(stdout)

            t0 = time()
            result = evaluate_checkpoint(ckpt_path, wildbg_lib;
                width=arch.width, blocks=arch.blocks,
                num_games=ARGS["num_games"],
                num_workers=ARGS["num_workers"],
                mcts_iters=ARGS["mcts_iters"],
                batch_size=ARGS["inference_batch_size"])
            eval_time = time() - t0

            println("  White:    $(round(result.white_avg, digits=3))")
            println("  Black:    $(round(result.black_avg, digits=3))")
            println("  Combined: $(round(result.combined, digits=3))")
            println("  Win%:     $(round(result.win_pct, digits=1))%")
            println("  Games:    $(result.total_games)")
            println("  Time:     $(round(eval_time / 60, digits=1)) min")
            flush(stdout)

            push!(results, (name=name, iters=iters, arch=arch,
                           combined=result.combined, win_pct=result.win_pct,
                           white=result.white_avg, black=result.black_avg,
                           time_min=eval_time/60))
        end

        # Summary table
        println("\n" * "=" ^ 70)
        println("BATCH EVALUATION SUMMARY (vs wildbg)")
        println("=" ^ 70)
        sort!(results, by=r -> r.combined, rev=true)
        println("Rank | Equity  | Win%  | Arch     | Iters | Session")
        println("-----|---------|-------|----------|-------|--------")
        for (rank, r) in enumerate(results)
            println("  $(lpad(rank, 2)) | $(lpad(round(r.combined, digits=3), 7)) | $(lpad(round(r.win_pct, digits=1), 5))% | $(r.arch.width)w×$(r.arch.blocks)b | $(lpad(r.iters, 5)) | $(r.name)")
        end
        println("=" ^ 70)

        # Save results
        results_path = joinpath(sessions_dir, "wildbg_eval_results_$(Dates.format(now(), "yyyymmdd_HHMMSS")).txt")
        open(results_path, "w") do f
            println(f, "# Batch Evaluation vs wildbg")
            println(f, "# Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
            println(f, "# Games per side: $(ARGS["num_games"])")
            println(f, "# MCTS iters: $(ARGS["mcts_iters"])")
            println(f, "# Workers: $(ARGS["num_workers"])")
            println(f, "")
            for r in results
                println(f, "$(r.name)\t$(r.iters)\t$(r.arch.width)x$(r.arch.blocks)\t$(r.combined)\t$(r.win_pct)\t$(r.white)\t$(r.black)")
            end
        end
        println("Results saved to: $results_path")

    else
        # Single checkpoint eval
        ckpt_path = ARGS["checkpoint"]
        println("Evaluating: $ckpt_path")
        println("Architecture: $(ARGS["width"])w×$(ARGS["blocks"])b")
        println("Games: $(ARGS["num_games"]) per side")
        println("MCTS: $(ARGS["mcts_iters"]) iterations")
        println("Workers: $(ARGS["num_workers"])")
        println("=" ^ 70)
        flush(stdout)

        t0 = time()
        result = evaluate_checkpoint(ckpt_path, wildbg_lib;
            width=ARGS["width"], blocks=ARGS["blocks"],
            num_games=ARGS["num_games"],
            num_workers=ARGS["num_workers"],
            mcts_iters=ARGS["mcts_iters"],
            batch_size=ARGS["inference_batch_size"])
        eval_time = time() - t0

        println("=" ^ 70)
        println("Results (vs wildbg):")
        println("  White:    $(round(result.white_avg, digits=3))")
        println("  Black:    $(round(result.black_avg, digits=3))")
        println("  Combined: $(round(result.combined, digits=3))")
        println("  Win%:     $(round(result.win_pct, digits=1))%")
        println("  Games:    $(result.total_games)")
        println("  Time:     $(round(eval_time / 60, digits=1)) min")
        println("=" ^ 70)
    end
end

main()
