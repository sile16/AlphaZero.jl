#!/usr/bin/env julia
"""
Evaluate AlphaZero checkpoints against wildbg (Rust ONNX backgammon engine).

Uses BackgammonNet.jl's WildbgBackend + BackendAgent for opponent play,
and the AlphaZero network for our agent's play via batched MCTS.

Supports both single-model and dual-model (contact + race) architectures.
Dual-model sessions are auto-detected by the presence of contact_latest.data.

Tracks NN value prediction error vs wildbg equity at every AZ decision point,
reported separately for contact and race positions.

Optionally uses GPU (Metal.jl) for a subset of workers (--gpu-workers).

Usage:
    julia --threads 30 --project scripts/eval_vs_wildbg.jl <checkpoint> [options...]

    # Single checkpoint (CPU only):
    julia --threads 30 --project scripts/eval_vs_wildbg.jl /path/to/latest.data --num-workers=24

    # With GPU workers (24 CPU + 6 GPU on M3 Max):
    julia --threads 30 --project scripts/eval_vs_wildbg.jl /path/to/latest.data \\
        --num-workers=24 --gpu-workers=6

    # Dual-model checkpoint (auto-detected):
    julia --threads 30 --project scripts/eval_vs_wildbg.jl /path/to/contact_latest.data

    # Batch eval (multiple checkpoints):
    julia --threads 30 --project scripts/eval_vs_wildbg.jl --batch /path/to/sessions/

Options:
    --obs-type=minimal_flat  Observation type (default: minimal_flat)
    --num-games=500        Games per side (total = 2x this)
    --width=128            Network width (single model or contact model)
    --blocks=3             Network blocks (single model or contact model)
    --race-width=128       Race network width (dual-model only)
    --race-blocks=3        Race network blocks (dual-model only)
    --num-workers=24       CPU worker threads
    --gpu-workers=0        GPU worker threads (Metal.jl, macOS only)
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
            default = "minimal_flat"
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
        "--race-width"
            help = "Race network width (dual-model)"
            arg_type = Int
            default = 128
        "--race-blocks"
            help = "Race network blocks (dual-model)"
            arg_type = Int
            default = 3
        "--num-workers"
            help = "CPU worker threads"
            arg_type = Int
            default = 24
        "--gpu-workers"
            help = "GPU worker threads (Metal.jl, macOS only)"
            arg_type = Int
            default = 0
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
using Printf

# BackgammonNet provides game + wildbg backend
using BackgammonNet

# Try to load Metal.jl for GPU support
const HAS_METAL = try
    @eval using Metal
    Metal.current_device()  # Verify actual GPU availability
    true
catch
    false
end

# Set up game
ENV["BACKGAMMON_OBS_TYPE"] = ARGS["obs_type"]
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
const gspec = GameSpec()
const NUM_ACTIONS = GI.num_actions(gspec)
const _state_dim = GI.state_dim(gspec)[1]

# ── Wildbg Library ────────────────────────────────────────────────────────

function find_wildbg_lib()
    if !isempty(ARGS["wildbg_lib"])
        return ARGS["wildbg_lib"]
    end
    candidates = [
        joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg.so"),
        joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg.dylib"),
        joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg_main.so"),
        joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg_main.dylib"),
        "/usr/local/lib/libwildbg.so",
        "/usr/local/lib/libwildbg.dylib",
    ]
    for c in candidates
        isfile(c) && return c
    end
    error("libwildbg not found. Pass --wildbg-lib=/path/to/libwildbg.so")
end

# ── Network Forward (CPU) ────────────────────────────────────────────────

function _forward_network(net, states, gspec)
    n = length(states)
    X = zeros(Float32, _state_dim, n)
    A = zeros(Float32, NUM_ACTIONS, n)
    for (i, s) in enumerate(states)
        v = GI.vectorize_state(gspec, s)
        X[:, i] .= vec(v)
        if !BackgammonNet.game_terminated(s)
            for action in BackgammonNet.legal_actions(s)
                if 1 <= action <= NUM_ACTIONS
                    A[action, i] = 1.0f0
                end
            end
        end
    end
    P_raw, V, _ = Network.convert_output_tuple(
        net, Network.forward_normalized(net, X, A))
    results = Vector{Tuple{Vector{Float32}, Float32}}(undef, n)
    for i in 1:n
        legal = @view(A[:, i]) .> 0
        results[i] = (P_raw[legal, i], V[1, i])
    end
    return results
end

# ── GPU Forward (Metal, lock-based) ──────────────────────────────────────

const GPU_LOCK = ReentrantLock()

function _forward_network_gpu(net_gpu, states, gspec, gpu_array_fn, sync_fn)
    n = length(states)
    X = zeros(Float32, _state_dim, n)
    A = zeros(Float32, NUM_ACTIONS, n)
    for (i, s) in enumerate(states)
        v = GI.vectorize_state(gspec, s)
        X[:, i] .= vec(v)
        if !BackgammonNet.game_terminated(s)
            for action in BackgammonNet.legal_actions(s)
                if 1 <= action <= NUM_ACTIONS
                    A[action, i] = 1.0f0
                end
            end
        end
    end
    local Pr_cpu, V_cpu
    lock(GPU_LOCK) do
        X_g = gpu_array_fn(X)
        A_g = gpu_array_fn(A)
        result = Network.forward_normalized(net_gpu, X_g, A_g)
        sync_fn()
        Pr_cpu = Array(result[1])
        V_cpu = Array(result[2])
    end
    A_bool = A .> 0
    results = Vector{Tuple{Vector{Float32}, Float32}}(undef, n)
    for i in 1:n
        results[i] = (Pr_cpu[@view(A_bool[:, i]), i], V_cpu[1, i])
    end
    return results
end

# ── Oracle Factories ─────────────────────────────────────────────────────

function make_cpu_oracles(contact_net, race_net)
    function batch_oracle(states::Vector)
        n = length(states)
        n == 0 && return Tuple{Vector{Float32}, Float32}[]
        if race_net === nothing
            return _forward_network(contact_net, states, gspec)
        end
        contact_states, contact_idxs = eltype(states)[], Int[]
        race_states, race_idxs = eltype(states)[], Int[]
        for (i, s) in enumerate(states)
            if s isa BackgammonNet.BackgammonGame && !BackgammonNet.is_contact_position(s)
                push!(race_states, s); push!(race_idxs, i)
            else
                push!(contact_states, s); push!(contact_idxs, i)
            end
        end
        results = Vector{Tuple{Vector{Float32}, Float32}}(undef, n)
        if !isempty(contact_states)
            cr = _forward_network(contact_net, contact_states, gspec)
            for (j, idx) in enumerate(contact_idxs); results[idx] = cr[j]; end
        end
        if !isempty(race_states)
            rr = _forward_network(race_net, race_states, gspec)
            for (j, idx) in enumerate(race_idxs); results[idx] = rr[j]; end
        end
        results
    end
    single_oracle(s) = batch_oracle([s])[1]
    single_oracle, batch_oracle
end

function make_gpu_oracles(cn_gpu, rn_gpu, gpu_array_fn, sync_fn)
    function batch_oracle(states::Vector)
        n = length(states)
        n == 0 && return Tuple{Vector{Float32}, Float32}[]
        if rn_gpu === nothing
            return _forward_network_gpu(cn_gpu, states, gspec, gpu_array_fn, sync_fn)
        end
        contact_states, contact_idxs = eltype(states)[], Int[]
        race_states, race_idxs = eltype(states)[], Int[]
        for (i, s) in enumerate(states)
            if s isa BackgammonNet.BackgammonGame && !BackgammonNet.is_contact_position(s)
                push!(race_states, s); push!(race_idxs, i)
            else
                push!(contact_states, s); push!(contact_idxs, i)
            end
        end
        results = Vector{Tuple{Vector{Float32}, Float32}}(undef, n)
        if !isempty(contact_states)
            cr = _forward_network_gpu(cn_gpu, contact_states, gspec, gpu_array_fn, sync_fn)
            for (j, idx) in enumerate(contact_idxs); results[idx] = cr[j]; end
        end
        if !isempty(race_states)
            rr = _forward_network_gpu(rn_gpu, race_states, gspec, gpu_array_fn, sync_fn)
            for (j, idx) in enumerate(race_idxs); results[idx] = rr[j]; end
        end
        results
    end
    single_oracle(s) = batch_oracle([s])[1]
    single_oracle, batch_oracle
end

# ── Raw NN Value (for value error tracking, always CPU) ──────────────────

"""Get raw NN value prediction for a position. Returns (value, is_race)."""
function nn_raw_value(contact_net, race_net, g)
    is_race = g isa BackgammonNet.BackgammonGame && !BackgammonNet.is_contact_position(g)
    net = (is_race && race_net !== nothing) ? race_net : contact_net
    r = _forward_network(net, [g], gspec)
    _, v = r[1]
    return Float64(v), is_race
end

# ── AlphaZero Agent ──────────────────────────────────────────────────────

struct AlphaZeroAgent <: BackgammonNet.AbstractAgent
    single_oracle::Any
    batch_oracle::Any
    mcts_params::MctsParams
    batch_size::Int
    gspec::Any
end

function BackgammonNet.agent_move(agent::AlphaZeroAgent, g::BackgammonGame)
    env = GI.init(agent.gspec)
    env.game = BackgammonNet.clone(g)

    player = BatchedMCTS.BatchedMctsPlayer(
        agent.gspec, agent.single_oracle, agent.mcts_params;
        batch_size=agent.batch_size, batch_oracle=agent.batch_oracle)

    actions, policy = BatchedMCTS.think(player, env)
    BatchedMCTS.reset_player!(player)

    return actions[argmax(policy)]
end

# ── Value Stats ──────────────────────────────────────────────────────────

struct PositionValueSample
    nn_val::Float64
    wb_val::Float64
    is_contact::Bool
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

function print_value_stats(stats; label="")
    stats === nothing && return
    @printf("  %-12s | n=%5d | MSE=%.4f | MAE=%.4f | bias=%+.4f | corr=%.4f | NN=%.3f±%.3f | WB=%.3f±%.3f\n",
            label, stats.n, stats.mse, stats.mae, stats.bias, stats.corr,
            stats.nn_mean, stats.nn_std, stats.wb_mean, stats.wb_std)
end

# ── Game Play ────────────────────────────────────────────────────────────

"""Play a single eval game. Returns (reward, value_samples)."""
function eval_game(az_agent::AlphaZeroAgent, wildbg_agent::BackgammonNet.BackendAgent,
                   az_is_white::Bool; seed::Int=1,
                   contact_net=nothing, race_net=nothing)
    rng = MersenneTwister(seed)
    g = BackgammonNet.initial_state()
    value_samples = PositionValueSample[]

    while !BackgammonNet.game_terminated(g)
        if BackgammonNet.is_chance_node(g)
            BackgammonNet.sample_chance!(g, rng)
        else
            cp = Int(g.current_player)
            is_az_turn = (cp == 0) == az_is_white

            # Collect value comparison at AZ decision points
            if is_az_turn && contact_net !== nothing
                nn_v, is_race = nn_raw_value(contact_net, race_net, g)
                wb_v = Float64(BackgammonNet.evaluate(wildbg_agent.backend, g))
                # Both nn_v and wb_v are from current player's perspective
                push!(value_samples, PositionValueSample(nn_v, wb_v, !is_race))
            end

            agent = is_az_turn ? az_agent : wildbg_agent
            action = BackgammonNet.agent_move(agent, g)
            BackgammonNet.apply_action!(g, action)
        end
    end

    white_reward = Float64(g.reward)
    az_reward = az_is_white ? white_reward : -white_reward
    return (reward=az_reward, value_samples=value_samples)
end

# ── Evaluate Checkpoint ──────────────────────────────────────────────────

function evaluate_checkpoint(checkpoint_path::String, wildbg_lib::String;
                             width::Int, blocks::Int, num_games::Int,
                             num_workers::Int, mcts_iters::Int, batch_size::Int,
                             gpu_workers::Int=0,
                             race_checkpoint_path::Union{String,Nothing}=nothing,
                             race_width::Int=128, race_blocks::Int=3)
    # Load contact/main network (CPU)
    contact_network = FluxLib.FCResNetMultiHead(
        gspec, FluxLib.FCResNetMultiHeadHP(width=width, num_blocks=blocks))
    FluxLib.load_weights(checkpoint_path, contact_network)
    contact_network = Flux.cpu(contact_network)

    # Load race network if dual-model (CPU)
    race_network = nothing
    if race_checkpoint_path !== nothing
        race_network = FluxLib.FCResNetMultiHead(
            gspec, FluxLib.FCResNetMultiHeadHP(width=race_width, num_blocks=race_blocks))
        FluxLib.load_weights(race_checkpoint_path, race_network)
        race_network = Flux.cpu(race_network)
    end

    mcts_params = MctsParams(
        num_iters_per_turn=mcts_iters,
        cpuct=1.5,
        temperature=ConstSchedule(0.0),
        dirichlet_noise_ϵ=0.0,
        dirichlet_noise_α=1.0)

    # Create CPU oracles and agent
    cpu_single, cpu_batch = make_cpu_oracles(contact_network, race_network)
    cpu_agent = AlphaZeroAgent(cpu_single, cpu_batch, mcts_params, batch_size, gspec)

    # Create GPU oracles and agent (if requested and available)
    gpu_agent = nothing
    if gpu_workers > 0
        if !HAS_METAL
            println("  WARNING: --gpu-workers=$gpu_workers but Metal.jl not available, using CPU only")
            gpu_workers = 0
        else
            println("  Metal device: $(Metal.current_device())")
            cn_gpu = Flux.gpu(contact_network)
            rn_gpu = race_network !== nothing ? Flux.gpu(race_network) : nothing

            # Warmup GPU
            X_w = Metal.MtlArray(randn(Float32, _state_dim, 10))
            A_w = Metal.MtlArray(zeros(Float32, NUM_ACTIONS, 10))
            Network.forward_normalized(cn_gpu, X_w, A_w)
            Metal.synchronize()
            println("  GPU warmed up")

            gpu_single, gpu_batch = make_gpu_oracles(cn_gpu, rn_gpu, Metal.MtlArray, Metal.synchronize)
            gpu_agent = AlphaZeroAgent(gpu_single, gpu_batch, mcts_params, batch_size, gspec)
        end
    end

    total_workers = num_workers + gpu_workers

    # Initialize per-thread wildbg backends
    lib_size = filesize(wildbg_lib)
    nets_variant = lib_size > 10_000_000 ? :large : :small
    if nets_variant == :large
        BackgammonNet.wildbg_set_lib_path!(large=wildbg_lib)
    else
        BackgammonNet.wildbg_set_lib_path!(small=wildbg_lib)
    end
    println("  wildbg nets: $nets_variant (lib $(round(lib_size/1e6, digits=1))MB)")
    println("  Workers: $num_workers CPU" * (gpu_workers > 0 ? " + $gpu_workers GPU" : ""))
    wildbg_agents = [begin
        wb = BackgammonNet.WildbgBackend(nets=nets_variant)
        BackgammonNet.open!(wb)
        BackgammonNet.BackendAgent(wb)
    end for _ in 1:total_workers]

    games_per_side = num_games

    # Per-game storage for value samples
    white_rewards = Vector{Float64}(undef, games_per_side)
    white_vsamples = Vector{Vector{PositionValueSample}}(undef, games_per_side)
    black_rewards = Vector{Float64}(undef, games_per_side)
    black_vsamples = Vector{Vector{PositionValueSample}}(undef, games_per_side)

    # Play as white
    println("  Playing $games_per_side games as white ($total_workers workers)...")
    flush(stdout)
    white_claimed = Threads.Atomic{Int}(0)
    Threads.@threads for tid in 1:total_workers
        wa = wildbg_agents[tid]
        agent = (gpu_agent !== nothing && tid > num_workers) ? gpu_agent : cpu_agent
        while true
            i = Threads.atomic_add!(white_claimed, 1) + 1
            i > games_per_side && break
            result = eval_game(agent, wa, true; seed=i,
                               contact_net=contact_network, race_net=race_network)
            white_rewards[i] = result.reward
            white_vsamples[i] = result.value_samples
        end
    end

    # Play as black
    println("  Playing $games_per_side games as black ($total_workers workers)...")
    flush(stdout)
    black_claimed = Threads.Atomic{Int}(0)
    Threads.@threads for tid in 1:total_workers
        wa = wildbg_agents[tid]
        agent = (gpu_agent !== nothing && tid > num_workers) ? gpu_agent : cpu_agent
        while true
            i = Threads.atomic_add!(black_claimed, 1) + 1
            i > games_per_side && break
            result = eval_game(agent, wa, false; seed=i + games_per_side,
                               contact_net=contact_network, race_net=race_network)
            black_rewards[i] = result.reward
            black_vsamples[i] = result.value_samples
        end
    end

    for wa in wildbg_agents
        BackgammonNet.close(wa.backend)
    end

    # Aggregate game results
    white_avg = mean(white_rewards)
    black_avg = mean(black_rewards)
    combined = (white_avg + black_avg) / 2
    total_games = 2 * games_per_side
    win_count = count(r -> r > 0, white_rewards) + count(r -> r > 0, black_rewards)
    win_pct = 100 * win_count / total_games

    # Aggregate value stats
    all_samples = PositionValueSample[]
    for vs in white_vsamples; append!(all_samples, vs); end
    for vs in black_vsamples; append!(all_samples, vs); end

    contact_samples = filter(s -> s.is_contact, all_samples)
    race_samples = filter(s -> !s.is_contact, all_samples)

    all_stats = compute_value_stats(all_samples)
    contact_stats = compute_value_stats(contact_samples)
    race_stats = compute_value_stats(race_samples)

    return (white_avg=white_avg, black_avg=black_avg, combined=combined,
            total_games=total_games, win_pct=win_pct,
            all_stats=all_stats, contact_stats=contact_stats, race_stats=race_stats)
end

# ── Checkpoint Discovery ─────────────────────────────────────────────────

"""Find all interesting checkpoints in a sessions directory."""
function find_checkpoints(sessions_dir::String)
    checkpoints = Tuple{String, String, Union{String,Nothing}, Int}[]

    for entry in readdir(sessions_dir)
        session_path = joinpath(sessions_dir, entry)
        isdir(session_path) || continue

        ckpt_dir = joinpath(session_path, "checkpoints")
        isdir(ckpt_dir) || continue

        contact_ckpt = joinpath(ckpt_dir, "contact_latest.data")
        race_ckpt = joinpath(ckpt_dir, "race_latest.data")
        latest = joinpath(ckpt_dir, "latest.data")

        contact_path = nothing
        race_path = nothing
        if isfile(contact_ckpt)
            contact_path = contact_ckpt
            race_path = isfile(race_ckpt) ? race_ckpt : nothing
        elseif isfile(latest)
            contact_path = latest
        else
            continue
        end

        iter_file = joinpath(ckpt_dir, "iter.txt")
        iters = if isfile(iter_file)
            parse(Int, strip(read(iter_file, String)))
        else
            iter_files = filter(f -> occursin(r"^(contact_)?iter_\d+\.data$", f), readdir(ckpt_dir))
            if isempty(iter_files)
                0
            else
                maximum(parse(Int, match(r"(\d+)", f).captures[1]) for f in iter_files)
            end
        end

        iters < 10 && continue
        push!(checkpoints, (entry, contact_path, race_path, iters))
    end

    sort!(checkpoints, by=x -> x[4], rev=true)
    return checkpoints
end

"""Detect network architecture from checkpoint."""
function detect_architecture(checkpoint_path::String)
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

"""Detect race network architecture from checkpoint."""
function detect_race_architecture(checkpoint_path::String)
    for (w, b) in [(128, 3), (64, 2), (256, 5)]
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

# ── Main ─────────────────────────────────────────────────────────────────

function main()
    wildbg_lib = find_wildbg_lib()
    println("wildbg library: $wildbg_lib")
    HAS_METAL && println("Metal.jl loaded: $(Metal.current_device())")

    gpu_workers = ARGS["gpu_workers"]

    if ARGS["batch"]
        sessions_dir = ARGS["checkpoint"]
        println("Batch evaluation of sessions in: $sessions_dir")
        println("=" ^ 70)

        checkpoints = find_checkpoints(sessions_dir)
        println("Found $(length(checkpoints)) checkpoints with 10+ iterations:\n")
        for (name, cpath, rpath, iters) in checkpoints
            dual = rpath !== nothing ? " [dual]" : ""
            println("  [$iters iter$dual] $name")
        end
        println()

        results = []
        for (name, ckpt_path, race_ckpt_path, iters) in checkpoints
            println("=" ^ 70)
            println("Evaluating: $name ($iters iterations)")
            println("  Checkpoint: $ckpt_path")

            arch = detect_architecture(ckpt_path)
            if arch === nothing
                println("  SKIP: Could not detect contact/main architecture")
                continue
            end

            race_arch = nothing
            if race_ckpt_path !== nothing
                race_arch = detect_race_architecture(race_ckpt_path)
                if race_arch === nothing
                    println("  SKIP: Could not detect race architecture")
                    continue
                end
                println("  Architecture: contact=$(arch.width)w×$(arch.blocks)b + race=$(race_arch.width)w×$(race_arch.blocks)b")
            else
                println("  Architecture: $(arch.width)w×$(arch.blocks)b")
            end
            flush(stdout)

            t0 = time()
            local result
            try
                result = evaluate_checkpoint(ckpt_path, wildbg_lib;
                    width=arch.width, blocks=arch.blocks,
                    num_games=ARGS["num_games"],
                    num_workers=ARGS["num_workers"],
                    gpu_workers=gpu_workers,
                    mcts_iters=ARGS["mcts_iters"],
                    batch_size=ARGS["inference_batch_size"],
                    race_checkpoint_path=race_ckpt_path,
                    race_width=race_arch !== nothing ? race_arch.width : 128,
                    race_blocks=race_arch !== nothing ? race_arch.blocks : 3)
            catch e
                println("  ERROR: $e")
                flush(stdout)
                continue
            end
            eval_time = time() - t0

            arch_str = race_arch !== nothing ?
                "$(arch.width)w×$(arch.blocks)b+$(race_arch.width)w×$(race_arch.blocks)b" :
                "$(arch.width)w×$(arch.blocks)b"

            println("  White:    $(round(result.white_avg, digits=3))")
            println("  Black:    $(round(result.black_avg, digits=3))")
            println("  Combined: $(round(result.combined, digits=3))")
            println("  Win%:     $(round(result.win_pct, digits=1))%")
            println("  Games:    $(result.total_games)")
            println("  Time:     $(round(eval_time / 60, digits=1)) min")

            # Print value stats
            println("  Value Error vs Wildbg:")
            print_value_stats(result.all_stats; label="All")
            print_value_stats(result.contact_stats; label="Contact")
            print_value_stats(result.race_stats; label="Race")
            flush(stdout)

            push!(results, (name=name, iters=iters, arch_str=arch_str,
                           combined=result.combined, win_pct=result.win_pct,
                           white=result.white_avg, black=result.black_avg,
                           time_min=eval_time/60,
                           all_stats=result.all_stats,
                           contact_stats=result.contact_stats,
                           race_stats=result.race_stats))
        end

        # Summary table
        println("\n" * "=" ^ 70)
        println("BATCH EVALUATION SUMMARY (vs wildbg)")
        println("=" ^ 70)
        sort!(results, by=r -> r.combined, rev=true)
        println("Rank | Equity  | Win%  | Arch              | Iters | Session")
        println("-----|---------|-------|-------------------|-------|--------")
        for (rank, r) in enumerate(results)
            println("  $(lpad(rank, 2)) | $(lpad(round(r.combined, digits=3), 7)) | $(lpad(round(r.win_pct, digits=1), 5))% | $(rpad(r.arch_str, 17)) | $(lpad(r.iters, 5)) | $(r.name)")
        end

        # Value stats summary
        println("\nVALUE ERROR SUMMARY (NN vs Wildbg)")
        println("-" ^ 70)
        @printf("  %-30s | %-6s | %-7s | %-7s | %-7s | %-7s\n",
                "Session", "Model", "MSE", "MAE", "Bias", "Corr")
        @printf("  %-30s |--------|---------|---------|---------|--------\n", "")
        for r in results
            name_short = length(r.name) > 30 ? r.name[end-29:end] : r.name
            if r.contact_stats !== nothing
                @printf("  %-30s | %-6s | %7.4f | %7.4f | %+7.4f | %7.4f\n",
                        name_short, "Contct", r.contact_stats.mse, r.contact_stats.mae,
                        r.contact_stats.bias, r.contact_stats.corr)
            end
            if r.race_stats !== nothing
                @printf("  %-30s | %-6s | %7.4f | %7.4f | %+7.4f | %7.4f\n",
                        "", "Race", r.race_stats.mse, r.race_stats.mae,
                        r.race_stats.bias, r.race_stats.corr)
            end
        end
        println("=" ^ 70)

        # Save results
        results_path = joinpath(sessions_dir, "wildbg_eval_results_$(Dates.format(now(), "yyyymmdd_HHMMSS")).txt")
        open(results_path, "w") do f
            println(f, "# Batch Evaluation vs wildbg")
            println(f, "# Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
            println(f, "# Games per side: $(ARGS["num_games"])")
            println(f, "# MCTS iters: $(ARGS["mcts_iters"])")
            println(f, "# Workers: $(ARGS["num_workers"]) CPU + $(gpu_workers) GPU")
            println(f, "")
            println(f, "# name\titers\tarch\tequity\twin_pct\twhite\tblack\tmse_all\tmae_all\tcorr_all\tmse_contact\tmae_contact\tcorr_contact\tmse_race\tmae_race\tcorr_race")
            for r in results
                as = r.all_stats
                cs = r.contact_stats
                rs = r.race_stats
                println(f, join([
                    r.name, r.iters, r.arch_str,
                    round(r.combined, digits=4), round(r.win_pct, digits=2),
                    round(r.white, digits=4), round(r.black, digits=4),
                    as !== nothing ? round(as.mse, digits=6) : "NA",
                    as !== nothing ? round(as.mae, digits=6) : "NA",
                    as !== nothing ? round(as.corr, digits=6) : "NA",
                    cs !== nothing ? round(cs.mse, digits=6) : "NA",
                    cs !== nothing ? round(cs.mae, digits=6) : "NA",
                    cs !== nothing ? round(cs.corr, digits=6) : "NA",
                    rs !== nothing ? round(rs.mse, digits=6) : "NA",
                    rs !== nothing ? round(rs.mae, digits=6) : "NA",
                    rs !== nothing ? round(rs.corr, digits=6) : "NA",
                ], "\t"))
            end
        end
        println("Results saved to: $results_path")

    else
        # Single checkpoint eval
        ckpt_path = ARGS["checkpoint"]
        race_ckpt_path = nothing

        # Auto-detect dual-model
        ckpt_dir = dirname(ckpt_path)
        ckpt_name = basename(ckpt_path)
        if startswith(ckpt_name, "contact_")
            race_name = replace(ckpt_name, "contact_" => "race_")
            race_candidate = joinpath(ckpt_dir, race_name)
            if isfile(race_candidate)
                race_ckpt_path = race_candidate
            end
        elseif ckpt_name == "latest.data"
            contact_candidate = joinpath(ckpt_dir, "contact_latest.data")
            race_candidate = joinpath(ckpt_dir, "race_latest.data")
            if isfile(contact_candidate) && isfile(race_candidate)
                ckpt_path = contact_candidate
                race_ckpt_path = race_candidate
            end
        end

        width = ARGS["width"]
        blocks = ARGS["blocks"]
        race_width = ARGS["race_width"]
        race_blocks = ARGS["race_blocks"]

        if race_ckpt_path !== nothing
            println("Evaluating (dual-model): $ckpt_path")
            println("  Race checkpoint: $race_ckpt_path")
            println("Architecture: contact=$(width)w×$(blocks)b + race=$(race_width)w×$(race_blocks)b")
        else
            println("Evaluating: $ckpt_path")
            println("Architecture: $(width)w×$(blocks)b")
        end
        println("Games: $(ARGS["num_games"]) per side")
        println("MCTS: $(ARGS["mcts_iters"]) iterations")
        println("Workers: $(ARGS["num_workers"]) CPU" * (gpu_workers > 0 ? " + $gpu_workers GPU" : ""))
        println("=" ^ 70)
        flush(stdout)

        t0 = time()
        result = evaluate_checkpoint(ckpt_path, wildbg_lib;
            width=width, blocks=blocks,
            num_games=ARGS["num_games"],
            num_workers=ARGS["num_workers"],
            gpu_workers=gpu_workers,
            mcts_iters=ARGS["mcts_iters"],
            batch_size=ARGS["inference_batch_size"],
            race_checkpoint_path=race_ckpt_path,
            race_width=race_width, race_blocks=race_blocks)
        eval_time = time() - t0

        println("=" ^ 70)
        println("Results (vs wildbg):")
        println("  White:    $(round(result.white_avg, digits=3))")
        println("  Black:    $(round(result.black_avg, digits=3))")
        println("  Combined: $(round(result.combined, digits=3))")
        println("  Win%:     $(round(result.win_pct, digits=1))%")
        println("  Games:    $(result.total_games)")
        println("  Time:     $(round(eval_time / 60, digits=1)) min")

        println("\nValue Error (NN vs Wildbg):")
        print_value_stats(result.all_stats; label="All")
        print_value_stats(result.contact_stats; label="Contact")
        print_value_stats(result.race_stats; label="Race")
        println("=" ^ 70)
    end
end

main()
