#!/usr/bin/env julia
"""
Evaluate AlphaZero checkpoints against GNU Backgammon (gnubg).

Adapted from scripts/eval_vs_wildbg.jl. Uses BackgammonNet.jl's
`GnubgCLibBackend` (in-process gnubg bridge) + ExternalAgent for opponent play,
and the AlphaZero network for our agent's play via batched MCTS.

Supports both single-model and dual-model (contact + race) architectures.
Dual-model sessions are auto-detected by the presence of contact_latest.data.

Tracks NN value prediction error vs gnubg equity at every AZ decision point,
reported separately for contact and race positions.

GNUbg ply is the search depth: ply-0 = 1-ply neural net (fast, ~wildbg strength),
ply-2 = 2-ply (much stronger, near-world-class, but SLOWER per move).

NOTE: the gnubg C bridge is a single shared, reference-counted runtime (threads=1)
and every gnubg call is globally serialized by an internal lock. Per-worker backend
instances all resolve to that same runtime, so gnubg moves are thread-safe but do
not run concurrently; parallelism still benefits the AZ (MCTS) side.

IMPORTANT (ply >= 2): gnubg's multi-ply evaluator spins up its OWN internal worker
thread pool, which can DEADLOCK (all pool threads stuck in pthread_cond_wait) on
certain positions. This is a gnubg-internal 2-ply threading bug, independent of the
number of Julia workers: it is hit far more often under many Julia workers (a 40-game
run hangs in minutes), and forcing --num-workers 1 greatly reduces — but does NOT fully
eliminate — the hang (an 80-game single-worker run still eventually locked up; a 6-game
one completed). ply-0 does not use that path and runs fine with many workers. To get
ply>=2 numbers reliably, run small single-worker batches (e.g. <=10 games/side) and
aggregate the ones that complete; expect ~6-7 games/min (gnubg is globally serialized,
so worker count barely affects throughput anyway).

Optionally uses GPU (Metal.jl) for a subset of workers (--gpu-workers).

Usage:
    julia --threads 14 --project scripts/eval_vs_gnubg.jl <checkpoint> [options...]

    # Dual-model checkpoint (auto-detected) vs gnubg ply-0:
    julia --threads 14 --project scripts/eval_vs_gnubg.jl \\
        sessions/contact-flywheel/checkpoints/contact_iter_140.data \\
        --obs-type min_plus_flat --width 256 --blocks 5 --race-width 128 --race-blocks 3 \\
        --num-games 100 --num-workers 12 --mcts-iters 800 --chance-mode passthrough --gnubg-ply 0

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
    --gnubg-ply=0          gnubg search ply (0 = 1-ply NN, 2 = 2-ply, up to 7)
    --batch                Batch mode: eval all latest.data in session dir
    --allow-race-checkpoint
                           Allow full-game eval of a race-only checkpoint
"""

using ArgParse

function parse_eval_args()
    s = ArgParseSettings(description="Evaluate against gnubg", autofix_names=true)

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
        "--gnubg-ply"
            help = "gnubg search ply (0 = 1-ply NN, 2 = 2-ply stronger, range 0:7). " *
                   "IMPORTANT: ply >= 2 deadlocks under multiple workers — use --num-workers 1."
            arg_type = Int
            default = 0
        "--batch"
            help = "Batch mode: eval all checkpoints in session directory"
            action = :store_true
        "--allow-race-checkpoint"
            help = "Allow evaluating a race-only checkpoint from full-game opening positions"
            action = :store_true
        "--inference-batch-size"
            help = "Inference batch size for MCTS"
            arg_type = Int
            default = 50
        "--inference-backend"
            help = "CPU inference backend: auto, fast, or flux"
            arg_type = String
            default = "auto"
        "--chance-mode"
            help = "Chance-node handling: passthrough (default) or exact_expectation. NOTE:
                    exact_expectation measured WORSE at deep search (46% vs 54% @800) — its
                    21-child expansions consume the iter budget, starving decision-tree depth
                    (the dominant strength lever). Kept for the record; passthrough is better."
            arg_type = String
            default = "passthrough"
    end

    return ArgParse.parse_args(s)
end

const ARGS = parse_eval_args()

function looks_like_race_only_checkpoint(path::AbstractString)::Bool
    base = lowercase(basename(path))
    dir = lowercase(dirname(path))

    return base == "race_latest.data" ||
           base == "race_train_latest.data" ||
           base == "race_best.data" ||
           occursin(r"^race_iter_\d+\.data$", base) ||
           (base == "latest.data" && occursin(r"(^|[/_-])race([/_-]|$)", dir))
end

function race_checkpoint_guard_message(path::AbstractString)::String
    return "Refusing full-game eval of likely race-only checkpoint: $path. " *
           "Use scripts/eval_race.jl for race checkpoints, pass contact_latest.data " *
           "for dual-model full-game eval, or override with --allow-race-checkpoint."
end

# Load packages
using AlphaZero
using AlphaZero: GI, MCTS, Network, FluxLib, MctsParams, ConstSchedule, BatchedMCTS, GameLoop
using AlphaZero.NetLib
import Flux
using Random
using Statistics
using Dates
using Printf
using Logging

# BackgammonNet provides game + gnubg backend
using BackgammonNet

# ── Fallback counter ─────────────────────────────────────────────────────
# The ExternalAgent path in AlphaZero.GameLoop falls back to a legal move when
# the backend can't produce one (for gnubg this happens on fully-blocked doubles
# positions where the engine must dance/pass — the fallback picks the only legal
# action, i.e. pass, so it is self-correcting). We wrap the eval in a logger that
# counts these warnings (bypassing their maxlog cap) so we can report the rate.
mutable struct _FallbackCountLogger <: AbstractLogger
    base::AbstractLogger
    n::Int
end
Logging.min_enabled_level(l::_FallbackCountLogger) = Logging.min_enabled_level(l.base)
Logging.shouldlog(l::_FallbackCountLogger, args...) = true
Logging.catch_exceptions(l::_FallbackCountLogger) = Logging.catch_exceptions(l.base)
function Logging.handle_message(l::_FallbackCountLogger, level, message, _module, group, id, file, line; kwargs...)
    if level == Logging.Warn && occursin("ExternalAgent move failed", string(message))
        l.n += 1
        # Count silently after the first few; the base logger still respects maxlog.
        l.n <= 3 && Logging.handle_message(l.base, level, message, _module, group, id, file, line; kwargs...)
        return
    end
    Logging.handle_message(l.base, level, message, _module, group, id, file, line; kwargs...)
end

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
const ORACLE_CFG = AlphaZero.BackgammonInference.OracleConfig(
    _state_dim, NUM_ACTIONS, gspec;
    vectorize_state! = vectorize_state_into!,
    route_state = s -> (s isa BackgammonNet.BackgammonGame && !BackgammonNet.is_contact_position(s) ? 2 : 1))

# ── GPU Forward (Metal, lock-based) ──────────────────────────────────────

const GPU_LOCK = ReentrantLock()

# ── Value Stats ──────────────────────────────────────────────────────────

struct PositionValueSample
    nn_val::Float64
    gnubg_val::Float64
    is_contact::Bool
end

function compute_value_stats(samples::Vector{PositionValueSample})
    isempty(samples) && return nothing

    nn = [s.nn_val for s in samples]
    gb = [s.gnubg_val for s in samples]

    mse = mean((nn .- gb) .^ 2)
    mae = mean(abs.(nn .- gb))
    bias = mean(nn) - mean(gb)
    corr = length(nn) >= 3 ? cor(nn, gb) : NaN

    return (n=length(samples), mse=mse, mae=mae, bias=bias, corr=corr,
            nn_mean=mean(nn), gb_mean=mean(gb), nn_std=std(nn), gb_std=std(gb))
end

function print_value_stats(stats; label="")
    stats === nothing && return
    @printf("  %-12s | n=%5d | MSE=%.4f | MAE=%.4f | bias=%+.4f | corr=%.4f | NN=%.3f±%.3f | GB=%.3f±%.3f\n",
            label, stats.n, stats.mse, stats.mae, stats.bias, stats.corr,
            stats.nn_mean, stats.nn_std, stats.gb_mean, stats.gb_std)
end

# ── Game Play ────────────────────────────────────────────────────────────

"""Play a single eval game using GameLoop.play_game(). Returns (reward, value_samples)."""
function eval_game(single_oracle, batch_oracle, gnubg_backend,
                   az_is_white::Bool; seed::Int=1,
                   value_batch_oracle=nothing,
                   gspec=nothing, mcts_params=nothing, batch_size::Int=50)
    rng = MersenneTwister(seed)

    # Initialize game from opening position
    env = GI.init(gspec)

    az = GameLoop.MctsAgent(single_oracle, batch_oracle, mcts_params, batch_size, gspec)
    gb = GameLoop.ExternalAgent(gnubg_backend)

    # Value comparison functions (only if value_batch_oracle provided)
    value_oracle_fn = nothing
    gnubg_value_fn = nothing
    if value_batch_oracle !== nothing
        # NN oracle V is normalized equity/3 ∈ [-1,1]; gnubg evaluate() returns
        # rule-aware equity in points. Scale NN back to raw points so
        # MSE/MAE/bias/corr compare like with like (in equity units).
        value_oracle_fn = env -> Float64(value_batch_oracle([env.game])[1][2]) * Float64(GI.reward_scale(gspec))
        gnubg_value_fn = env -> Float64(BackgammonNet.evaluate(gnubg_backend, env.game))
    end

    w, b = az_is_white ? (az, gb) : (gb, az)
    result = GameLoop.play_game(w, b, env;
        record_value_comparison=(value_batch_oracle !== nothing),
        value_oracle=value_oracle_fn,
        opponent_value_fn=gnubg_value_fn,
        rng=rng,
        temperature_fn=_ -> 0.0)

    az_reward = az_is_white ? result.reward : -result.reward
    value_samples = [PositionValueSample(s.nn_val, s.opponent_val, s.is_contact)
                     for s in result.value_samples]
    return (reward=az_reward, value_samples=value_samples)
end

# ── Evaluate Checkpoint ──────────────────────────────────────────────────

function evaluate_checkpoint(checkpoint_path::String;
                             width::Int, blocks::Int, num_games::Int,
                             num_workers::Int, mcts_iters::Int, batch_size::Int,
                             gnubg_ply::Int=0,
                             inference_backend::Union{Symbol, AbstractString}="auto",
                             gpu_workers::Int=0,
                             race_checkpoint_path::Union{String,Nothing}=nothing,
                             race_width::Int=128, race_blocks::Int=3,
                             chance_mode::Symbol=:passthrough)
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
        dirichlet_noise_α=1.0,
        # EVAL-ONLY: exact expectimax over dice outcomes at chance nodes (default).
        chance_mode=chance_mode)

    backend = AlphaZero.BackgammonInference.resolve_cpu_backend(inference_backend)
    println("  CPU inference: $(AlphaZero.BackgammonInference.cpu_backend_summary(backend))")

    cpu_single, cpu_batch = AlphaZero.BackgammonInference.make_cpu_oracles(
        backend, contact_network, ORACLE_CFG;
        secondary_net=race_network, batch_size=batch_size)
    _, value_batch_oracle = AlphaZero.BackgammonInference.make_cpu_oracles(
        :flux, contact_network, ORACLE_CFG;
        secondary_net=race_network, batch_size=1)

    # Create GPU oracles (if requested and available)
    gpu_single = nothing
    gpu_batch = nothing
    gpu_server = nothing
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

            gpu_single, gpu_batch, gpu_server = AlphaZero.BackgammonInference.make_gpu_server_oracles(
                cn_gpu, ORACLE_CFG;
                secondary_net_gpu=rn_gpu,
                batch_size=batch_size,
                num_workers=gpu_workers,
                gpu_array_fn=Metal.MtlArray,
                sync_fn=Metal.synchronize,
                gpu_lock=GPU_LOCK)
        end
    end

    total_workers = num_workers + gpu_workers

    # Initialize per-worker gnubg backends. The gnubg C bridge is a single shared,
    # reference-counted runtime (threads=1) guarded by an internal global lock, so
    # these instances all resolve to the same engine and gnubg calls are serialized.
    println("  gnubg ply: $gnubg_ply")
    println("  Workers: $num_workers CPU" * (gpu_workers > 0 ? " + $gpu_workers GPU" : ""))
    gnubg_backends = [begin
        gb = BackgammonNet.GnubgCLibBackend(ply=gnubg_ply, threads=1)
        BackgammonNet.open!(gb)
        gb
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
        gb = gnubg_backends[tid]
        use_gpu = gpu_single !== nothing && tid > num_workers
        so = use_gpu ? gpu_single : cpu_single
        bo = use_gpu ? gpu_batch : cpu_batch
        while true
            i = Threads.atomic_add!(white_claimed, 1) + 1
            i > games_per_side && break
            result = eval_game(so, bo, gb, true; seed=i,
                               value_batch_oracle=value_batch_oracle,
                               gspec=gspec, mcts_params=mcts_params, batch_size=batch_size)
            white_rewards[i] = result.reward
            white_vsamples[i] = result.value_samples
        end
    end

    # Play as black
    println("  Playing $games_per_side games as black ($total_workers workers)...")
    flush(stdout)
    black_claimed = Threads.Atomic{Int}(0)
    Threads.@threads for tid in 1:total_workers
        gb = gnubg_backends[tid]
        use_gpu = gpu_single !== nothing && tid > num_workers
        so = use_gpu ? gpu_single : cpu_single
        bo = use_gpu ? gpu_batch : cpu_batch
        while true
            i = Threads.atomic_add!(black_claimed, 1) + 1
            i > games_per_side && break
            result = eval_game(so, bo, gb, false; seed=i + games_per_side,
                               value_batch_oracle=value_batch_oracle,
                               gspec=gspec, mcts_params=mcts_params, batch_size=batch_size)
            black_rewards[i] = result.reward
            black_vsamples[i] = result.value_samples
        end
    end

    for gb in gnubg_backends
        close(gb)
    end
    gpu_single !== nothing && close(gpu_server)

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
    gnubg_ply = ARGS["gnubg_ply"]
    println("gnubg backend: GnubgCLibBackend(ply=$gnubg_ply)")
    HAS_METAL && println("Metal.jl loaded: $(Metal.current_device())")

    # ply >= 2 deadlocks under multiple worker threads (gnubg's multi-ply evaluator
    # is not safe to invoke from varying OS threads). Force single worker.
    if gnubg_ply >= 2 && ARGS["num_workers"] > 1
        @warn "gnubg ply=$gnubg_ply can deadlock in its internal 2-ply thread pool; forcing " *
              "--num-workers=1 to greatly reduce (not fully eliminate) the hang. Prefer small " *
              "game batches at ply>=2. gnubg is globally serialized, so throughput is ~unchanged."
        ARGS["num_workers"] = 1
        ARGS["gpu_workers"] = 0
    end

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

            if race_ckpt_path === nothing &&
               !ARGS["allow_race_checkpoint"] &&
               looks_like_race_only_checkpoint(ckpt_path)
                println("  SKIP: $(race_checkpoint_guard_message(ckpt_path))")
                flush(stdout)
                continue
            end

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
                result = evaluate_checkpoint(ckpt_path;
                    width=arch.width, blocks=arch.blocks,
                    num_games=ARGS["num_games"],
                    num_workers=ARGS["num_workers"],
                    gpu_workers=gpu_workers,
                    mcts_iters=ARGS["mcts_iters"],
                    gnubg_ply=gnubg_ply,
                    chance_mode=Symbol(ARGS["chance_mode"]),
                    batch_size=ARGS["inference_batch_size"],
                    inference_backend=ARGS["inference_backend"],
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
            println("  Time:     $(round(eval_time, digits=2)) s ($(round(eval_time / 60, digits=3)) min)")
            println("  Rate:     $(round(result.total_games / (eval_time / 60), digits=1)) games/min")

            # Print value stats
            println("  Value Error vs gnubg:")
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
        println("BATCH EVALUATION SUMMARY (vs gnubg ply-$gnubg_ply)")
        println("=" ^ 70)
        sort!(results, by=r -> r.combined, rev=true)
        println("Rank | Equity  | Win%  | Arch              | Iters | Session")
        println("-----|---------|-------|-------------------|-------|--------")
        for (rank, r) in enumerate(results)
            println("  $(lpad(rank, 2)) | $(lpad(round(r.combined, digits=3), 7)) | $(lpad(round(r.win_pct, digits=1), 5))% | $(rpad(r.arch_str, 17)) | $(lpad(r.iters, 5)) | $(r.name)")
        end

        # Value stats summary
        println("\nVALUE ERROR SUMMARY (NN vs gnubg)")
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
        results_path = joinpath(sessions_dir, "gnubg_eval_results_$(Dates.format(now(), "yyyymmdd_HHMMSS")).txt")
        open(results_path, "w") do f
            println(f, "# Batch Evaluation vs gnubg")
            println(f, "# Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
            println(f, "# gnubg ply: $gnubg_ply")
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
            elseif isfile(joinpath(ckpt_dir, "race_latest.data"))
                # Fall back to race_latest.data copied into the checkpoints dir.
                race_ckpt_path = joinpath(ckpt_dir, "race_latest.data")
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

        if race_ckpt_path === nothing &&
           !ARGS["allow_race_checkpoint"] &&
           looks_like_race_only_checkpoint(ckpt_path)
            error(race_checkpoint_guard_message(ckpt_path))
        end

        if race_ckpt_path !== nothing
            println("Evaluating (dual-model): $ckpt_path")
            println("  Race checkpoint: $race_ckpt_path")
            println("Architecture: contact=$(width)w×$(blocks)b + race=$(race_width)w×$(race_blocks)b")
        else
            println("Evaluating: $ckpt_path")
            println("Architecture: $(width)w×$(blocks)b")
        end
        println("Opponent: gnubg ply-$gnubg_ply")
        println("Games: $(ARGS["num_games"]) per side")
        println("MCTS: $(ARGS["mcts_iters"]) iterations")
        println("Workers: $(ARGS["num_workers"]) CPU" * (gpu_workers > 0 ? " + $gpu_workers GPU" : ""))
        println("CPU inference: $(AlphaZero.BackgammonInference.cpu_backend_summary(ARGS["inference_backend"]))")
        println("=" ^ 70)
        flush(stdout)

        t0 = time()
        result = evaluate_checkpoint(ckpt_path;
            width=width, blocks=blocks,
            num_games=ARGS["num_games"],
            num_workers=ARGS["num_workers"],
            gpu_workers=gpu_workers,
            mcts_iters=ARGS["mcts_iters"],
            gnubg_ply=gnubg_ply,
            chance_mode=Symbol(ARGS["chance_mode"]),
            batch_size=ARGS["inference_batch_size"],
            inference_backend=ARGS["inference_backend"],
            race_checkpoint_path=race_ckpt_path,
            race_width=race_width, race_blocks=race_blocks)
        eval_time = time() - t0

        println("=" ^ 70)
        println("Results (vs gnubg ply-$gnubg_ply):")
        println("  White:    $(round(result.white_avg, digits=3))")
        println("  Black:    $(round(result.black_avg, digits=3))")
        println("  Combined: $(round(result.combined, digits=3))")
        println("  Win%:     $(round(result.win_pct, digits=1))%")
        println("  Games:    $(result.total_games)")
        println("  Time:     $(round(eval_time, digits=2)) s ($(round(eval_time / 60, digits=3)) min)")
        println("  Rate:     $(round(result.total_games / (eval_time / 60), digits=1)) games/min")

        println("\nValue Error (NN vs gnubg):")
        print_value_stats(result.all_stats; label="All")
        print_value_stats(result.contact_stats; label="Contact")
        print_value_stats(result.race_stats; label="Race")
        println("=" ^ 70)
    end
end

let cl = _FallbackCountLogger(global_logger(), 0)
    Logging.with_logger(cl) do
        main()
    end
    if cl.n > 0
        println("\nNOTE: gnubg move fallback triggered $(cl.n) time(s) total " *
                "(blocked/dance doubles positions; fallback plays the forced legal move).")
    else
        println("\nNOTE: gnubg move fallback never triggered (clean integration).")
    end
end
