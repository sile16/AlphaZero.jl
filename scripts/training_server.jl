#!/usr/bin/env julia
"""
Distributed training server.

Runs on Jarvis (RTX 4090). Accepts self-play samples via HTTP,
trains models on GPU, serves weights to clients.

Architecture:
- HTTP thread: Accepts samples, serves weights (async I/O)
- Training thread: Gradient updates on GPU, reanalyze, eval

Usage:
    julia --threads 4 --project scripts/training_server.jl \\
        --port 9090 \\
        --data-dir /home/sile/alphazero-server \\
        --api-key my-secret-key \\
        --contact-width 256 --contact-blocks 5 \\
        --total-iterations 200 \\
        --games-per-iteration 500
"""

using ArgParse
using Dates
using Random
using Serialization
using Statistics: mean, cor, std
using TensorBoardLogger
using Logging: with_logger

function parse_args()
    s = ArgParseSettings(
        description="Distributed AlphaZero training server",
        autofix_names=true
    )

    @add_arg_table! s begin
        # Server
        "--port"
            help = "HTTP server port"
            arg_type = Int
            default = 9090
        "--data-dir"
            help = "Directory for checkpoints, buffer, logs (outside git)"
            arg_type = String
            default = "/home/sile/alphazero-server"
        "--api-key"
            help = "API key for client authentication"
            arg_type = String
            default = "alphazero-dev-key"

        # Model architecture
        "--contact-width"
            arg_type = Int
            default = 256
        "--contact-blocks"
            arg_type = Int
            default = 5
        "--race-width"
            arg_type = Int
            default = 128
        "--race-blocks"
            arg_type = Int
            default = 3

        # Training
        "--total-iterations"
            arg_type = Int
            default = 200
        "--learning-rate"
            arg_type = Float64
            default = 0.001
        "--l2-reg"
            arg_type = Float64
            default = 0.0001
        "--batch-size"
            arg_type = Int
            default = 256
        "--buffer-capacity"
            arg_type = Int
            default = 600000
        "--games-per-iteration"
            help = "Number of games worth of samples per training iteration"
            arg_type = Int
            default = 500
        "--training-steps"
            help = "Gradient steps per iteration (0 = auto: games_per_iteration * 200 / batch_size)"
            arg_type = Int
            default = 0
        "--seed"
            arg_type = Int
            default = 42
        "--training-mode"
            help = "Training mode: 'dual' (contact+race), 'race' (race-only)"
            arg_type = String
            default = "dual"

        # PER
        "--use-per"
            action = :store_true
        "--per-alpha"
            arg_type = Float64
            default = 0.6
        "--per-beta"
            arg_type = Float64
            default = 0.4
        "--per-epsilon"
            arg_type = Float64
            default = 0.01

        # Reanalyze
        "--use-reanalyze"
            action = :store_true
        "--reanalyze-fraction"
            arg_type = Float64
            default = 0.25
        "--reanalyze-blend"
            help = "EMA blend factor for reanalyze (0.0-1.0, lower = less aggressive)"
            arg_type = Float64
            default = 0.5

        # Learning rate schedule
        "--lr-schedule"
            help = "LR schedule: 'constant' or 'cosine'"
            arg_type = String
            default = "constant"
        "--lr-min"
            help = "Minimum LR for cosine schedule"
            arg_type = Float64
            default = 0.0001

        # Self-play config (served to clients)
        "--mcts-iters"
            arg_type = Int
            default = 400
        "--inference-batch-size"
            arg_type = Int
            default = 50
        "--cpuct"
            arg_type = Float64
            default = 2.0
        "--dirichlet-alpha"
            arg_type = Float64
            default = 0.3
        "--dirichlet-epsilon"
            arg_type = Float64
            default = 0.25

        # Temperature scheduling (served to clients)
        "--temp-move-cutoff"
            arg_type = Int
            default = 20
        "--temp-final"
            arg_type = Float64
            default = 0.1
        "--temp-iter-decay"
            action = :store_true
        "--temp-iter-final"
            arg_type = Float64
            default = 0.3

        # Bear-off (always enabled — clients load table locally)
        "--bearoff-hard-targets"
            action = :store_true
        "--bearoff-truncation"
            action = :store_true

        # Starting positions (for race-only or custom start mode)
        "--start-positions-file"
            help = "File with starting positions (portable tuples on NFS). Empty = use default opening."
            arg_type = String
            default = ""
        "--eval-positions-file"
            help = "File with fixed eval positions (portable tuples on NFS). Empty = no position-based eval."
            arg_type = String
            default = ""

        # Bootstrap (pre-fill buffer with expert games before self-play)
        "--bootstrap-file"
            help = "File with pre-converted training samples (NamedTuples on NFS). Loaded into buffer at startup."
            arg_type = String
            default = ""
        "--bootstrap-max-samples"
            help = "Max samples to load from bootstrap file (0 = all, capped at buffer capacity)"
            arg_type = Int
            default = 0

        # Eval
        "--eval-interval"
            help = "Run eval every N iterations (0 = disabled)"
            arg_type = Int
            default = 10
        "--eval-games"
            help = "Number of eval positions (each played from both sides)"
            arg_type = Int
            default = 100
        "--eval-mcts-iters"
            help = "MCTS iterations for eval games"
            arg_type = Int
            default = 200
        "--eval-workers"
            help = "Number of CPU threads for eval (0 = auto: nthreads - 2)"
            arg_type = Int
            default = 4
        "--wildbg-lib"
            help = "Path to libwildbg shared library"
            arg_type = String
            default = ""

        # Checkpoints
        "--checkpoint-interval"
            arg_type = Int
            default = 10
        "--buffer-checkpoint-interval"
            help = "Save full buffer every N iterations (0 = disabled)"
            arg_type = Int
            default = 50
        "--resume"
            help = "Resume from checkpoint directory"
            arg_type = String
            default = ""
    end

    return ArgParse.parse_args(s)
end

const ARGS = parse_args()
const DATA_DIR = ARGS["data_dir"]
const CHECKPOINT_DIR = joinpath(DATA_DIR, "checkpoints")
const TB_DIR = joinpath(DATA_DIR, "tb")

# Create directories
mkpath(CHECKPOINT_DIR)
mkpath(TB_DIR)
mkpath(joinpath(DATA_DIR, "buffer"))

# Set seed
Random.seed!(ARGS["seed"])

println("=" ^ 60)
println("AlphaZero Distributed Training Server")
println("=" ^ 60)
println("Port: $(ARGS["port"])")
println("Data dir: $DATA_DIR")
println("Contact model: $(ARGS["contact_width"])w×$(ARGS["contact_blocks"])b")
println("Race model: $(ARGS["race_width"])w×$(ARGS["race_blocks"])b")
println("Buffer capacity: $(ARGS["buffer_capacity"])")
println("Training mode: $(ARGS["training_mode"])")
if !isempty(ARGS["start_positions_file"])
    println("Start positions: $(ARGS["start_positions_file"])")
end
if !isempty(ARGS["eval_positions_file"])
    println("Eval positions: $(ARGS["eval_positions_file"])")
end
if !isempty(ARGS["bootstrap_file"])
    println("Bootstrap: $(ARGS["bootstrap_file"])")
    if ARGS["bootstrap_max_samples"] > 0
        println("Bootstrap max: $(ARGS["bootstrap_max_samples"])")
    end
end
println("PER: $(ARGS["use_per"])")
println("Reanalyze: $(ARGS["use_reanalyze"]) (blend=$(ARGS["reanalyze_blend"]))")
println("LR: $(ARGS["learning_rate"]) (schedule=$(ARGS["lr_schedule"]), min=$(ARGS["lr_min"]))")
println("=" ^ 60)
flush(stdout)

# Load packages
using AlphaZero
using AlphaZero: GI, MCTS, Network, FluxLib, MctsParams, LearningParams, Adam, BatchedMCTS
using AlphaZero: CONSTANT_WEIGHT, losses, ConstSchedule
# Note: NetLib not needed - using FluxLib directly for network creation
import Flux
import CUDA

# Check GPU
const USE_GPU = CUDA.functional()
if USE_GPU
    CUDA.allowscalar(false)
    println("\nGPU: $(CUDA.name(CUDA.device()))")
else
    println("\nWARNING: No GPU detected! Training will be slow on CPU.")
end
flush(stdout)

# Include shared distributed code
include(joinpath(@__DIR__, "..", "src", "distributed", "buffer.jl"))
include(joinpath(@__DIR__, "..", "src", "distributed", "protocol.jl"))
include(joinpath(@__DIR__, "..", "src", "distributed", "server.jl"))

# Game setup
const GAME_NAME = "backgammon-deterministic"
if GAME_NAME == "backgammon-deterministic"
    ENV["BACKGAMMON_OBS_TYPE"] = "minimal_flat"
    include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
else
    error("Unknown game: $GAME_NAME")
end
const gspec = GameSpec()
const NUM_ACTIONS = GI.num_actions(gspec)
const _state_dim = let env = GI.init(gspec); length(vec(GI.vectorize_state(gspec, GI.current_state(env)))); end

# Network setup
const CONTACT_WIDTH = ARGS["contact_width"]
const CONTACT_BLOCKS = ARGS["contact_blocks"]
const RACE_WIDTH = ARGS["race_width"]
const RACE_BLOCKS = ARGS["race_blocks"]
const BATCH_SIZE = ARGS["batch_size"]
const BUFFER_CAPACITY = ARGS["buffer_capacity"]
const LEARNING_RATE = Float32(ARGS["learning_rate"])
const L2_REG = Float32(ARGS["l2_reg"])
const USE_PER = ARGS["use_per"]
const PER_ALPHA = Float32(ARGS["per_alpha"])
const PER_BETA_INIT = Float32(ARGS["per_beta"])
const PER_EPSILON = Float32(ARGS["per_epsilon"])
const USE_REANALYZE = ARGS["use_reanalyze"]
const REANALYZE_FRACTION = ARGS["reanalyze_fraction"]
const REANALYZE_BLEND = Float32(ARGS["reanalyze_blend"])
const LR_SCHEDULE = ARGS["lr_schedule"]
const LR_MIN = Float32(ARGS["lr_min"])
const EVAL_INTERVAL = ARGS["eval_interval"]
const EVAL_GAMES = ARGS["eval_games"]
const EVAL_MCTS_ITERS = ARGS["eval_mcts_iters"]
const EVAL_WORKERS = ARGS["eval_workers"]

# ── Eval Setup (wildbg on CPU) ─────────────────────────────────────────
using BackgammonNet
using StaticArrays

function find_wildbg_lib_server()
    if !isempty(ARGS["wildbg_lib"])
        return ARGS["wildbg_lib"]
    end
    candidates = [
        joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg.so"),
        joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg.dylib"),
        joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg_main.so"),
        joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg_main.dylib"),
    ]
    for c in candidates
        isfile(c) && return c
    end
    return ""  # No wildbg found — eval disabled
end

const WILDBG_LIB = find_wildbg_lib_server()
const EVAL_ENABLED = EVAL_INTERVAL > 0 && !isempty(WILDBG_LIB) && !isempty(ARGS["eval_positions_file"])

# Load eval positions
const EVAL_POSITIONS = if EVAL_ENABLED && isfile(ARGS["eval_positions_file"])
    pos = Serialization.deserialize(ARGS["eval_positions_file"])
    n = EVAL_GAMES > 0 ? min(EVAL_GAMES, length(pos)) : length(pos)
    pos[1:n]
else
    Tuple[]
end

if EVAL_ENABLED
    # Set up wildbg
    lib_size = filesize(WILDBG_LIB)
    nets_variant = lib_size > 10_000_000 ? :large : :small
    if nets_variant == :large
        BackgammonNet.wildbg_set_lib_path!(large=WILDBG_LIB)
    else
        BackgammonNet.wildbg_set_lib_path!(small=WILDBG_LIB)
    end
    println("Eval: $(length(EVAL_POSITIONS)) positions × 2 sides, $(EVAL_MCTS_ITERS) MCTS iters, $(EVAL_WORKERS) workers")
    println("Eval: wildbg $nets_variant ($(round(lib_size/1e6, digits=1))MB), every $EVAL_INTERVAL iters")
else
    if EVAL_INTERVAL > 0
        println("Eval: DISABLED (wildbg_lib=$(isempty(WILDBG_LIB) ? "not found" : WILDBG_LIB), eval_positions=$(ARGS["eval_positions_file"]))")
    else
        println("Eval: disabled (eval-interval=0)")
    end
end

# PositionValueSample is now provided by GameLoop module
const PositionValueSample = AlphaZero.GameLoop.PositionValueSample

function _eval_forward_network(net, states)
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

function eval_race_game_server(single_oracle, batch_oracle, wildbg_backend,
                               position_data::Tuple, eval_net;
                               seed::Int=1, az_is_white::Bool=true,
                               eval_mcts_iters::Int=EVAL_MCTS_ITERS)
    eval_mcts_params = MctsParams(
        num_iters_per_turn=eval_mcts_iters,
        cpuct=1.5,
        temperature=ConstSchedule(0.0),
        dirichlet_noise_ϵ=0.0,
        dirichlet_noise_α=1.0)

    az = AlphaZero.GameLoop.MctsAgent(single_oracle, batch_oracle, eval_mcts_params, 50, gspec)
    wb = AlphaZero.GameLoop.ExternalAgent(wildbg_backend)

    value_oracle_fn = function(env)
        result = _eval_forward_network(eval_net, [env.game])
        return Float64(result[1][2])  # value head output
    end
    wildbg_value_fn = function(env)
        return Float64(BackgammonNet.evaluate(wildbg_backend, env.game))
    end

    w, b = az_is_white ? (az, wb) : (wb, az)

    # Initialize game from position tuple
    p0, p1, cp = position_data
    g = BackgammonGame(p0, p1, SVector{2,Int8}(0, 0), Int8(0), cp, false, 0.0f0;
                       obs_type=:minimal_flat)
    env = GI.init(gspec)
    env.game = g

    rng = MersenneTwister(seed)
    result = AlphaZero.GameLoop.play_game(w, b, env;
        record_value_comparison=true,
        value_oracle=value_oracle_fn,
        opponent_value_fn=wildbg_value_fn,
        rng=rng)

    # Convert to expected return format (map opponent_val -> wb_val field name)
    value_samples = result.value_samples
    az_reward = az_is_white ? result.reward : -result.reward

    return (reward=az_reward, value_samples=value_samples)
end

"""Run eval vs wildbg on fixed positions using CPU threads.
Returns (equity, win_pct, value_mse, value_corr) or nothing if eval disabled."""
function run_eval!(network_to_eval, iter::Int)
    !EVAL_ENABLED && return nothing
    isempty(EVAL_POSITIONS) && return nothing

    # Copy network to CPU for eval
    eval_net = Flux.cpu(network_to_eval)

    batch_oracle(states::Vector) = _eval_forward_network(eval_net, states)
    single_oracle(s) = batch_oracle([s])[1]

    n_pos = length(EVAL_POSITIONS)
    n_total = 2 * n_pos  # both sides
    rewards = Vector{Float64}(undef, n_total)
    vsamples = Vector{Vector{PositionValueSample}}(undef, n_total)

    # Create per-worker wildbg backends
    n_workers = min(EVAL_WORKERS, Threads.nthreads() - 1)
    n_workers = max(1, n_workers)

    wildbg_backends = [begin
        wb = BackgammonNet.WildbgBackend(nets=nets_variant)
        BackgammonNet.open!(wb)
        wb
    end for _ in 1:n_workers]

    t_start = time()
    claimed = Threads.Atomic{Int}(0)

    Threads.@threads for tid in 1:n_workers
        wb = wildbg_backends[tid]
        while true
            job = Threads.atomic_add!(claimed, 1) + 1
            job > n_total && break
            if job <= n_pos
                pos_idx = job
                az_white = true
            else
                pos_idx = job - n_pos
                az_white = false
            end
            result = eval_race_game_server(single_oracle, batch_oracle, wb,
                                           EVAL_POSITIONS[pos_idx], eval_net;
                                           seed=job + iter * 10000, az_is_white=az_white)
            rewards[job] = result.reward
            vsamples[job] = result.value_samples
        end
    end

    elapsed = time() - t_start

    for wb in wildbg_backends
        BackgammonNet.close(wb)
    end

    # Compute stats
    avg_equity = mean(rewards)
    win_pct = 100 * count(r -> r > 0, rewards) / n_total
    white_equity = mean(rewards[1:n_pos])
    black_equity = mean(rewards[n_pos+1:end])

    all_vs = PositionValueSample[]
    for vs in vsamples; append!(all_vs, vs); end
    value_mse = NaN
    value_corr = NaN
    if length(all_vs) >= 3
        nn = [s.nn_val for s in all_vs]
        wb = [s.opponent_val for s in all_vs]
        value_mse = mean((nn .- wb) .^ 2)
        value_corr = cor(nn, wb)
    end

    @info "Eval iter $iter: equity=$(round(avg_equity, digits=3)) win%=$(round(win_pct, digits=1)) " *
          "white=$(round(white_equity, digits=3)) black=$(round(black_equity, digits=3)) " *
          "value_mse=$(round(value_mse, digits=4)) corr=$(round(value_corr, digits=4)) " *
          "$(n_total) games in $(round(elapsed, digits=1))s"

    return (equity=avg_equity, win_pct=win_pct, white_equity=white_equity, black_equity=black_equity,
            value_mse=value_mse, value_corr=value_corr, n_games=n_total, elapsed=elapsed)
end

# Create networks
println("\nCreating networks...")
contact_network = FluxLib.FCResNetMultiHead(
    gspec, FluxLib.FCResNetMultiHeadHP(width=CONTACT_WIDTH, num_blocks=CONTACT_BLOCKS))
race_network = FluxLib.FCResNetMultiHead(
    gspec, FluxLib.FCResNetMultiHeadHP(width=RACE_WIDTH, num_blocks=RACE_BLOCKS))

println("Contact model parameters: $(sum(length(p) for p in Flux.params(contact_network)))")
println("Race model parameters: $(sum(length(p) for p in Flux.params(race_network)))")

# Move to GPU if available
if USE_GPU
    contact_network = Network.to_gpu(contact_network)
    race_network = Network.to_gpu(race_network)
    println("Models moved to GPU")
end

# Resume from checkpoint if specified
START_ITER = 0
if !isempty(ARGS["resume"])
    resume_dir = ARGS["resume"]
    # Try dual-model checkpoints first
    contact_path = joinpath(resume_dir, "contact_latest.data")
    race_path = joinpath(resume_dir, "race_latest.data")
    if isfile(contact_path) && isfile(race_path)
        FluxLib.load_weights(contact_path, contact_network)
        FluxLib.load_weights(race_path, race_network)
        iter_file = joinpath(resume_dir, "iter.txt")
        if isfile(iter_file)
            START_ITER = parse(Int, strip(read(iter_file, String)))
        end
        println("Resumed from $resume_dir at iteration $START_ITER")
        # Buffer loading deferred to after replay_buffer is created (see RESUME_BUFFER_DIR below)
        global RESUME_BUFFER_DIR = joinpath(resume_dir, "..")
    else
        # Try single-model checkpoint
        single_path = joinpath(resume_dir, "latest.data")
        if isfile(single_path)
            FluxLib.load_weights(single_path, contact_network)
            FluxLib.load_weights(single_path, race_network)
            iter_file = joinpath(resume_dir, "iter.txt")
            if isfile(iter_file)
                START_ITER = parse(Int, strip(read(iter_file, String)))
            end
            println("Resumed single-model from $resume_dir at iteration $START_ITER")
        else
            error("No checkpoint found at $resume_dir")
        end
    end
end

# Optimizers
contact_opt = Flux.AdamW(LEARNING_RATE, (0.9f0, 0.999f0), L2_REG)
contact_opt_state = Flux.setup(contact_opt, contact_network)
race_opt = Flux.AdamW(LEARNING_RATE, (0.9f0, 0.999f0), L2_REG)
race_opt_state = Flux.setup(race_opt, race_network)

"""Update learning rate based on schedule. Returns current LR."""
function update_lr!(opt_state, iter::Int, total_iters::Int)
    if LR_SCHEDULE == "cosine"
        # Cosine annealing: lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t/T))
        progress = Float32(iter) / Float32(total_iters)
        lr = LR_MIN + 0.5f0 * (LEARNING_RATE - LR_MIN) * (1f0 + cos(Float32(π) * progress))
    else
        lr = LEARNING_RATE
    end
    Flux.adjust!(opt_state; eta=lr)
    return lr
end

# Learning params (for loss function)
const LEARNING_PARAMS = LearningParams(
    use_gpu=USE_GPU,
    use_position_averaging=false,
    samples_weighing_policy=CONSTANT_WEIGHT,
    optimiser=Adam(lr=LEARNING_RATE),
    l2_regularization=0f0,
    rewards_renormalization=1f0,
    nonvalidity_penalty=1f0,
    batch_size=BATCH_SIZE,
    loss_computation_batch_size=BATCH_SIZE,
    min_checkpoints_per_epoch=1,
    max_batches_per_checkpoint=100,
    num_checkpoints=1
)

# PER buffer (columnar, pre-allocated)
replay_buffer = PERBuffer(BUFFER_CAPACITY, _state_dim, NUM_ACTIONS;
                           beta_init=PER_BETA_INIT, annealing_iters=ARGS["total_iterations"])

# Bootstrap: pre-fill buffer with expert games
if !isempty(ARGS["bootstrap_file"])
    let bootstrap_path = ARGS["bootstrap_file"],
        t0 = time()

        println("\nLoading bootstrap data from: $bootstrap_path")
        flush(stdout)
        bootstrap_samples = Serialization.deserialize(bootstrap_path)

        # Production bootstrap artifacts currently deserialize to
        # `Vector{NamedTuple}` with fields:
        #   state, policy, value, equity, has_equity, is_chance, is_contact, is_bearoff
        #
        # `value` is already in the same player-relative convention used by the
        # learner. `equity` contains joint cumulative 5-head values (same
        # convention as the NN and self-play targets). The loader below preserves
        # that layout and only repacks it into columnar arrays for the replay buffer.
        n_bootstrap = length(bootstrap_samples)
        max_load = ARGS["bootstrap_max_samples"] > 0 ?
            min(ARGS["bootstrap_max_samples"], BUFFER_CAPACITY) : min(n_bootstrap, BUFFER_CAPACITY)

        chunk_size = 10000
        loaded = 0
        for start_idx in 1:chunk_size:max_load
            end_idx = min(start_idx + chunk_size - 1, max_load)
            chunk = bootstrap_samples[start_idx:end_idx]
            n = length(chunk)

            states = hcat([s.state for s in chunk]...)
            policies_raw = hcat([s.policy for s in chunk]...)
            if size(policies_raw, 1) < NUM_ACTIONS
                policies = zeros(Float32, NUM_ACTIONS, n)
                policies[1:size(policies_raw, 1), :] .= policies_raw
            else
                policies = policies_raw[1:NUM_ACTIONS, :]
            end
            values = Float32[s.value for s in chunk]
            equities = hcat([s.equity for s in chunk]...)
            has_equity = Bool[s.has_equity for s in chunk]
            is_contact = Bool[s.is_contact for s in chunk]
            is_bearoff = Bool[s.is_bearoff for s in chunk]

            per_add_batch!(replay_buffer, states, policies, values,
                           equities, has_equity, is_contact, is_bearoff)
            loaded += n
        end

        bootstrap_samples = nothing
        GC.gc()
        t_load = time() - t0
        println("  Loaded $loaded / $n_bootstrap bootstrap samples in $(round(t_load, digits=1))s")
        println("  Buffer size: $(buf_length(replay_buffer))")
        flush(stdout)
    end
end

# Load buffer checkpoint if resuming (must happen after replay_buffer is created)
if @isdefined(RESUME_BUFFER_DIR)
    buf_path = joinpath(RESUME_BUFFER_DIR, "buffer", "buffer_iter_$START_ITER.jls")
    if isfile(buf_path)
        load_buffer!(replay_buffer, buf_path)
        println("Loaded buffer checkpoint from $buf_path")
    else
        buf_dir = joinpath(RESUME_BUFFER_DIR, "buffer")
        if isdir(buf_dir)
            buf_files = filter(f -> startswith(f, "buffer_iter_") && endswith(f, ".jls"), readdir(buf_dir))
            buf_iters = [parse(Int, match(r"buffer_iter_(\d+)\.jls", f)[1]) for f in buf_files]
            valid = filter(i -> i <= START_ITER, buf_iters)
            if !isempty(valid)
                best_iter = maximum(valid)
                load_buffer!(replay_buffer, joinpath(buf_dir, "buffer_iter_$best_iter.jls"))
                println("Loaded buffer checkpoint from iteration $best_iter")
            else
                println("No buffer checkpoint found, starting with empty buffer")
            end
        else
            println("No buffer directory found, starting with empty buffer")
        end
    end
end

# Training functions (extracted from train_distributed.jl)

"""Prepare training batch from columnar buffer extract.

Accepts the NamedTuple from extract_batch (columnar matrices) instead of
a Vector of NamedTuples — avoids per-sample allocation entirely.

The returned batch matches the same learner-side contract as
`AlphaZero.convert_samples`:
- `V` is the scalar player-relative target used by the single-head fallback and
  for TD error computation
- `EqWin/EqGW/EqBGW/EqGL/EqBGL` are the five joint cumulative equity heads
- `HasEquity` gates whether the multi-head BCEWithLogits losses apply to that sample
"""
function prepare_batch_columnar(col_data, num_actions, use_gpu_flag, net)
    n = size(col_data.states, 2)
    W = ones(Float32, 1, n)
    X = col_data.states           # Already (state_dim, n)
    P = col_data.policies         # Already (num_actions, n)
    V = reshape(col_data.values, 1, n)

    A = zeros(Float32, num_actions, n)
    IsChance = zeros(Float32, 1, n)
    @inbounds for i in 1:n
        if col_data.is_chance[i]
            A[:, i] .= 1.0f0
            IsChance[1, i] = 1.0f0
        else
            for j in 1:num_actions
                A[j, i] = col_data.policies[j, i] > 0 ? 1.0f0 : 0.0f0
            end
        end
    end

    eq_heads = AlphaZero.split_equity_targets(col_data.equities, col_data.has_equity)
    EqWin, EqGW, EqBGW, EqGL, EqBGL, HasEquity =
        eq_heads.EqWin, eq_heads.EqGW, eq_heads.EqBGW, eq_heads.EqGL, eq_heads.EqBGL, eq_heads.HasEquity

    batch_data = (; W, X, A, P, V, IsChance, EqWin, EqGW, EqBGW, EqGL, EqBGL, HasEquity)

    if use_gpu_flag
        batch_data = Network.convert_input_tuple(net, batch_data)
    end

    return batch_data
end

function compute_td_errors(nn, batch_data)
    X, A, V = batch_data.X, batch_data.A, batch_data.V
    is_multihead = nn isa FluxLib.FCResNetMultiHead
    if is_multihead
        # forward_normalized_multihead returns raw logits; apply sigmoid for equity
        _, L̂_win, L̂_gw, L̂_bgw, L̂_gl, L̂_bgl, _ =
            FluxLib.forward_normalized_multihead(nn, X, A)
        equity = FluxLib.compute_equity(
            Flux.sigmoid.(L̂_win), Flux.sigmoid.(L̂_gw), Flux.sigmoid.(L̂_bgw),
            Flux.sigmoid.(L̂_gl), Flux.sigmoid.(L̂_bgl))
        V̂_combined = equity ./ 3f0
        td = abs.(Flux.cpu(V̂_combined) .- Flux.cpu(V))
    else
        _, V̂, _ = Network.forward_normalized(nn, X, A)
        td = abs.(Flux.cpu(V̂) .- Flux.cpu(V))
    end
    return Float32.(vec(td))
end

function _train_model_on_samples!(buf_indices::Vector{Int}, network, opt_state)
    n = length(buf_indices)
    n < BATCH_SIZE && return (avg_loss=0.0, avg_Lp=0.0, avg_Lv=0.0, avg_Linv=0.0, num_batches=0)

    if ARGS["training_steps"] > 0
        max_batches = ARGS["training_steps"]
    else
        max_batches = max(1, ARGS["games_per_iteration"] * 200 ÷ BATCH_SIZE)
    end
    num_batches = min(max(1, n ÷ BATCH_SIZE), max_batches)
    total_loss = 0.0
    total_Lp = 0.0
    total_Lv = 0.0
    total_Linv = 0.0

    for _ in 1:num_batches
        # PER: sample proportional to priorities within this model's partition
        # Uniform: sample randomly from the partition
        if USE_PER
            batch_buf_indices, is_weights = per_sample_partition(
                replay_buffer, buf_indices, BATCH_SIZE, PER_ALPHA, PER_EPSILON)
        else
            sample_idx = rand(1:n, BATCH_SIZE)
            batch_buf_indices = buf_indices[sample_idx]
            is_weights = ones(Float32, BATCH_SIZE)
        end

        # Extract columnar data from buffer
        col_data = extract_batch(replay_buffer, batch_buf_indices)
        batch_data = prepare_batch_columnar(col_data, NUM_ACTIONS, USE_GPU, network)

        # IS weights: scale sample weights by importance sampling correction
        W_is = reshape(is_weights, 1, BATCH_SIZE)
        if USE_GPU
            W_is = Flux.gpu(W_is)
        end
        Wmean = mean(W_is)
        Hp = 0.0f0

        # Replace uniform weights with IS weights for PER-corrected loss
        batch_data_per = merge(batch_data, (W=W_is,))

        loss_fn(nn) = losses(nn, LEARNING_PARAMS, Wmean, Hp, batch_data_per)[1]
        loss, grads = Flux.withgradient(loss_fn, network)
        Flux.update!(opt_state, network, grads[1])

        L, Lp, Lv, _, Linv = losses(network, LEARNING_PARAMS, Wmean, Hp, batch_data_per)
        total_loss += Float64(L)
        total_Lp += Float64(Lp)
        total_Lv += Float64(Lv)
        total_Linv += Float64(Linv)

        # Update PER priorities with TD-errors
        if USE_PER
            td_errors = compute_td_errors(network, batch_data_per)
            per_update_priorities!(replay_buffer, batch_buf_indices, td_errors)
        end
    end

    return (avg_loss=total_loss / num_batches, avg_Lp=total_Lp / num_batches,
            avg_Lv=total_Lv / num_batches, avg_Linv=total_Linv / num_batches,
            num_batches=num_batches)
end

function train_on_buffer!()
    n_buf = buf_length(replay_buffer)
    n_buf < BATCH_SIZE && return (contact=(avg_loss=0.0,), race=(avg_loss=0.0,))

    if USE_PER
        per_anneal_beta!(replay_buffer)
    end

    # Partition buffer indices by contact/race (single lock acquisition)
    parts = partition_indices(replay_buffer)

    if ARGS["training_mode"] == "race"
        # Race-only mode: train race on ALL samples
        contact_result = (avg_loss=0.0, avg_Lp=0.0, avg_Lv=0.0, avg_Linv=0.0, num_batches=0)
        race_result = _train_model_on_samples!(parts.all, race_network, race_opt_state)
    else
        contact_result = _train_model_on_samples!(parts.contact, contact_network, contact_opt_state)
        race_result = _train_model_on_samples!(parts.race, race_network, race_opt_state)
    end

    return (contact=contact_result, race=race_result)
end

function reanalyze_buffer!()
    USE_REANALYZE || return 0
    n = buf_length(replay_buffer)
    n == 0 && return 0

    num_to_reanalyze = max(1, round(Int, n * REANALYZE_FRACTION))
    reanalyze_indices = randperm(n)[1:min(num_to_reanalyze, n)]

    batch_size = min(2048, length(reanalyze_indices))
    total_updated = 0

    for batch_start in 1:batch_size:length(reanalyze_indices)
        batch_end = min(batch_start + batch_size - 1, length(reanalyze_indices))
        batch_indices = reanalyze_indices[batch_start:batch_end]

        # Extract columnar data once (lock-free)
        col_data = extract_batch(replay_buffer, batch_indices)

        for (is_contact_flag, net) in [(true, contact_network), (false, race_network)]
            # Filter to matching model type, skip bearoff samples (exact table values)
            sub_mask = [col_data.is_contact[j] == is_contact_flag && !col_data.is_bearoff[j]
                        for j in 1:length(batch_indices)]
            any(sub_mask) || continue

            sub_local_idx = findall(sub_mask)
            sub_buf_indices = batch_indices[sub_local_idx]

            # Slice from already-extracted data (no second buffer read)
            sub_col = (
                states = col_data.states[:, sub_local_idx],
                policies = col_data.policies[:, sub_local_idx],
                values = col_data.values[sub_local_idx],
                equities = col_data.equities[:, sub_local_idx],
                has_equity = col_data.has_equity[sub_local_idx],
                is_chance = col_data.is_chance[sub_local_idx],
                is_contact = col_data.is_contact[sub_local_idx],
                is_bearoff = col_data.is_bearoff[sub_local_idx],
            )
            sub_batch_data = prepare_batch_columnar(sub_col, NUM_ACTIONS, USE_GPU, net)

            # forward_normalized_multihead returns raw logits; apply sigmoid
            _, L̂_win, L̂_gw, L̂_bgw, L̂_gl, L̂_bgl, _ =
                FluxLib.forward_normalized_multihead(net, sub_batch_data.X, sub_batch_data.A)
            V̂_win = Flux.sigmoid.(L̂_win)
            V̂_gw = Flux.sigmoid.(L̂_gw)
            V̂_bgw = Flux.sigmoid.(L̂_bgw)
            V̂_gl = Flux.sigmoid.(L̂_gl)
            V̂_bgl = Flux.sigmoid.(L̂_bgl)
            equity = FluxLib.compute_equity(V̂_win, V̂_gw, V̂_bgw, V̂_gl, V̂_bgl)
            new_values = Float32.(vec(Flux.cpu(equity ./ 3f0)))

            new_eq_win = Float32.(vec(Flux.cpu(V̂_win)))
            new_eq_gw = Float32.(vec(Flux.cpu(V̂_gw)))
            new_eq_bgw = Float32.(vec(Flux.cpu(V̂_bgw)))
            new_eq_gl = Float32.(vec(Flux.cpu(V̂_gl)))
            new_eq_bgl = Float32.(vec(Flux.cpu(V̂_bgl)))

            # Write blended values back (lock-free)
            reanalyze_update!(replay_buffer, sub_buf_indices,
                              new_values, new_eq_win, new_eq_gw, new_eq_bgw, new_eq_gl, new_eq_bgl;
                              α_blend=REANALYZE_BLEND)

            total_updated += length(sub_buf_indices)
        end

        # Let GC clean up batch temporaries
        col_data = nothing
    end

    return total_updated
end

# TensorBoard logger
const TB_LOGGER = if START_ITER > 0
    lg = TBLogger(TB_DIR, tb_append)
    lg.global_step = START_ITER
    println("TensorBoard: appending from step $START_ITER")
    lg
else
    TBLogger(TB_DIR, tb_overwrite)
end

# Log config
with_logger(TB_LOGGER) do
    git_commit = try strip(read(`git rev-parse HEAD`, String)) catch; "unknown" end
    cmd = "julia " * join(Base.ARGS, " ")
    params_lines = ["## Hyperparameters\n"]
    for (k, v) in sort(collect(ARGS), by=first)
        push!(params_lines, "- **$k**: `$v`")
    end
    repro_text = """
    ## Distributed Training Server
    - **Git commit**: `$(git_commit)`
    - **Command**: `$(cmd)`
    - **Julia version**: `$(VERSION)`
    - **Date**: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    - **GPU**: $(USE_GPU ? "$(CUDA.name(CUDA.device()))" : "CPU only")
    $(join(params_lines, "\n"))
    """
    @info "config" text=repro_text log_step_increment=0
end

# Server config (served to clients via GET /api/config)
const SERVER_CONFIG = Dict{String, Any}(
    "mcts_iters" => ARGS["mcts_iters"],
    "inference_batch_size" => ARGS["inference_batch_size"],
    "cpuct" => ARGS["cpuct"],
    "dirichlet_alpha" => ARGS["dirichlet_alpha"],
    "dirichlet_epsilon" => ARGS["dirichlet_epsilon"],
    "contact_width" => CONTACT_WIDTH,
    "contact_blocks" => CONTACT_BLOCKS,
    "race_width" => RACE_WIDTH,
    "race_blocks" => RACE_BLOCKS,
    "state_dim" => _state_dim,
    "num_actions" => NUM_ACTIONS,
    "game" => GAME_NAME,
    "temp_move_cutoff" => ARGS["temp_move_cutoff"],
    "temp_final" => ARGS["temp_final"],
    "temp_iter_decay" => ARGS["temp_iter_decay"],
    "temp_iter_final" => ARGS["temp_iter_final"],
    "total_iterations" => ARGS["total_iterations"],
    "use_bearoff" => true,  # always enabled
    "bearoff_hard_targets" => ARGS["bearoff_hard_targets"],
    "bearoff_truncation" => ARGS["bearoff_truncation"],
    "training_mode" => ARGS["training_mode"],
    "start_positions_file" => basename(ARGS["start_positions_file"]),
    "eval_positions_file" => basename(ARGS["eval_positions_file"]),
    "seed" => ARGS["seed"],
)

# Initialize server state
server_state = ServerState(api_key=ARGS["api_key"], config=SERVER_CONFIG)
server_state.iteration[] = START_ITER

# Cache initial weights
update_weight_cache!(server_state, contact_network, race_network;
                     contact_width=CONTACT_WIDTH, contact_blocks=CONTACT_BLOCKS,
                     race_width=RACE_WIDTH, race_blocks=RACE_BLOCKS)

# Start HTTP server
http_server = start_server!(server_state, replay_buffer; port=ARGS["port"])
println("\nServer listening on port $(ARGS["port"])")
println("Waiting for self-play samples...")
flush(stdout)

"""Collect server-side performance stats (GPU + CPU)."""
function collect_server_stats()
    stats = Dict{String, Float64}(
        "gpu_percent" => 0.0,
        "gpu_memory_used_gb" => 0.0,
        "gpu_memory_total_gb" => 0.0,
        "cpu_percent" => 0.0,
    )

    # GPU stats via nvidia-smi (Linux with NVIDIA GPU)
    if USE_GPU
        try
            gpu_util = strip(read(`nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits`, String))
            stats["gpu_percent"] = parse(Float64, gpu_util)
        catch e
            @debug "Failed to read GPU utilization" exception=e
        end

        try
            gpu_mem = strip(read(`nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits`, String))
            parts = split(gpu_mem, ",")
            if length(parts) >= 2
                stats["gpu_memory_used_gb"] = round(parse(Float64, strip(parts[1])) / 1024, digits=2)
                stats["gpu_memory_total_gb"] = round(parse(Float64, strip(parts[2])) / 1024, digits=2)
            end
        catch e
            @debug "Failed to read GPU memory" exception=e
        end
    end

    # CPU stats via /proc/stat (Linux)
    if Sys.islinux()
        try
            lines1 = readlines("/proc/stat")
            cpu1 = parse.(Int, split(lines1[1])[2:end])
            sleep(0.1)
            lines2 = readlines("/proc/stat")
            cpu2 = parse.(Int, split(lines2[1])[2:end])
            idle1 = cpu1[4]; idle2 = cpu2[4]
            total1 = sum(cpu1); total2 = sum(cpu2)
            dt = total2 - total1
            if dt > 0
                stats["cpu_percent"] = round(100.0 * (1.0 - (idle2 - idle1) / dt), digits=1)
            end
        catch e
            @debug "Failed to read CPU stats" exception=e
        end
    elseif Sys.isapple()
        try
            cpu_output = strip(read(`ps -A -o %cpu`, String))
            lines = split(cpu_output, '\n')
            total_cpu = 0.0
            for line in lines[2:end]
                s = strip(line)
                isempty(s) && continue
                total_cpu += parse(Float64, s)
            end
            ncpu = parse(Int, strip(read(`sysctl -n hw.ncpu`, String)))
            stats["cpu_percent"] = round(total_cpu / ncpu, digits=1)
        catch e
            @debug "Failed to read CPU stats" exception=e
        end
    end

    return stats
end

# Background task: expire stale eval checkouts (no job timeout — jobs complete naturally)
@async begin
    while true
        sleep(60)
        lock(EVAL_LOCK) do
            job = EVAL_JOB[]
            job === nothing && return
            expired = EvalManager.expire_stale_checkouts!(job; lease_seconds=EVAL_CHECKOUT_LEASE)
            if expired > 0
                println("Eval: expired $expired stale checkout(s) for iter $(job.iter)")
            end
        end
    end
end

# Samples threshold for one iteration
const SAMPLES_PER_ITERATION = ARGS["games_per_iteration"] * 200  # ~200 samples per game

# Main training loop — runs on a spawned thread so the main thread's
# libuv event loop stays active for HTTP.jl to handle requests.
training_task = Threads.@spawn begin

# Eval at iter 0 (bootstrap weights) — baseline before any self-play training
if EVAL_ENABLED && START_ITER == 0
    lock(EVAL_LOCK) do
        wv = ARGS["training_mode"] == "race" ? server_state.race_version[] : server_state.contact_version[]
        n_pos = length(EVAL_POSITIONS)
        EVAL_JOB[] = EvalManager.create_eval_job(0, n_pos, wv; chunk_size=EVAL_CHUNK_SIZE)
        println("Eval job created for iter 0 (bootstrap baseline): $(length(EVAL_JOB[].chunks)) chunks")
    end
end

for iter in (START_ITER + 1):ARGS["total_iterations"]
    # Wait for enough new samples (offset by START_ITER so resume works)
    target_samples = (iter - START_ITER) * SAMPLES_PER_ITERATION
    while server_state.total_samples[] < target_samples
        cur = server_state.total_samples[]
        pct = round(100 * cur / target_samples, digits=1)
        n_clients = length(server_state.clients)
        print("\rIteration $iter: waiting for samples ($cur / $target_samples = $pct%, $n_clients clients)  ")
        flush(stdout)
        sleep(5)
    end
    println()

    iter_start = time()

    # Update learning rate
    current_lr = update_lr!(contact_opt_state, iter, ARGS["total_iterations"])
    update_lr!(race_opt_state, iter, ARGS["total_iterations"])

    # Train on buffer (GPU)
    t0 = time()
    train_result = train_on_buffer!()
    t_train = time() - t0

    contact_loss = train_result.contact.avg_loss
    race_loss = train_result.race.avg_loss
    avg_loss = ARGS["training_mode"] == "race" ? race_loss : (contact_loss + race_loss) / 2

    # Reanalyze (GPU)
    t0 = time()
    n_reanalyzed = reanalyze_buffer!()
    t_reanalyze = time() - t0

    iter_time = time() - iter_start

    # Update server state
    server_state.iteration[] = iter
    server_state.contact_loss = contact_loss
    server_state.race_loss = race_loss

    # Update weight cache
    update_weight_cache!(server_state, contact_network, race_network;
                         contact_width=CONTACT_WIDTH, contact_blocks=CONTACT_BLOCKS,
                         race_width=RACE_WIDTH, race_blocks=RACE_BLOCKS)

    # Log to console
    grad_steps = train_result.contact.num_batches + train_result.race.num_batches
    @info "Iteration $iter" avg_loss contact_loss race_loss grad_steps buffer_size=buf_length(replay_buffer) total_samples=server_state.total_samples[] n_clients=length(server_state.clients) iter_time t_train t_reanalyze n_reanalyzed

    # Collect server and cluster stats
    server_stats = collect_server_stats()
    cluster_stats = get_cluster_stats(server_state)

    # Log to TensorBoard
    with_logger(TB_LOGGER) do
        @info "loss/avg" value=avg_loss log_step_increment=0
        @info "loss/contact" value=contact_loss log_step_increment=0
        @info "loss/race" value=race_loss log_step_increment=0
        # Per-component losses
        if train_result.contact.num_batches > 0
            @info "loss/contact_policy" value=train_result.contact.avg_Lp log_step_increment=0
            @info "loss/contact_value" value=train_result.contact.avg_Lv log_step_increment=0
            @info "loss/contact_invalid" value=train_result.contact.avg_Linv log_step_increment=0
        end
        if train_result.race.num_batches > 0
            @info "loss/race_policy" value=train_result.race.avg_Lp log_step_increment=0
            @info "loss/race_value" value=train_result.race.avg_Lv log_step_increment=0
            @info "loss/race_invalid" value=train_result.race.avg_Linv log_step_increment=0
        end
        @info "perf/train_s" value=t_train log_step_increment=0
        @info "perf/reanalyze_s" value=t_reanalyze log_step_increment=0
        @info "perf/iter_time" value=iter_time log_step_increment=0
        @info "buffer/size" value=buf_length(replay_buffer) log_step_increment=0
        @info "buffer/total_samples" value=server_state.total_samples[] log_step_increment=0
        buf_parts = partition_indices(replay_buffer)
        @info "buffer/contact_samples" value=length(buf_parts.contact) log_step_increment=0
        @info "buffer/race_samples" value=length(buf_parts.race) log_step_increment=0
        if USE_PER
            @info "per/beta" value=replay_buffer.beta log_step_increment=0
        end
        @info "train/lr" value=current_lr log_step_increment=0
        @info "train/gradient_steps" value=(train_result.contact.num_batches + train_result.race.num_batches) log_step_increment=0

        # Cluster performance
        @info "cluster/total_games_per_sec" value=cluster_stats.total_games_per_sec log_step_increment=0
        @info "cluster/total_samples_per_sec" value=cluster_stats.total_samples_per_sec log_step_increment=0
        @info "cluster/total_clients" value=cluster_stats.total_clients log_step_increment=0

        # Per-client stats
        for (cid, cstats) in cluster_stats.per_client
            @info "client/$(cid)/games_per_sec" value=cstats["games_per_sec"] log_step_increment=0
            @info "client/$(cid)/cpu_percent" value=cstats["cpu_percent"] log_step_increment=0
        end

        # Server stats
        @info "server/gpu_percent" value=server_stats["gpu_percent"] log_step_increment=0
        @info "server/gpu_memory_gb" value=server_stats["gpu_memory_used_gb"] log_step_increment=0
        @info "server/cpu_percent" value=server_stats["cpu_percent"] log_step_increment=0

        # Buffer reward distribution (sanity check: gammon/backgammon rates)
        n_buf = buf_length(replay_buffer)
        if n_buf > 0
            vals = @view replay_buffer.values[1:n_buf]
            n_bg_loss = count(v -> v <= -2.5f0, vals)
            n_g_loss  = count(v -> -2.5f0 < v <= -1.5f0, vals)
            n_loss    = count(v -> -1.5f0 < v < -0.5f0, vals)
            n_win     = count(v -> 0.5f0 < v < 1.5f0, vals)
            n_g_win   = count(v -> 1.5f0 <= v < 2.5f0, vals)
            n_bg_win  = count(v -> v >= 2.5f0, vals)
            @info "buffer/reward_bg_loss" value=n_bg_loss/n_buf log_step_increment=0
            @info "buffer/reward_g_loss" value=n_g_loss/n_buf log_step_increment=0
            @info "buffer/reward_loss" value=n_loss/n_buf log_step_increment=0
            @info "buffer/reward_win" value=n_win/n_buf log_step_increment=0
            @info "buffer/reward_g_win" value=n_g_win/n_buf log_step_increment=0
            @info "buffer/reward_bg_win" value=n_bg_win/n_buf log_step_increment=0
            @info "buffer/gammon_rate" value=(n_g_loss+n_bg_loss+n_g_win+n_bg_win)/n_buf log_step_increment=0
            @info "buffer/backgammon_rate" value=(n_bg_loss+n_bg_win)/n_buf log_step_increment=1
        else
            @info "placeholder" value=0 log_step_increment=1
        end
    end

    # Bearoff accuracy: NN equity vs exact table targets on bearoff positions
    # Measures how well the NN has learned bearoff evaluation
    try
        n_buf = buf_length(replay_buffer)
        bearoff_mask = findall(i -> replay_buffer.is_bearoff[i] && replay_buffer.has_equity[i], 1:n_buf)
        n_bo = length(bearoff_mask)
        if n_bo >= 100
            n_sample = min(1000, n_bo)
            sample_idx = bearoff_mask[randperm(n_sample)]

            # Get states and targets
            bo_states = replay_buffer.states[:, sample_idx]
            bo_eq_targets = replay_buffer.equities[:, sample_idx]  # 5×n, exact table values

            # NN forward pass (on GPU)
            bo_states_gpu = Flux.gpu(bo_states)
            nn = ARGS["training_mode"] == "race" ? race_network : contact_network
            P_hat, V_hat = Network.forward(nn, bo_states_gpu)
            nn_equity = Vector{Float32}(vec(Flux.cpu(V_hat)))  # normalized [-1,1]

            # Table equity from stored targets (joint formula, normalized)
            table_equity = Float32[
                ((2*bo_eq_targets[1,i] - 1) + (bo_eq_targets[2,i] - bo_eq_targets[4,i]) +
                 (bo_eq_targets[3,i] - bo_eq_targets[5,i])) / 3.0f0
                for i in 1:n_sample]

            # Compute MSE and correlation
            diffs = nn_equity .- table_equity
            bo_mse = sum(diffs .^ 2) / n_sample
            bo_mae = sum(abs.(diffs)) / n_sample

            nn_mean = sum(nn_equity) / n_sample
            tbl_mean = sum(table_equity) / n_sample
            nn_dev = nn_equity .- nn_mean
            tbl_dev = table_equity .- tbl_mean
            cov_val = sum(nn_dev .* tbl_dev) / n_sample
            nn_std = sqrt(sum(nn_dev .^ 2) / n_sample)
            tbl_std = sqrt(sum(tbl_dev .^ 2) / n_sample)
            bo_corr = (nn_std > 0 && tbl_std > 0) ? cov_val / (nn_std * tbl_std) : 0.0f0

            with_logger(TB_LOGGER) do
                @info "bearoff/nn_vs_table_mse" value=bo_mse log_step_increment=0
                @info "bearoff/nn_vs_table_mae" value=bo_mae log_step_increment=0
                @info "bearoff/nn_vs_table_corr" value=bo_corr log_step_increment=0
                @info "bearoff/n_samples" value=n_bo log_step_increment=0
            end
            @info "Bearoff accuracy" mse=round(bo_mse, digits=6) mae=round(bo_mae, digits=4) corr=round(bo_corr, digits=4) n_bearoff=n_bo n_sampled=n_sample
        end
    catch e
        @warn "Bearoff accuracy computation failed" exception=e
    end

    # Save client stats
    save_client_stats(server_state, joinpath(DATA_DIR, "clients.json"))

    # Checkpoint
    if iter % ARGS["checkpoint_interval"] == 0
        FluxLib.save_weights(joinpath(CHECKPOINT_DIR, "contact_iter_$iter.data"),
                             Flux.cpu(contact_network))
        FluxLib.save_weights(joinpath(CHECKPOINT_DIR, "race_iter_$iter.data"),
                             Flux.cpu(race_network))
        FluxLib.save_weights(joinpath(CHECKPOINT_DIR, "contact_latest.data"),
                             Flux.cpu(contact_network))
        FluxLib.save_weights(joinpath(CHECKPOINT_DIR, "race_latest.data"),
                             Flux.cpu(race_network))
        open(joinpath(CHECKPOINT_DIR, "iter.txt"), "w") do f
            print(f, iter)
        end
        @info "Saved checkpoint at iteration $iter"
    end

    # Buffer checkpoint (large — only every N iterations)
    buf_interval = ARGS["buffer_checkpoint_interval"]
    if buf_interval > 0 && iter % buf_interval == 0
        buf_path = joinpath(DATA_DIR, "buffer", "buffer_iter_$iter.jls")
        @info "Saving buffer checkpoint..." iter=iter size=replay_buffer.size
        t0 = time()
        save_buffer(replay_buffer, buf_path)
        @info "Buffer checkpoint saved" path=buf_path elapsed=round(time()-t0, digits=1)
    end

    # Distributed eval: create a non-blocking eval job for clients to work on
    if EVAL_ENABLED && iter % EVAL_INTERVAL == 0
        lock(EVAL_LOCK) do
            wv = ARGS["training_mode"] == "race" ? server_state.race_version[] : server_state.contact_version[]
            n_pos = length(EVAL_POSITIONS)
            EVAL_JOB[] = EvalManager.create_eval_job(iter, n_pos, wv; chunk_size=EVAL_CHUNK_SIZE)
            println("Eval job created for iter $iter: $(length(EVAL_JOB[].chunks)) chunks, $n_pos positions × 2 sides")
        end
    end

    # Check if a previous eval job completed — finalize and log results
    lock(EVAL_LOCK) do
        job = EVAL_JOB[]
        if job !== nothing && EvalManager.is_complete(job)
            result = EvalManager.finalize_eval(job)
            @info "Eval completed iter $(job.iter)" equity=round(result.equity, digits=3) win_pct=round(result.win_pct * 100, digits=1) games=result.num_games
            with_logger(TB_LOGGER) do
                @info "eval/equity" value=result.equity log_step_increment=0
                @info "eval/win_pct" value=result.win_pct * 100 log_step_increment=0
                @info "eval/white_equity" value=result.white_equity log_step_increment=0
                @info "eval/black_equity" value=result.black_equity log_step_increment=0
                @info "eval/value_mse" value=result.value_mse log_step_increment=0
                @info "eval/value_corr" value=result.value_corr log_step_increment=0
                @info "eval/games" value=result.num_games log_step_increment=0
            end
            EVAL_JOB[] = nothing
        end
    end
end

println("\nTraining complete!")
println("Checkpoints at: $CHECKPOINT_DIR")
println("TensorBoard: tensorboard --logdir $TB_DIR")

end # Threads.@spawn

# Keep server running — main thread runs libuv event loop for HTTP.jl
println("Server running on port $(ARGS["port"]). Training loop started on background thread.")
println("Press Ctrl+C to stop.")
try
    wait(training_task)
catch e
    e isa InterruptException || rethrow()
    println("\nShutting down...")
end
