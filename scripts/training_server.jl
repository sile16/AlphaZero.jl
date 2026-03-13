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
using Statistics
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
        "--seed"
            arg_type = Int
            default = 42

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

        # Bear-off (clients load locally)
        "--use-bearoff"
            action = :store_true
        "--bearoff-hard-targets"
            action = :store_true
        "--bearoff-truncation"
            action = :store_true

        # Eval
        "--eval-interval"
            help = "Run eval every N iterations (0 = disabled)"
            arg_type = Int
            default = 10
        "--eval-games"
            arg_type = Int
            default = 100

        # Checkpoints
        "--checkpoint-interval"
            arg_type = Int
            default = 10
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
println("PER: $(ARGS["use_per"])")
println("Reanalyze: $(ARGS["use_reanalyze"])")
println("=" ^ 60)
flush(stdout)

# Load packages
using AlphaZero
using AlphaZero: GI, MCTS, Network, FluxLib, MctsParams, LearningParams, Adam
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
    ENV["BACKGAMMON_OBS_TYPE"] = "minimal"
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

# PER buffer
replay_buffer = USE_PER ?
    PERBuffer(BUFFER_CAPACITY; beta_init=PER_BETA_INIT, annealing_iters=ARGS["total_iterations"]) :
    PERBuffer(BUFFER_CAPACITY)  # Still use PERBuffer for thread safety

# Training functions (extracted from train_distributed.jl)

function prepare_batch(batch, num_actions, use_gpu_flag, net)
    n = length(batch)
    W = ones(Float32, 1, n)
    X = hcat([s.state for s in batch]...)
    P = hcat([s.policy for s in batch]...)
    V = reshape(Float32[s.value for s in batch], 1, n)

    A = zeros(Float32, num_actions, n)
    IsChance = zeros(Float32, 1, n)
    for i in 1:n
        if batch[i].is_chance
            A[:, i] .= 1.0f0
            IsChance[1, i] = 1.0f0
        else
            A[:, i] .= Float32.(batch[i].policy .> 0)
        end
    end

    EqWin = zeros(Float32, 1, n)
    EqGW = zeros(Float32, 1, n)
    EqBGW = zeros(Float32, 1, n)
    EqGL = zeros(Float32, 1, n)
    EqBGL = zeros(Float32, 1, n)
    HasEquity = zeros(Float32, 1, n)
    for i in 1:n
        if batch[i].has_equity
            eq = batch[i].equity
            EqWin[1, i] = eq[1]
            EqGW[1, i] = eq[2]
            EqBGW[1, i] = eq[3]
            EqGL[1, i] = eq[4]
            EqBGL[1, i] = eq[5]
            HasEquity[1, i] = 1.0f0
        end
    end

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
        _, V̂_win, V̂_gw, V̂_bgw, V̂_gl, V̂_bgl, _ =
            FluxLib.forward_normalized_multihead(nn, X, A)
        equity = FluxLib.compute_equity(V̂_win, V̂_gw, V̂_bgw, V̂_gl, V̂_bgl)
        V̂_combined = equity ./ 3f0
        td = abs.(Flux.cpu(V̂_combined) .- Flux.cpu(V))
    else
        _, V̂, _ = Network.forward_normalized(nn, X, A)
        td = abs.(Flux.cpu(V̂) .- Flux.cpu(V))
    end
    return Float32.(vec(td))
end

function _train_model_on_samples!(samples_subset, indices_subset, network, opt_state)
    n = length(samples_subset)
    n < BATCH_SIZE && return (avg_loss=0.0, avg_Lp=0.0, avg_Lv=0.0, avg_Linv=0.0, num_batches=0)

    max_batches = max(1, ARGS["games_per_iteration"] * 200 ÷ BATCH_SIZE)
    num_batches = min(max(1, n ÷ BATCH_SIZE), max_batches)
    total_loss = 0.0
    total_Lp = 0.0
    total_Lv = 0.0
    total_Linv = 0.0

    for _ in 1:num_batches
        batch_idx = rand(1:n, BATCH_SIZE)
        batch = [samples_subset[i] for i in batch_idx]
        buf_indices = [indices_subset[i] for i in batch_idx]

        batch_data = prepare_batch(batch, NUM_ACTIONS, USE_GPU, network)
        Wmean = mean(batch_data.W)
        Hp = 0.0f0

        loss_fn(nn) = losses(nn, LEARNING_PARAMS, Wmean, Hp, batch_data)[1]
        loss, grads = Flux.withgradient(loss_fn, network)
        Flux.update!(opt_state, network, grads[1])

        L, Lp, Lv, _, Linv = losses(network, LEARNING_PARAMS, Wmean, Hp, batch_data)
        total_loss += Float64(L)
        total_Lp += Float64(Lp)
        total_Lv += Float64(Lv)
        total_Linv += Float64(Linv)

        # Update PER priorities
        if USE_PER
            td_errors = compute_td_errors(network, batch_data)
            per_update_priorities!(replay_buffer, buf_indices, td_errors)
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

    # Get snapshot of samples (under lock)
    all_samples, contact_samples, contact_indices, race_samples, race_indices = lock(replay_buffer.lock) do
        all = replay_buffer.samples
        cs, ci, rs, ri = [], Int[], [], Int[]
        for i in 1:length(all)
            s = all[i]
            if s.is_contact
                push!(cs, s); push!(ci, i)
            else
                push!(rs, s); push!(ri, i)
            end
        end
        (all, cs, ci, rs, ri)
    end

    contact_result = _train_model_on_samples!(contact_samples, contact_indices, contact_network, contact_opt_state)
    race_result = _train_model_on_samples!(race_samples, race_indices, race_network, race_opt_state)

    return (contact=contact_result, race=race_result)
end

function reanalyze_buffer!()
    USE_REANALYZE || return 0
    samples = lock(replay_buffer.lock) do
        replay_buffer.samples
    end
    n = length(samples)
    n == 0 && return 0

    num_to_reanalyze = max(1, round(Int, n * REANALYZE_FRACTION))
    reanalyze_indices = randperm(n)[1:min(num_to_reanalyze, n)]

    batch_size = min(4096, length(reanalyze_indices))
    total_updated = 0

    for batch_start in 1:batch_size:length(reanalyze_indices)
        batch_end = min(batch_start + batch_size - 1, length(reanalyze_indices))
        batch_indices = reanalyze_indices[batch_start:batch_end]
        batch = lock(replay_buffer.lock) do
            [samples[i] for i in batch_indices]
        end

        for (is_contact_flag, net) in [(true, contact_network), (false, race_network)]
            sub_batch_map = [(j, bi) for (j, (bi, s)) in enumerate(zip(batch_indices, batch)) if s.is_contact == is_contact_flag]
            isempty(sub_batch_map) && continue

            sub_batch = [batch[j] for (j, _) in sub_batch_map]
            sub_batch_data = prepare_batch(sub_batch, NUM_ACTIONS, USE_GPU, net)

            _, V̂_win, V̂_gw, V̂_bgw, V̂_gl, V̂_bgl, _ =
                FluxLib.forward_normalized_multihead(net, sub_batch_data.X, sub_batch_data.A)
            equity = FluxLib.compute_equity(V̂_win, V̂_gw, V̂_bgw, V̂_gl, V̂_bgl)
            new_values = Float32.(vec(Flux.cpu(equity ./ 3f0)))

            new_eq_win = Float32.(vec(Flux.cpu(V̂_win)))
            new_eq_gw = Float32.(vec(Flux.cpu(V̂_gw)))
            new_eq_bgw = Float32.(vec(Flux.cpu(V̂_bgw)))
            new_eq_gl = Float32.(vec(Flux.cpu(V̂_gl)))
            new_eq_bgl = Float32.(vec(Flux.cpu(V̂_bgl)))

            α_blend = 0.5f0
            lock(replay_buffer.lock) do
                for (k, (_, buf_idx)) in enumerate(sub_batch_map)
                    old = samples[buf_idx]
                    old_val = old.value
                    blended_val = (1f0 - α_blend) * old_val + α_blend * new_values[k]

                    if old.has_equity
                        old_eq = old.equity
                        new_eq = Float32[
                            (1f0 - α_blend) * old_eq[1] + α_blend * new_eq_win[k],
                            (1f0 - α_blend) * old_eq[2] + α_blend * new_eq_gw[k],
                            (1f0 - α_blend) * old_eq[3] + α_blend * new_eq_bgw[k],
                            (1f0 - α_blend) * old_eq[4] + α_blend * new_eq_gl[k],
                            (1f0 - α_blend) * old_eq[5] + α_blend * new_eq_bgl[k],
                        ]
                        samples[buf_idx] = merge(old, (value=blended_val, equity=new_eq))
                    else
                        samples[buf_idx] = merge(old, (value=blended_val,))
                    end

                    if USE_PER
                        td_error = abs(new_values[k] - old_val)
                        replay_buffer.priorities[buf_idx] = td_error
                    end
                    total_updated += 1
                end
            end
        end
    end

    return total_updated
end

# TensorBoard logger
const TB_LOGGER = TBLogger(TB_DIR, tb_overwrite)

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
    "use_bearoff" => ARGS["use_bearoff"],
    "bearoff_hard_targets" => ARGS["bearoff_hard_targets"],
    "bearoff_truncation" => ARGS["bearoff_truncation"],
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

# Samples threshold for one iteration
const SAMPLES_PER_ITERATION = ARGS["games_per_iteration"] * 200  # ~200 samples per game

# Main training loop — runs on a spawned thread so the main thread's
# libuv event loop stays active for HTTP.jl to handle requests.
training_task = Threads.@spawn begin

for iter in (START_ITER + 1):ARGS["total_iterations"]
    # Wait for enough new samples
    target_samples = iter * SAMPLES_PER_ITERATION
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

    # Train on buffer (GPU)
    t0 = time()
    train_result = train_on_buffer!()
    t_train = time() - t0

    contact_loss = train_result.contact.avg_loss
    race_loss = train_result.race.avg_loss
    avg_loss = (contact_loss + race_loss) / 2

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
    @info "Iteration $iter" avg_loss contact_loss race_loss buffer_size=buf_length(replay_buffer) total_samples=server_state.total_samples[] n_clients=length(server_state.clients) iter_time t_train t_reanalyze n_reanalyzed

    # Collect server and cluster stats
    server_stats = collect_server_stats()
    cluster_stats = get_cluster_stats(server_state)

    # Log to TensorBoard
    with_logger(TB_LOGGER) do
        @info "loss/avg" value=avg_loss log_step_increment=0
        @info "loss/contact" value=contact_loss log_step_increment=0
        @info "loss/race" value=race_loss log_step_increment=0
        @info "perf/train_s" value=t_train log_step_increment=0
        @info "perf/reanalyze_s" value=t_reanalyze log_step_increment=0
        @info "perf/iter_time" value=iter_time log_step_increment=0
        @info "buffer/size" value=buf_length(replay_buffer) log_step_increment=0
        @info "buffer/total_samples" value=server_state.total_samples[] log_step_increment=0

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
        @info "server/cpu_percent" value=server_stats["cpu_percent"] log_step_increment=1
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
