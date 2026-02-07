#!/usr/bin/env julia

"""
Distributed training for AlphaZero.jl using Julia's Distributed stdlib.

Each worker runs on a separate process with its own network copy for inference.
The coordinator manages the replay buffer, training, and weight distribution.

Usage:
    # Local single-machine (spawns worker processes)
    julia --project scripts/train_cluster.jl \\
        --game=backgammon-deterministic \\
        --network-type=fcresnet-multihead \\
        --num-workers=4 \\
        --total-iterations=50

    # With GnuBG evaluation
    julia --project scripts/train_cluster.jl \\
        --game=backgammon-deterministic \\
        --eval-vs-gnubg \\
        --num-workers=4

    # View TensorBoard logs
    tensorboard --logdir=sessions/<session>/tensorboard
"""

using ArgParse
using CUDA
using Dates
using Distributed
using Random

# Configure CUDA BEFORE loading AlphaZero
if CUDA.functional()
    CUDA.allowscalar(false)
    @info "CUDA configuration:"
    @info "  Device: $(CUDA.name(CUDA.device()))"
    @info "  Memory: $(round(CUDA.total_memory() / 1e9, digits=2)) GB total"
    @info "  Free:   $(round(CUDA.available_memory() / 1e9, digits=2)) GB available"
end

using AlphaZero
using AlphaZero.Cluster
using AlphaZero: GI, Network, FluxLib, ReanalyzeConfig
using AlphaZero.TensorBoard
import Flux
using Statistics: mean
using Base.Threads

# Pre-load game modules at top level to avoid world age issues
const GAMES_DIR = joinpath(@__DIR__, "..", "games")
include(joinpath(GAMES_DIR, "backgammon-deterministic", "main.jl"))
include(joinpath(GAMES_DIR, "backgammon", "main.jl"))

# Check if GnuBG evaluation is requested (load GnubgPlayer if so)
function check_gnubg_mode()
    return "--eval-vs-gnubg" in ARGS
end

const LOAD_GNUBG = check_gnubg_mode()

if LOAD_GNUBG
    @info "Loading GnubgPlayer for evaluation..."
    include(joinpath(@__DIR__, "GnubgPlayer.jl"))
    using .GnubgPlayer
end

function parse_args()
    s = ArgParseSettings(
        description="Distributed AlphaZero training using Julia Distributed",
        autofix_names=true
    )

    @add_arg_table! s begin
        # Game configuration
        "--game"
            help = "Game to train on"
            arg_type = String
            default = "backgammon-deterministic"

        # Network configuration
        "--network-type"
            help = "Network type: simple, resnet, fcresnet, fcresnet-multihead"
            arg_type = String
            default = "fcresnet-multihead"
        "--network-width"
            help = "Network width"
            arg_type = Int
            default = 128
        "--network-blocks"
            help = "Number of residual blocks"
            arg_type = Int
            default = 3
        "--load-network"
            help = "Path to network checkpoint to load"
            arg_type = String
            default = nothing

        # Training configuration
        "--num-workers"
            help = "Number of self-play workers (local processes)"
            arg_type = Int
            default = 4
        "--total-iterations"
            help = "Total training iterations"
            arg_type = Int
            default = 50
        "--games-per-iteration"
            help = "Games to collect per iteration"
            arg_type = Int
            default = 50
        "--batch-size"
            help = "Training batch size"
            arg_type = Int
            default = 256

        # MCTS configuration
        "--mcts-iters"
            help = "MCTS iterations per move"
            arg_type = Int
            default = 100
        "--cpuct"
            help = "MCTS exploration constant"
            arg_type = Float64
            default = 2.0

        # Buffer configuration
        "--buffer-capacity"
            help = "Replay buffer capacity"
            arg_type = Int
            default = 100000

        # Learning configuration
        "--learning-rate"
            help = "Learning rate"
            arg_type = Float64
            default = 1e-3
        "--l2-reg"
            help = "L2 regularization"
            arg_type = Float64
            default = 1e-4

        # Checkpoint configuration
        "--checkpoint-interval"
            help = "Iterations between checkpoints"
            arg_type = Int
            default = 10
        "--checkpoint-dir"
            help = "Directory for checkpoints"
            arg_type = String
            default = "checkpoints"

        # Logging configuration
        "--host-id"
            help = "Host identifier for multi-machine tracking"
            arg_type = String
            default = nothing
        "--no-tensorboard"
            help = "Disable TensorBoard logging"
            action = :store_true

        # Evaluation
        "--eval-interval"
            help = "Run evaluation every N iterations (0 to disable)"
            arg_type = Int
            default = 10
        "--eval-games"
            help = "Number of games per evaluation"
            arg_type = Int
            default = 50
        "--eval-mcts-iters"
            help = "MCTS iterations for evaluation (default: same as training)"
            arg_type = Int
            default = 0

        # Final evaluation (parallel)
        "--final-eval-games"
            help = "Number of games for final evaluation (0 to skip)"
            arg_type = Int
            default = 1000
        "--final-eval-workers"
            help = "Number of parallel workers for final evaluation (0 = use all threads)"
            arg_type = Int
            default = 0

        # GnuBG evaluation
        "--eval-vs-gnubg"
            help = "Evaluate against GnuBG during training (in addition to random)"
            action = :store_true
        "--gnubg-ply"
            help = "GnuBG lookahead depth for evaluation (0=neural net only, 1=1-ply)"
            arg_type = Int
            default = 0
        "--gnubg-eval-games"
            help = "Number of games per GnuBG evaluation (default: same as --eval-games)"
            arg_type = Int
            default = 0

        # Reproducibility
        "--seed"
            help = "Random seed for reproducibility (0 = use random seed)"
            arg_type = Int
            default = 0

        # Prioritized Experience Replay (PER)
        "--per"
            help = "Enable Prioritized Experience Replay (Schaul et al. 2016)"
            action = :store_true
        "--per-alpha"
            help = "PER priority exponent (0=uniform, 1=full prioritization)"
            arg_type = Float64
            default = 0.6
        "--per-beta"
            help = "PER importance sampling initial beta (anneals to 1.0)"
            arg_type = Float64
            default = 0.4
        "--per-epsilon"
            help = "PER small constant for numerical stability"
            arg_type = Float64
            default = 0.01

        # Reanalyze (MuZero-style)
        "--reanalyze"
            help = "Enable MuZero-style reanalysis of replay buffer samples"
            action = :store_true
        "--reanalyze-batch-size"
            help = "Batch size for reanalysis"
            arg_type = Int
            default = 256
        "--reanalyze-alpha"
            help = "Blend factor for reanalysis (0=keep old, 1=use new)"
            arg_type = Float64
            default = 0.5
        "--reanalyze-interval"
            help = "Run reanalysis every N training iterations"
            arg_type = Int
            default = 1

        # Architecture
        "--distributed"
            help = "Use fully concurrent architecture (async reanalyze + eval)"
            action = :store_true

        # Misc
        "--no-gpu"
            help = "Disable GPU (CPU only)"
            action = :store_true
        "--verbose"
            help = "Verbose output"
            action = :store_true
    end

    return ArgParse.parse_args(s)
end

function get_game_spec(game_name::String)
    if game_name == "connect-four"
        return Examples.games["connect-four"]
    elseif game_name == "tictactoe"
        return Examples.games["tictactoe"]
    elseif game_name == "backgammon-deterministic"
        return BackgammonDeterministic.GameSpec()
    elseif game_name == "backgammon"
        return Backgammon.GameSpec()
    else
        error("Unknown game: $game_name. Available: connect-four, tictactoe, backgammon-deterministic, backgammon")
    end
end

function create_network_constructor(args)
    network_type = args["network_type"]
    width = args["network_width"]
    blocks = args["network_blocks"]

    if network_type == "simple"
        return gspec -> NetLib.SimpleNet(gspec, NetLib.SimpleNetHP(width=width, depth_common=blocks))
    elseif network_type == "resnet"
        return gspec -> NetLib.ResNet(gspec, NetLib.ResNetHP(num_blocks=blocks, num_filters=width))
    elseif network_type == "fcresnet"
        return gspec -> NetLib.FCResNet(gspec, NetLib.FCResNetHP(width=width, num_blocks=blocks))
    elseif network_type == "fcresnet-multihead"
        return gspec -> NetLib.FCResNetMultiHead(gspec, NetLib.FCResNetMultiHeadHP(width=width, num_blocks=blocks))
    else
        error("Unknown network type: $network_type")
    end
end

"""
Evaluate current network against a random player (single-threaded).
Returns Dict with eval metrics.
"""
function evaluate_vs_random(network, gspec, num_games::Int, mcts_iters::Int, use_gpu::Bool)
    # Create evaluation network
    eval_network = Network.copy(network, on_gpu=use_gpu, test_mode=true)

    # Create MCTS params for evaluation (lower temperature, no noise)
    eval_mcts = MctsParams(
        num_iters_per_turn=mcts_iters,
        cpuct=1.5,
        temperature=ConstSchedule(0.2),
        dirichlet_noise_ϵ=0.0,
        dirichlet_noise_α=0.3,
        gamma=1.0
    )

    az_player = MctsPlayer(gspec, eval_network, eval_mcts)
    random_player = RandomPlayer()

    games_per_side = num_games ÷ 2

    # Play as white
    rewards_white = Float64[]
    for _ in 1:games_per_side
        trace = play_game(gspec, TwoPlayers(az_player, random_player))
        push!(rewards_white, total_reward(trace))
        reset_player!(az_player)
    end

    # Play as black
    rewards_black = Float64[]
    for _ in 1:games_per_side
        trace = play_game(gspec, TwoPlayers(random_player, az_player))
        push!(rewards_black, -total_reward(trace))  # Negate for black's perspective
        reset_player!(az_player)
    end

    avg_white = isempty(rewards_white) ? 0.0 : sum(rewards_white) / length(rewards_white)
    avg_black = isempty(rewards_black) ? 0.0 : sum(rewards_black) / length(rewards_black)
    combined = (avg_white + avg_black) / 2

    return Dict{String, Any}(
        "eval/vs_random_white" => avg_white,
        "eval/vs_random_black" => avg_black,
        "eval/vs_random_combined" => combined,
        "eval/games" => num_games
    )
end

"""
Parallel evaluation using multiple threads.
Each thread gets its own network copy for thread safety.
Returns Dict with eval metrics.
"""
function parallel_evaluate_vs_random(network, gspec, num_games::Int, mcts_iters::Int, use_gpu::Bool; num_workers::Int=Threads.nthreads())
    games_per_side = num_games ÷ 2

    # Thread-safe result storage
    results_lock = ReentrantLock()
    rewards_white = Float64[]
    rewards_black = Float64[]
    sizehint!(rewards_white, games_per_side)
    sizehint!(rewards_black, games_per_side)

    # Progress tracking
    games_done = Threads.Atomic{Int}(0)
    total_games = num_games

    # MCTS params for evaluation
    eval_mcts = MctsParams(
        num_iters_per_turn=mcts_iters,
        cpuct=1.5,
        temperature=ConstSchedule(0.2),
        dirichlet_noise_ϵ=0.0,
        dirichlet_noise_α=0.3,
        gamma=1.0
    )

    # Function to play games as a specific color
    function play_games_threaded!(as_white::Bool, num_to_play::Int)
        # Divide work among threads
        games_per_worker = cld(num_to_play, num_workers)

        tasks = Task[]
        for worker_id in 1:num_workers
            task = Threads.@spawn begin
                # Each thread gets its own network copy (CPU for thread safety)
                worker_network = Network.copy(network, on_gpu=false, test_mode=true)
                worker_player = MctsPlayer(gspec, worker_network, eval_mcts)
                random_player = RandomPlayer()

                local_rewards = Float64[]

                start_idx = (worker_id - 1) * games_per_worker + 1
                end_idx = min(worker_id * games_per_worker, num_to_play)

                for _ in start_idx:end_idx
                    if as_white
                        trace = play_game(gspec, TwoPlayers(worker_player, random_player))
                        push!(local_rewards, total_reward(trace))
                    else
                        trace = play_game(gspec, TwoPlayers(random_player, worker_player))
                        push!(local_rewards, -total_reward(trace))
                    end
                    reset_player!(worker_player)

                    # Update progress
                    Threads.atomic_add!(games_done, 1)
                    done = games_done[]
                    if done % max(1, total_games ÷ 10) == 0
                        @info "Evaluation progress: $done / $total_games games"
                    end
                end

                # Merge results
                lock(results_lock) do
                    if as_white
                        append!(rewards_white, local_rewards)
                    else
                        append!(rewards_black, local_rewards)
                    end
                end
            end
            push!(tasks, task)
        end

        # Wait for all threads
        for task in tasks
            wait(task)
        end
    end

    @info "Starting parallel evaluation with $num_workers workers ($num_games games total)..."

    # Play as white
    @info "Playing $games_per_side games as white..."
    play_games_threaded!(true, games_per_side)

    # Play as black
    @info "Playing $games_per_side games as black..."
    play_games_threaded!(false, games_per_side)

    avg_white = isempty(rewards_white) ? 0.0 : sum(rewards_white) / length(rewards_white)
    avg_black = isempty(rewards_black) ? 0.0 : sum(rewards_black) / length(rewards_black)
    combined = (avg_white + avg_black) / 2

    return Dict{String, Any}(
        "eval/vs_random_white" => avg_white,
        "eval/vs_random_black" => avg_black,
        "eval/vs_random_combined" => combined,
        "eval/games" => num_games,
        "eval/workers" => num_workers
    )
end

"""
Evaluate current network against GnuBG (single-threaded).
Returns Dict with eval metrics.
"""
function evaluate_vs_gnubg(network, gspec, num_games::Int, mcts_iters::Int, use_gpu::Bool; gnubg_ply::Int=0)
    if !LOAD_GNUBG
        @warn "GnuBG evaluation requested but GnubgPlayer not loaded"
        return Dict{String, Any}()
    end

    # Create evaluation network
    eval_network = Network.copy(network, on_gpu=use_gpu, test_mode=true)

    # Create MCTS params for evaluation (lower temperature, no noise)
    eval_mcts = MctsParams(
        num_iters_per_turn=mcts_iters,
        cpuct=1.5,
        temperature=ConstSchedule(0.2),
        dirichlet_noise_ϵ=0.0,
        dirichlet_noise_α=0.3,
        gamma=1.0
    )

    az_player = MctsPlayer(gspec, eval_network, eval_mcts)
    gnubg_player = GnubgPlayer.GnubgBaseline(ply=gnubg_ply)

    games_per_side = num_games ÷ 2

    # Play as white against GnuBG
    rewards_white = Float64[]
    for i in 1:games_per_side
        trace = play_game(gspec, TwoPlayers(az_player, gnubg_player))
        push!(rewards_white, total_reward(trace))
        reset_player!(az_player)

        # Progress reporting
        if i % max(1, games_per_side ÷ 5) == 0
            @info "GnuBG eval as white: $i / $games_per_side"
        end
    end

    # Play as black against GnuBG
    rewards_black = Float64[]
    for i in 1:games_per_side
        trace = play_game(gspec, TwoPlayers(gnubg_player, az_player))
        push!(rewards_black, -total_reward(trace))  # Negate for black's perspective
        reset_player!(az_player)

        # Progress reporting
        if i % max(1, games_per_side ÷ 5) == 0
            @info "GnuBG eval as black: $i / $games_per_side"
        end
    end

    avg_white = isempty(rewards_white) ? 0.0 : sum(rewards_white) / length(rewards_white)
    avg_black = isempty(rewards_black) ? 0.0 : sum(rewards_black) / length(rewards_black)
    combined = (avg_white + avg_black) / 2

    # Calculate win rates
    wins_white = count(r -> r > 0, rewards_white)
    wins_black = count(r -> r > 0, rewards_black)
    wr_white = isempty(rewards_white) ? 0.0 : wins_white / length(rewards_white)
    wr_black = isempty(rewards_black) ? 0.0 : wins_black / length(rewards_black)
    wr_combined = (wr_white + wr_black) / 2

    return Dict{String, Any}(
        "eval/vs_gnubg$(gnubg_ply)ply_white" => avg_white,
        "eval/vs_gnubg$(gnubg_ply)ply_black" => avg_black,
        "eval/vs_gnubg$(gnubg_ply)ply_combined" => combined,
        "eval/vs_gnubg$(gnubg_ply)ply_wr_white" => wr_white,
        "eval/vs_gnubg$(gnubg_ply)ply_wr_black" => wr_black,
        "eval/vs_gnubg$(gnubg_ply)ply_wr_combined" => wr_combined,
        "eval/gnubg_games" => num_games,
        "eval/gnubg_ply" => gnubg_ply
    )
end

"""
Main entry point.
"""
function main()
    args = parse_args()

    if args["verbose"]
        ENV["JULIA_DEBUG"] = "AlphaZero"
    end

    # GnuBG availability is determined at load time via LOAD_GNUBG constant
    eval_vs_gnubg = args["eval_vs_gnubg"] && LOAD_GNUBG
    if args["eval_vs_gnubg"] && !LOAD_GNUBG
        @warn "GnuBG evaluation was requested but GnubgPlayer failed to load"
    end

    # Get git commit hash for reproducibility
    git_hash = try
        strip(read(`git rev-parse HEAD`, String))
    catch
        "unknown"
    end
    git_dirty = try
        !isempty(strip(read(`git status --porcelain`, String)))
    catch
        false
    end

    # Initialize random seed for reproducibility
    seed = args["seed"]
    if seed == 0
        seed = Int(rand(UInt32))  # Generate random seed as Int for type compatibility
    end
    Random.seed!(seed)
    @info "Random seed: $seed"

    @info "=" ^ 60
    @info "AlphaZero Training (TensorBoard logging)"
    @info "=" ^ 60
    @info "Git commit: $(git_hash[1:min(8, length(git_hash))])$(git_dirty ? " (dirty)" : "")"
    @info "Seed: $seed"

    # Load game
    gspec = get_game_spec(args["game"])
    @info "Game: $(args["game"])"

    # Create network constructor
    network_constructor = create_network_constructor(args)

    # Create a temporary network to check parameters
    temp_network = network_constructor(gspec)
    num_params = Network.num_parameters(temp_network)
    @info "Network: $(args["network_type"]) (width=$(args["network_width"]), blocks=$(args["network_blocks"]), params=$num_params)"

    # Check if loading from checkpoint
    load_network_path = args["load_network"]
    if !isnothing(load_network_path)
        @info "Will load weights from: $load_network_path"
        # Verify file exists
        if !isfile(load_network_path)
            error("Checkpoint file not found: $load_network_path")
        end
    end

    # Determine host_id
    host_id = args["host_id"]
    if isnothing(host_id)
        host_id = gethostname()
    end

    use_gpu = !args["no_gpu"] && CUDA.functional()

    # Configuration
    @info "Configuration:"
    @info "  Workers: $(args["num_workers"])"
    @info "  Buffer capacity: $(args["buffer_capacity"])"
    @info "  Batch size: $(args["batch_size"])"
    @info "  Games per iteration: $(args["games_per_iteration"])"
    @info "  MCTS iterations: $(args["mcts_iters"])"
    @info "  Total iterations: $(args["total_iterations"])"
    @info "  GPU enabled: $use_gpu"
    @info "  Host ID: $host_id"

    tb_enabled = !args["no_tensorboard"]
    @info "  TensorBoard: $(tb_enabled ? "enabled" : "disabled")"

    eval_interval = args["eval_interval"]
    eval_mcts = args["eval_mcts_iters"] > 0 ? args["eval_mcts_iters"] : args["mcts_iters"]
    if eval_interval > 0
        @info "  Eval: every $eval_interval iters, $(args["eval_games"]) games, $eval_mcts MCTS iters"
    else
        @info "  Eval: disabled"
    end

    final_eval_games = args["final_eval_games"]
    final_eval_workers = args["final_eval_workers"] > 0 ? args["final_eval_workers"] : nthreads()
    if final_eval_games > 0
        @info "  Final eval: $final_eval_games games, $final_eval_workers parallel workers"
    else
        @info "  Final eval: disabled"
    end

    # GnuBG evaluation config
    gnubg_ply = args["gnubg_ply"]
    gnubg_eval_games = args["gnubg_eval_games"] > 0 ? args["gnubg_eval_games"] : args["eval_games"]
    if eval_vs_gnubg && eval_interval > 0
        @info "  GnuBG eval: every $eval_interval iters, $gnubg_eval_games games, $(gnubg_ply)-ply"
    end

    if use_gpu
        @info "  GPU memory: $(round(CUDA.available_memory() / 1e9, digits=2)) GB free"
    end

    @info "=" ^ 60

    # Create session directory
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    session_dir = joinpath("sessions", "cluster_$(timestamp)")
    checkpoint_dir = joinpath(session_dir, args["checkpoint_dir"])
    tb_dir = joinpath(session_dir, "tensorboard")
    mkpath(checkpoint_dir)
    mkpath(tb_dir)
    @info "Session directory: $session_dir"

    # Initialize TensorBoard
    if tb_enabled
        run_name = "$(args["game"])-$(args["network_type"])-$(Dates.format(now(), "mmdd-HHMM"))"
        tb_init(logdir=tb_dir, run_name=run_name)

        # Log configuration
        config = Dict{String, Any}(
            "game" => args["game"],
            "network_type" => args["network_type"],
            "network_width" => args["network_width"],
            "network_blocks" => args["network_blocks"],
            "network_params" => num_params,
            "num_workers" => args["num_workers"],
            "total_iterations" => args["total_iterations"],
            "games_per_iteration" => args["games_per_iteration"],
            "batch_size" => args["batch_size"],
            "learning_rate" => args["learning_rate"],
            "l2_reg" => args["l2_reg"],
            "mcts_iters" => args["mcts_iters"],
            "cpuct" => args["cpuct"],
            "buffer_capacity" => args["buffer_capacity"],
            "eval_interval" => eval_interval,
            "eval_games" => args["eval_games"],
            "eval_vs_gnubg" => eval_vs_gnubg,
            "gnubg_ply" => gnubg_ply,
            "host_id" => host_id,
            "use_gpu" => use_gpu,
            "git_commit" => git_hash,
            "seed" => seed,
        )
        tb_log_config(config)
    end

    # Save reproducibility info to session directory
    open(joinpath(session_dir, "run_info.txt"), "w") do f
        println(f, "# AlphaZero.jl Training Run")
        println(f, "timestamp: $timestamp")
        println(f, "git_commit: $git_hash")
        println(f, "git_dirty: $git_dirty")
        println(f, "seed: $seed")
        println(f, "")
        println(f, "# Arguments")
        for (k, v) in sort(collect(args), by=first)
            println(f, "$k: $v")
        end
    end

    # Track start time
    start_time = time()

    # TensorBoard logging callback
    tb_log_fn = if tb_enabled
        (metrics) -> begin
            try
                # Add system metrics periodically
                if haskey(metrics, "train/iteration") && metrics["train/iteration"] % 5 == 0
                    cuda_mod = use_gpu ? CUDA : nothing
                    sys_metrics = all_system_metrics(host_id=host_id, cuda_module=cuda_mod)
                    merge!(metrics, sys_metrics)
                end
                tb_log(metrics, step=get(metrics, "train/iteration", nothing))
            catch e
                @warn "TensorBoard log failed" exception=e
            end
        end
    else
        nothing
    end

    # Create MCTS params
    mcts_params = MctsParams(
        num_iters_per_turn=args["mcts_iters"],
        cpuct=args["cpuct"],
        dirichlet_noise_ϵ=0.25,
        dirichlet_noise_α=1.0,
        temperature=ConstSchedule(1.0),
        gamma=1.0
    )

    # Create learning params
    learning_params = LearningParams(
        batch_size=args["batch_size"],
        loss_computation_batch_size=args["batch_size"],
        optimiser=Adam(lr=args["learning_rate"]),
        l2_regularization=Float32(args["l2_reg"]),
        use_gpu=use_gpu,
        samples_weighing_policy=CONSTANT_WEIGHT,
        min_checkpoints_per_epoch=1,
        max_batches_per_checkpoint=100,
        num_checkpoints=1
    )

    # Evaluation callback
    eval_fn = if eval_interval > 0
        (network) -> begin
            results = Dict{String, Any}()
            eval_start = time()

            # Always evaluate vs random (fast baseline)
            @info "Running evaluation ($(args["eval_games"]) games vs random)..."
            random_results = evaluate_vs_random(network, gspec, args["eval_games"], eval_mcts, use_gpu)
            merge!(results, random_results)
            @info "vs Random: white=$(round(results["eval/vs_random_white"], digits=3)), " *
                  "black=$(round(results["eval/vs_random_black"], digits=3)), " *
                  "combined=$(round(results["eval/vs_random_combined"], digits=3))"

            # Optionally evaluate vs GnuBG (meaningful generalization test)
            if eval_vs_gnubg
                @info "Running GnuBG evaluation ($gnubg_eval_games games vs $(gnubg_ply)-ply)..."
                gnubg_results = evaluate_vs_gnubg(network, gspec, gnubg_eval_games, eval_mcts, use_gpu; gnubg_ply=gnubg_ply)
                merge!(results, gnubg_results)
                @info "vs GnuBG $(gnubg_ply)-ply: white=$(round(gnubg_results["eval/vs_gnubg$(gnubg_ply)ply_white"], digits=3)), " *
                      "black=$(round(gnubg_results["eval/vs_gnubg$(gnubg_ply)ply_black"], digits=3)), " *
                      "combined=$(round(gnubg_results["eval/vs_gnubg$(gnubg_ply)ply_combined"], digits=3)) " *
                      "(wr=$(round(100*gnubg_results["eval/vs_gnubg$(gnubg_ply)ply_wr_combined"], digits=1))%)"
            end

            results["eval/time_s"] = time() - eval_start
            results
        end
    else
        nothing
    end

    # Create PER config
    per_config = Cluster.PrioritizedSamplingConfig(
        enabled=args["per"],
        alpha=Float32(args["per_alpha"]),
        beta=Float32(args["per_beta"]),
        beta_annealing_steps=args["total_iterations"],  # Anneal over full training
        epsilon=Float32(args["per_epsilon"]),
        initial_priority=1.0f0
    )

    # Create Reanalyze config
    reanalyze_config = ReanalyzeConfig(
        enabled=args["reanalyze"],
        batch_size=args["reanalyze_batch_size"],
        update_interval=args["reanalyze_interval"],
        reanalyze_alpha=Float32(args["reanalyze_alpha"]),
        max_reanalyze_count=5,
        prioritize_high_td=true,
        log_interval=10
    )

    # Run training
    coord = nothing
    try
        if args["distributed"]
            # Use new concurrent architecture
            @info "Using distributed (concurrent) architecture"
            coord = start_distributed_training(
                gspec,
                network_constructor,
                learning_params,
                mcts_params;
                num_workers=args["num_workers"],
                buffer_capacity=args["buffer_capacity"],
                batch_size=args["batch_size"],
                checkpoint_interval=args["checkpoint_interval"],
                total_iterations=args["total_iterations"],
                games_per_iteration=args["games_per_iteration"],
                use_gpu=use_gpu,
                checkpoint_dir=checkpoint_dir,
                wandb_log=tb_log_fn,
                eval_games=args["eval_games"],
                eval_interval=eval_interval,
                seed=seed,
                load_network_path=load_network_path,
                per_config=per_config,
                reanalyze_config=reanalyze_config
            )
        else
            # Use original sequential architecture
            coord = start_cluster_training(
                gspec,
                network_constructor,
                learning_params,
                mcts_params;
                num_workers=args["num_workers"],
                buffer_capacity=args["buffer_capacity"],
                batch_size=args["batch_size"],
                checkpoint_interval=args["checkpoint_interval"],
                total_iterations=args["total_iterations"],
                games_per_iteration=args["games_per_iteration"],
                use_gpu=use_gpu,
                checkpoint_dir=checkpoint_dir,
                wandb_log=tb_log_fn,
                eval_fn=eval_fn,
                eval_interval=eval_interval,
                seed=seed,
                load_network_path=load_network_path,
                per_config=per_config,
                reanalyze_config=reanalyze_config
            )
        end
    catch e
        if e isa InterruptException
            @info "Training interrupted"
        else
            @error "Training error" exception=(e, catch_backtrace())
        end
    end

    total_time = time() - start_time

    @info "=" ^ 60
    @info "Training Summary:"
    if !isnothing(coord)
        @info "  Total iterations: $(coord.iteration)"
        @info "  Total games: $(coord.total_games)"
        @info "  Total samples: $(coord.total_samples)"
        @info "  Final buffer size: $(length(coord.buffer))"
    end
    @info "  Total time: $(round(total_time / 60, digits=2)) minutes"
    @info "  Session saved to: $session_dir"
    if tb_enabled
        @info "  TensorBoard: tensorboard --logdir=$tb_dir"
    end
    @info "=" ^ 60

    # Final evaluation (parallel for speed)
    final_eval_games = args["final_eval_games"]
    if final_eval_games > 0 && !isnothing(coord)
        final_eval_workers = args["final_eval_workers"]
        if final_eval_workers <= 0
            final_eval_workers = nthreads()
        end

        @info "=" ^ 60
        @info "Running parallel final evaluation..."
        @info "  Games: $final_eval_games"
        @info "  Workers: $final_eval_workers"
        @info "  MCTS iters: $eval_mcts"
        @info "=" ^ 60

        eval_start = time()
        final_results = parallel_evaluate_vs_random(
            coord.network, gspec, final_eval_games, eval_mcts, use_gpu;
            num_workers=final_eval_workers
        )
        eval_time = time() - eval_start

        @info "=" ^ 60
        @info "Final Evaluation Results (vs Random):"
        @info "  White: $(round(final_results["eval/vs_random_white"], digits=3))"
        @info "  Black: $(round(final_results["eval/vs_random_black"], digits=3))"
        @info "  Combined: $(round(final_results["eval/vs_random_combined"], digits=3))"
        @info "  Time: $(round(eval_time / 60, digits=2)) minutes"
        @info "  Speed: $(round(final_eval_games / eval_time * 60, digits=1)) games/min"
        @info "=" ^ 60

        if tb_enabled
            final_metrics = Dict{String, Any}(
                "eval/final_vs_random_white" => final_results["eval/vs_random_white"],
                "eval/final_vs_random_black" => final_results["eval/vs_random_black"],
                "eval/final_vs_random_combined" => final_results["eval/vs_random_combined"],
                "eval/final_games" => final_eval_games,
                "eval/final_workers" => final_eval_workers,
                "eval/final_time_s" => eval_time,
                "summary/total_iterations" => !isnothing(coord) ? coord.iteration : 0,
                "summary/total_games" => !isnothing(coord) ? coord.total_games : 0,
                "summary/total_samples" => !isnothing(coord) ? coord.total_samples : 0,
                "summary/total_time_min" => total_time / 60,
            )
            tb_log(final_metrics)
        end

        # Final GnuBG evaluation (if enabled)
        if eval_vs_gnubg
            @info "=" ^ 60
            @info "Running final GnuBG evaluation..."
            @info "  Games: $gnubg_eval_games"
            @info "  GnuBG ply: $gnubg_ply"
            @info "  MCTS iters: $eval_mcts"
            @info "=" ^ 60

            gnubg_eval_start = time()
            gnubg_final_results = evaluate_vs_gnubg(
                coord.network, gspec, gnubg_eval_games, eval_mcts, use_gpu;
                gnubg_ply=gnubg_ply
            )
            gnubg_eval_time = time() - gnubg_eval_start

            @info "=" ^ 60
            @info "Final Evaluation Results (vs GnuBG $(gnubg_ply)-ply):"
            @info "  White: $(round(gnubg_final_results["eval/vs_gnubg$(gnubg_ply)ply_white"], digits=3)) (wr=$(round(100*gnubg_final_results["eval/vs_gnubg$(gnubg_ply)ply_wr_white"], digits=1))%)"
            @info "  Black: $(round(gnubg_final_results["eval/vs_gnubg$(gnubg_ply)ply_black"], digits=3)) (wr=$(round(100*gnubg_final_results["eval/vs_gnubg$(gnubg_ply)ply_wr_black"], digits=1))%)"
            @info "  Combined: $(round(gnubg_final_results["eval/vs_gnubg$(gnubg_ply)ply_combined"], digits=3)) (wr=$(round(100*gnubg_final_results["eval/vs_gnubg$(gnubg_ply)ply_wr_combined"], digits=1))%)"
            @info "  Time: $(round(gnubg_eval_time / 60, digits=2)) minutes"
            @info "=" ^ 60

            if tb_enabled
                gnubg_metrics = Dict{String, Any}(
                    "eval/final_vs_gnubg$(gnubg_ply)ply_white" => gnubg_final_results["eval/vs_gnubg$(gnubg_ply)ply_white"],
                    "eval/final_vs_gnubg$(gnubg_ply)ply_black" => gnubg_final_results["eval/vs_gnubg$(gnubg_ply)ply_black"],
                    "eval/final_vs_gnubg$(gnubg_ply)ply_combined" => gnubg_final_results["eval/vs_gnubg$(gnubg_ply)ply_combined"],
                    "eval/final_vs_gnubg$(gnubg_ply)ply_wr_white" => gnubg_final_results["eval/vs_gnubg$(gnubg_ply)ply_wr_white"],
                    "eval/final_vs_gnubg$(gnubg_ply)ply_wr_black" => gnubg_final_results["eval/vs_gnubg$(gnubg_ply)ply_wr_black"],
                    "eval/final_vs_gnubg$(gnubg_ply)ply_wr_combined" => gnubg_final_results["eval/vs_gnubg$(gnubg_ply)ply_wr_combined"],
                    "eval/final_gnubg_games" => gnubg_eval_games,
                    "eval/final_gnubg_ply" => gnubg_ply,
                    "eval/final_gnubg_time_s" => gnubg_eval_time,
                )
                tb_log(gnubg_metrics)
            end
        end
    end

    # Finish TensorBoard
    if tb_enabled
        tb_finish()
        @info "TensorBoard logging finished"
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
