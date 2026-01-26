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

    # Multi-machine with SSH (TODO)
    julia --project scripts/train_cluster.jl \\
        --game=backgammon-deterministic \\
        --worker-hosts=worker1,worker2
"""

using ArgParse
using CUDA
using Dates
using Distributed

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
using AlphaZero: GI, Network, FluxLib
using AlphaZero.Wandb
import Flux
using Statistics: mean
using Base.Threads

# Preload PythonCall for wandb (avoids world age issues)
using PythonCall

# Pre-load game modules at top level to avoid world age issues
const GAMES_DIR = joinpath(@__DIR__, "..", "games")
include(joinpath(GAMES_DIR, "backgammon-deterministic", "main.jl"))
include(joinpath(GAMES_DIR, "backgammon", "main.jl"))

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

        # WandB configuration
        "--wandb-project"
            help = "WandB project name"
            arg_type = String
            default = "alphazero-jl"
        "--wandb-run-name"
            help = "WandB run name (auto-generated if not provided)"
            arg_type = String
            default = nothing
        "--host-id"
            help = "Host identifier for multi-machine tracking"
            arg_type = String
            default = nothing
        "--no-wandb"
            help = "Disable WandB logging"
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
Main entry point.
"""
function main()
    args = parse_args()

    if args["verbose"]
        ENV["JULIA_DEBUG"] = "AlphaZero"
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

    @info "=" ^ 60
    @info "Distributed AlphaZero Training (Julia Distributed)"
    @info "=" ^ 60
    @info "Git commit: $(git_hash[1:min(8, length(git_hash))])$(git_dirty ? " (dirty)" : "")"

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
    if !isnothing(args["load_network"])
        @info "Will load weights from: $(args["load_network"])"
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

    wandb_enabled = !args["no_wandb"]
    @info "  WandB: $(wandb_enabled ? args["wandb_project"] : "disabled")"

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

    if use_gpu
        @info "  GPU memory: $(round(CUDA.available_memory() / 1e9, digits=2)) GB free"
    end

    @info "=" ^ 60

    # Initialize wandb
    if wandb_enabled
        @info "Initializing WandB..."
        try
            run_name = args["wandb_run_name"]
            if isnothing(run_name)
                run_name = "cluster-$(args["game"])-$(args["network_type"])-$(Dates.format(now(), "mmdd-HHMM"))"
            end

            config = Dict{String, Any}(
                # Game
                "game" => args["game"],
                # Network
                "network_type" => args["network_type"],
                "network_width" => args["network_width"],
                "network_blocks" => args["network_blocks"],
                "network_params" => num_params,
                # Training
                "num_workers" => args["num_workers"],
                "total_iterations" => args["total_iterations"],
                "games_per_iteration" => args["games_per_iteration"],
                "batch_size" => args["batch_size"],
                "learning_rate" => args["learning_rate"],
                "l2_reg" => args["l2_reg"],
                # MCTS
                "mcts_iters" => args["mcts_iters"],
                "cpuct" => args["cpuct"],
                # Buffer
                "buffer_capacity" => args["buffer_capacity"],
                # Evaluation
                "eval_interval" => eval_interval,
                "eval_games" => args["eval_games"],
                "eval_mcts_iters" => eval_mcts,
                # Final evaluation
                "final_eval_games" => final_eval_games,
                "final_eval_workers" => final_eval_workers,
                # System
                "host_id" => host_id,
                "use_gpu" => use_gpu,
                "training_mode" => "cluster",
                # Reproducibility
                "git_commit" => git_hash,
                "git_dirty" => git_dirty,
            )

            wandb_init(project=args["wandb_project"], name=run_name, config=config)
            @info "WandB initialized: project=$(args["wandb_project"]), run=$run_name"
        catch e
            @warn "Failed to initialize wandb, continuing without logging" exception=e
            wandb_enabled = false
        end
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

    # Create session directory
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    session_dir = joinpath("sessions", "cluster_$(timestamp)")
    checkpoint_dir = joinpath(session_dir, args["checkpoint_dir"])
    mkpath(checkpoint_dir)
    @info "Session directory: $session_dir"

    # Save reproducibility info to session directory
    open(joinpath(session_dir, "run_info.txt"), "w") do f
        println(f, "# AlphaZero.jl Training Run")
        println(f, "timestamp: $timestamp")
        println(f, "git_commit: $git_hash")
        println(f, "git_dirty: $git_dirty")
        println(f, "")
        println(f, "# Arguments")
        for (k, v) in sort(collect(args), by=first)
            println(f, "$k: $v")
        end
    end

    # Track start time
    start_time = time()

    # WandB logging callback
    wandb_log_fn = if wandb_enabled
        (metrics) -> begin
            try
                # Add system metrics periodically
                if haskey(metrics, "train/iteration") && metrics["train/iteration"] % 5 == 0
                    cuda_mod = use_gpu ? CUDA : nothing
                    sys_metrics = all_system_metrics(host_id=host_id, cuda_module=cuda_mod)
                    merge!(metrics, sys_metrics)
                end
                wandb_log(metrics, step=get(metrics, "train/iteration", nothing))
            catch e
                @warn "WandB log failed" exception=e
            end
        end
    else
        nothing
    end

    # Evaluation callback
    eval_fn = if eval_interval > 0
        (network) -> begin
            @info "Running evaluation ($(args["eval_games"]) games vs random)..."
            eval_start = time()
            results = evaluate_vs_random(network, gspec, args["eval_games"], eval_mcts, use_gpu)
            results["eval/time_s"] = time() - eval_start
            @info "Eval results: white=$(round(results["eval/vs_random_white"], digits=3)), " *
                  "black=$(round(results["eval/vs_random_black"], digits=3)), " *
                  "combined=$(round(results["eval/vs_random_combined"], digits=3))"
            results
        end
    else
        nothing
    end

    # Run training
    coord = nothing
    try
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
            wandb_log=wandb_log_fn,
            eval_fn=eval_fn,
            eval_interval=eval_interval
        )
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
        @info "Final Evaluation Results:"
        @info "  White: $(round(final_results["eval/vs_random_white"], digits=3))"
        @info "  Black: $(round(final_results["eval/vs_random_black"], digits=3))"
        @info "  Combined: $(round(final_results["eval/vs_random_combined"], digits=3))"
        @info "  Time: $(round(eval_time / 60, digits=2)) minutes"
        @info "  Speed: $(round(final_eval_games / eval_time * 60, digits=1)) games/min"
        @info "=" ^ 60

        if wandb_enabled
            final_metrics = Dict{String, Any}(
                "eval/final_vs_random_white" => final_results["eval/vs_random_white"],
                "eval/final_vs_random_black" => final_results["eval/vs_random_black"],
                "eval/final_vs_random_combined" => final_results["eval/vs_random_combined"],
                "eval/final_games" => final_eval_games,
                "eval/final_workers" => final_eval_workers,
                "eval/final_time_s" => eval_time,
                "summary/total_iterations" => coord.iteration,
                "summary/total_games" => coord.total_games,
                "summary/total_samples" => coord.total_samples,
                "summary/total_time_min" => total_time / 60,
            )
            try
                wandb_log(final_metrics)
            catch e
                @warn "Failed to log final metrics to wandb" exception=e
            end
        end
    end

    # Finish wandb
    if wandb_enabled
        try
            wandb_finish()
            @info "WandB run finished"
        catch e
            @warn "Failed to finish wandb" exception=e
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
