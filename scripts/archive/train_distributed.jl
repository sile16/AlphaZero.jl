#!/usr/bin/env julia

"""
Distributed training script for AlphaZero.jl using ZMQ-based architecture.

This is the PRIMARY training script. It uses the full distributed framework
with ZMQ communication, which can scale from single-machine to multi-server.

Usage:
    # Single-server mode (all components on one machine):
    julia --project --threads=auto scripts/train_distributed.jl \
        --game=backgammon-deterministic \
        --network-type=fcresnet-multihead \
        --num-workers=4 \
        --total-iterations=300

    # Coordinator mode (for multi-server setup):
    julia --project scripts/train_distributed.jl --coordinator \
        --inference-port=5555 \
        --replay-port=5556 \
        --num-workers=0

    # Worker mode (run on remote servers):
    julia --project scripts/train_distributed.jl --worker \
        --inference-server=tcp://coordinator:5555 \
        --replay-server=tcp://coordinator:5556 \
        --worker-id=worker-server2
"""

using ArgParse
using CUDA
using Dates

# Configure CUDA BEFORE loading AlphaZero
if CUDA.functional()
    CUDA.allowscalar(false)
    @info "CUDA configuration:"
    @info "  Device: $(CUDA.name(CUDA.device()))"
    @info "  Memory: $(round(CUDA.total_memory() / 1e9, digits=2)) GB total"
    @info "  Free:   $(round(CUDA.available_memory() / 1e9, digits=2)) GB available"
end

using AlphaZero
using AlphaZero.Distributed
using AlphaZero: GI, Network, FluxLib
using AlphaZero.Wandb
using Statistics: mean

# Preload PythonCall for wandb (avoids world age issues)
using PythonCall

# Pre-load game modules at TOP LEVEL to avoid world age issues (Julia 1.12+)
const GAMES_DIR = joinpath(@__DIR__, "..", "games")
include(joinpath(GAMES_DIR, "backgammon-deterministic", "main.jl"))
include(joinpath(GAMES_DIR, "backgammon", "main.jl"))

function parse_args()
    s = ArgParseSettings(
        description="Distributed AlphaZero training with ZMQ",
        autofix_names=true
    )

    @add_arg_table! s begin
        # Mode selection
        "--coordinator"
            help = "Run as coordinator only (for multi-server setup)"
            action = :store_true
        "--worker"
            help = "Run as worker only (connects to remote coordinator)"
            action = :store_true

        # Game configuration
        "--game"
            help = "Game to train on"
            arg_type = String
            default = "connect-four"

        # Network configuration
        "--network-type"
            help = "Network type: simple, resnet, fcresnet, fcresnet-multihead"
            arg_type = String
            default = "simple"
        "--network-width"
            help = "Network width"
            arg_type = Int
            default = 128
        "--network-blocks"
            help = "Number of residual blocks"
            arg_type = Int
            default = 4
        "--load-network"
            help = "Path to network checkpoint to load"
            arg_type = String
            default = nothing

        # Training configuration
        "--num-workers"
            help = "Number of local self-play workers"
            arg_type = Int
            default = 4
        "--total-iterations"
            help = "Total training iterations"
            arg_type = Int
            default = 100
        "--batch-size"
            help = "Training batch size"
            arg_type = Int
            default = 256
        "--batches-per-iteration"
            help = "Training batches per iteration"
            arg_type = Int
            default = 100

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
        "--min-samples"
            help = "Minimum samples before training starts"
            arg_type = Int
            default = 5000

        # Learning configuration
        "--learning-rate"
            help = "Learning rate"
            arg_type = Float64
            default = 1e-3
        "--l2-reg"
            help = "L2 regularization"
            arg_type = Float64
            default = 1e-4

        # Network ports (for distributed)
        "--inference-port"
            help = "Port for ZMQ inference server"
            arg_type = Int
            default = 5555
        "--replay-port"
            help = "Port for ZMQ replay buffer"
            arg_type = Int
            default = 5556
        "--command-port"
            help = "Port for worker commands"
            arg_type = Int
            default = 5557

        # Worker mode options
        "--inference-server"
            help = "Inference server address (tcp://host:port)"
            arg_type = String
            default = nothing
        "--replay-server"
            help = "Replay buffer address (tcp://host:port)"
            arg_type = String
            default = nothing
        "--coordinator-server"
            help = "Coordinator address for commands (tcp://host:port)"
            arg_type = String
            default = nothing
        "--worker-id"
            help = "Unique worker identifier"
            arg_type = String
            default = nothing

        # Output
        "--session-dir"
            help = "Directory for session data"
            arg_type = String
            default = "sessions"
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
            help = "MCTS iterations for evaluation (0 = use training value)"
            arg_type = Int
            default = 0

        # Misc
        "--no-gpu"
            help = "Disable GPU"
            action = :store_true
        "--verbose"
            help = "Verbose output"
            action = :store_true
    end

    return ArgParse.parse_args(s)
end

function get_game_spec(game_name::String)
    # Game modules are loaded at top level to avoid world age issues
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

function create_network(gspec, args)
    if !isnothing(args["load_network"])
        @info "Loading network from $(args["load_network"])"
        nn_bytes = read(args["load_network"])
        weights = FluxLib.deserialize_weights(nn_bytes)
        # Need to create network first, then load weights
        network_type = args["network_type"]
        width = args["network_width"]
        blocks = args["network_blocks"]

        if network_type == "fcresnet-multihead"
            hp = FluxLib.FCResNetMultiHeadHP(width=width, num_blocks=blocks)
            nn = FluxLib.FCResNetMultiHead(gspec, hp)
        else
            error("Loading only supported for fcresnet-multihead currently")
        end
        FluxLib.load_weights!(nn, weights)
        return nn
    end

    network_type = args["network_type"]
    width = args["network_width"]
    blocks = args["network_blocks"]

    @info "Creating $network_type network (width=$width, blocks=$blocks)"

    if network_type == "simple"
        hp = NetLib.SimpleNetHP(width=width, depth_common=blocks)
        return NetLib.SimpleNet(gspec, hp)
    elseif network_type == "resnet"
        hp = NetLib.ResNetHP(num_blocks=blocks, num_filters=width)
        return NetLib.ResNet(gspec, hp)
    elseif network_type == "fcresnet"
        hp = FluxLib.FCResNetHP(width=width, num_blocks=blocks)
        return FluxLib.FCResNet(gspec, hp)
    elseif network_type == "fcresnet-multihead"
        hp = FluxLib.FCResNetMultiHeadHP(width=width, num_blocks=blocks)
        return FluxLib.FCResNetMultiHead(gspec, hp)
    else
        error("Unknown network type: $network_type")
    end
end

function create_mcts_params(args)
    return MctsParams(
        num_iters_per_turn=args["mcts_iters"],
        cpuct=args["cpuct"],
        dirichlet_noise_ϵ=0.25,
        dirichlet_noise_α=1.0,
        temperature=ConstSchedule(1.0),
        gamma=1.0
    )
end

function create_learning_params(args)
    return LearningParams(
        batch_size=args["batch_size"],
        loss_computation_batch_size=args["batch_size"],
        optimiser=Adam(lr=args["learning_rate"]),
        l2_regularization=Float32(args["l2_reg"]),
        use_gpu=!args["no_gpu"] && CUDA.functional(),
        samples_weighing_policy=CONSTANT_WEIGHT,
        min_checkpoints_per_epoch=1,
        max_batches_per_checkpoint=args["batches_per_iteration"],
        num_checkpoints=1
    )
end

"""
Evaluate current network against a random player.
"""
function evaluate_vs_random(gspec, network, num_games::Int, mcts_iters::Int; use_gpu::Bool=true)
    eval_network = Network.copy(network, on_gpu=use_gpu, test_mode=true)

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

    rewards_white = Float64[]
    for _ in 1:games_per_side
        trace = play_game(gspec, TwoPlayers(az_player, random_player))
        push!(rewards_white, total_reward(trace))
        reset_player!(az_player)
    end

    rewards_black = Float64[]
    for _ in 1:games_per_side
        trace = play_game(gspec, TwoPlayers(random_player, az_player))
        push!(rewards_black, -total_reward(trace))
        reset_player!(az_player)
    end

    avg_white = isempty(rewards_white) ? 0.0 : mean(rewards_white)
    avg_black = isempty(rewards_black) ? 0.0 : mean(rewards_black)
    combined = (avg_white + avg_black) / 2

    return (avg_white, avg_black, combined)
end

"""
Run training as coordinator with all components.
"""
function run_as_coordinator(args)
    gspec = get_game_spec(args["game"])
    network = create_network(gspec, args)
    mcts_params = create_mcts_params(args)
    learning_params = create_learning_params(args)

    use_gpu = !args["no_gpu"] && CUDA.functional()
    host_id = isnothing(args["host_id"]) ? gethostname() : args["host_id"]
    wandb_enabled = !args["no_wandb"]

    # Create session directory
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    session_dir = joinpath(args["session_dir"], "distributed_$(timestamp)")
    mkpath(session_dir)
    checkpoint_dir = joinpath(session_dir, "checkpoints")
    mkpath(checkpoint_dir)

    @info "=" ^ 60
    @info "Distributed AlphaZero Training (ZMQ)"
    @info "=" ^ 60
    @info "Configuration:"
    @info "  Game: $(args["game"])"
    @info "  Network: $(args["network_type"]) ($(args["network_width"])x$(args["network_blocks"]))"
    @info "  Workers: $(args["num_workers"])"
    @info "  Iterations: $(args["total_iterations"])"
    @info "  MCTS iters: $(args["mcts_iters"])"
    @info "  Buffer: $(args["buffer_capacity"])"
    @info "  Session: $session_dir"
    @info "  GPU: $use_gpu"
    @info "  Host ID: $host_id"
    @info "  WandB: $(wandb_enabled ? args["wandb_project"] : "disabled")"
    eval_interval = args["eval_interval"]
    if eval_interval > 0
        eval_mcts = args["eval_mcts_iters"] > 0 ? args["eval_mcts_iters"] : args["mcts_iters"]
        @info "  Eval: every $eval_interval iters, $(args["eval_games"]) games"
    end
    @info "=" ^ 60

    # Initialize wandb
    if wandb_enabled
        @info "Initializing WandB..."
        try
            run_name = args["wandb_run_name"]
            if isnothing(run_name)
                run_name = "$(args["game"])-$(args["network_type"])-$(Dates.format(now(), "mmdd-HHMM"))"
            end

            config = Dict{String, Any}(
                "game" => args["game"],
                "network_type" => args["network_type"],
                "network_width" => args["network_width"],
                "network_blocks" => args["network_blocks"],
                "network_params" => Network.num_parameters(network),
                "num_workers" => args["num_workers"],
                "total_iterations" => args["total_iterations"],
                "batch_size" => args["batch_size"],
                "batches_per_iteration" => args["batches_per_iteration"],
                "learning_rate" => args["learning_rate"],
                "l2_reg" => args["l2_reg"],
                "mcts_iters" => args["mcts_iters"],
                "cpuct" => args["cpuct"],
                "buffer_capacity" => args["buffer_capacity"],
                "min_samples" => args["min_samples"],
                "eval_interval" => args["eval_interval"],
                "eval_games" => args["eval_games"],
                "host_id" => host_id,
                "use_gpu" => use_gpu,
                "distributed" => true,
                "inference_port" => args["inference_port"],
                "replay_port" => args["replay_port"],
            )

            wandb_init(project=args["wandb_project"], name=run_name, config=config)
            @info "WandB initialized: $(args["wandb_project"])/$run_name"
        catch e
            @warn "Failed to initialize wandb" exception=e
            wandb_enabled = false
        end
    end

    # Create distributed components
    @info "Starting distributed components..."

    # Inference server config
    inference_config = InferenceServerConfig(
        endpoint=EndpointConfig(port=args["inference_port"]),
        batch_size=64,
        use_gpu=use_gpu
    )

    # Replay buffer config
    replay_config = ReplayBufferConfig(
        endpoint=EndpointConfig(port=args["replay_port"]),
        capacity=args["buffer_capacity"],
        min_samples_for_training=args["min_samples"]
    )

    # Training config
    training_config = TrainingConfig(
        replay_endpoint=EndpointConfig(host="localhost", port=args["replay_port"]),
        coordinator_endpoint=EndpointConfig(host="localhost", port=args["command_port"]),
        learning_params=learning_params,
        batch_size=args["batch_size"],
        checkpoint_dir=checkpoint_dir,
        use_gpu=use_gpu,
        wandb_project=wandb_enabled ? args["wandb_project"] : nothing
    )

    # Coordinator config
    coordinator_config = CoordinatorConfig(
        inference_config=inference_config,
        replay_config=replay_config,
        training_config=training_config,
        command_endpoint=EndpointConfig(port=args["command_port"]),
        num_local_workers=args["num_workers"],
        session_dir=session_dir,
        total_iterations=args["total_iterations"],
        games_per_iteration=0,  # Continuous self-play
        wandb_project=wandb_enabled ? args["wandb_project"] : nothing
    )

    # Create and run coordinator
    coordinator = DistributedCoordinator(coordinator_config, gspec, network)

    start_time = time()

    try
        # Start all components
        start_inference_server!(coordinator)
        start_replay_manager!(coordinator)
        start_local_workers!(coordinator, mcts_params)

        @info "All components started, beginning training loop..."

        # Custom training loop with wandb logging and evaluation
        iteration = 0
        last_games = 0
        last_time = time()

        while iteration < args["total_iterations"]
            # Wait for enough samples
            buffer_stats = get_buffer_stats(coordinator.replay_manager)
            if buffer_stats.size < args["min_samples"]
                @info "Waiting for samples... ($(buffer_stats.size)/$(args["min_samples"]))"
                sleep(2.0)
                continue
            end

            iteration += 1
            iter_start = time()

            # Training step
            training_step!(coordinator.training_process)

            iter_time = time() - iter_start
            buffer_stats = get_buffer_stats(coordinator.replay_manager)
            training_stats = get_training_stats(coordinator.training_process)

            # Calculate games per minute
            current_time = time()
            games_since_last = buffer_stats.total_games - last_games
            elapsed = current_time - last_time
            games_per_min = elapsed > 0 ? (games_since_last / elapsed) * 60 : 0.0

            # Log to wandb
            if wandb_enabled
                metrics = Dict{String, Any}(
                    "train/loss" => training_stats.last_loss,
                    "train/iteration" => iteration,
                    "train/iteration_time_s" => iter_time,
                    "buffer/size" => buffer_stats.size,
                    "buffer/capacity_pct" => (buffer_stats.size / args["buffer_capacity"]) * 100,
                    "games/total" => buffer_stats.total_games,
                    "games/per_minute" => games_per_min,
                    "samples/total" => buffer_stats.total_samples,
                    "workers/active" => args["num_workers"],
                )

                # System metrics every 5 iterations
                if iteration % 5 == 0
                    cuda_mod = use_gpu ? CUDA : nothing
                    sys_metrics = all_system_metrics(host_id=host_id, cuda_module=cuda_mod)
                    merge!(metrics, sys_metrics)
                end

                try
                    wandb_log(metrics, step=iteration)
                catch e
                    @warn "Failed to log to wandb" exception=e
                end
            end

            last_games = buffer_stats.total_games
            last_time = current_time

            @info "Iteration $iteration/$(args["total_iterations"]): loss=$(round(training_stats.last_loss, digits=4)), buffer=$(buffer_stats.size), games=$(buffer_stats.total_games)"

            # Save checkpoint
            if iteration % 10 == 0
                save_checkpoint(coordinator.training_process, iteration)
                @info "Saved checkpoint at iteration $iteration"
            end

            # Evaluation
            if eval_interval > 0 && iteration % eval_interval == 0
                eval_games = args["eval_games"]
                eval_mcts = args["eval_mcts_iters"] > 0 ? args["eval_mcts_iters"] : args["mcts_iters"]

                @info "Running evaluation ($eval_games games vs random)..."
                eval_start = time()
                avg_white, avg_black, combined = evaluate_vs_random(
                    gspec, coordinator.training_process.network, eval_games, eval_mcts; use_gpu=use_gpu
                )
                eval_time = time() - eval_start

                @info "Eval: white=$(round(avg_white, digits=3)), black=$(round(avg_black, digits=3)), combined=$(round(combined, digits=3))"

                if wandb_enabled
                    eval_metrics = Dict{String, Any}(
                        "eval/vs_random_white" => avg_white,
                        "eval/vs_random_black" => avg_black,
                        "eval/vs_random_combined" => combined,
                        "eval/games" => eval_games,
                        "eval/time_s" => eval_time,
                    )
                    try
                        wandb_log(eval_metrics, step=iteration)
                    catch e
                        @warn "Failed to log eval metrics" exception=e
                    end
                end
            end

            # Reclaim GPU memory
            if use_gpu && iteration % 10 == 0
                CUDA.reclaim()
            end
        end

        # Final checkpoint
        save_checkpoint(coordinator.training_process, iteration)

        # Final evaluation
        if eval_interval > 0
            eval_games = args["eval_games"]
            eval_mcts = args["eval_mcts_iters"] > 0 ? args["eval_mcts_iters"] : args["mcts_iters"]

            @info "Running final evaluation..."
            avg_white, avg_black, combined = evaluate_vs_random(
                gspec, coordinator.training_process.network, eval_games, eval_mcts; use_gpu=use_gpu
            )
            @info "Final eval: white=$(round(avg_white, digits=3)), black=$(round(avg_black, digits=3)), combined=$(round(combined, digits=3))"

            if wandb_enabled
                wandb_log(Dict{String, Any}(
                    "eval/final_vs_random_white" => avg_white,
                    "eval/final_vs_random_black" => avg_black,
                    "eval/final_vs_random_combined" => combined,
                ))
            end
        end

        total_time = time() - start_time
        buffer_stats = get_buffer_stats(coordinator.replay_manager)

        # Final summary
        if wandb_enabled
            wandb_log(Dict{String, Any}(
                "summary/total_iterations" => iteration,
                "summary/total_games" => buffer_stats.total_games,
                "summary/total_samples" => buffer_stats.total_samples,
                "summary/total_time_min" => total_time / 60,
            ))
        end

        @info "=" ^ 60
        @info "Training Summary:"
        @info "  Iterations: $iteration"
        @info "  Games: $(buffer_stats.total_games)"
        @info "  Samples: $(buffer_stats.total_samples)"
        @info "  Time: $(round(total_time / 60, digits=1)) minutes"
        @info "  Session: $session_dir"
        @info "=" ^ 60

    catch e
        if e isa InterruptException
            @info "Training interrupted"
        else
            @error "Training error" exception=(e, catch_backtrace())
        end
    finally
        @info "Shutting down components..."
        shutdown_coordinator(coordinator)

        if wandb_enabled
            try
                wandb_finish()
                @info "WandB run finished"
            catch end
        end
    end
end

"""
Run as remote worker connecting to coordinator.
"""
function run_as_worker(args)
    if isnothing(args["inference_server"])
        error("--inference-server required for worker mode")
    end
    if isnothing(args["replay_server"])
        error("--replay-server required for worker mode")
    end

    worker_id = args["worker_id"]
    if isnothing(worker_id)
        worker_id = "worker_$(gethostname())_$(getpid())"
    end

    gspec = get_game_spec(args["game"])
    mcts_params = create_mcts_params(args)

    # Parse endpoints
    function parse_endpoint(s)
        m = match(r"tcp://([^:]+):(\d+)", s)
        isnothing(m) && error("Invalid endpoint: $s (expected tcp://host:port)")
        return EndpointConfig(host=m.captures[1], port=parse(Int, m.captures[2]))
    end

    inference_ep = parse_endpoint(args["inference_server"])
    replay_ep = parse_endpoint(args["replay_server"])

    coordinator_ep = if !isnothing(args["coordinator_server"])
        parse_endpoint(args["coordinator_server"])
    else
        EndpointConfig(host=inference_ep.host, port=args["command_port"])
    end

    config = WorkerConfig(
        worker_id=worker_id,
        inference_endpoint=inference_ep,
        replay_endpoint=replay_ep,
        coordinator_endpoint=coordinator_ep,
        mcts_params=mcts_params
    )

    @info "Starting worker: $worker_id"
    @info "  Inference: $(args["inference_server"])"
    @info "  Replay: $(args["replay_server"])"

    worker = SelfPlayWorker(config, gspec)

    try
        run_worker(worker)
    catch e
        if e isa InterruptException
            @info "Worker interrupted"
        else
            @error "Worker error" exception=(e, catch_backtrace())
        end
    finally
        shutdown_worker(worker)
    end
end

function main()
    args = parse_args()

    if args["verbose"]
        ENV["JULIA_DEBUG"] = "AlphaZero"
    end

    if args["coordinator"] && args["worker"]
        error("Cannot run as both --coordinator and --worker")
    end

    if args["worker"]
        run_as_worker(args)
    else
        # Default: run as coordinator (single-server or multi-server coordinator)
        run_as_coordinator(args)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
