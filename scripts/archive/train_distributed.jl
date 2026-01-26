#!/usr/bin/env julia

"""
Distributed training script for AlphaZero.jl

Usage:
    # On coordinator server (runs all components):
    julia --project scripts/train_distributed.jl --coordinator \\
        --game backgammon-deterministic \\
        --inference-port 5555 \\
        --replay-port 5556 \\
        --num-workers 4 \\
        --wandb-project alphazero-distributed

    # On remote worker server:
    julia --project scripts/train_distributed.jl --worker \\
        --inference-server tcp://coordinator:5555 \\
        --replay-server tcp://coordinator:5556 \\
        --worker-id worker-server2
"""

using ArgParse
using AlphaZero
using AlphaZero.Distributed

function parse_args()
    s = ArgParseSettings(
        description="Distributed AlphaZero training",
        autofix_names=true
    )

    @add_arg_table! s begin
        "--coordinator"
            help = "Run as coordinator (manages all components)"
            action = :store_true
        "--worker"
            help = "Run as worker (self-play only)"
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
            default = 6
        "--load-network"
            help = "Path to network checkpoint to load"
            arg_type = String
            default = nothing

        # Coordinator options
        "--inference-port"
            help = "Port for inference server"
            arg_type = Int
            default = 5555
        "--replay-port"
            help = "Port for replay buffer"
            arg_type = Int
            default = 5556
        "--command-port"
            help = "Port for worker commands"
            arg_type = Int
            default = 5557
        "--num-workers"
            help = "Number of local workers to spawn"
            arg_type = Int
            default = 4
        "--total-iterations"
            help = "Total training iterations"
            arg_type = Int
            default = 100
        "--games-per-iteration"
            help = "Games per training iteration"
            arg_type = Int
            default = 500
        "--session-dir"
            help = "Directory for session data"
            arg_type = String
            default = "sessions"

        # Worker options
        "--inference-server"
            help = "Inference server address (tcp://host:port)"
            arg_type = String
            default = nothing
        "--replay-server"
            help = "Replay buffer address (tcp://host:port)"
            arg_type = String
            default = nothing
        "--coordinator-server"
            help = "Coordinator address (tcp://host:port)"
            arg_type = String
            default = nothing
        "--worker-id"
            help = "Unique worker identifier"
            arg_type = String
            default = nothing

        # MCTS configuration
        "--mcts-iters"
            help = "MCTS iterations per move"
            arg_type = Int
            default = 400
        "--cpuct"
            help = "MCTS exploration constant"
            arg_type = Float64
            default = 1.0
        "--dirichlet-noise"
            help = "Dirichlet noise epsilon"
            arg_type = Float64
            default = 0.25

        # Learning configuration
        "--batch-size"
            help = "Training batch size"
            arg_type = Int
            default = 2048
        "--learning-rate"
            help = "Learning rate"
            arg_type = Float64
            default = 1e-3
        "--l2-reg"
            help = "L2 regularization"
            arg_type = Float64
            default = 1e-4

        # Buffer configuration
        "--buffer-capacity"
            help = "Replay buffer capacity"
            arg_type = Int
            default = 500000
        "--min-samples"
            help = "Minimum samples before training starts"
            arg_type = Int
            default = 10000

        # Misc
        "--wandb-project"
            help = "WandB project name"
            arg_type = String
            default = nothing
        "--use-gpu"
            help = "Use GPU for inference/training"
            action = :store_true
        "--verbose"
            help = "Verbose output"
            action = :store_true
    end

    return parse_args(s)
end

function get_game_spec(game_name::String)
    if game_name == "connect-four"
        return Examples.games["connect-four"]
    elseif game_name == "tictactoe"
        return Examples.games["tictactoe"]
    elseif game_name == "backgammon-deterministic"
        # Load backgammon game
        include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "main.jl"))
        return Backgammon.GameSpec()
    elseif game_name == "backgammon"
        include(joinpath(@__DIR__, "..", "games", "backgammon", "main.jl"))
        return Backgammon.GameSpec()
    else
        error("Unknown game: $game_name")
    end
end

function create_network(gspec, args)
    if !isnothing(args["load_network"])
        @info "Loading network from $(args["load_network"])"
        return Network.load(args["load_network"])
    end

    network_type = args["network_type"]
    width = args["network_width"]
    blocks = args["network_blocks"]

    if network_type == "simple"
        hp = NetLib.SimpleNetHP(width=width, depth_common=blocks)
        return NetLib.SimpleNet(gspec; hp)
    elseif network_type == "resnet"
        hp = NetLib.ResNetHP(num_blocks=blocks, num_filters=width)
        return NetLib.ResNet(gspec; hp)
    elseif network_type == "fcresnet"
        hp = NetLib.FCResNetHP(width=width, num_blocks=blocks)
        return NetLib.FCResNet(gspec; hp)
    elseif network_type == "fcresnet-multihead"
        hp = NetLib.FCResNetMultiHeadHP(width=width, num_blocks=blocks)
        return NetLib.FCResNetMultiHead(gspec; hp)
    else
        error("Unknown network type: $network_type")
    end
end

function create_mcts_params(args)
    return MctsParams(
        num_iters_per_turn=args["mcts_iters"],
        cpuct=args["cpuct"],
        dirichlet_noise_ϵ=args["dirichlet_noise"],
        temperature=ConstSchedule(1.0)
    )
end

function create_learning_params(args)
    return LearningParams(
        batch_size=args["batch_size"],
        optimiser=Adam(lr=args["learning_rate"]),
        l2_regularization=args["l2_reg"],
        use_gpu=args["use_gpu"]
    )
end

function run_coordinator(args)
    @info "Starting distributed training coordinator"

    gspec = get_game_spec(args["game"])
    network = create_network(gspec, args)
    mcts_params = create_mcts_params(args)
    learning_params = create_learning_params(args)

    # Create configuration
    inference_config = InferenceServerConfig(
        endpoint=EndpointConfig(port=args["inference_port"]),
        batch_size=64,
        use_gpu=args["use_gpu"]
    )

    replay_config = ReplayBufferConfig(
        endpoint=EndpointConfig(port=args["replay_port"]),
        capacity=args["buffer_capacity"],
        min_samples_for_training=args["min_samples"]
    )

    training_config = TrainingConfig(
        replay_endpoint=EndpointConfig(host="localhost", port=args["replay_port"]),
        coordinator_endpoint=EndpointConfig(host="localhost", port=args["command_port"]),
        learning_params=learning_params,
        batch_size=args["batch_size"],
        checkpoint_dir=joinpath(args["session_dir"], "checkpoints"),
        use_gpu=args["use_gpu"],
        wandb_project=args["wandb_project"]
    )

    coordinator_config = CoordinatorConfig(
        inference_config=inference_config,
        replay_config=replay_config,
        training_config=training_config,
        command_endpoint=EndpointConfig(port=args["command_port"]),
        num_local_workers=args["num_workers"],
        session_dir=args["session_dir"],
        total_iterations=args["total_iterations"],
        games_per_iteration=args["games_per_iteration"],
        wandb_project=args["wandb_project"]
    )

    coordinator = DistributedCoordinator(coordinator_config, gspec, network)

    # Run training
    stats = run_coordinator(coordinator, mcts_params)

    @info "Training complete" stats
end

function parse_endpoint(endpoint_str::String)
    # Parse tcp://host:port format
    m = match(r"tcp://([^:]+):(\d+)", endpoint_str)
    if isnothing(m)
        error("Invalid endpoint format: $endpoint_str (expected tcp://host:port)")
    end
    return (host=m.captures[1], port=parse(Int, m.captures[2]))
end

function run_worker_mode(args)
    @info "Starting distributed worker"

    # Validate required args
    if isnothing(args["inference_server"])
        error("--inference-server required for worker mode")
    end
    if isnothing(args["replay_server"])
        error("--replay-server required for worker mode")
    end
    if isnothing(args["worker_id"])
        # Generate unique worker ID
        args["worker_id"] = "worker_$(gethostname())_$(getpid())"
    end

    gspec = get_game_spec(args["game"])
    mcts_params = create_mcts_params(args)

    # Parse endpoints
    inference_ep = parse_endpoint(args["inference_server"])
    replay_ep = parse_endpoint(args["replay_server"])

    coordinator_ep = if !isnothing(args["coordinator_server"])
        parse_endpoint(args["coordinator_server"])
    else
        (host=inference_ep.host, port=args["command_port"])
    end

    config = WorkerConfig(
        worker_id=args["worker_id"],
        inference_endpoint=EndpointConfig(host=inference_ep.host, port=inference_ep.port),
        replay_endpoint=EndpointConfig(host=replay_ep.host, port=replay_ep.port),
        coordinator_endpoint=EndpointConfig(host=coordinator_ep.host, port=coordinator_ep.port),
        mcts_params=mcts_params
    )

    worker = SelfPlayWorker(config, gspec)

    # Run worker
    stats = run_worker(worker)

    @info "Worker stopped" stats
end

function main()
    args = parse_args()

    if args["verbose"]
        ENV["JULIA_DEBUG"] = "AlphaZero"
    end

    if args["coordinator"] && args["worker"]
        error("Cannot run as both coordinator and worker")
    end

    if args["coordinator"]
        run_coordinator(args)
    elseif args["worker"]
        run_worker_mode(args)
    else
        # Default: run as coordinator
        @info "No mode specified, running as coordinator"
        run_coordinator(args)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
