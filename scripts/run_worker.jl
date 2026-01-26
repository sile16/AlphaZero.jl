#!/usr/bin/env julia

"""
Remote worker script for distributed AlphaZero training.

This script runs on remote servers with their own GPUs. Workers:
- Load a copy of the neural network locally
- Run MCTS self-play with local GPU inference
- Submit game samples to the coordinator's replay buffer
- Receive periodic weight updates from the coordinator

Usage (local GPU inference - recommended):
    julia --project scripts/run_worker.jl \\
        --coordinator 192.168.1.100 \\
        --worker-id worker-server2 \\
        --game backgammon-deterministic \\
        --network-type fcresnet-multihead \\
        --use-gpu

Usage (remote inference - for CPU-only workers):
    julia --project scripts/run_worker.jl \\
        --coordinator 192.168.1.100 \\
        --worker-id worker-cpu1 \\
        --game backgammon-deterministic \\
        --remote-inference

Multiple workers on same server:
    for i in 1 2 3 4; do
        julia --project scripts/run_worker.jl \\
            --coordinator 192.168.1.100 \\
            --worker-id worker-server2-\$i \\
            --game backgammon-deterministic &
    done
"""

using ArgParse
using AlphaZero
using AlphaZero.Distributed

function parse_args()
    s = ArgParseSettings(
        description="Distributed AlphaZero worker",
        autofix_names=true
    )

    @add_arg_table! s begin
        # Required
        "--coordinator"
            help = "Coordinator hostname or IP"
            arg_type = String
            required = true

        # Worker identity
        "--worker-id"
            help = "Unique worker identifier"
            arg_type = String
            default = nothing

        # Game configuration
        "--game"
            help = "Game to train on"
            arg_type = String
            default = "connect-four"

        # Network configuration (for local inference)
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

        # Inference mode
        "--remote-inference"
            help = "Use remote inference server (for CPU-only workers)"
            action = :store_true
        "--use-gpu"
            help = "Use GPU for local inference"
            action = :store_true

        # Connection options
        "--inference-port"
            help = "Inference server port on coordinator (for remote inference)"
            arg_type = Int
            default = 5555
        "--replay-port"
            help = "Replay buffer port on coordinator"
            arg_type = Int
            default = 5556
        "--command-port"
            help = "Command/weight update port on coordinator"
            arg_type = Int
            default = 5557

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
        "--temperature"
            help = "MCTS temperature"
            arg_type = Float64
            default = 1.0

        # Worker behavior
        "--games-per-batch"
            help = "Games to complete before submitting samples"
            arg_type = Int
            default = 10
        "--heartbeat-interval"
            help = "Seconds between heartbeats"
            arg_type = Float64
            default = 30.0

        # Misc
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

function main()
    args = parse_args()

    if args["verbose"]
        ENV["JULIA_DEBUG"] = "AlphaZero"
    end

    # Generate worker ID if not provided
    worker_id = if !isnothing(args["worker_id"])
        args["worker_id"]
    else
        "worker_$(gethostname())_$(getpid())"
    end

    @info "Starting worker: $worker_id"
    @info "Connecting to coordinator: $(args["coordinator"])"

    # Load game
    gspec = get_game_spec(args["game"])
    @info "Game: $(args["game"])"

    # Determine inference mode
    use_local_inference = !args["remote_inference"]

    # Create network for local inference
    network = if use_local_inference
        @info "Using local GPU inference"
        nn = create_network(gspec, args)
        if args["use_gpu"]
            @info "Moving network to GPU..."
            nn = Network.to_gpu(nn)
        end
        nn
    else
        @info "Using remote inference server"
        nothing
    end

    # Create MCTS params
    mcts_params = MctsParams(
        num_iters_per_turn=args["mcts_iters"],
        cpuct=args["cpuct"],
        dirichlet_noise_ϵ=args["dirichlet_noise"],
        temperature=ConstSchedule(args["temperature"])
    )

    # Create worker configuration
    config = WorkerConfig(
        worker_id=worker_id,
        inference_endpoint=EndpointConfig(
            host=args["coordinator"],
            port=args["inference_port"]
        ),
        replay_endpoint=EndpointConfig(
            host=args["coordinator"],
            port=args["replay_port"]
        ),
        coordinator_endpoint=EndpointConfig(
            host=args["coordinator"],
            port=args["command_port"]
        ),
        mcts_params=mcts_params,
        games_per_batch=args["games_per_batch"],
        heartbeat_interval_s=args["heartbeat_interval"],
        use_gpu=args["use_gpu"]
    )

    # Create worker
    worker = SelfPlayWorker(config, gspec; network=network)

    @info "Worker configuration:"
    @info "  Inference mode:   $(use_local_inference ? "local" : "remote")"
    @info "  Replay buffer:    tcp://$(args["coordinator"]):$(args["replay_port"])"
    @info "  Weight updates:   tcp://$(args["coordinator"]):$(args["command_port"])"
    @info "  MCTS iterations:  $(args["mcts_iters"])"
    @info "  Games per batch:  $(args["games_per_batch"])"
    if !use_local_inference
        @info "  Inference server: tcp://$(args["coordinator"]):$(args["inference_port"])"
    end

    # Handle interrupts gracefully
    @info "Starting self-play loop (Ctrl+C to stop)..."

    try
        stats = run_worker(worker)
        @info "Worker stopped normally"
        @info "Final stats:" stats
    catch e
        if e isa InterruptException
            @info "Worker interrupted by user"
        else
            @error "Worker error" exception=(e, catch_backtrace())
        end
    finally
        shutdown_worker(worker)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
