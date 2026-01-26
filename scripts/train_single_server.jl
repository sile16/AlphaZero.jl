#!/usr/bin/env julia

"""
Single-server distributed training for AlphaZero.jl

Runs all components (workers, replay buffer, training) in one Julia process,
sharing the GPU efficiently. This is useful for testing the distributed
architecture on a single machine.

Key features:
- All components share the same GPU with lazy memory allocation
- Workers run in separate threads for CPU parallelism
- Training and workers time-slice GPU access

Usage:
    julia --project --threads=auto scripts/train_single_server.jl \\
        --game connect-four \\
        --num-workers 4 \\
        --total-iterations 50

For backgammon with multi-head network:
    julia --project --threads=auto scripts/train_single_server.jl \\
        --game backgammon-deterministic \\
        --network-type fcresnet-multihead \\
        --num-workers 4
"""

using ArgParse
using CUDA
using Dates

# Configure CUDA for memory sharing BEFORE loading AlphaZero
# This ensures lazy allocation so multiple components can share GPU
if CUDA.functional()
    # Allow scalar operations temporarily for setup
    CUDA.allowscalar(false)

    @info "CUDA configuration:"
    @info "  Device: $(CUDA.name(CUDA.device()))"
    @info "  Memory: $(round(CUDA.total_memory() / 1e9, digits=2)) GB total"
    @info "  Free:   $(round(CUDA.available_memory() / 1e9, digits=2)) GB available"
end

using AlphaZero
using AlphaZero.Distributed
using AlphaZero: GI, Network, losses, Trace, FluxLib
using AlphaZero.Wandb
import Flux
using Statistics: mean

# Preload PythonCall for wandb (avoids world age issues)
using PythonCall

# Pre-load game modules at top level to avoid world age issues
const GAMES_DIR = joinpath(@__DIR__, "..", "games")
include(joinpath(GAMES_DIR, "backgammon-deterministic", "main.jl"))
include(joinpath(GAMES_DIR, "backgammon", "main.jl"))

function parse_args()
    s = ArgParseSettings(
        description="Single-server distributed AlphaZero training",
        autofix_names=true
    )

    @add_arg_table! s begin
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
            help = "Number of self-play workers"
            arg_type = Int
            default = 4
        "--total-iterations"
            help = "Total training iterations"
            arg_type = Int
            default = 50
        "--batch-size"
            help = "Training batch size"
            arg_type = Int
            default = 256
        "--batches-per-iteration"
            help = "Training batches per iteration"
            arg_type = Int
            default = 50

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
            default = 1000
        "--games-per-batch"
            help = "Games per worker batch"
            arg_type = Int
            default = 5

        # Learning configuration
        "--learning-rate"
            help = "Learning rate"
            arg_type = Float64
            default = 1e-3
        "--l2-reg"
            help = "L2 regularization"
            arg_type = Float64
            default = 1e-4

        # Output
        "--session-dir"
            help = "Directory for session data"
            arg_type = String
            default = "sessions"
        "--wandb-project"
            help = "WandB project name (enables wandb logging)"
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

function create_network(gspec, args)
    if !isnothing(args["load_network"])
        @info "Loading network from $(args["load_network"])"
        return Network.load(args["load_network"])
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
        hp = NetLib.FCResNetHP(width=width, num_blocks=blocks)
        return NetLib.FCResNet(gspec, hp)
    elseif network_type == "fcresnet-multihead"
        hp = NetLib.FCResNetMultiHeadHP(width=width, num_blocks=blocks)
        return NetLib.FCResNetMultiHead(gspec, hp)
    else
        error("Unknown network type: $network_type")
    end
end

"""
Single-server training coordinator that runs everything in-process.
"""
mutable struct SingleServerTrainer
    gspec::AbstractGameSpec
    network::AbstractNetwork
    mcts_params::MctsParams
    learning_params::LearningParams
    buffer::Vector{SerializedSample}
    buffer_capacity::Int
    buffer_lock::ReentrantLock
    iteration::Int
    total_games::Int
    total_samples::Int
    running::Bool
    use_gpu::Bool
    session_dir::String
    args::Dict
    wandb_enabled::Bool
    host_id::Union{String,Nothing}
    start_time::Float64
end

function SingleServerTrainer(gspec, network, args)
    mcts_params = MctsParams(
        num_iters_per_turn=args["mcts_iters"],
        cpuct=args["cpuct"],
        dirichlet_noise_ϵ=0.25,
        dirichlet_noise_α=1.0,
        temperature=ConstSchedule(1.0),
        gamma=1.0
    )

    learning_params = LearningParams(
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

    # Create session directory
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    session_dir = joinpath(args["session_dir"], "single_server_$(timestamp)")
    mkpath(session_dir)

    # Determine host_id (for multi-machine tracking)
    host_id = args["host_id"]
    if isnothing(host_id)
        host_id = gethostname()
    end

    return SingleServerTrainer(
        gspec,
        network,
        mcts_params,
        learning_params,
        SerializedSample[],
        args["buffer_capacity"],
        ReentrantLock(),
        0,
        0,
        0,
        true,
        !args["no_gpu"] && CUDA.functional(),
        session_dir,
        args,
        !args["no_wandb"] && !isnothing(args["wandb_project"]),  # wandb_enabled
        host_id,
        time()  # start_time
    )
end

"""
Add samples to the replay buffer (thread-safe).
"""
function add_samples!(trainer::SingleServerTrainer, samples::Vector{SerializedSample})
    lock(trainer.buffer_lock) do
        append!(trainer.buffer, samples)
        # Trim to capacity (remove oldest)
        if length(trainer.buffer) > trainer.buffer_capacity
            deleteat!(trainer.buffer, 1:(length(trainer.buffer) - trainer.buffer_capacity))
        end
        trainer.total_samples += length(samples)
    end
end

"""
Get a random batch for training.
"""
function sample_batch(trainer::SingleServerTrainer, batch_size::Int)
    lock(trainer.buffer_lock) do
        n = length(trainer.buffer)
        if n < batch_size
            return nothing
        end
        indices = rand(1:n, batch_size)
        return [trainer.buffer[i] for i in indices]
    end
end

"""
Run a self-play worker (called in a thread).
"""
function run_self_play_worker!(
    trainer::SingleServerTrainer,
    worker_id::Int,
    games_per_batch::Int
)
    # Create a copy of the network for this worker (CPU for MCTS, inference on shared GPU)
    worker_network = Network.copy(trainer.network, on_gpu=false, test_mode=true)

    @info "Worker $worker_id started"

    games_played = 0
    pending_samples = SerializedSample[]

    while trainer.running
        try
            # Sync weights periodically (every batch)
            if games_played % games_per_batch == 0 && games_played > 0
                lock(trainer.buffer_lock) do
                    # Copy current network weights
                    for (wp, mp) in zip(Network.params(worker_network), Network.params(trainer.network))
                        copyto!(wp, Array(mp))
                    end
                end
            end

            # Create player with worker's network copy
            player = MctsPlayer(trainer.gspec, worker_network, trainer.mcts_params)

            # Play a game
            trace = play_game(trainer.gspec, player)
            games_played += 1

            # Convert to samples
            samples = trace_to_serialized_samples(trainer, trace)
            append!(pending_samples, samples)

            # Submit batch
            if length(pending_samples) >= games_per_batch * 30  # Approx samples per game
                add_samples!(trainer, pending_samples)
                trainer.total_games += games_per_batch
                empty!(pending_samples)

                if worker_id == 1 && games_played % 10 == 0
                    @info "Worker $worker_id: $games_played games, buffer size: $(length(trainer.buffer))"
                end
            end

            # Reset player
            reset_player!(player)

        catch e
            if !(e isa InterruptException)
                @error "Worker $worker_id error" exception=(e, catch_backtrace())
            end
            break
        end
    end

    # Submit remaining samples
    if !isempty(pending_samples)
        add_samples!(trainer, pending_samples)
    end

    @info "Worker $worker_id stopped after $games_played games"
end

"""
Convert trace to serialized samples.
"""
function trace_to_serialized_samples(trainer::SingleServerTrainer, trace::Trace)
    gamma = trainer.mcts_params.gamma
    n = length(trace)
    samples = SerializedSample[]
    sizehint!(samples, n)

    # Compute cumulative rewards
    wr = 0.0
    cumulative_rewards = zeros(n)
    for i in reverse(1:n)
        wr = gamma * wr + trace.rewards[i]
        cumulative_rewards[i] = wr
    end

    has_outcome = !isnothing(trace.outcome)

    num_actions = GI.num_actions(trainer.gspec)

    for i in 1:n
        state = trace.states[i]
        policy = trace.policies[i]
        is_chance = trace.is_chance[i]

        wp = GI.white_playing(trainer.gspec, state)
        z = wp ? cumulative_rewards[i] : -cumulative_rewards[i]
        t = Float32(n - i + 1)

        state_arr = GI.vectorize_state(trainer.gspec, state)
        state_vec = Vector{Float32}(vec(state_arr))

        # Expand sparse policy to full action space
        actions_mask = GI.actions_mask(GI.init(trainer.gspec, state))
        full_policy = zeros(Float32, num_actions)
        if !is_chance && !isempty(policy)
            full_policy[actions_mask] = Float32.(policy)
        end
        policy_vec = full_policy

        equity = if has_outcome
            outcome = trace.outcome
            won = outcome.white_won == wp
            if won
                MultiHeadValue(1.0f0, outcome.is_gammon ? 1.0f0 : 0.0f0,
                              outcome.is_backgammon ? 1.0f0 : 0.0f0, 0.0f0, 0.0f0)
            else
                MultiHeadValue(0.0f0, 0.0f0, 0.0f0,
                              outcome.is_gammon ? 1.0f0 : 0.0f0,
                              outcome.is_backgammon ? 1.0f0 : 0.0f0)
            end
        else
            nothing
        end

        push!(samples, SerializedSample(
            state=state_vec, policy=policy_vec, value=Float32(z),
            turn=t, is_chance=is_chance, equity=equity
        ))
    end

    return samples
end

"""
Convert serialized samples to training batch format.
Uses Flux.batch to properly handle state shapes.
"""
function prepare_training_batch(trainer::SingleServerTrainer, samples::Vector{SerializedSample})
    n = length(samples)
    state_shape = GI.state_dim(trainer.gspec)
    policy_dim = GI.num_actions(trainer.gspec)

    # Prepare individual arrays for batching
    ws = Vector{Vector{Float32}}(undef, n)
    xs = Vector{Array{Float32}}(undef, n)
    as = Vector{Vector{Float32}}(undef, n)
    ps = Vector{Vector{Float32}}(undef, n)
    vs = Vector{Vector{Float32}}(undef, n)
    is_chances = Vector{Vector{Float32}}(undef, n)
    eq_wins = Vector{Vector{Float32}}(undef, n)
    eq_gws = Vector{Vector{Float32}}(undef, n)
    eq_bgws = Vector{Vector{Float32}}(undef, n)
    eq_gls = Vector{Vector{Float32}}(undef, n)
    eq_bgls = Vector{Vector{Float32}}(undef, n)
    has_equitys = Vector{Vector{Float32}}(undef, n)

    for (i, s) in enumerate(samples)
        # Reshape flattened state back to original shape
        xs[i] = reshape(s.state, state_shape)

        # Weight
        ws[i] = Float32[1]

        # Value
        vs[i] = Float32[s.value]

        # Is chance
        is_chances[i] = Float32[s.is_chance ? 1.0f0 : 0.0f0]

        # Policy (already expanded to full action space)
        ps[i] = s.policy

        # Action mask: derive from policy (non-zero = valid action)
        # For chance nodes, all actions are "valid" (mask of ones)
        if s.is_chance
            as[i] = ones(Float32, policy_dim)
        else
            as[i] = Float32.(s.policy .> 0)
        end

        # Equity targets
        if !isnothing(s.equity)
            eq_wins[i] = Float32[s.equity.p_win]
            eq_gws[i] = Float32[s.equity.p_gammon_win]
            eq_bgws[i] = Float32[s.equity.p_bg_win]
            eq_gls[i] = Float32[s.equity.p_gammon_loss]
            eq_bgls[i] = Float32[s.equity.p_bg_loss]
            has_equitys[i] = Float32[1.0f0]
        else
            eq_wins[i] = Float32[0.0f0]
            eq_gws[i] = Float32[0.0f0]
            eq_bgws[i] = Float32[0.0f0]
            eq_gls[i] = Float32[0.0f0]
            eq_bgls[i] = Float32[0.0f0]
            has_equitys[i] = Float32[0.0f0]
        end
    end

    # Use Flux.batch to properly stack arrays
    W = Flux.batch(ws)
    X = Flux.batch(xs)
    A = Flux.batch(as)
    P = Flux.batch(ps)
    V = Flux.batch(vs)
    IsChance = Flux.batch(is_chances)
    EqWin = Flux.batch(eq_wins)
    EqGW = Flux.batch(eq_gws)
    EqBGW = Flux.batch(eq_bgws)
    EqGL = Flux.batch(eq_gls)
    EqBGL = Flux.batch(eq_bgls)
    HasEquity = Flux.batch(has_equitys)

    f32(arr) = convert(AbstractArray{Float32}, arr)
    batch = map(f32, (; W, X, A, P, V, IsChance, EqWin, EqGW, EqBGW, EqGL, EqBGL, HasEquity))

    if trainer.use_gpu
        batch = Network.convert_input_tuple(trainer.network, batch)
    end

    return batch
end

"""
Run training iterations.
"""
function run_training!(trainer::SingleServerTrainer)
    @info "Training process started"

    # Move network to GPU for training
    if trainer.use_gpu
        trainer.network = Network.to_gpu(trainer.network)
        CUDA.reclaim()  # Free any unused GPU memory
        @info "GPU memory after network load: $(round(CUDA.available_memory() / 1e9, digits=2)) GB free"
    end

    params = trainer.learning_params
    opt_state = Flux.setup(Flux.Adam(params.optimiser.lr), trainer.network)

    total_loss = 0.0
    batches_trained = 0
    last_games = 0
    last_time = time()

    while trainer.running && trainer.iteration < trainer.args["total_iterations"]
        # Wait for enough samples
        if length(trainer.buffer) < trainer.args["min_samples"]
            @info "Waiting for samples... ($(length(trainer.buffer))/$(trainer.args["min_samples"]))"
            sleep(2.0)
            continue
        end

        trainer.iteration += 1
        iteration_loss = 0.0
        iteration_start = time()

        # Train for several batches
        for batch_idx in 1:trainer.args["batches_per_iteration"]
            samples = sample_batch(trainer, params.batch_size)
            if isnothing(samples)
                break
            end

            batch_data = prepare_training_batch(trainer, samples)

            # Compute loss and gradients
            Wmean = mean(batch_data.W)
            Hp = 0.0f0

            loss_fn(nn) = losses(nn, params, Wmean, Hp, batch_data)[1]

            l, grads = Flux.withgradient(loss_fn, trainer.network)
            Flux.update!(opt_state, trainer.network, grads[1])

            loss_val = Float64(l)
            iteration_loss += loss_val
            total_loss += loss_val
            batches_trained += 1
        end

        avg_iter_loss = iteration_loss / trainer.args["batches_per_iteration"]
        iteration_time = time() - iteration_start

        # Calculate games per minute
        current_time = time()
        elapsed_since_last = current_time - last_time
        games_since_last = trainer.total_games - last_games
        games_per_min = elapsed_since_last > 0 ? (games_since_last / elapsed_since_last) * 60 : 0.0

        # Log to wandb
        if trainer.wandb_enabled
            metrics = Dict{String, Any}(
                "train/loss" => avg_iter_loss,
                "train/iteration" => trainer.iteration,
                "train/iteration_time_s" => iteration_time,
                "train/batches_trained" => batches_trained,
                "buffer/size" => length(trainer.buffer),
                "buffer/capacity_pct" => (length(trainer.buffer) / trainer.buffer_capacity) * 100,
                "games/total" => trainer.total_games,
                "games/per_minute" => games_per_min,
                "samples/total" => trainer.total_samples,
                "workers/active" => trainer.args["num_workers"],
            )

            # Add system metrics every 5 iterations (not too frequent)
            if trainer.iteration % 5 == 0
                cuda_mod = trainer.use_gpu ? CUDA : nothing
                sys_metrics = all_system_metrics(host_id=trainer.host_id, cuda_module=cuda_mod)
                merge!(metrics, sys_metrics)
            end

            try
                wandb_log(metrics, step=trainer.iteration)
            catch e
                @warn "Failed to log to wandb" exception=e
            end
        end

        # Update tracking for games/min calculation
        last_games = trainer.total_games
        last_time = current_time

        # Reclaim GPU memory periodically
        if trainer.use_gpu && trainer.iteration % 10 == 0
            CUDA.reclaim()
        end

        # Log progress
        @info "Iteration $(trainer.iteration)/$(trainer.args["total_iterations"]): " *
              "loss=$(round(avg_iter_loss, digits=4)), " *
              "buffer=$(length(trainer.buffer)), " *
              "games=$(trainer.total_games)"

        # Save checkpoint periodically
        if trainer.iteration % 10 == 0
            save_checkpoint(trainer)
        end

        # Brief pause to let workers catch up
        sleep(0.1)
    end

    # Final checkpoint
    save_checkpoint(trainer)

    # Log final summary to wandb
    if trainer.wandb_enabled
        total_time = time() - trainer.start_time
        final_metrics = Dict{String, Any}(
            "summary/total_iterations" => trainer.iteration,
            "summary/total_games" => trainer.total_games,
            "summary/total_samples" => trainer.total_samples,
            "summary/total_time_min" => total_time / 60,
            "summary/avg_loss" => batches_trained > 0 ? total_loss / batches_trained : 0.0,
        )
        try
            wandb_log(final_metrics)
        catch e
            @warn "Failed to log final metrics to wandb" exception=e
        end
    end

    @info "Training complete: $(trainer.iteration) iterations, $(batches_trained) batches"
end

"""
Save a training checkpoint.
"""
function save_checkpoint(trainer::SingleServerTrainer)
    checkpoint_dir = joinpath(trainer.session_dir, "checkpoints")
    mkpath(checkpoint_dir)

    # Save network weights
    cpu_network = Network.to_cpu(trainer.network)
    FluxLib.save_weights(joinpath(checkpoint_dir, "network_iter$(trainer.iteration).data"), cpu_network)
    FluxLib.save_weights(joinpath(checkpoint_dir, "latest.data"), cpu_network)

    @info "Saved checkpoint at iteration $(trainer.iteration)"
end

"""
Main entry point.
"""
function main()
    args = parse_args()

    if args["verbose"]
        ENV["JULIA_DEBUG"] = "AlphaZero"
    end

    @info "=" ^ 60
    @info "Single-Server Distributed AlphaZero Training"
    @info "=" ^ 60

    # Check threading
    num_threads = Threads.nthreads()
    @info "Julia threads: $num_threads"
    if num_threads < args["num_workers"] + 1
        @warn "Recommend at least $(args["num_workers"] + 1) threads for optimal performance"
        @warn "Run with: julia --threads=$(args["num_workers"] + 1)"
    end

    # Load game
    gspec = get_game_spec(args["game"])
    @info "Game: $(args["game"])"

    # Create network
    network = create_network(gspec, args)

    # Create trainer
    trainer = SingleServerTrainer(gspec, network, args)

    @info "Configuration:"
    @info "  Workers: $(args["num_workers"])"
    @info "  Buffer capacity: $(args["buffer_capacity"])"
    @info "  Batch size: $(args["batch_size"])"
    @info "  MCTS iterations: $(args["mcts_iters"])"
    @info "  Total iterations: $(args["total_iterations"])"
    @info "  Session dir: $(trainer.session_dir)"
    @info "  GPU enabled: $(trainer.use_gpu)"
    @info "  Host ID: $(trainer.host_id)"
    @info "  WandB: $(trainer.wandb_enabled ? args["wandb_project"] : "disabled")"

    if trainer.use_gpu
        @info "  GPU memory: $(round(CUDA.available_memory() / 1e9, digits=2)) GB free"
    end

    @info "=" ^ 60

    # Initialize wandb
    if trainer.wandb_enabled
        @info "Initializing WandB..."
        try
            run_name = args["wandb_run_name"]
            if isnothing(run_name)
                run_name = "$(args["game"])-$(args["network_type"])-$(Dates.format(now(), "mmdd-HHMM"))"
            end

            config = Dict{String, Any}(
                # Game
                "game" => args["game"],
                # Network
                "network_type" => args["network_type"],
                "network_width" => args["network_width"],
                "network_blocks" => args["network_blocks"],
                "network_params" => Network.num_parameters(network),
                # Training
                "num_workers" => args["num_workers"],
                "total_iterations" => args["total_iterations"],
                "batch_size" => args["batch_size"],
                "batches_per_iteration" => args["batches_per_iteration"],
                "learning_rate" => args["learning_rate"],
                "l2_reg" => args["l2_reg"],
                # MCTS
                "mcts_iters" => args["mcts_iters"],
                "cpuct" => args["cpuct"],
                # Buffer
                "buffer_capacity" => args["buffer_capacity"],
                "min_samples" => args["min_samples"],
                "games_per_batch" => args["games_per_batch"],
                # System
                "host_id" => trainer.host_id,
                "use_gpu" => trainer.use_gpu,
                "julia_threads" => Threads.nthreads(),
            )

            wandb_init(project=args["wandb_project"], name=run_name, config=config)
            @info "WandB initialized: project=$(args["wandb_project"]), run=$run_name"
        catch e
            @warn "Failed to initialize wandb, continuing without logging" exception=e
            trainer.wandb_enabled = false
        end
    end

    # Start worker threads
    worker_tasks = Task[]
    for i in 1:args["num_workers"]
        task = Threads.@spawn run_self_play_worker!(
            trainer, i, args["games_per_batch"]
        )
        push!(worker_tasks, task)
    end

    # Run training in main thread
    try
        run_training!(trainer)
    catch e
        if e isa InterruptException
            @info "Training interrupted"
        else
            @error "Training error" exception=(e, catch_backtrace())
        end
    finally
        trainer.running = false

        # Wait for workers
        @info "Waiting for workers to finish..."
        for task in worker_tasks
            try
                wait(task)
            catch end
        end
    end

    @info "=" ^ 60
    @info "Training Summary:"
    @info "  Total iterations: $(trainer.iteration)"
    @info "  Total games: $(trainer.total_games)"
    @info "  Total samples: $(trainer.total_samples)"
    @info "  Final buffer size: $(length(trainer.buffer))"
    @info "  Session saved to: $(trainer.session_dir)"
    @info "=" ^ 60

    # Finish wandb
    if trainer.wandb_enabled
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
