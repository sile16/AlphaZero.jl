#!/usr/bin/env julia
#####
##### Multi-head training v2 - ALWAYS replaces network
##### Evaluation still runs for tracking, but doesn't gate network selection
##### With wandb logging for experiment tracking
#####

using Pkg
Pkg.activate(dirname(@__DIR__))

# Initialize wandb BEFORE loading other AlphaZero modules
using AlphaZero.Wandb: wandb_available, wandb_init, wandb_log, wandb_finish
using AlphaZero.Wandb: params_to_config

const USE_WANDB = wandb_available()
if USE_WANDB
    println("Wandb available - will log metrics")
else
    println("Wandb not available - metrics will only be logged locally")
end

using AlphaZero
using AlphaZero.FluxLib
using Dates

# Use deterministic backgammon
const GAMES_DIR = joinpath(@__DIR__, "..", "games")
include(joinpath(GAMES_DIR, "backgammon-deterministic", "main.jl"))
using .BackgammonDeterministic
const GameModule = BackgammonDeterministic

println("=" ^ 70)
println("MULTI-HEAD TRAINING v2 (always_replace=true)")
println("=" ^ 70)
println("Start time: $(now())")
println("Network is ALWAYS replaced - eval is for tracking only")
println("=" ^ 70)

# Multi-head network - matching SimpleNet size
netparams = FluxLib.FCResNetMultiHeadHP(
    width = 128,
    num_blocks = 3,
    depth_phead = 1,
    depth_vhead = 1,
    share_value_trunk = true
)

# Self-play parameters
self_play = SelfPlayParams(
    sim = SimParams(
        num_games = 100,
        num_workers = 16,
        batch_size = 16,
        use_gpu = true,
        reset_every = 4,
        flip_probability = 0.0,
        alternate_colors = false
    ),
    mcts = MctsParams(
        num_iters_per_turn = 100,
        cpuct = 1.5,
        temperature = ConstSchedule(1.0),
        dirichlet_noise_ϵ = 0.25,
        dirichlet_noise_α = 0.3
    )
)

# Learning parameters
learning = LearningParams(
    use_gpu = true,
    use_position_averaging = false,
    samples_weighing_policy = LOG_WEIGHT,
    l2_regularization = 1e-4,
    optimiser = CyclicNesterov(
        lr_base = 1e-3,
        lr_high = 1e-2,
        lr_low = 1e-3,
        momentum_high = 0.9,
        momentum_low = 0.8
    ),
    batch_size = 64,
    loss_computation_batch_size = 2048,
    nonvalidity_penalty = 1.0,
    min_checkpoints_per_epoch = 0,
    max_batches_per_checkpoint = 1000,
    num_checkpoints = 1,
    rewards_renormalization = 1.0
)

# Arena parameters - run evaluation for tracking, but ALWAYS replace network
arena = ArenaParams(
    sim = SimParams(
        num_games = 20,
        num_workers = 12,
        batch_size = 12,
        use_gpu = true,
        reset_every = 1,
        flip_probability = 0.0,
        alternate_colors = true
    ),
    mcts = MctsParams(
        num_iters_per_turn = 100,
        cpuct = 1.5,
        temperature = ConstSchedule(0.2),
        dirichlet_noise_ϵ = 0.05,
        dirichlet_noise_α = 0.3
    ),
    update_threshold = 0.0,
    always_replace = true  # KEY CHANGE: Always replace network
)

# Memory analysis
memory = MemAnalysisParams(num_game_stages = 4)

# Full params
params = Params(
    arena = arena,
    self_play = self_play,
    learning = learning,
    memory_analysis = memory,
    num_iters = 128,
    use_symmetries = false,
    ternary_outcome = false,
    mem_buffer_size = PLSchedule(100_000)
)

# Create experiment
experiment = Experiment(
    name = "backgammon-multihead-v2",
    gspec = GameModule.GameSpec(),
    params = params,
    mknet = FluxLib.FCResNetMultiHead,
    netparams = netparams,
    benchmark = Benchmark.Evaluation[]
)

# Print network info
println("\nNetwork architecture: FCResNetMultiHead")
println("  Width: $(netparams.width)")
println("  Blocks: $(netparams.num_blocks)")
println("  Policy head depth: $(netparams.depth_phead)")
println("  Value head depth: $(netparams.depth_vhead)")

# Create network and show parameter count
gspec = GameModule.GameSpec()
nn = FluxLib.FCResNetMultiHead(gspec, netparams)
println("  Total parameters: $(Network.num_parameters(nn))")

# Test forward pass
println("\nTesting forward pass...")
game = GI.init(gspec)
state = GI.current_state(game)
x = GI.vectorize_state(gspec, state)
x_batch = reshape(Float32.(x), size(x)..., 1)

p, v_win, v_gw, v_bgw, v_gl, v_bgl = FluxLib.forward_multihead(nn, x_batch)
println("  Initial P(win): $(v_win[1])")
equity = FluxLib.compute_equity(v_win[1], v_gw[1], v_bgw[1], v_gl[1], v_bgl[1])
println("  Initial equity: $(equity)")

println("\n" * "=" ^ 70)
println("Key settings:")
println("  MCTS sims/turn: 100")
println("  CPUCT: 1.5")
println("  always_replace: TRUE (eval is for tracking only)")
println("  Memory buffer: 100K")
println("=" ^ 70)

# Initialize wandb
if USE_WANDB
    config = params_to_config(params, nn)
    config["network/type"] = "FCResNetMultiHead"
    config["network/width"] = netparams.width
    config["network/num_blocks"] = netparams.num_blocks
    config["experiment/version"] = "v2"
    config["experiment/always_replace"] = true

    wandb_init(
        project = "alphazero-backgammon",
        name = "multihead-v2-$(Dates.format(now(), "mmdd_HHMM"))",
        config = config
    )
    println("\nWandb run initialized")
end

println("\nStarting training...")

# Generate unique session name
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
session_dir = "sessions/bg-multihead-v2-$timestamp"

# Run training
start_time = time()
session = Session(experiment, dir=session_dir, autosave=true, save_intermediate=true)
resume!(session)
elapsed = time() - start_time

println("\n" * "=" ^ 70)
println("Training completed!")
println("=" ^ 70)
println("Total time: $(round(elapsed/3600, digits=2)) hours")
println("Iterations completed: $(session.env.itc)")
println("Session saved to: $session_dir")

# Final evaluation
println("\nFinal network evaluation:")
nn_final = session.env.bestnn
p, v_win, v_gw, v_bgw, v_gl, v_bgl = FluxLib.forward_multihead(nn_final, x_batch)
println("  P(win): $(v_win[1])")
println("  P(gammon|win): $(v_gw[1])")
println("  P(bg|win): $(v_bgw[1])")
println("  P(gammon|loss): $(v_gl[1])")
println("  P(bg|loss): $(v_bgl[1])")
equity = FluxLib.compute_equity(v_win[1], v_gw[1], v_bgw[1], v_gl[1], v_bgl[1])
println("  Computed equity: $(equity)")

# Log final metrics to wandb
if USE_WANDB
    wandb_log(Dict(
        "final/p_win" => Float64(v_win[1]),
        "final/p_gammon_win" => Float64(v_gw[1]),
        "final/p_bg_win" => Float64(v_bgw[1]),
        "final/p_gammon_loss" => Float64(v_gl[1]),
        "final/p_bg_loss" => Float64(v_bgl[1]),
        "final/equity" => Float64(equity),
        "final/iterations" => session.env.itc,
        "final/elapsed_hours" => elapsed / 3600
    ))
    wandb_finish()
    println("\nWandb run finished")
end

println("\nEnd time: $(now())")
println("=" ^ 70)
