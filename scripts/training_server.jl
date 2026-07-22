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
        --data-dir ./sessions/alphazero-server \\
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
            default = get(ENV, "ALPHAZERO_DATA_DIR",
                          joinpath(dirname(@__DIR__), "sessions", "alphazero-server"))
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
        "--mcts-budget-mode"
            help = "Self-play MCTS budget mode: constant, progressive, or turn_progressive"
            arg_type = String
            default = "constant"
        "--progressive-sim-min"
            help = "Minimum self-play simulations for mcts_budget_mode=progressive"
            arg_type = Int
            default = 0
        "--progressive-sim-max"
            help = "Maximum self-play simulations for mcts_budget_mode=progressive"
            arg_type = Int
            default = 0
        "--turn-sim-min"
            help = "Minimum opening simulations for mcts_budget_mode=turn_progressive"
            arg_type = Int
            default = 0
        "--turn-sim-target"
            help = "Target simulations for mcts_budget_mode=turn_progressive"
            arg_type = Int
            default = 0
        "--ramp-turns-initial"
            help = "Initial-iteration turns to reach target for mcts_budget_mode=turn_progressive"
            arg_type = Int
            default = 0
        "--ramp-turns-final"
            help = "Final-iteration turns to reach target for mcts_budget_mode=turn_progressive"
            arg_type = Int
            default = 0
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
            help = "Validated BackgammonNet backgammon_training_v4 artifact (raw checker action equities). Legacy sample formats are rejected."
            arg_type = String
            default = ""
        "--bootstrap-max-samples"
            help = "Max samples to load from bootstrap file (0 = all, capped at buffer capacity)"
            arg_type = Int
            default = 0
        "--bootstrap-train-iters"
            help = "Train this many iters on bootstrap, then clear buffer and switch to pure self-play (0 = never clear)"
            arg_type = Int
            default = 0
        "--bootstrap-only"
            help = "Train only on preloaded bootstrap buffer; do not wait for self-play samples"
            action = :store_true

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
        "--eval-backend-quality"
            help = "BackgammonNet WildBG quality profile used by distributed eval clients"
            arg_type = String
            default = "high"

        # Bearoff table (fail-fast: required unless --no-bearoff)
        "--no-bearoff"
            help = "Explicitly run WITHOUT the exact bearoff table — disables bearoff fixed-eval, promotion gate, and exact bearoff targets. WITHOUT this flag the server REQUIRES a local k=7 table and FAILS FAST if none is found, rather than silently running as if bearoff were active (which would corrupt result interpretation)."
            action = :store_true

        # Fixed bearoff eval
        "--bearoff-eval-interval"
            help = "Run fixed-set bearoff eval every N iterations (0 = disabled)"
            arg_type = Int
            default = 10
        "--bearoff-eval-positions"
            help = "Number of fixed bearoff decision states for raw NN bearoff eval"
            arg_type = Int
            default = 200
        "--bearoff-eval-mcts-positions"
            help = "Number of fixed bearoff decision states for NN+MCTS bearoff eval"
            arg_type = Int
            default = 50
        "--bearoff-eval-mcts-iters"
            help = "MCTS iterations for fixed bearoff eval"
            arg_type = Int
            default = 600
        "--bearoff-eval-rollouts-per-start"
            help = "Random rollouts per race start when building fixed bearoff eval cache"
            arg_type = Int
            default = 2

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
        "--preflight"
            help = "Validate the complete run contract and artifacts, write a report, then exit without serving or training"
            action = :store_true
        "--eval-manifest"
            help = "Optional immutable evaluation manifest JSON; defaults to <eval-positions-file>.manifest.json when present"
            arg_type = String
            default = ""

        # Weight promotion gate
        "--no-promotion-gate"
            help = "Disable the weight promotion gate (publish every iteration unconditionally). The gate is otherwise ENABLED whenever the fixed bearoff eval is enabled (it is the gate signal)."
            action = :store_true
        "--gate-tolerance"
            help = "Promotion-gate fractional tolerance: race value MAE may be up to this fraction worse than best-so-far and still publish (e.g. 0.10 = 10%)."
            arg_type = Float64
            default = 0.10
    end

    return ArgParse.parse_args(s)
end

const ARGS = parse_args()
ARGS["eval_backend_quality"] in ("min", "low", "high", "max") ||
    error("--eval-backend-quality must be min, low, high, or max")
ARGS["mcts_budget_mode"] in ("constant", "progressive", "turn_progressive") ||
    error("Unsupported --mcts-budget-mode=" * string(ARGS["mcts_budget_mode"]))
if ARGS["mcts_budget_mode"] == "progressive"
    ARGS["progressive_sim_min"] > 0 && ARGS["progressive_sim_max"] > 0 ||
        error("--mcts-budget-mode=progressive requires --progressive-sim-min and --progressive-sim-max")
elseif ARGS["mcts_budget_mode"] == "turn_progressive"
    ARGS["turn_sim_min"] > 0 && ARGS["turn_sim_target"] > 0 &&
        ARGS["ramp_turns_initial"] > 0 && ARGS["ramp_turns_final"] > 0 ||
        error("--mcts-budget-mode=turn_progressive requires --turn-sim-min, --turn-sim-target, --ramp-turns-initial, and --ramp-turns-final")
end
const DATA_DIR = ARGS["data_dir"]
const CHECKPOINT_DIR = joinpath(DATA_DIR, "checkpoints")
const TB_DIR = joinpath(DATA_DIR, "tb")

# Create directories
mkpath(CHECKPOINT_DIR)
mkpath(TB_DIR)
mkpath(joinpath(DATA_DIR, "buffer"))

# Set seed
Random.seed!(ARGS["seed"])
const TRAIN_RNG = Ref(MersenneTwister(ARGS["seed"]))

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
println("Bootstrap only: $(ARGS["bootstrap_only"])")
println("MCTS budget mode: $(ARGS["mcts_budget_mode"])")
println("LR: $(ARGS["learning_rate"]) (schedule=$(ARGS["lr_schedule"]), min=$(ARGS["lr_min"]))")
println("=" ^ 60)
flush(stdout)

# Load packages
using AlphaZero
using AlphaZero: GI, MCTS, Network, FluxLib, MctsParams, LearningParams, Adam, BatchedMCTS
using AlphaZero: CONSTANT_WEIGHT, losses, ConstSchedule
using AlphaZero.BackgammonInference
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
include(joinpath(@__DIR__, "..", "src", "distributed", "numerical_safety.jl"))
include(joinpath(@__DIR__, "..", "src", "distributed", "checkpoint_manager.jl"))
include(joinpath(@__DIR__, "..", "src", "distributed", "preflight.jl"))
include(joinpath(@__DIR__, "..", "src", "distributed", "eval_manifest.jl"))
using .CheckpointManager
using .Preflight
using .EvalManifest
include(joinpath(@__DIR__, "..", "src", "distributed", "server.jl"))
include(joinpath(@__DIR__, "..", "src", "distributed", "promotion_gate.jl"))

# Game setup
const GAME_NAME = "backgammon-deterministic"
if GAME_NAME == "backgammon-deterministic"
    ENV["BACKGAMMON_OBS_TYPE"] = get(ENV, "BACKGAMMON_OBS_TYPE", "min_plus_flat")
    include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
else
    error("Unknown game: $GAME_NAME")
end
const gspec = GameSpec()
# C-scale: single source of the raw→[-1,1] normalization (= GI.reward_scale = 3.0 for
# backgammon). Buffer/targets are RAW [-reward_scale, reward_scale]; network output +
# MCTS live in [-1,1]. Deriving every /3 from reward_scale(gspec) stops the constant
# from drifting if the game's reward scale ever changes.
const REWARD_SCALE = Float32(GI.reward_scale(gspec))
const NUM_ACTIONS = GI.num_actions(gspec)
const ML_CONTRACT = backgammon_ml_contract(gspec)
const CONTRACT_FINGERPRINT = contract_fingerprint(ML_CONTRACT)
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
const BEAROFF_EVAL_INTERVAL = ARGS["bearoff_eval_interval"]
const BEAROFF_EVAL_POSITIONS = ARGS["bearoff_eval_positions"]
const BEAROFF_EVAL_MCTS_POSITIONS = ARGS["bearoff_eval_mcts_positions"]
const BEAROFF_EVAL_MCTS_ITERS = ARGS["bearoff_eval_mcts_iters"]
const BEAROFF_EVAL_ROLLOUTS_PER_START = ARGS["bearoff_eval_rollouts_per_start"]

# ── Distributed evaluation setup ────────────────────────────────────────
using BackgammonNet
using BackgammonNet: BearoffK7, bearoff_turn_value
using StaticArrays

const EVAL_ENABLED = EVAL_INTERVAL > 0 && !isempty(ARGS["eval_positions_file"])
EVAL_ENABLED && !isfile(ARGS["eval_positions_file"]) && error(
    "Evaluation positions file does not exist: $(ARGS["eval_positions_file"])")
const BEAROFF_START_POSITIONS = if !isempty(ARGS["eval_positions_file"]) && isfile(ARGS["eval_positions_file"])
    Serialization.deserialize(ARGS["eval_positions_file"])
else
    Tuple[]
end

# Load eval positions
const EVAL_POSITIONS = if EVAL_ENABLED && isfile(ARGS["eval_positions_file"])
    pos = Serialization.deserialize(ARGS["eval_positions_file"])
    n = EVAL_GAMES > 0 ? min(EVAL_GAMES, length(pos)) : length(pos)
    pos[1:n]
else
    Tuple[]
end

const EVAL_MANIFEST_PATH = if !isempty(ARGS["eval_manifest"])
    ARGS["eval_manifest"]
elseif !isempty(ARGS["eval_positions_file"])
    candidate = ARGS["eval_positions_file"] * ".manifest.json"
    isfile(candidate) ? candidate : ""
else
    ""
end

function _eval_position_game(position)
    position isa BackgammonNet.BackgammonGame && return position
    p0, p1, cp = position
    return backgammon_game(p0, p1, SVector{2,Int8}(0, 0), Int8(0), cp,
                           false, 0.0f0; observation_type=:minimal_flat)
end

function _eval_position_fingerprint(position)
    state = BackgammonNet.game_state_fingerprint(_eval_position_game(position))
    return bytes2hex(SHA.sha256(codeunits(repr(state))))
end

if EVAL_ENABLED
    isempty(EVAL_MANIFEST_PATH) && error(
        "Evaluation is enabled but no immutable manifest was found. Run " *
        "scripts/build_eval_manifest.jl --input $(ARGS["eval_positions_file"])")
    validate_eval_manifest(EVAL_MANIFEST_PATH, ARGS["eval_positions_file"], EVAL_POSITIONS;
        contract_fingerprint=CONTRACT_FINGERPRINT,
        fingerprint=_eval_position_fingerprint)
end

function find_bearoff_dir_server()
    dir = BackgammonNet.default_bearoff_k7_dir()
    isdir(dir) && isfile(joinpath(dir, "bearoff_k7_c14.bin")) && return dir
    error("Bearoff k7 directory not found at $dir. Set BACKGAMMONNET_BEAROFF_K7_DIR " *
          "or place the table under $(BackgammonNet.default_bearoff_root()).")
end

# Fail-fast: the bearoff table is REQUIRED unless --no-bearoff is explicitly set.
# find_bearoff_dir_server() errors if no local table is found — we never silently
# fall back, so results are never misread as bearoff-anchored when they are not.
const BEAROFF_DIR = if ARGS["no_bearoff"]
    @warn "════════════════════════════════════════════════════════════════════\n" *
          "  --no-bearoff SET: running WITHOUT the exact bearoff table.\n" *
          "  Bearoff fixed-eval, promotion gate, and exact bearoff targets are\n" *
          "  DISABLED. Do NOT interpret results as bearoff-anchored.\n" *
          "════════════════════════════════════════════════════════════════════"
    ""
else
    find_bearoff_dir_server()  # errors (fail-fast) if no local k=7 table is present
end
const BEAROFF_FIXED_EVAL_ENABLED =
    ARGS["training_mode"] == "race" &&
    BEAROFF_EVAL_INTERVAL > 0 &&
    !isempty(BEAROFF_DIR) &&
    !isempty(BEAROFF_START_POSITIONS)

if EVAL_ENABLED
    println("Eval: $(length(EVAL_POSITIONS)) positions × 2 sides, $(EVAL_MCTS_ITERS) MCTS iters")
    println("Eval: distributed WildBG quality=$(ARGS["eval_backend_quality"]), every $EVAL_INTERVAL iters")
else
    println("Eval: disabled (set --eval-interval and --eval-positions-file to enable)")
end

if BEAROFF_FIXED_EVAL_ENABLED
    println("Bearoff eval: fixed canonical set, every $(BEAROFF_EVAL_INTERVAL) iters, raw=$(BEAROFF_EVAL_POSITIONS), mcts=$(BEAROFF_EVAL_MCTS_POSITIONS) @ $(BEAROFF_EVAL_MCTS_ITERS)")
else
    println("Bearoff eval: disabled")
end

# ── Weight promotion gate ──────────────────────────────────────────────────
# Gates PUBLICATION (served weight version + race_latest.data) on the fixed
# bearoff eval's value MAE (lower is better). Training is never gated. See
# src/distributed/promotion_gate.jl for the full rationale.
# The gate can only work when the bearoff eval (its signal) is enabled.
const GATE_ENABLED = BEAROFF_FIXED_EVAL_ENABLED && !ARGS["no_promotion_gate"]
const GATE_METRIC_NAME = "value_mae"      # race value head MAE vs exact k=7 table (normalized)
const GATE_TOL_FRAC = ARGS["gate_tolerance"]
const GATE_TOL_ABS = 0.003                # absolute floor (normalized eq ≈ 0.009 points)
const GATE_STATE = Ref(GateState())        # persists across iterations; seeded on --resume
if GATE_ENABLED
    println("Promotion gate: ENABLED — metric=$(GATE_METRIC_NAME), tol=$(round(100*GATE_TOL_FRAC, digits=1))% + $(GATE_TOL_ABS) abs. Publication held on regression; training never blocks. (race model only; contact publishes with race.)")
elseif ARGS["no_promotion_gate"]
    println("Promotion gate: DISABLED (--no-promotion-gate) — publishing every iteration unconditionally.")
else
    println("Promotion gate: DISABLED (fixed bearoff eval off — no gate signal available). Publishing every iteration unconditionally.")
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
    if net isa FluxLib.FCResNetMultiHead
        P_raw, Lw, Lgw, Lbgw, Lgl, Lbgl, _ = Network.convert_output_tuple(
            net, FluxLib.forward_normalized_multihead(net, X, A))
        V = zeros(Float32, 1, n)
        @inbounds for i in 1:n
            heads = (
                Float32(Flux.sigmoid(Lw[1, i])),
                Float32(Flux.sigmoid(Lgw[1, i])),
                Float32(Flux.sigmoid(Lbgw[1, i])),
                Float32(Flux.sigmoid(Lgl[1, i])),
                Float32(Flux.sigmoid(Lbgl[1, i])),
            )
            V[1, i] = BackgammonNet.search_value(states[i], heads; mode=:auto)
        end
    else
        P_raw, V, _ = Network.convert_output_tuple(
            net, Network.forward_normalized(net, X, A))
    end
    results = Vector{Tuple{Vector{Float32}, Float32}}(undef, n)
    for i in 1:n
        legal = @view(A[:, i]) .> 0
        results[i] = (P_raw[legal, i], V[1, i])
    end
    return results
end

@inline normalized_points_server(v::Real) = Float64(v) / Float64(REWARD_SCALE)

function start_game_from_tuple_server(position_data::Tuple{UInt128, UInt128, Int8}, seed::Int)
    p0, p1, cp = position_data
    game = backgammon_game(
        p0, p1, SVector{2, Int8}(0, 0), Int8(0), cp, false, 0.0f0;
        observation_type=:minimal_flat)
    return GameEnv(game, MersenneTwister(seed))
end

function make_bearoff_state_key(state::BackgammonNet.BackgammonGame)
    return (state.p0, state.p1, state.dice[1], state.dice[2], state.remaining_actions, state.current_player)
end

function build_bearoff_eval_positions(cache_path::String)
    isfile(cache_path) && return Serialization.deserialize(cache_path)
    isempty(BEAROFF_START_POSITIONS) && return BackgammonNet.BackgammonGame[]

    rng = MersenneTwister(ARGS["seed"])
    seen = Set{Tuple{UInt128, UInt128, Int8, Int8, Int8, Int8}}()
    candidates = BackgammonNet.BackgammonGame[]
    wanted = max(BEAROFF_EVAL_POSITIONS, BEAROFF_EVAL_MCTS_POSITIONS)

    for (i, start_pos) in enumerate(BEAROFF_START_POSITIONS)
        for r in 1:BEAROFF_EVAL_ROLLOUTS_PER_START
            env = start_game_from_tuple_server(start_pos, ARGS["seed"] + 100_000 * r + i)
            while !GI.game_terminated(env)
                if GI.is_chance_node(env)
                    BackgammonNet.sample_chance!(env.game, env.rng)
                    continue
                end
                state = GI.current_state(env)
                if state.phase == BackgammonNet.PHASE_CHECKER_PLAY &&
                   state.remaining_actions == Int8(1) &&
                   BearoffK7.is_bearoff_position(state.p0, state.p1) &&
                   length(BackgammonNet.legal_actions(state)) > 1
                    key = make_bearoff_state_key(state)
                    if !(key in seen)
                        push!(seen, key)
                        push!(candidates, state)
                    end
                end
                acts = BackgammonNet.legal_actions(env.game)
                isempty(acts) && break
                GI.play!(env, rand(rng, acts))
            end
        end
    end

    if length(candidates) < wanted
        error("Only found $(length(candidates)) canonical bearoff states, need $wanted")
    end
    order = randperm(rng, length(candidates))[1:wanted]
    positions = [candidates[i] for i in order]
    mkpath(dirname(cache_path))
    Serialization.serialize(cache_path, positions)
    return positions
end

function exact_bearoff_action_values(state::BackgammonNet.BackgammonGame, table)
    actions = BackgammonNet.legal_actions(state)
    mover = Int(state.current_player)
    work = BackgammonNet.clone(state)
    action_values = Dict{Int, Float64}()
    for action in actions
        BackgammonNet.copy_state!(work, state)
        BackgammonNet.apply_action!(work, action)
        # Turn-aware exact value: handles terminal rewards (gammon multiplier),
        # completed turns (opponent pre-dice lookup), and doubles mid-turn states
        # (recursion) — see BackgammonNet.bearoff_turn_value for the doubles pitfall.
        move_val = normalized_points_server(bearoff_turn_value(table, work, mover))
        action_values[action] = move_val
    end
    isempty(action_values) && error("No exact bearoff move values computed")
    vals = collect(values(action_values))
    best = maximum(vals)
    tol = 1e-8
    optimal = sort([a for (a, v) in action_values if best - v <= tol])
    nonbest = [v for v in vals if best - v > tol]
    second_best = isempty(nonbest) ? best : maximum(nonbest)
    margin = best - second_best
    return (action_values=action_values, best_value=best, optimal_actions=optimal, margin=margin)
end

function nn_greedy_bearoff_action(state, value_oracle)
    actions = BackgammonNet.legal_actions(state)
    mover = Int(state.current_player)
    # A4: score TERMINAL children (a move that bears off the last checker = an
    # immediate, known win/loss) from the exact game.reward — NOT the NN. Passing
    # decisive moves through the value network corrupts the greedy pick and the
    # nn_top1/nn_regret metric on exactly the moves that matter most. Only
    # non-terminal children go to the NN.
    values = Vector{Float64}(undef, length(actions))
    succs = BackgammonNet.BackgammonGame[]
    succ_idx = Int[]
    for (i, action) in enumerate(actions)
        g = BackgammonNet.clone(state)
        BackgammonNet.apply_action!(g, action)
        if g.terminated
            white_r = Float64(g.reward)               # reward is white-relative
            mover_r = mover == 0 ? white_r : -white_r # carries gammon multiplier
            values[i] = normalized_points_server(mover_r)
        else
            push!(succs, g); push!(succ_idx, i)
        end
    end
    if !isempty(succs)
        evals = value_oracle(succs)
        for (k, i) in enumerate(succ_idx)
            v = Float64(evals[k][2])
            # value_oracle is from the successor's current_player perspective.
            Int(succs[k].current_player) != mover && (v = -v)
            values[i] = v
        end
    end
    best_action = actions[1]
    best_value = -Inf
    for (i, action) in enumerate(actions)
        if values[i] > best_value
            best_value = values[i]
            best_action = action
        end
    end
    return best_action
end

function bearoff_policy_stats(state, policy_oracle, exact)
    actions = BackgammonNet.legal_actions(state)
    policy, _ = policy_oracle(state)
    order = sortperm(policy; rev=true)
    ranked_actions = actions[order]
    optimal_set = Set(exact.optimal_actions)
    opt_mass = 0.0
    expected_regret = 0.0
    for (a, p) in zip(actions, policy)
        p64 = Float64(p)
        if a in optimal_set
            opt_mass += p64
        end
        expected_regret += p64 * (exact.best_value - exact.action_values[a])
    end
    best_rank = length(actions) + 1
    for (rank, a) in enumerate(ranked_actions)
        if a in optimal_set
            best_rank = rank
            break
        end
    end
    return (
        top1 = ranked_actions[1] in optimal_set,
        top3 = any(a in optimal_set for a in ranked_actions[1:min(3, end)]),
        top5 = any(a in optimal_set for a in ranked_actions[1:min(5, end)]),
        best_rank = best_rank,
        opt_mass = opt_mass,
        expected_regret = expected_regret,
        top1_prob = Float64(policy[order[1]]),
    )
end

function mcts_bearoff_action(state, player, seed::Int)
    env = GameEnv(BackgammonNet.clone(state), MersenneTwister(seed))
    try
        actions, policy = think(player, env)
        return actions[argmax(policy)]
    finally
        reset_player!(player)
    end
end

function run_bearoff_eval!(network_to_eval, iter::Int)
    !BEAROFF_FIXED_EVAL_ENABLED && return nothing

    cache_path = joinpath(DATA_DIR, "bearoff_eval_positions_$(max(BEAROFF_EVAL_POSITIONS, BEAROFF_EVAL_MCTS_POSITIONS))_seed$(ARGS["seed"]).jls")
    positions = build_bearoff_eval_positions(cache_path)
    isempty(positions) && return nothing
    table = BearoffK7.BearoffTable(BEAROFF_DIR)
    eval_net = Flux.cpu(network_to_eval)

    batch_oracle(states::Vector) = _eval_forward_network(eval_net, states)
    policy_oracle(s) = batch_oracle([s])[1]
    value_oracle(states::Vector) = batch_oracle(states)
    mcts_params = MctsParams(
        num_iters_per_turn=BEAROFF_EVAL_MCTS_ITERS,
        cpuct=1.5,
        temperature=ConstSchedule(0.0),
        dirichlet_noise_ϵ=0.0,
        dirichlet_noise_α=1.0)
    player = MctsPlayer(gspec, policy_oracle, mcts_params)

    n_raw = min(BEAROFF_EVAL_POSITIONS, length(positions))
    raw_positions = @view positions[1:n_raw]
    n_mcts = min(BEAROFF_EVAL_MCTS_POSITIONS, length(positions))
    mcts_positions = @view positions[1:n_mcts]

    exact_values = Float64[]
    nn_values = Float64[]
    nn_wrong = Bool[]
    nn_regret = Float64[]
    policy_top1 = Bool[]
    policy_top3 = Bool[]
    policy_top5 = Bool[]
    policy_best_rank = Int[]
    policy_opt_mass = Float64[]
    policy_expected_regret = Float64[]
    policy_top1_prob = Float64[]
    margins = Float64[]
    n_opt_actions = Int[]

    for (idx, state) in enumerate(raw_positions)
        exact = exact_bearoff_action_values(state, table)
        push!(exact_values, exact.best_value)
        push!(margins, exact.margin)
        push!(n_opt_actions, length(exact.optimal_actions))
        push!(nn_values, Float64(value_oracle([state])[1][2]))

        pd = bearoff_policy_stats(state, policy_oracle, exact)
        push!(policy_top1, pd.top1)
        push!(policy_top3, pd.top3)
        push!(policy_top5, pd.top5)
        push!(policy_best_rank, pd.best_rank)
        push!(policy_opt_mass, pd.opt_mass)
        push!(policy_expected_regret, pd.expected_regret)
        push!(policy_top1_prob, pd.top1_prob)

        nn_move = nn_greedy_bearoff_action(state, value_oracle)
        regret = exact.best_value - exact.action_values[nn_move]
        push!(nn_regret, regret)
        push!(nn_wrong, !(nn_move in exact.optimal_actions))
    end

    diffs = nn_values .- exact_values
    result = Dict{String, Float64}(
        "value_mae" => mean(abs.(diffs)),
        "value_rmse" => sqrt(mean(diffs .^ 2)),
        "value_bias" => mean(diffs),
        "value_corr" => (length(nn_values) >= 3 && std(nn_values) > 0 && std(exact_values) > 0) ? cor(nn_values, exact_values) : 0.0,
        "policy_top1" => mean(policy_top1),
        "policy_top3" => mean(policy_top3),
        "policy_top5" => mean(policy_top5),
        "policy_best_rank" => mean(policy_best_rank),
        "policy_opt_mass" => mean(policy_opt_mass),
        "policy_expected_regret" => mean(policy_expected_regret),
        "policy_top1_prob" => mean(policy_top1_prob),
        "nn_top1" => 1 - mean(nn_wrong),
        "nn_wrong" => mean(nn_wrong),
        "nn_regret" => mean(nn_regret),
        "tie_rate" => mean(n_opt_actions .> 1),
        "avg_margin" => mean(margins),
    )

    if BEAROFF_EVAL_MCTS_POSITIONS > 0
        mcts_wrong = Bool[]
        mcts_regret = Float64[]
        for (idx, state) in enumerate(mcts_positions)
            exact = exact_bearoff_action_values(state, table)
            move = mcts_bearoff_action(state, player, ARGS["seed"] + iter * 10000 + idx)
            regret = exact.best_value - exact.action_values[move]
            push!(mcts_regret, regret)
            push!(mcts_wrong, !(move in exact.optimal_actions))
        end
        result["mcts_top1"] = 1 - mean(mcts_wrong)
        result["mcts_wrong"] = mean(mcts_wrong)
        result["mcts_regret"] = mean(mcts_regret)
        result["mcts_regret_gt_001"] = mean(mcts_regret .> 0.01)
    end

    return result
end

# Create networks
println("\nCreating networks...")
contact_network = FluxLib.FCResNetMultiHead(
    gspec, FluxLib.FCResNetMultiHeadHP(width=CONTACT_WIDTH, num_blocks=CONTACT_BLOCKS))
race_network = FluxLib.FCResNetMultiHead(
    gspec, FluxLib.FCResNetMultiHeadHP(width=RACE_WIDTH, num_blocks=RACE_BLOCKS))

println("Contact model parameters: $(sum(length, Flux.trainables(contact_network)))")
println("Race model parameters: $(sum(length, Flux.trainables(race_network)))")

# Move to GPU if available
if USE_GPU
    contact_network = Network.to_gpu(contact_network)
    race_network = Network.to_gpu(race_network)
    println("Models moved to GPU")
end

# Resume from checkpoint if specified
START_ITER = 0
RESUME_BUNDLE = nothing
if !isempty(ARGS["resume"])
    resume_dir = ARGS["resume"]
    bundle_roots = (resume_dir, joinpath(resume_dir, "checkpoints"))
    for root in bundle_roots
        candidate = latest_valid_checkpoint(root; required_files=[
            "contact_train.data", "race_train.data", "optimizer_state.jls", "rng_state.jls"])
        if candidate !== nothing
            global RESUME_BUNDLE = candidate
            break
        end
    end
    if RESUME_BUNDLE !== nothing
        manifest = validate_checkpoint_bundle(RESUME_BUNDLE)
        metadata = get(manifest, "metadata", Dict{String,Any}())
        saved_contract = String(get(metadata, "contract_fingerprint", ""))
        saved_contract == CONTRACT_FINGERPRINT || error(
            "Resume checkpoint ML contract mismatch: saved=$saved_contract current=$CONTRACT_FINGERPRINT")
        for (key, current) in (("contact_width", CONTACT_WIDTH),
                               ("contact_blocks", CONTACT_BLOCKS),
                               ("race_width", RACE_WIDTH),
                               ("race_blocks", RACE_BLOCKS))
            Int(get(metadata, key, current)) == current || error(
                "Resume checkpoint architecture mismatch for $key")
        end
        FluxLib.load_weights(joinpath(RESUME_BUNDLE, "contact_train.data"), contact_network)
        FluxLib.load_weights(joinpath(RESUME_BUNDLE, "race_train.data"), race_network)
        START_ITER = Int(manifest["iteration"])
        if GATE_ENABLED && isfile(joinpath(RESUME_BUNDLE, "gate_state.json"))
            GATE_STATE[] = load_gate_state(joinpath(RESUME_BUNDLE, "gate_state.json"))
        end
        println("Resume: selected newest valid transactional bundle $RESUME_BUNDLE")
        println("Resumed training weights at iteration $START_ITER")
    else
        error("No valid transactional checkpoint bundle found under $resume_dir")
    end
end

# Optimizers
contact_opt = Flux.AdamW(LEARNING_RATE, (0.9f0, 0.999f0), L2_REG)
contact_opt_state = Flux.setup(contact_opt, contact_network)
race_opt = Flux.AdamW(LEARNING_RATE, (0.9f0, 0.999f0), L2_REG)
race_opt_state = Flux.setup(race_opt, race_network)

if RESUME_BUNDLE !== nothing
    optimizer_state = Serialization.deserialize(joinpath(RESUME_BUNDLE, "optimizer_state.jls"))
    contact_opt_state = USE_GPU ? Flux.gpu(optimizer_state["contact"]) : optimizer_state["contact"]
    race_opt_state = USE_GPU ? Flux.gpu(optimizer_state["race"]) : optimizer_state["race"]
    TRAIN_RNG[] = Serialization.deserialize(joinpath(RESUME_BUNDLE, "rng_state.jls"))
    println("Resume: restored AdamW moments and learner RNG state")
end

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

# Bootstrap artifacts are accepted only through BackgammonNet's fail-closed
# backgammon_training_v4 loader. Historical vectors and raw columnar dumps are
# intentionally unsupported: they cannot prove the current observation, policy,
# value-head, perspective, or provenance contracts.
if !isempty(ARGS["bootstrap_file"])
    let bootstrap_path = ARGS["bootstrap_file"],
        t0 = time()

        println("\nLoading canonical bootstrap artifact from: $bootstrap_path")
        flush(stdout)
        artifact = BackgammonNet.load_training_artifact(bootstrap_path)
        if !isempty(artifact.cube_states)
            @warn "Skipping $(length(artifact.cube_states)) cube-policy samples: " *
                  "the current AlphaZero network has only the 676-way checker head"
        end

        OBSERVATION_FORMAT === :flat || error(
            "canonical bootstrap ingestion requires BACKGAMMON_OBS_TYPE=*_flat; " *
            "got $(OBSERVATION_TYPE)")
        n_bootstrap = length(artifact.states)
        max_load = ARGS["bootstrap_max_samples"] > 0 ?
            min(ARGS["bootstrap_max_samples"], BUFFER_CAPACITY) :
            min(n_bootstrap, BUFFER_CAPACITY)

        chunk_size = 10000
        loaded = 0
        for start_idx in 1:chunk_size:max_load
            end_idx = min(start_idx + chunk_size - 1, max_load)
            n = end_idx - start_idx + 1
            batch = BackgammonNet.TrainingBatch(
                n; kind=BackgammonNet.ACTION_TYPE_CHECKERS,
                tier=OBSERVATION_TIER, format=:flat)
            BackgammonNet.fill_training_batch!(batch, artifact, start_idx:end_idx)

            states = @view batch.observations[:, 1:n]
            policies = @view batch.policies[:, 1:n]
            values = @view batch.value_scalars[1:n]
            equities = @view batch.value_heads[:, 1:n]
            has_equity = fill(true, n)
            is_chance = Vector{Bool}(undef, n)
            is_contact = Vector{Bool}(undef, n)
            is_bearoff = Vector{Bool}(undef, n)
            for j in 1:n
                game = artifact.states[start_idx + j - 1]
                is_chance[j] = BackgammonNet.is_chance_node(game)
                is_contact[j] = BackgammonNet.is_contact_position(game)
                is_bearoff[j] = BearoffK7.is_bearoff_position(game.p0, game.p1)
            end

            per_add_batch!(replay_buffer, states, policies, values,
                           equities, has_equity, is_chance, is_contact, is_bearoff)
            loaded += n
        end

        artifact = nothing
        GC.gc()
        t_load = time() - t0
        println("  Loaded $loaded / $n_bootstrap bootstrap samples in $(round(t_load, digits=1))s")
        println("  Buffer size: $(buf_length(replay_buffer))")
        let parts = partition_indices(replay_buffer)
            println("  Partition: contact=$(length(parts.contact)) race=$(length(parts.race))")
            if ARGS["training_mode"] == "race" && !isempty(parts.contact)
                @warn "Race-only mode but bootstrap contains $(length(parts.contact)) contact samples — they will be excluded from training"
            end
        end
        flush(stdout)
    end
end

# Load buffer checkpoint if resuming (must happen after replay_buffer is created)
if RESUME_BUNDLE !== nothing && isfile(joinpath(RESUME_BUNDLE, "buffer.jls"))
    load_buffer!(replay_buffer, joinpath(RESUME_BUNDLE, "buffer.jls"))
    println("Resume: restored replay buffer from transactional bundle")
elseif RESUME_BUNDLE !== nothing
    println("Resume: bundle has no replay buffer — starting warm weights with a cold buffer")
end

# Training functions (extracted from train_distributed.jl)

"""Prepare training batch from columnar buffer extract.

Accepts the NamedTuple from extract_batch (columnar matrices) instead of
a Vector of NamedTuples — avoids per-sample allocation entirely.

The returned batch matches the same learner-side contract as
`AlphaZero.convert_samples`:
- `V` is the scalar player-relative target used by the single-head value path and
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
        V̂_combined = equity ./ REWARD_SCALE
        V_normalized = V ./ REWARD_SCALE  # Buffer V is RAW [-reward_scale, reward_scale] → [-1,1]
        td = abs.(Flux.cpu(V̂_combined) .- Flux.cpu(V_normalized))
    else
        _, V̂, _ = Network.forward_normalized(nn, X, A)
        td = abs.(Flux.cpu(V̂) .- Flux.cpu(V))
    end
    return Float32.(vec(td))
end

function _train_model_on_samples!(buf_indices::Vector{Int}, network, opt_state;
                                  expect_contact::Union{Nothing, Bool}=nothing,
                                  current_iteration::Int=0)
    n = length(buf_indices)
    n < BATCH_SIZE && return (
        avg_loss=0.0, avg_Lp=0.0, avg_Lv=0.0, avg_Linv=0.0,
        num_batches=0, skipped_batches=0, nonfinite_batches=0,
        td_error_mean=0.0, td_error_count=0,
        sample_age_mean=0.0, sample_age_max=0, sample_age_count=0)

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
    n_masked_skipped = 0
    n_nonfinite = 0
    total_td_error = 0.0
    n_td_errors = 0
    total_sample_age = 0
    max_sample_age = 0
    n_sample_ages = 0

    for _ in 1:num_batches
        # PER: sample proportional to priorities within this model's partition
        # Uniform: sample randomly from the partition
        if USE_PER
            batch_buf_indices, is_weights = per_sample_partition(
                replay_buffer, buf_indices, BATCH_SIZE, PER_ALPHA, PER_EPSILON;
                rng=TRAIN_RNG[])
        else
            sample_idx = rand(TRAIN_RNG[], 1:n, BATCH_SIZE)
            batch_buf_indices = buf_indices[sample_idx]
            is_weights = ones(Float32, BATCH_SIZE)
        end

        # Extract columnar data from buffer
        col_data = extract_batch(replay_buffer, batch_buf_indices)

        # Guard against stale partition membership: with a full circular buffer,
        # an index can be overwritten (race → contact) between partitioning and
        # extraction. Zero the loss weight of any sample whose CURRENT flag no
        # longer matches this model's partition instead of training on it.
        if expect_contact !== nothing && !all(col_data.is_contact .== expect_contact)
            is_weights = is_weights .* Float32.(col_data.is_contact .== expect_contact)
            # If the mask zeroed EVERY sample, Wmean would be 0 and the loss would
            # divide by it (NaN). Skip this batch instead. (Normal path — no
            # partition mismatch — never enters this branch, so it stays alloc-free.)
            if !any(!iszero, is_weights)
                n_masked_skipped += 1
                if n_masked_skipped == 1 || n_masked_skipped % 100 == 0
                    @warn "Skipped all-stale-partition batch (every sample masked out)" expect_contact n_masked_skipped
                end
                continue
            end
        end

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
        # Inspect the same pre-update batch components plus every numeric gradient
        # leaf before mutating either the model or optimizer state. A rejected
        # batch cannot poison weights, optimizer moments, or PER priorities.
        L, Lp, Lv, _, Linv = losses(network, LEARNING_PARAMS, Wmean, Hp, batch_data_per)
        Lf, Lpf, Lvf, Linvf = Float64(L), Float64(Lp), Float64(Lv), Float64(Linv)
        finite_scalars = isfinite(Float64(loss)) && isfinite(Lf) &&
            isfinite(Lpf) && isfinite(Lvf) && isfinite(Linvf)
        if !finite_scalars || !_all_finite_gradient(grads[1])
            n_nonfinite += 1
            if n_nonfinite == 1 || n_nonfinite % 100 == 0
                @warn "Rejected non-finite training batch before optimizer update" expect_contact n_nonfinite
            end
            continue
        end
        Flux.update!(opt_state, network, grads[1])

        total_loss += Lf
        total_Lp += Lpf
        total_Lv += Lvf
        total_Linv += Linvf
        for source_iteration in col_data.source_iterations
            if source_iteration >= 0
                age = max(0, current_iteration - Int(source_iteration))
                total_sample_age += age
                max_sample_age = max(max_sample_age, age)
                n_sample_ages += 1
            end
        end

        # Update PER priorities with TD-errors
        if USE_PER
            td_errors = compute_td_errors(network, batch_data_per)
            total_td_error += sum(td_errors)
            n_td_errors += length(td_errors)
            per_update_priorities!(replay_buffer, batch_buf_indices, td_errors)
        end
    end

    # Divide by batches actually trained on (skipped all-masked batches excluded).
    processed = num_batches - n_masked_skipped - n_nonfinite
    processed == 0 && return (
        avg_loss=0.0, avg_Lp=0.0, avg_Lv=0.0, avg_Linv=0.0,
        num_batches=0, skipped_batches=n_masked_skipped,
        nonfinite_batches=n_nonfinite, td_error_mean=0.0,
        td_error_count=n_td_errors, sample_age_mean=0.0,
        sample_age_max=max_sample_age, sample_age_count=n_sample_ages)
    return (avg_loss=total_loss / processed, avg_Lp=total_Lp / processed,
            avg_Lv=total_Lv / processed, avg_Linv=total_Linv / processed,
            num_batches=processed, skipped_batches=n_masked_skipped,
            nonfinite_batches=n_nonfinite,
            td_error_mean=n_td_errors == 0 ? 0.0 : total_td_error / n_td_errors,
            td_error_count=n_td_errors,
            sample_age_mean=n_sample_ages == 0 ? 0.0 : total_sample_age / n_sample_ages,
            sample_age_max=max_sample_age, sample_age_count=n_sample_ages)
end

function train_on_buffer!(current_iteration::Int)
    n_buf = buf_length(replay_buffer)
    if n_buf < BATCH_SIZE
        empty_result = (
            avg_loss=0.0, avg_Lp=0.0, avg_Lv=0.0, avg_Linv=0.0,
            num_batches=0, skipped_batches=0, nonfinite_batches=0,
            td_error_mean=0.0, td_error_count=0,
            sample_age_mean=0.0, sample_age_max=0, sample_age_count=0)
        return (contact=empty_result, race=empty_result)
    end

    if USE_PER
        per_anneal_beta!(replay_buffer)
    end

    # Partition buffer indices by contact/race (single lock acquisition)
    parts = partition_indices(replay_buffer)

    if ARGS["training_mode"] == "race"
        # Race-only mode: train race ONLY on race samples. Training on parts.all
        # would contaminate the race net with contact states if the buffer holds
        # any (e.g. a full-game bootstrap artifact).
        n_contact = length(parts.contact)
        if n_contact > 0
            @warn "Race-only mode: buffer holds $(n_contact) contact samples — excluded from training" maxlog=10
        end
        contact_result = (
            avg_loss=0.0, avg_Lp=0.0, avg_Lv=0.0, avg_Linv=0.0,
            num_batches=0, skipped_batches=0, nonfinite_batches=0,
            td_error_mean=0.0, td_error_count=0,
            sample_age_mean=0.0, sample_age_max=0, sample_age_count=0)
        race_result = _train_model_on_samples!(parts.race, race_network, race_opt_state;
                                               expect_contact=false,
                                               current_iteration=current_iteration)
    else
        contact_result = _train_model_on_samples!(parts.contact, contact_network, contact_opt_state;
                                                  expect_contact=true,
                                                  current_iteration=current_iteration)
        race_result = _train_model_on_samples!(parts.race, race_network, race_opt_state;
                                               expect_contact=false,
                                               current_iteration=current_iteration)
    end

    return (contact=contact_result, race=race_result)
end

function reanalyze_buffer!()
    USE_REANALYZE || return 0
    n = buf_length(replay_buffer)
    n == 0 && return 0

    num_to_reanalyze = max(1, round(Int, n * REANALYZE_FRACTION))
    reanalyze_indices = randperm(TRAIN_RNG[], n)[1:min(num_to_reanalyze, n)]

    batch_size = min(2048, length(reanalyze_indices))
    total_updated = 0
    total_skipped = 0

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
            # Generation snapshot captured at extract time — reanalyze_update!
            # skips any of these slots overwritten during the NN inference below.
            sub_generations = col_data.generations[sub_local_idx]

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
            new_values = Float32.(vec(Flux.cpu(equity)))  # Keep raw [-3,3] scale to match selfplay

            new_eq_win = Float32.(vec(Flux.cpu(V̂_win)))
            new_eq_gw = Float32.(vec(Flux.cpu(V̂_gw)))
            new_eq_bgw = Float32.(vec(Flux.cpu(V̂_bgw)))
            new_eq_gl = Float32.(vec(Flux.cpu(V̂_gl)))
            new_eq_bgl = Float32.(vec(Flux.cpu(V̂_bgl)))

            # Write blended values back (skips slots overwritten since extraction)
            skipped = reanalyze_update!(replay_buffer, sub_buf_indices, sub_generations,
                              new_values, new_eq_win, new_eq_gw, new_eq_bgw, new_eq_gl, new_eq_bgl;
                              α_blend=REANALYZE_BLEND)

            total_skipped += skipped
            total_updated += length(sub_buf_indices) - skipped
        end

        # Let GC clean up batch temporaries
        col_data = nothing
    end

    if total_skipped > 0
        @info "Reanalyze skipped stale slots (overwritten during inference)" total_skipped total_updated
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

include(joinpath(@__DIR__, "..", "src", "distributed", "tensorboard_dashboard.jl"))
const TB_DASHBOARD_LAYOUT = install_tensorboard_dashboard!(TB_LOGGER)

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
    @info "00_run/config" text=repro_text log_step_increment=0
end

function finalize_eval_job!(job; source::String="training-loop")
    # finalize_eval aggregates client-submitted value arrays and can throw
    # (e.g. DimensionMismatch on a malformed submission). This runs inside the
    # training task, so guard it and always clear the job.
    saved_step = TB_LOGGER.global_step
    try
        result = EvalManager.finalize_eval(job)
        @info "Eval completed iter $(job.iter)" source equity=round(result.equity, digits=3) win_pct=round(result.win_pct * 100, digits=1) games=result.num_games
        # Log to TB at the eval iteration (not the current training iteration).
        TB_LOGGER.global_step = job.iter
        with_logger(TB_LOGGER) do
            @info "05_eval_strength/equity" value=result.equity log_step_increment=0
            @info "05_eval_strength/win_pct" value=result.win_pct * 100 log_step_increment=0
            @info "05_eval_strength/white_equity" value=result.white_equity log_step_increment=0
            @info "05_eval_strength/black_equity" value=result.black_equity log_step_increment=0
            @info "05_eval_strength/contact_value_mse" value=result.contact_value_mse log_step_increment=0
            @info "05_eval_strength/contact_value_corr" value=result.contact_value_corr log_step_increment=0
            @info "05_eval_strength/race_value_mse" value=result.race_value_mse log_step_increment=0
            @info "05_eval_strength/race_value_corr" value=result.race_value_corr log_step_increment=0
        end
    catch e
        @error "finalize_eval failed; discarding eval job for iter $(job.iter)" source exception=(e, catch_backtrace())
    finally
        TB_LOGGER.global_step = saved_step
        EVAL_JOB[] = nothing
    end
end

function wait_for_final_eval!()
    deadline = time() + EVAL_JOB_TIMEOUT
    announced = false
    while true
        wait_more = false
        lock(EVAL_LOCK) do
            job = EVAL_JOB[]
            if job === nothing
                return
            elseif EvalManager.is_complete(job)
                finalize_eval_job!(job; source="final-wait")
                return
            elseif time() >= deadline
                st = EvalManager.status(job)
                @warn "Final eval timed out; discarding unfinished eval job" iter=job.iter completed=st.completed total_chunks=st.total_chunks timeout_s=EVAL_JOB_TIMEOUT
                EVAL_JOB[] = nothing
                return
            else
                if !announced
                    st = EvalManager.status(job)
                    println("Waiting for final eval iter $(job.iter) to complete ($(st.completed)/$(st.total_chunks) chunks)...")
                    flush(stdout)
                    announced = true
                end
                wait_more = true
            end
        end
        wait_more || return
        sleep(10)
    end
end

# Server config (served to clients via GET /api/config)
const SERVER_CONFIG = Dict{String, Any}(
    "protocol_version" => DISTRIBUTED_PROTOCOL_VERSION,
    "ml_contract" => ML_CONTRACT,
    "contract_fingerprint" => CONTRACT_FINGERPRINT,
    "observation_type" => String(OBSERVATION_TYPE),
    "cube_enabled" => CUBE_ENABLED,
    "jacoby_enabled" => JACOBY_ENABLED,
    "tavla_enabled" => TAVLA_ENABLED,
    "mcts_iters" => ARGS["mcts_iters"],
    "mcts_budget_mode" => ARGS["mcts_budget_mode"],
    "progressive_sim_min" => ARGS["progressive_sim_min"],
    "progressive_sim_max" => ARGS["progressive_sim_max"],
    "turn_sim_min" => ARGS["turn_sim_min"],
    "turn_sim_target" => ARGS["turn_sim_target"],
    "ramp_turns_initial" => ARGS["ramp_turns_initial"],
    "ramp_turns_final" => ARGS["ramp_turns_final"],
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
    # F3: propagate the server's bearoff state so --no-bearoff is CLUSTER-WIDE.
    # When bearoff is on, clients must have the local table (they fail-fast otherwise);
    # when off, clients run without it. Never let table presence silently diverge
    # training targets across clients.
    "use_bearoff" => !isempty(BEAROFF_DIR),
    "bearoff_hard_targets" => !isempty(BEAROFF_DIR) && ARGS["bearoff_hard_targets"],
    "bearoff_truncation" => !isempty(BEAROFF_DIR) && ARGS["bearoff_truncation"],
    "bootstrap_only" => ARGS["bootstrap_only"],
    "training_mode" => ARGS["training_mode"],
    "start_positions_file" => basename(ARGS["start_positions_file"]),
    "eval_positions_file" => basename(ARGS["eval_positions_file"]),
    "eval_manifest_file" => basename(EVAL_MANIFEST_PATH),
    "eval_data_dir" => isempty(ARGS["eval_positions_file"]) ? "" : dirname(abspath(ARGS["eval_positions_file"])),
    "data_dir" => abspath(DATA_DIR),
    "seed" => ARGS["seed"],
    "eval_mcts_iters" => ARGS["eval_mcts_iters"],
    "eval_backend_quality" => ARGS["eval_backend_quality"],
)

# Initialize server state
server_state = ServerState(api_key=ARGS["api_key"], config=SERVER_CONFIG)
server_state.iteration[] = START_ITER

# Cache initial weights. Transactional resume restores the last PUBLISHED blobs,
# which may intentionally differ from training weights after a promotion block.
if RESUME_BUNDLE !== nothing &&
   isfile(joinpath(RESUME_BUNDLE, "contact_published.weights")) &&
   isfile(joinpath(RESUME_BUNDLE, "race_published.weights"))
    contact_bytes = read(joinpath(RESUME_BUNDLE, "contact_published.weights"))
    race_bytes = read(joinpath(RESUME_BUNDLE, "race_published.weights"))
    deserialize_weights_with_header(contact_bytes)
    deserialize_weights_with_header(race_bytes)
    metadata = checkpoint_manifest(RESUME_BUNDLE)["metadata"]
    server_state.contact_weight_bytes = contact_bytes
    server_state.race_weight_bytes = race_bytes
    server_state.contact_version[] = Int(get(metadata, "contact_version", 1))
    server_state.race_version[] = Int(get(metadata, "race_version", 1))
    version = max(server_state.contact_version[], server_state.race_version[])
    server_state.weight_history[version] = (copy(contact_bytes), copy(race_bytes))
    println("Resume: restored last-good published weights separately from training state")
else
    update_weight_cache!(server_state, contact_network, race_network;
                         contact_width=CONTACT_WIDTH, contact_blocks=CONTACT_BLOCKS,
                         race_width=RACE_WIDTH, race_blocks=RACE_BLOCKS)
end

function _git_revision(path::AbstractString)
    try
        return strip(read(`git -C $path rev-parse HEAD`, String))
    catch
        return "unknown"
    end
end

function _git_dirty(path::AbstractString)
    try
        return !isempty(strip(read(`git -C $path status --porcelain`, String)))
    catch
        return nothing
    end
end

function _artifact_identity(path::AbstractString)
    isempty(path) && return nothing
    isfile(path) || return Dict("path" => abspath(path), "missing" => true)
    digest = open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
    return Dict("path" => abspath(path), "bytes" => filesize(path), "sha256" => digest)
end

const RUN_PROVENANCE = Dict{String,Any}(
    "alphazero_commit" => _git_revision(dirname(@__DIR__)),
    "alphazero_dirty" => _git_dirty(dirname(@__DIR__)),
    "backgammonnet_version" => string(Base.pkgversion(BackgammonNet)),
    "backgammonnet_commit" => _git_revision(dirname(pathof(BackgammonNet))),
    "backgammonnet_dirty" => _git_dirty(dirname(pathof(BackgammonNet))),
    "julia_version" => string(VERSION),
    "contract_fingerprint" => CONTRACT_FINGERPRINT,
    "config_fingerprint" => contract_fingerprint(SERVER_CONFIG),
    "seed" => ARGS["seed"],
    "artifacts" => Dict(
        "bootstrap" => _artifact_identity(ARGS["bootstrap_file"]),
        "start_positions" => _artifact_identity(ARGS["start_positions_file"]),
        "eval_positions" => _artifact_identity(ARGS["eval_positions_file"]),
        "eval_manifest" => _artifact_identity(EVAL_MANIFEST_PATH),
    ),
)
SERVER_CONFIG["run_provenance"] = RUN_PROVENANCE

function _atomic_copy(source::AbstractString, destination::AbstractString)
    temporary = destination * ".tmp"
    cp(source, temporary; force=true)
    mv(temporary, destination; force=true)
    return destination
end

function save_training_checkpoint_bundle!(iteration::Int;
                                          include_buffer::Bool=false,
                                          reason::String="periodic")
    published_contact, published_race = lock(server_state.weight_lock) do
        (copy(server_state.contact_weight_bytes), copy(server_state.race_weight_bytes))
    end
    writers = Dict{String,Function}(
        "contact_train.data" => path -> FluxLib.save_weights(path, Flux.cpu(contact_network)),
        "race_train.data" => path -> FluxLib.save_weights(path, Flux.cpu(race_network)),
        "optimizer_state.jls" => path -> Serialization.serialize(path, Dict(
            "contact" => Flux.cpu(contact_opt_state),
            "race" => Flux.cpu(race_opt_state))),
        "rng_state.jls" => path -> Serialization.serialize(path, TRAIN_RNG[]),
        "run_config.json" => path -> open(path, "w") do io
            JSON.print(io, Dict("server_config" => SERVER_CONFIG,
                                "provenance" => RUN_PROVENANCE))
        end,
        "contact_published.weights" => path -> write(path, published_contact),
        "race_published.weights" => path -> write(path, published_race),
    )
    if GATE_ENABLED
        writers["gate_state.json"] = path -> save_gate_state(
            path, GATE_STATE[]; metric_name=GATE_METRIC_NAME,
            tol_frac=GATE_TOL_FRAC, tol_abs=GATE_TOL_ABS)
    end
    include_buffer && (writers["buffer.jls"] = path -> save_buffer(replay_buffer, path))
    metadata = merge(copy(RUN_PROVENANCE), Dict{String,Any}(
        "reason" => reason,
        "includes_buffer" => include_buffer,
        "contact_version" => server_state.contact_version[],
        "race_version" => server_state.race_version[],
        "contact_width" => CONTACT_WIDTH,
        "contact_blocks" => CONTACT_BLOCKS,
        "race_width" => RACE_WIDTH,
        "race_blocks" => RACE_BLOCKS,
        "training_mode" => ARGS["training_mode"],
    ))
    bundle = write_checkpoint_bundle!(CHECKPOINT_DIR, iteration, writers; metadata)
    return bundle
end

function perform_preflight!()
    report_path = joinpath(DATA_DIR, "preflight_report.json")
    checks = Pair{String,Function}[
        "Julia version" => function()
            VERSION >= v"1.12.6" || error("Julia >= 1.12.6 required, got $VERSION")
            return string(VERSION)
        end,
        "BackgammonNet version" => function()
            version = Base.pkgversion(BackgammonNet)
            version >= v"0.7.0" || error("BackgammonNet >= 0.7.0 required, got $version")
            return string(version)
        end,
        "BackgammonNet revision" => function()
            package_root = dirname(dirname(pathof(BackgammonNet)))
            revision = _git_revision(package_root)
            revision == "unknown" && error(
                "BackgammonNet source must be a Git checkout for reproducible validation")
            dirty = _git_dirty(package_root)
            dirty === false || error(
                "BackgammonNet checkout must be clean before validation (dirty=$dirty)")
            return revision
        end,
        "ML contract" => function()
            contract_fingerprint(backgammon_ml_contract(gspec)) == CONTRACT_FINGERPRINT ||
                error("ML contract fingerprint is unstable")
            return Dict("fingerprint" => CONTRACT_FINGERPRINT,
                        "state_dim" => _state_dim, "num_actions" => NUM_ACTIONS)
        end,
        "21 chance outcomes" => function()
            GI.num_chance_outcomes(gspec) == 21 || error(
                "wrapper reports $(GI.num_chance_outcomes(gspec)) chance outcomes")
            length(BackgammonNet.DICE_OUTCOMES) == 21 || error(
                "BackgammonNet reports $(length(BackgammonNet.DICE_OUTCOMES)) dice outcomes")
            length(BackgammonNet.DICE_PROBS) == 21 || error("dice probability length mismatch")
            isapprox(sum(BackgammonNet.DICE_PROBS), 1; atol=1e-12) || error(
                "dice probabilities do not sum to one")
            return 21
        end,
        "value-head probability contract" => function()
            BackgammonNet.check_probability_contract(
                (0.55, 0.15, 0.03, 0.12, 0.02); label="preflight")
            return String(BackgammonNet.VALUE_HEAD_CONTRACT)
        end,
        "contact and race inference" => function()
            env = GI.init(gspec)
            state = Float32.(reshape(vec(GI.vectorize_state(gspec, GI.current_state(env))), :, 1))
            input = USE_GPU ? Flux.gpu(state) : state
            for (name, network) in (("contact", contact_network), ("race", race_network))
                policy, value = Network.forward(network, input)
                size(policy, 1) == NUM_ACTIONS || error("$name policy width mismatch")
                all(isfinite, Flux.cpu(policy)) || error("$name policy is non-finite")
                all(isfinite, Flux.cpu(value)) || error("$name value is non-finite")
            end
            return "finite"
        end,
        "distributed protocol round-trip" => function()
            batch = SampleBatch(Int32(1), zeros(Float32, _state_dim, 1),
                zeros(Float32, NUM_ACTIONS, 1), zeros(Float32, 1),
                zeros(Float32, 5, 1), falses(1), trues(1), falses(1), falses(1))
            envelope = unpack_samples_envelope(pack_samples(batch;
                contract_fingerprint=CONTRACT_FINGERPRINT,
                batch_id="preflight-1", source_iteration=START_ITER))
            validate_sample_envelope!(envelope, CONTRACT_FINGERPRINT)
            return DISTRIBUTED_PROTOCOL_VERSION
        end,
        "weight serialization checksum" => function()
            bytes = serialize_weights_with_header(Flux.cpu(contact_network),
                WeightHeader(0x01, Int32(START_ITER), Int32(CONTACT_WIDTH),
                             Int32(CONTACT_BLOCKS), UInt64(0)))
            header, _ = deserialize_weights_with_header(bytes)
            return Dict("iteration" => Int(header.iteration), "bytes" => length(bytes))
        end,
        "configured artifacts" => function()
            for (label, path) in (("bootstrap", ARGS["bootstrap_file"]),
                                  ("start positions", ARGS["start_positions_file"]),
                                  ("eval positions", ARGS["eval_positions_file"]))
                isempty(path) || isfile(path) || error("$label artifact missing: $path")
            end
            return "present or intentionally disabled"
        end,
        "data directory writable" => function()
            path, io = mktemp(DATA_DIR)
            close(io)
            rm(path)
            return abspath(DATA_DIR)
        end,
    ]
    report = run_preflight!(checks, report_path; metadata=RUN_PROVENANCE)
    println("Preflight passed ($(length(report["checks"])) checks): $report_path")
    return report
end

if ARGS["preflight"]
    perform_preflight!()
    exit(0)
end

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

# Observability snapshots are cumulative on the HTTP path. Taking one delta per
# training iteration keeps TensorBoard series stable and avoids hot-path logging.
metrics_previous = server_metrics_snapshot(server_state)
metrics_previous_samples = server_state.total_samples[]
metrics_previous_games = server_state.total_games[]
metrics_previous_time = time()

# Eval at iter 0 (bootstrap weights) — baseline before any self-play training
if EVAL_ENABLED && START_ITER == 0
    lock(EVAL_LOCK) do
        wv = ARGS["training_mode"] == "race" ? server_state.race_version[] : server_state.contact_version[]
        # Pin current weights in history so eval clients can download by version
        lock(server_state.weight_lock) do
            if !haskey(server_state.weight_history, wv) &&
               !isempty(server_state.contact_weight_bytes) && !isempty(server_state.race_weight_bytes)
                server_state.weight_history[wv] = (copy(server_state.contact_weight_bytes),
                                                    copy(server_state.race_weight_bytes))
            end
        end
        n_pos = length(EVAL_POSITIONS)
        EVAL_JOB[] = EvalManager.create_eval_job(0, n_pos, wv; chunk_size=EVAL_CHUNK_SIZE)
        println("Eval job created for iter 0 (bootstrap baseline): $(length(EVAL_JOB[].chunks)) chunks")
    end
end

for iter in (START_ITER + 1):ARGS["total_iterations"]
    server_state.shutdown_requested[] && break
    # Wait for enough new samples (offset by START_ITER so resume works)
    if !ARGS["bootstrap_only"]
        target_samples = (iter - START_ITER) * SAMPLES_PER_ITERATION
        while server_state.total_samples[] < target_samples &&
              !server_state.shutdown_requested[]
            cur = server_state.total_samples[]
            pct = round(100 * cur / target_samples, digits=1)
            n_clients = length(server_state.clients)
            print("\rIteration $iter: waiting for samples ($cur / $target_samples = $pct%, $n_clients clients)  ")
            flush(stdout)
            sleep(5)
        end
        server_state.shutdown_requested[] && break
        println()
    end

    iter_start = time()

    # Bootstrap phase complete — clear buffer to switch to pure self-play
    if !ARGS["bootstrap_only"] && ARGS["bootstrap_train_iters"] > 0 && iter == ARGS["bootstrap_train_iters"] + 1
        println("\n*** Bootstrap phase complete ($(ARGS["bootstrap_train_iters"]) iters). Clearing buffer for pure self-play. ***")
        flush(stdout)
        clear_for_selfplay!(replay_buffer)
        # Wait for self-play to fill buffer with at least 1 iter worth of data
        min_samples = SAMPLES_PER_ITERATION
        println("Waiting for $min_samples self-play samples before resuming training...")
        while buf_length(replay_buffer) < min_samples
            cur = buf_length(replay_buffer)
            print("\rBuffer refill: $cur / $min_samples ($(round(100*cur/min_samples, digits=1))%)  ")
            flush(stdout)
            sleep(5)
        end
        println("\nBuffer refilled. Resuming training with pure self-play data.")
        flush(stdout)
    end

    # Update learning rate
    current_lr = update_lr!(contact_opt_state, iter, ARGS["total_iterations"])
    update_lr!(race_opt_state, iter, ARGS["total_iterations"])

    # Train on buffer (GPU)
    t0 = time()
    train_result = train_on_buffer!(iter)
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

    # Weight PUBLICATION is deferred until after the fixed bearoff eval below, so
    # the promotion gate can act on THIS iteration's eval before serving new
    # weights. See "Weight publication (gated)" further down.

    # Log to console
    grad_steps = train_result.contact.num_batches + train_result.race.num_batches
    skipped_batches = train_result.contact.skipped_batches + train_result.race.skipped_batches
    nonfinite_batches = train_result.contact.nonfinite_batches + train_result.race.nonfinite_batches
    @info "Iteration $iter" avg_loss contact_loss race_loss grad_steps skipped_batches nonfinite_batches buffer_size=buf_length(replay_buffer) total_samples=server_state.total_samples[] n_clients=length(server_state.clients) iter_time t_train t_reanalyze n_reanalyzed

    # Collect server and cluster stats
    server_stats = collect_server_stats()
    cluster_stats = get_cluster_stats(server_state)
    metrics_now = server_metrics_snapshot(server_state)
    metrics_now_time = time()
    metrics_elapsed = max(metrics_now_time - metrics_previous_time, 1e-6)
    metric_delta(name) = getproperty(metrics_now, name) - getproperty(metrics_previous, name)
    new_samples = server_state.total_samples[] - metrics_previous_samples
    new_games = server_state.total_games[] - metrics_previous_games
    new_requests = metric_delta(:upload_requests)
    new_simulations = metric_delta(:mcts_simulations)
    new_tree_hits = metric_delta(:tree_hits)
    new_tree_misses = metric_delta(:tree_misses)
    new_nn_evals = metric_delta(:nn_evaluations)
    new_oracle_calls = metric_delta(:oracle_calls)
    new_bearoff_hits = metric_delta(:bearoff_hits)
    new_bearoff_misses = metric_delta(:bearoff_misses)
    new_search_ns = metric_delta(:search_ns)

    # Log to TensorBoard
    with_logger(TB_LOGGER) do
        @info "01_ml_loss/overall" value=avg_loss log_step_increment=0
        @info "01_ml_loss/contact_total" value=contact_loss log_step_increment=0
        @info "01_ml_loss/race_total" value=race_loss log_step_increment=0
        # Per-component losses
        if train_result.contact.num_batches > 0
            @info "01_ml_loss/contact_policy" value=train_result.contact.avg_Lp log_step_increment=0
            @info "01_ml_loss/contact_value" value=train_result.contact.avg_Lv log_step_increment=0
            @info "01_ml_loss/contact_invalid" value=train_result.contact.avg_Linv log_step_increment=0
        end
        if train_result.race.num_batches > 0
            @info "01_ml_loss/race_policy" value=train_result.race.avg_Lp log_step_increment=0
            @info "01_ml_loss/race_value" value=train_result.race.avg_Lv log_step_increment=0
            @info "01_ml_loss/race_invalid" value=train_result.race.avg_Linv log_step_increment=0
        end
        @info "02_ml_perf/train_seconds" value=t_train log_step_increment=0
        @info "02_ml_perf/reanalyze_seconds" value=t_reanalyze log_step_increment=0
        @info "02_ml_perf/iteration_seconds" value=iter_time log_step_increment=0
        buf_parts = partition_indices(replay_buffer)
        @info "04_data/contact_samples" value=length(buf_parts.contact) log_step_increment=0
        @info "04_data/race_samples" value=length(buf_parts.race) log_step_increment=0
        age_count = train_result.contact.sample_age_count + train_result.race.sample_age_count
        if age_count > 0
            age_mean = (
                train_result.contact.sample_age_mean * train_result.contact.sample_age_count +
                train_result.race.sample_age_mean * train_result.race.sample_age_count) / age_count
            age_max = max(train_result.contact.sample_age_max,
                          train_result.race.sample_age_max)
            @info "04_data/train_sample_age_mean" value=age_mean log_step_increment=0
            @info "04_data/train_sample_age_max" value=age_max log_step_increment=0
        end
        if USE_PER
            @info "02_ml_perf/per_beta" value=replay_buffer.beta log_step_increment=0
        end
        @info "02_ml_perf/learning_rate" value=current_lr log_step_increment=0
        @info "02_ml_perf/samples_per_sec" value=(grad_steps * BATCH_SIZE / max(t_train, 1e-6)) log_step_increment=0
        @info "02_ml_perf/skipped_batches" value=skipped_batches log_step_increment=0
        @info "02_ml_perf/nonfinite_batches" value=nonfinite_batches log_step_increment=0
        td_count = train_result.contact.td_error_count + train_result.race.td_error_count
        if td_count > 0
            td_mean = (
                train_result.contact.td_error_mean * train_result.contact.td_error_count +
                train_result.race.td_error_mean * train_result.race.td_error_count) / td_count
            @info "02_ml_perf/per_td_error_mean" value=td_mean log_step_increment=0
        end

        # Cluster performance
        @info "03_selfplay_perf/samples_per_sec" value=(new_samples / metrics_elapsed) log_step_increment=0
        @info "03_selfplay_perf/active_clients" value=cluster_stats.total_clients log_step_increment=0

        # Compact self-play/search dashboard. Detailed cumulative counters remain
        # available via /api/status; dynamic per-client series stay in /api/clients.
        @info "03_selfplay_perf/games_per_sec" value=(new_games / metrics_elapsed) log_step_increment=0
        if new_search_ns > 0
            @info "03_selfplay_perf/mcts_sims_per_sec" value=(new_simulations / (new_search_ns / 1e9)) log_step_increment=0
        end
        if new_simulations > 0
            @info "03_selfplay_perf/nn_evals_per_sim" value=(new_nn_evals / new_simulations) log_step_increment=0
        end
        if new_oracle_calls > 0
            @info "03_selfplay_perf/oracle_batch_size" value=(new_nn_evals / new_oracle_calls) log_step_increment=0
        end
        tree_probes = new_tree_hits + new_tree_misses
        if tree_probes > 0
            @info "03_selfplay_perf/tree_hit_rate" value=(new_tree_hits / tree_probes) log_step_increment=0
        end
        bearoff_probes = new_bearoff_hits + new_bearoff_misses
        if bearoff_probes > 0
            @info "03_selfplay_perf/bearoff_hit_rate" value=(new_bearoff_hits / bearoff_probes) log_step_increment=0
        end
        if new_requests > 0
            @info "08_reliability/upload_latency_ms" value=(metric_delta(:upload_ns) / new_requests / 1e6) log_step_increment=0
        end
        @info "08_reliability/duplicate_batches" value=metric_delta(:duplicate_batches) log_step_increment=0
        @info "08_reliability/rejected_batches" value=metric_delta(:rejected_batches) log_step_increment=0

        # Server stats
        @info "07_system/gpu_percent" value=server_stats["gpu_percent"] log_step_increment=0
        @info "07_system/gpu_memory_gb" value=server_stats["gpu_memory_used_gb"] log_step_increment=0
        @info "07_system/cpu_percent" value=server_stats["cpu_percent"] log_step_increment=0

        # Buffer reward distribution (sanity check: gammon/backgammon rates)
        n_buf = buf_length(replay_buffer)
        if n_buf > 0
            n_bg_loss = 0; n_g_loss = 0; n_loss = 0
            n_win = 0; n_g_win = 0; n_bg_win = 0
            n_equity = 0; n_chance = 0; n_bearoff = 0
            @inbounds for i in 1:n_buf
                v = replay_buffer.values[i]
                if v <= -2.5f0
                    n_bg_loss += 1
                elseif v <= -1.5f0
                    n_g_loss += 1
                elseif v < -0.5f0
                    n_loss += 1
                elseif 0.5f0 < v < 1.5f0
                    n_win += 1
                elseif 1.5f0 <= v < 2.5f0
                    n_g_win += 1
                elseif v >= 2.5f0
                    n_bg_win += 1
                end
                n_equity += replay_buffer.has_equity[i]
                n_chance += replay_buffer.is_chance[i]
                n_bearoff += replay_buffer.is_bearoff[i]
            end
            @info "04_data/win_rate" value=(n_win+n_g_win+n_bg_win)/n_buf log_step_increment=0
            @info "04_data/gammon_rate" value=(n_g_loss+n_bg_loss+n_g_win+n_bg_win)/n_buf log_step_increment=0
            @info "04_data/equity_label_rate" value=n_equity/n_buf log_step_increment=0
            @info "04_data/chance_sample_rate" value=n_chance/n_buf log_step_increment=0
            @info "04_data/bearoff_sample_rate" value=n_bearoff/n_buf log_step_increment=0
            @info "04_data/backgammon_rate" value=(n_bg_loss+n_bg_win)/n_buf log_step_increment=1
        else
            @info "04_data/backgammon_rate" value=0 log_step_increment=1
        end
    end

    metrics_previous = metrics_now
    metrics_previous_samples = server_state.total_samples[]
    metrics_previous_games = server_state.total_games[]
    metrics_previous_time = metrics_now_time

    # Bearoff accuracy: NN equity vs exact table targets on bearoff positions
    # Measures how well the NN has learned bearoff evaluation
    try
        n_buf = buf_length(replay_buffer)
        bearoff_mask = findall(i -> replay_buffer.is_bearoff[i] && replay_buffer.has_equity[i], 1:n_buf)
        n_bo = length(bearoff_mask)
        if n_bo >= 100
            n_sample = min(1000, n_bo)
            # Draw a random subset of ALL bearoff slots, not just the first n_sample
            # (randperm(n_sample) only permutes 1:n_sample → oldest slots only).
            sample_idx = bearoff_mask[randperm(TRAIN_RNG[], n_bo)[1:n_sample]]

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
                 (bo_eq_targets[3,i] - bo_eq_targets[5,i])) / REWARD_SCALE
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
                @info "06_eval_bearoff/learned_value_mae" value=bo_mae log_step_increment=0
                @info "06_eval_bearoff/learned_value_corr" value=bo_corr log_step_increment=0
                @info "06_eval_bearoff/learned_samples" value=n_bo log_step_increment=0
            end
            @info "Bearoff accuracy" mse=round(bo_mse, digits=6) mae=round(bo_mae, digits=4) corr=round(bo_corr, digits=4) n_bearoff=n_bo n_sampled=n_sample
        end
    catch e
        @warn "Bearoff accuracy computation failed" exception=e
    end

    # Fixed-set bearoff eval on canonical post-dice bearoff decision states
    if BEAROFF_FIXED_EVAL_ENABLED && iter % BEAROFF_EVAL_INTERVAL == 0
        gate_updated_this_eval = false   # did a gate decision run before any throw?
        try
            bo_result = run_bearoff_eval!(race_network, iter)
            if bo_result !== nothing
                with_logger(TB_LOGGER) do
                    @info "06_eval_bearoff/fixed_value_mae" value=bo_result["value_mae"] log_step_increment=0
                    @info "06_eval_bearoff/fixed_value_bias" value=bo_result["value_bias"] log_step_increment=0
                    @info "06_eval_bearoff/fixed_value_corr" value=bo_result["value_corr"] log_step_increment=0
                    @info "06_eval_bearoff/fixed_policy_top1" value=100 * bo_result["policy_top1"] log_step_increment=0
                    @info "06_eval_bearoff/fixed_policy_top3" value=100 * bo_result["policy_top3"] log_step_increment=0
                    @info "06_eval_bearoff/fixed_policy_opt_mass" value=100 * bo_result["policy_opt_mass"] log_step_increment=0
                    @info "06_eval_bearoff/fixed_policy_expected_regret" value=bo_result["policy_expected_regret"] log_step_increment=0
                    @info "06_eval_bearoff/fixed_nn_top1" value=100 * bo_result["nn_top1"] log_step_increment=0
                    @info "06_eval_bearoff/fixed_nn_regret" value=bo_result["nn_regret"] log_step_increment=0
                    if haskey(bo_result, "mcts_top1")
                        @info "06_eval_bearoff/fixed_mcts_top1" value=100 * bo_result["mcts_top1"] log_step_increment=0
                        @info "06_eval_bearoff/fixed_mcts_regret" value=bo_result["mcts_regret"] log_step_increment=0
                        @info "06_eval_bearoff/fixed_mcts_regret_gt_001" value=100 * bo_result["mcts_regret_gt_001"] log_step_increment=0
                    end
                end
                @info "Fixed bearoff eval" iter=iter value_mae=round(bo_result["value_mae"], digits=4) value_corr=round(bo_result["value_corr"], digits=4) policy_top1=round(100 * bo_result["policy_top1"], digits=1) nn_top1=round(100 * bo_result["nn_top1"], digits=1) mcts_top1=round(100 * get(bo_result, "mcts_top1", NaN), digits=1)

                # ── Promotion gate decision (race model) ────────────────────
                # Gate on value MAE (lower is better). Updates persistent
                # GATE_STATE; the publication step below reads GATE_STATE.
                if GATE_ENABLED
                    prev_best = GATE_STATE[].best_metric
                    dec = gate_evaluate(GATE_STATE[], bo_result[GATE_METRIC_NAME];
                                        tol_frac=GATE_TOL_FRAC, tol_abs=GATE_TOL_ABS)
                    GATE_STATE[] = dec.state
                    gate_updated_this_eval = true   # eval produced a decision; catch must not override
                    with_logger(TB_LOGGER) do
                        @info "09_promotion/metric" value=dec.metric log_step_increment=0
                        @info "09_promotion/best_metric" value=(isfinite(dec.best_metric) ? dec.best_metric : dec.metric) log_step_increment=0
                        @info "09_promotion/threshold" value=(isfinite(dec.threshold) ? dec.threshold : dec.metric) log_step_increment=0
                        @info "09_promotion/blocked" value=dec.state.n_blocked log_step_increment=0
                        @info "09_promotion/eval_failures" value=dec.state.n_eval_failures log_step_increment=0
                    end
                    if dec.publish
                        @info "Promotion gate PASS — publication allowed" iter=iter metric=round(dec.metric, digits=5) best=round(dec.best_metric, digits=5) threshold=round(dec.threshold, digits=5) improved=dec.improved
                    else
                        @warn "Promotion gate BLOCK — regression detected, holding last-good weights" iter=iter metric=round(dec.metric, digits=5) best=round(dec.best_metric, digits=5) threshold=round(dec.threshold, digits=5) n_blocked=dec.state.n_blocked
                    end
                    # Save best-so-far checkpoint (for rollback) + sidecar on improvement
                    if dec.improved
                        FluxLib.save_weights(joinpath(CHECKPOINT_DIR, "race_best.data"),
                                             Flux.cpu(race_network))
                        save_gate_state(joinpath(CHECKPOINT_DIR, "gate_state.json"), GATE_STATE[];
                                        metric_name=GATE_METRIC_NAME, tol_frac=GATE_TOL_FRAC, tol_abs=GATE_TOL_ABS)
                        @info "Saved race_best.data (new best $(GATE_METRIC_NAME)=$(round(dec.best_metric, digits=5)), prev=$(isfinite(prev_best) ? round(prev_best, digits=5) : "none"))"
                    end
                end
            end
        catch e
            @warn "Fixed bearoff eval failed" exception=e
            # The eval SIGNAL itself failed before producing a gate decision.
            # Calibrated fail-closed (finding 2): if the gate has EVER been
            # calibrated (a finite best exists), BLOCK — a persistently broken
            # eval must not keep publishing untested weights. If never calibrated
            # (cold start), publish (fail-open) so a startup failure doesn't stall
            # the run before any baseline exists.
            if GATE_ENABLED && !gate_updated_this_eval
                dec = gate_on_eval_error(GATE_STATE[])
                GATE_STATE[] = dec.state
                with_logger(TB_LOGGER) do
                    @info "09_promotion/eval_failures" value=dec.state.n_eval_failures log_step_increment=0
                    @info "09_promotion/blocked" value=dec.state.n_blocked log_step_increment=0
                end
                if dec.publish
                    @warn "Promotion gate: eval signal FAILED, no baseline yet (cold start) — publishing (fail-open)" iter=iter n_eval_failures=dec.state.n_eval_failures
                else
                    @warn "Promotion gate: eval signal FAILED — holding last-good published weights (calibrated fail-closed)" iter=iter n_eval_failures=dec.state.n_eval_failures best=round(dec.best_metric, digits=5)
                end
            end
        end
    end

    # ── Weight publication (gated) ─────────────────────────────────────────
    # Publish = bump served weight version (clients pull + self-play with them).
    # Held back on a failed gate so a regressed model can't poison the buffer.
    # When the gate is disabled this is unconditional (original behavior).
    publish_this_iter = !GATE_ENABLED || GATE_STATE[].last_published
    if publish_this_iter
        update_weight_cache!(server_state, contact_network, race_network;
                             contact_width=CONTACT_WIDTH, contact_blocks=CONTACT_BLOCKS,
                             race_width=RACE_WIDTH, race_blocks=RACE_BLOCKS)
    end
    if GATE_ENABLED
        with_logger(TB_LOGGER) do
            @info "09_promotion/published" value=(publish_this_iter ? 1 : 0) log_step_increment=0
        end
    end

    # Save client stats
    save_client_stats(server_state, joinpath(DATA_DIR, "clients.json"))

    # Checkpoint
    if iter % ARGS["checkpoint_interval"] == 0
        buf_interval = ARGS["buffer_checkpoint_interval"]
        include_buffer = buf_interval > 0 && iter % buf_interval == 0
        bundle = save_training_checkpoint_bundle!(iter; include_buffer)
        _atomic_copy(joinpath(bundle, "contact_train.data"),
                     joinpath(CHECKPOINT_DIR, "contact_iter_$iter.data"))
        _atomic_copy(joinpath(bundle, "race_train.data"),
                     joinpath(CHECKPOINT_DIR, "race_iter_$iter.data"))
        # *_latest.data are the PUBLISHED weights — only overwrite when the gate
        # permits publication this iteration (else keep serving last-good).
        if publish_this_iter
            _atomic_copy(joinpath(bundle, "contact_train.data"),
                         joinpath(CHECKPOINT_DIR, "contact_latest.data"))
            _atomic_copy(joinpath(bundle, "race_train.data"),
                         joinpath(CHECKPOINT_DIR, "race_latest.data"))
        else
            @warn "Gate held publication — leaving race_latest.data / contact_latest.data at last-good weights" iter=iter
        end
        if GATE_ENABLED
            _atomic_copy(joinpath(bundle, "gate_state.json"),
                         joinpath(CHECKPOINT_DIR, "gate_state.json"))
        end
        @info "Saved transactional checkpoint" iter bundle include_buffer
    end

    # If the buffer cadence does not coincide with the model cadence, retain a
    # separately atomic buffer snapshot. Coincident snapshots live in the bundle.
    buf_interval = ARGS["buffer_checkpoint_interval"]
    if buf_interval > 0 && iter % buf_interval == 0 &&
       iter % ARGS["checkpoint_interval"] != 0
        buf_path = joinpath(DATA_DIR, "buffer", "buffer_iter_$iter.jls")
        @info "Saving buffer checkpoint..." iter=iter size=replay_buffer.size
        t0 = time()
        temporary = buf_path * ".tmp"
        save_buffer(replay_buffer, temporary)
        mv(temporary, buf_path; force=true)
        @info "Buffer checkpoint saved" path=buf_path elapsed=round(time()-t0, digits=1)
    end

    # Check if a previous eval job completed — finalize and log results BEFORE creating new job
    lock(EVAL_LOCK) do
        job = EVAL_JOB[]
        if job !== nothing && EvalManager.is_complete(job)
            finalize_eval_job!(job; source="training-loop")
        end
    end

    # Distributed eval: create a non-blocking eval job for clients to work on.
    # Never replace a running eval — let it finish so results are valid.
    if EVAL_ENABLED && iter % EVAL_INTERVAL == 0
        lock(EVAL_LOCK) do
            if EVAL_JOB[] !== nothing
                st = EvalManager.status(EVAL_JOB[])
                println("Eval iter $(EVAL_JOB[].iter) still running ($(st.completed)/$(st.total_chunks) chunks) — skipping eval at iter $iter")
            else
                wv = ARGS["training_mode"] == "race" ? server_state.race_version[] : server_state.contact_version[]
                n_pos = length(EVAL_POSITIONS)
                EVAL_JOB[] = EvalManager.create_eval_job(iter, n_pos, wv; chunk_size=EVAL_CHUNK_SIZE)
                # Pin weights in history for eval clients
                lock(server_state.weight_lock) do
                    if !haskey(server_state.weight_history, wv) &&
                       !isempty(server_state.contact_weight_bytes) && !isempty(server_state.race_weight_bytes)
                        server_state.weight_history[wv] = (copy(server_state.contact_weight_bytes),
                                                            copy(server_state.race_weight_bytes))
                    end
                end
                println("Eval job created for iter $iter: $(length(EVAL_JOB[].chunks)) chunks, $n_pos positions × 2 sides")
            end
        end
    end
end

server_state.shutdown_requested[] || wait_for_final_eval!()

server_state.accepting_samples[] = false
final_iteration = server_state.iteration[]
final_bundle = save_training_checkpoint_bundle!(final_iteration;
    include_buffer=true, reason=server_state.shutdown_requested[] ? "drain" : "complete")
@info "Final transactional checkpoint ready" iteration=final_iteration bundle=final_bundle

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
    println("\nDrain requested; waiting for a safe checkpoint boundary...")
    server_state.accepting_samples[] = false
    server_state.shutdown_requested[] = true
    wait(training_task)
finally
    close(http_server)
end
