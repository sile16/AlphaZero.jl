#!/usr/bin/env julia
"""
Self-play client for distributed training.

Connects to the training server, downloads model weights,
runs MCTS self-play locally, and uploads samples.

Uses a shared CPU inference backend with platform-adaptive selection.
Self-play infrastructure extracted from train_distributed.jl.

Usage:
    julia --threads 30 --project scripts/selfplay_client.jl \\
        --server http://localhost:9090 \\
        --api-key alphazero-dev-key \\
        --num-workers 22

Note: Use SSH tunnel for remote servers:
    ssh -f -N -L 9090:localhost:9090 jarvis
"""

using ArgParse
using Dates
using Random
using Statistics

function parse_args()
    s = ArgParseSettings(
        description="Self-play client for distributed AlphaZero training",
        autofix_names=true
    )

    @add_arg_table! s begin
        "--server"
            help = "Training server URL (use localhost with SSH tunnel)"
            arg_type = String
            default = "http://localhost:9090"
        "--api-key"
            help = "API key for server authentication"
            arg_type = String
            default = "alphazero-dev-key"
        "--client-name"
            help = "Human-readable client name"
            arg_type = String
            default = ""
        "--num-workers"
            help = "Number of CPU self-play workers"
            arg_type = Int
            default = 22
        "--upload-interval"
            help = "Upload samples every N games"
            arg_type = Int
            default = 10
        "--seed"
            arg_type = Int
            default = 0
        "--gpu-workers"
            help = "Number of GPU workers (Metal, Mac only). Runs alongside CPU workers."
            arg_type = Int
            default = 0
        "--inference-backend"
            help = "CPU inference backend: auto, fast, or flux"
            arg_type = String
            default = "auto"
        "--eval-capable"
            help = "Enable eval mode (client does eval when server has eval jobs)"
            action = :store_true
        "--eval-mcts-iters"
            help = "MCTS iterations for eval games"
            arg_type = Int
            default = 600
        "--wildbg-lib"
            help = "Path to wildbg shared library (for eval)"
            arg_type = String
            default = ""
        "--eval-positions-file"
            help = "Path to fixed eval positions file (portable tuples)"
            arg_type = String
            default = "/homeshare/projects/AlphaZero.jl/eval_data/race_eval_2000.jls"
    end

    return ArgParse.parse_args(s)
end

const ARGS = parse_args()
const SERVER_URL = ARGS["server"]
const NUM_WORKERS = ARGS["num_workers"]
const GPU_WORKERS = ARGS["gpu_workers"]
const USE_GPU = GPU_WORKERS > 0
const EVAL_CAPABLE = ARGS["eval_capable"]
const EVAL_MCTS_ITERS = ARGS["eval_mcts_iters"]
const PAUSE_SELFPLAY = Threads.Atomic{Bool}(false)
const ACTIVE_SELFPLAY_GAMES = Threads.Atomic{Int}(0)

println("=" ^ 60)
println("AlphaZero Self-Play Client")
println("=" ^ 60)

# Load packages
using AlphaZero
using AlphaZero: GI, MCTS, Network, FluxLib, MctsParams, LearningParams
using AlphaZero: BatchedMCTS, Util
using AlphaZero: ConstSchedule
using AlphaZero: GameLoop
import Flux
import BackgammonNet
import JSON3
const CPU_INFERENCE_BACKEND = AlphaZero.BackgammonInference.resolve_cpu_backend(ARGS["inference_backend"])

println("Server: $SERVER_URL")
println("Workers: $NUM_WORKERS CPU" * (GPU_WORKERS > 0 ? " + $GPU_WORKERS GPU" : ""))
println("GPU: $(GPU_WORKERS > 0 ? "Metal ($GPU_WORKERS workers)" : "disabled")")
println("CPU inference: $(AlphaZero.BackgammonInference.cpu_backend_summary(CPU_INFERENCE_BACKEND))")
println("Eval capable: $EVAL_CAPABLE" * (EVAL_CAPABLE ? " ($(EVAL_MCTS_ITERS) MCTS iters)" : ""))
println("=" ^ 60)
flush(stdout)

# Include shared modules
include(joinpath(@__DIR__, "..", "src", "distributed", "buffer.jl"))
include(joinpath(@__DIR__, "..", "src", "distributed", "protocol.jl"))
include(joinpath(@__DIR__, "..", "src", "distributed", "client.jl"))

# Connect to server and get config
client_name = isempty(ARGS["client_name"]) ? "julia-$(gethostname())-$(getpid())" : ARGS["client_name"]
client = SelfPlayClient(SERVER_URL, ARGS["api_key"];
                        client_id=client_name, upload_threshold=ARGS["upload_interval"] * 200)

println("\nConnecting to server...")
reg = register!(client; name=client_name)
if !reg.success
    error("Failed to register with server")
end

# Use server-assigned seed (unique per client) or fall back to CLI arg
const MAIN_SEED = if reg.assigned_seed !== nothing
    println("Using server-assigned seed: $(reg.assigned_seed)")
    Int(reg.assigned_seed)
elseif ARGS["seed"] > 0
    println("Using CLI seed: $(ARGS["seed"])")
    ARGS["seed"]
else
    println("Using random seed")
    nothing
end
if MAIN_SEED !== nothing
    Random.seed!(MAIN_SEED)
end
println("Registered as: $client_name")

# Fetch config from server
config = fetch_config!(client)
println("\nServer config:")
for (k, v) in sort(collect(config), by=first)
    println("  $k: $v")
end
flush(stdout)

# Extract config
const GAME_NAME = config["game"]
const CONTACT_WIDTH = Int(config["contact_width"])
const CONTACT_BLOCKS = Int(config["contact_blocks"])
const RACE_WIDTH = Int(config["race_width"])
const RACE_BLOCKS = Int(config["race_blocks"])
const MCTS_ITERS = Int(config["mcts_iters"])
const INFERENCE_BATCH_SIZE = Int(config["inference_batch_size"])
const NUM_ACTIONS = Int(config["num_actions"])

# Temperature scheduling
const TEMP_MOVE_CUTOFF = Int(get(config, "temp_move_cutoff", 20))
const TEMP_FINAL = Float64(get(config, "temp_final", 0.1))
const TEMP_ITER_DECAY = Bool(get(config, "temp_iter_decay", false))
const TEMP_ITER_FINAL = Float64(get(config, "temp_iter_final", 0.3))
const TOTAL_ITERS = Int(get(config, "total_iterations", 200))

# Bear-off config (always enabled — mandatory for backgammon training)
const BEAROFF_HARD_TARGETS = Bool(get(config, "bearoff_hard_targets", false))
const BEAROFF_TRUNCATION = Bool(get(config, "bearoff_truncation", false))

# Game setup
if GAME_NAME == "backgammon-deterministic"
    ENV["BACKGAMMON_OBS_TYPE"] = "minimal_flat"
    include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
else
    error("Unknown game: $GAME_NAME")
end
const gspec = GameSpec()
const _state_dim = let env = GI.init(gspec); length(vec(GI.vectorize_state(gspec, GI.current_state(env)))); end
const ORACLE_CFG = AlphaZero.BackgammonInference.OracleConfig(
    _state_dim, NUM_ACTIONS, gspec;
    vectorize_state! = vectorize_state_into!,
    route_state = s -> (s isa BackgammonNet.BackgammonGame && !BackgammonNet.is_contact_position(s) ? 2 : 1))

# Create networks (CPU for self-play inference)
println("\nCreating networks...")
contact_network = FluxLib.FCResNetMultiHead(
    gspec, FluxLib.FCResNetMultiHeadHP(width=CONTACT_WIDTH, num_blocks=CONTACT_BLOCKS))
race_network = FluxLib.FCResNetMultiHead(
    gspec, FluxLib.FCResNetMultiHeadHP(width=RACE_WIDTH, num_blocks=RACE_BLOCKS))

println("Contact model: $(CONTACT_WIDTH)w×$(CONTACT_BLOCKS)b ($(sum(length(p) for p in Flux.params(contact_network))) params)")
println("Race model: $(RACE_WIDTH)w×$(RACE_BLOCKS)b ($(sum(length(p) for p in Flux.params(race_network))) params)")

# Download initial weights from server
println("\nDownloading initial weights from server...")
if !sync_weights!(client, contact_network, race_network)
    println("No weights available yet (new run). Starting with random weights.")
end
flush(stdout)

# Set up BLAS for single-threaded per-worker CPU inference (always needed)
import LinearAlgebra; LinearAlgebra.BLAS.set_num_threads(1)
println("CPU inference: BLAS threads=1, backend=$(AlphaZero.BackgammonInference.cpu_backend_summary(CPU_INFERENCE_BACKEND))")

# GPU setup (Metal.jl for Mac) — runs alongside CPU workers
if USE_GPU
    @eval using Metal
    println("Metal GPU: $(Metal.current_device())")

    # Move networks to GPU
    const contact_network_gpu = Flux.gpu(contact_network)
    const race_network_gpu = Flux.gpu(race_network)
    println("Networks moved to GPU ($GPU_WORKERS workers)")

    # Global GPU lock — Metal is NOT thread-safe
    const GPU_LOCK = ReentrantLock()
end

#####
##### Bear-off table (mandatory — loaded locally, too large to serve over HTTP)
#####

const BEAROFF_SRC_DIR = let
    local_path = joinpath(homedir(), "github", "BackgammonNet.jl", "src", "bearoff_k6.jl")
    pkg_path = joinpath(dirname(pathof(BackgammonNet)), "bearoff_k6.jl")
    if isfile(local_path)
        dirname(local_path)
    elseif isfile(pkg_path)
        dirname(pkg_path)
    else
        error("Cannot find bearoff_k6.jl. Expected at:\n  $local_path\n  $pkg_path")
    end
end

include(joinpath(BEAROFF_SRC_DIR, "bearoff_k6.jl"))
using .BearoffK6

const BEAROFF_TABLE = let
    table_dir = joinpath(BEAROFF_SRC_DIR, "..", "tools", "bearoff_twosided", "bearoff_k6_twosided")
    if !isdir(table_dir)
        error("Bear-off table not found at: $table_dir")
    end
    println("Loading k=6 bear-off table from $table_dir ...")
    t = BearoffTable(table_dir)
    println("  c14: $(t.c14_pairs) pairs ($(round(length(t.c14_data)/1e9, digits=1)) GB)")
    println("  c15: $(t.c15_pairs) pairs ($(round(length(t.c15_data)/1e9, digits=1)) GB)")
    flush(stdout)
    t
end

"""Look up exact bear-off equity from precomputed table (pre-dice, valid at chance nodes)."""
function bearoff_table_equity(game::BackgammonNet.BackgammonGame)
    r = BearoffK6.lookup(BEAROFF_TABLE, game)
    eq = Float32[r.p_win, r.p_gammon_win, r.p_bg_win, r.p_gammon_loss, r.p_bg_loss]
    value = BearoffK6.compute_equity(r)
    return (value=value, equity=eq)
end

"""
Compute exact post-dice bear-off value via move enumeration.

At a decision node (specific dice rolled), enumerate all legal moves, look up each
resulting position in the bear-off table, and return the best (max) value.
This gives the exact Q(board, dice) = max_move V(result(board, dice, move)),
where V is the opponent's pre-dice table value (negated for perspective flip).

Returns (value, equity) where value is white-relative scalar equity and
equity is the 5-element probability vector, or nothing if not a bear-off position.
"""
function bearoff_post_dice_equity(game::BackgammonNet.BackgammonGame, table)
    if !BearoffK6.is_bearoff_position(game.p0, game.p1)
        return nothing
    end
    if BackgammonNet.is_chance_node(game)
        # Pre-dice: table value is exact. lookup() returns mover-relative;
        # convert to white-relative for consistency with the function's contract.
        r = BearoffK6.lookup(table, game)
        mover_val = BearoffK6.compute_equity(r)
        mover_eq = Float32[r.p_win, r.p_gammon_win, r.p_bg_win, r.p_gammon_loss, r.p_bg_loss]
        if game.current_player == 0
            return (value=mover_val, equity=mover_eq)
        else
            return (value=-mover_val,
                    equity=Float32[1.0f0 - mover_eq[1], mover_eq[4], mover_eq[5], mover_eq[2], mover_eq[3]])
        end
    end

    # Decision node: enumerate all legal moves, find the best resulting position
    actions = BackgammonNet.legal_actions(game)
    if isempty(actions)
        return nothing
    end

    # Compute in mover's perspective (maximize), then convert to white-relative
    best_mover_value = -Inf32
    best_mover_equity = nothing  # 5-elem vector from mover's perspective
    bg_copy = BackgammonNet.clone(game)
    mover = game.current_player  # 0 (P0/white) or 1 (P1/black)

    for action in actions
        BackgammonNet.copy_state!(bg_copy, game)
        BackgammonNet.apply_action!(bg_copy, action)

        local mover_val::Float32
        local mover_eq::Vector{Float32}

        if bg_copy.terminated
            # Mover bore off all checkers — they win (simple win, no gammons at terminal bearoff)
            mover_val = 1.0f0
            mover_eq = Float32[1.0, 0.0, 0.0, 0.0, 0.0]
        elseif BearoffK6.is_bearoff_position(bg_copy.p0, bg_copy.p1)
            # After move, opponent's turn at a chance node.
            # lookup() returns from bg_copy.current_player's perspective (the opponent).
            r_opp = BearoffK6.lookup(table, bg_copy)
            # Negate for mover's perspective
            mover_val = -BearoffK6.compute_equity(r_opp)
            # Flip probabilities: mover's P(win) = opponent's P(loss), etc.
            mover_eq = Float32[1.0f0 - r_opp.p_win, r_opp.p_gammon_loss, r_opp.p_bg_loss,
                               r_opp.p_gammon_win, r_opp.p_bg_win]
        else
            continue
        end

        if mover_val > best_mover_value
            best_mover_value = mover_val
            best_mover_equity = mover_eq
        end
    end

    if best_mover_equity === nothing
        return nothing
    end

    # Convert mover-relative to white-relative
    if mover == 0
        # Mover is white — already white-relative
        return (value=best_mover_value, equity=best_mover_equity)
    else
        # Mover is black — flip to white perspective
        white_value = -best_mover_value
        white_eq = Float32[1.0f0 - best_mover_equity[1], best_mover_equity[4], best_mover_equity[5],
                           best_mover_equity[2], best_mover_equity[3]]
        return (value=white_value, equity=white_eq)
    end
end

"""
Create bear-off evaluator for MCTS.

At chance nodes (pre-dice): returns the pre-dice table value directly (exact).
At decision nodes (post-dice): enumerates all legal moves, looks up each resulting
position in the bear-off table, and returns the best value (exact post-dice Q-value).

Returns white-relative equity or nothing if not a bear-off position.
"""
function make_bearoff_evaluator(table)
    return function(game_env)
        bg = game_env.game
        if !BearoffK6.is_bearoff_position(bg.p0, bg.p1)
            return nothing
        end

        if BackgammonNet.is_chance_node(bg)
            # Pre-dice: table value is exact (mover-relative → white-relative)
            r = BearoffK6.lookup(table, bg)
            mover_equity = Float64(BearoffK6.compute_equity(r))
            return bg.current_player == 0 ? mover_equity : -mover_equity
        end

        # Decision node (post-dice): enumerate moves to get exact Q(board, dice)
        actions = BackgammonNet.legal_actions(bg)
        if isempty(actions)
            return nothing
        end

        # Allocate per-call (thread-safe; clone is cheap — just field copies + buffer alloc)
        bg_copy = BackgammonNet.clone(bg)

        best_value = -Inf
        for action in actions
            BackgammonNet.copy_state!(bg_copy, bg)
            BackgammonNet.apply_action!(bg_copy, action)

            if bg_copy.terminated
                # Terminal: current player won (bore off all checkers)
                move_val = 1.0
            elseif BearoffK6.is_bearoff_position(bg_copy.p0, bg_copy.p1)
                # Opponent's turn (chance node). Look up their pre-dice equity, negate.
                r_opp = BearoffK6.lookup(table, bg_copy)
                move_val = -Float64(BearoffK6.compute_equity(r_opp))
            else
                continue
            end

            if move_val > best_value
                best_value = move_val
            end
        end

        if best_value == -Inf
            return nothing
        end

        # Convert from current-player-relative to white-relative
        return bg.current_player == 0 ? best_value : -best_value
    end
end

const BEAROFF_EVALUATOR = make_bearoff_evaluator(BEAROFF_TABLE)

#####
##### Custom starting positions (loaded from NFS file if configured)
#####

using Serialization
using StaticArrays

const TRAINING_MODE = get(config, "training_mode", "dual")
const START_POSITIONS_FILE = get(config, "start_positions_file", "")

const START_POSITIONS = if !isempty(START_POSITIONS_FILE)
    if !isfile(START_POSITIONS_FILE)
        error("Start positions file not found: $START_POSITIONS_FILE")
    end
    tuples = Serialization.deserialize(START_POSITIONS_FILE)
    println("Loaded $(length(tuples)) starting positions from $START_POSITIONS_FILE")
    flush(stdout)
    tuples  # Vector{Tuple{UInt128, UInt128, Int8}}
else
    nothing
end

"""Initialize a game environment from configured starting positions or default opening."""
function init_game(rng::AbstractRNG)
    env = GI.init(gspec)
    if hasproperty(env, :rng)
        env.rng = rng
    end
    if START_POSITIONS !== nothing
        # Pick a random starting position, create a BackgammonGame at chance node (pre-dice)
        p0, p1, cp = START_POSITIONS[rand(rng, 1:length(START_POSITIONS))]
        game = BackgammonNet.BackgammonGame(
            p0, p1, SVector{2,Int8}(0, 0), Int8(0), cp, false, 0.0f0;
            obs_type=:minimal_flat)
        GI.set_state!(env, game)
        # Roll initial dice
        BackgammonNet.sample_chance!(env.game, rng)
    end
    return env
end

#####
##### CPU inference backend
#####

# Extract FastWeights only when the selected CPU backend uses them
const CONTACT_FAST_WEIGHTS = CPU_INFERENCE_BACKEND == :fast ? AlphaZero.FastInference.extract_fast_weights(contact_network) : nothing
const RACE_FAST_WEIGHTS = CPU_INFERENCE_BACKEND == :fast ? AlphaZero.FastInference.extract_fast_weights(race_network) : nothing
if CPU_INFERENCE_BACKEND == :fast
    println("Fast forward (contact): $(CONTACT_FAST_WEIGHTS.num_blocks) res blocks, $(CONTACT_FAST_WEIGHTS.num_policy_layers) policy layers")
    println("Fast forward (race): $(RACE_FAST_WEIGHTS.num_blocks) res blocks, $(RACE_FAST_WEIGHTS.num_policy_layers) policy layers")
end

const CPU_ORACLES = if CPU_INFERENCE_BACKEND == :fast
    AlphaZero.BackgammonInference.make_cpu_oracles(
        CPU_INFERENCE_BACKEND, contact_network, ORACLE_CFG;
        secondary_net=race_network, batch_size=INFERENCE_BATCH_SIZE,
        primary_fw=CONTACT_FAST_WEIGHTS, secondary_fw=RACE_FAST_WEIGHTS)
else
    AlphaZero.BackgammonInference.make_cpu_oracles(
        CPU_INFERENCE_BACKEND, contact_network, ORACLE_CFG;
        secondary_net=race_network, batch_size=INFERENCE_BATCH_SIZE)
end
const CPU_SINGLE_ORACLE = CPU_ORACLES[1]
const CPU_BATCH_ORACLE = CPU_ORACLES[2]
const GPU_ORACLES = if USE_GPU
    AlphaZero.BackgammonInference.make_gpu_server_oracles(
        contact_network_gpu, ORACLE_CFG;
        secondary_net_gpu=race_network_gpu,
        batch_size=INFERENCE_BATCH_SIZE,
        num_workers=GPU_WORKERS,
        gpu_array_fn=Metal.MtlArray,
        sync_fn=Metal.synchronize,
        gpu_lock=GPU_LOCK)
else
    nothing
end
const GPU_SINGLE_ORACLE = USE_GPU ? GPU_ORACLES[1] : nothing
const GPU_BATCH_ORACLE = USE_GPU ? GPU_ORACLES[2] : nothing
const GPU_ORACLE_SERVER = USE_GPU ? GPU_ORACLES[3] : nothing

function refresh_fast_weights!()
    if CPU_INFERENCE_BACKEND == :fast
        AlphaZero.FastInference.refresh_fast_weights!(CONTACT_FAST_WEIGHTS, contact_network)
        AlphaZero.FastInference.refresh_fast_weights!(RACE_FAST_WEIGHTS, race_network)
    end

    # Also update GPU networks if enabled
    if USE_GPU
        lock(GPU_LOCK) do
            Flux.loadmodel!(contact_network_gpu, Flux.gpu(contact_network))
            Flux.loadmodel!(race_network_gpu, Flux.gpu(race_network))
        end
    end
end

#####
##### Helper functions
#####

# Shared iteration counter (updated from server)
const CURRENT_ITERATION = Threads.Atomic{Int}(1)

function sample_from_policy(policy::AbstractVector{<:Real}, rng)
    r = rand(rng)
    cumsum = 0.0
    for i in 1:length(policy)
        cumsum += policy[i]
        if r <= cumsum
            return i
        end
    end
    return length(policy)
end

function get_temperature(move_num::Int)
    τ = if TEMP_MOVE_CUTOFF > 0 && move_num > TEMP_MOVE_CUTOFF
        TEMP_FINAL
    else
        1.0
    end
    if TEMP_ITER_DECAY
        iter = CURRENT_ITERATION[]
        progress = clamp((iter - 1) / max(TOTAL_ITERS - 1, 1), 0.0, 1.0)
        iter_τ = 1.0 + progress * (TEMP_ITER_FINAL - 1.0)
        τ *= iter_τ
    end
    return τ
end

function _sample_chance(rng, outcomes)
    r = rand(rng)
    acc = 0.0
    @inbounds for i in eachindex(outcomes)
        acc += outcomes[i][2]
        if r <= acc
            return i
        end
    end
    return length(outcomes)
end

function convert_trace_to_samples(gspec, states, policies, trace_actions, rewards, is_chance, final_reward, outcome; rng=nothing,
        bearoff_equity=nothing, bearoff_wp=nothing,
        first_bearoff_equity=nothing, first_bearoff_wp=nothing)
    n = length(states)
    samples = []
    num_actions = GI.num_actions(gspec)

    probs_white = if bearoff_equity !== nothing
        if bearoff_wp
            copy(bearoff_equity)
        else
            Float32[1.0f0 - bearoff_equity[1], bearoff_equity[4], bearoff_equity[5],
                    bearoff_equity[2], bearoff_equity[3]]
        end
    else
        nothing
    end

    first_bo_probs_white = if first_bearoff_equity !== nothing
        if first_bearoff_wp
            copy(first_bearoff_equity)
        else
            Float32[1.0f0 - first_bearoff_equity[1], first_bearoff_equity[4], first_bearoff_equity[5],
                    first_bearoff_equity[2], first_bearoff_equity[3]]
        end
    else
        nothing
    end

    for i in 1:n
        state = states[i]
        policy = policies[i]
        actions = trace_actions[i]
        is_ch = is_chance[i]
        wp = GI.white_playing(gspec, state)

        z = wp ? final_reward : -final_reward
        eq = zeros(Float32, 5)
        has_eq = false

        if bearoff_equity !== nothing
            has_eq = true
            if wp
                eq = copy(probs_white)
            else
                eq = Float32[1.0f0 - probs_white[1], probs_white[4], probs_white[5],
                             probs_white[2], probs_white[3]]
            end
        elseif !isnothing(outcome)
            has_eq = true
            won = outcome.white_won == wp
            if won
                eq[1] = 1.0f0
                eq[2] = outcome.is_gammon ? 1.0f0 : 0.0f0
                eq[3] = outcome.is_backgammon ? 1.0f0 : 0.0f0
            else
                eq[4] = outcome.is_gammon ? 1.0f0 : 0.0f0
                eq[5] = outcome.is_backgammon ? 1.0f0 : 0.0f0
            end
        end

        is_bearoff_pos = false
        if bearoff_equity === nothing && state isa BackgammonNet.BackgammonGame &&
                BearoffK6.is_bearoff_position(state.p0, state.p1)
            # Use post-dice move enumeration for exact Q(board, dice) values
            bo = bearoff_post_dice_equity(state, BEAROFF_TABLE)
            if bo !== nothing
                z = wp ? bo.value : -bo.value
                eq = copy(bo.equity)
                if !wp
                    eq = Float32[1.0f0 - bo.equity[1], bo.equity[4], bo.equity[5], bo.equity[2], bo.equity[3]]
                end
                has_eq = true
                is_bearoff_pos = true
            end
        end

        if !is_bearoff_pos && first_bo_probs_white !== nothing
            has_eq = true
            if wp
                eq = copy(first_bo_probs_white)
            else
                eq = Float32[1.0f0 - first_bo_probs_white[1], first_bo_probs_white[4], first_bo_probs_white[5],
                             first_bo_probs_white[2], first_bo_probs_white[3]]
            end
        end

        state_arr = GI.vectorize_state(gspec, state)
        state_vec = Vector{Float32}(vec(state_arr))

        full_policy = zeros(Float32, num_actions)
        if !is_ch && !isempty(policy) && !isempty(actions)
            for (j, a) in enumerate(actions)
                if j <= length(policy)
                    full_policy[a] = policy[j]
                end
            end
        end

        is_contact = if state isa BackgammonNet.BackgammonGame
            BackgammonNet.is_contact_position(state)
        else
            true
        end

        push!(samples, (
            state=state_vec,
            policy=full_policy,
            value=z,
            equity=eq,
            has_equity=has_eq,
            is_chance=is_ch,
            is_contact=is_contact,
            is_bearoff=is_bearoff_pos,
        ))
    end

    return samples
end

#####
##### Worker functions (self-contained, run on worker threads)
#####

"""
Extract arrays from a GameResult trace for convert_trace_to_samples().

Filters out single-action forced moves to match the original behavior where
only multi-action decision points were recorded in the trace.
"""
function _extract_trace_arrays(result::GameLoop.GameResult)
    trace_states = []
    trace_policies = Vector{Float32}[]
    trace_actions = Vector{Int}[]
    trace_rewards = Float32[]
    trace_is_chance = Bool[]

    for entry in result.trace
        # Skip single-action forced moves (original code didn't record them)
        if length(entry.legal_actions) <= 1
            continue
        end
        push!(trace_states, entry.state)
        push!(trace_policies, entry.policy)
        push!(trace_actions, entry.legal_actions)
        push!(trace_rewards, 0.0f0)
        push!(trace_is_chance, entry.is_chance)
    end

    return (states=trace_states, policies=trace_policies, actions=trace_actions,
            rewards=trace_rewards, is_chance=trace_is_chance)
end

"""Core game-playing loop with shared CPU inference backend."""
function _play_games_loop(vworker_id::Int, games_claimed::Threads.Atomic{Int}, total_games::Int,
                          rng::MersenneTwister)
    n_bearoff_truncated = 0

    mcts_params = MctsParams(
        num_iters_per_turn=MCTS_ITERS,
        cpuct=Float64(config["cpuct"]),
        temperature=ConstSchedule(1.0),
        dirichlet_noise_ϵ=Float64(config["dirichlet_epsilon"]),
        dirichlet_noise_α=Float64(config["dirichlet_alpha"]))

    az_agent = GameLoop.MctsAgent(
        CPU_SINGLE_ORACLE, CPU_BATCH_ORACLE,
        mcts_params, INFERENCE_BATCH_SIZE, gspec;
        bearoff_eval=BEAROFF_EVALUATOR)

    all_samples = []
    while Threads.atomic_add!(games_claimed, 1) < total_games
        env = init_game(rng)

        # Capture full bearoff equity (5-element vector) via closure,
        # since GameResult only stores the scalar value.
        first_bearoff_bo = Ref{Any}(nothing)
        first_bearoff_wp = Ref{Bool}(true)
        function bearoff_lookup_with_capture(game)
            if !BearoffK6.is_bearoff_position(game.p0, game.p1)
                return nothing
            end
            bo = bearoff_table_equity(game)
            if first_bearoff_bo[] === nothing
                first_bearoff_bo[] = bo
                first_bearoff_wp[] = (game.current_player == 0)
            end
            return bo
        end

        result = GameLoop.play_game(az_agent, az_agent, env;
            record_trace=true,
            bearoff_truncation=BEAROFF_TRUNCATION,
            bearoff_lookup=bearoff_lookup_with_capture,
            rng=rng,
            temperature_fn=get_temperature)

        # Extract trace arrays for convert_trace_to_samples
        tr = _extract_trace_arrays(result)

        if result.bearoff_truncated
            n_bearoff_truncated += 1
            bo = first_bearoff_bo[]
            wp = first_bearoff_wp[]
            final_reward = Float32(wp ? bo.value : -bo.value)
            samples = convert_trace_to_samples(
                gspec, tr.states, tr.policies, tr.actions, tr.rewards, tr.is_chance,
                final_reward, nothing; rng=rng,
                bearoff_equity=bo.equity, bearoff_wp=wp)
        elseif first_bearoff_bo[] !== nothing
            bo = first_bearoff_bo[]
            wp = first_bearoff_wp[]
            final_reward = Float32(wp ? bo.value : -bo.value)
            outcome = GI.game_outcome(env)
            samples = convert_trace_to_samples(
                gspec, tr.states, tr.policies, tr.actions, tr.rewards, tr.is_chance,
                final_reward, outcome; rng=rng,
                first_bearoff_equity=bo.equity, first_bearoff_wp=wp)
        else
            final_reward = Float32(result.reward)
            outcome = GI.game_outcome(env)
            samples = convert_trace_to_samples(
                gspec, tr.states, tr.policies, tr.actions, tr.rewards, tr.is_chance,
                final_reward, outcome; rng=rng)
        end
        append!(all_samples, samples)
    end

    return (samples=all_samples, n_bearoff_truncated=n_bearoff_truncated)
end

"""Play games on a worker thread with the shared CPU inference backend."""
function worker_play_games(worker_id::Int, games_claimed::Threads.Atomic{Int}, total_games::Int,
                           rng::MersenneTwister)
    sub_rng = MersenneTwister(rand(rng, UInt))

    result = _play_games_loop(worker_id, games_claimed, total_games, sub_rng)
    return result.samples
end

"""Play games on a worker thread with GPU inference (Metal, serialized via lock)."""
function worker_play_games_gpu(worker_id::Int, games_claimed::Threads.Atomic{Int}, total_games::Int,
                                rng::MersenneTwister)
    sub_rng = MersenneTwister(rand(rng, UInt))

    n_bearoff_truncated = 0

    mcts_params = MctsParams(
        num_iters_per_turn=MCTS_ITERS,
        cpuct=Float64(config["cpuct"]),
        temperature=ConstSchedule(1.0),
        dirichlet_noise_ϵ=Float64(config["dirichlet_epsilon"]),
        dirichlet_noise_α=Float64(config["dirichlet_alpha"]))

    az_agent = GameLoop.MctsAgent(
        GPU_SINGLE_ORACLE, GPU_BATCH_ORACLE,
        mcts_params, INFERENCE_BATCH_SIZE, gspec;
        bearoff_eval=BEAROFF_EVALUATOR)

    all_samples = []
    while Threads.atomic_add!(games_claimed, 1) < total_games
        env = init_game(sub_rng)

        # Capture full bearoff equity (5-element vector) via closure
        first_bearoff_bo = Ref{Any}(nothing)
        first_bearoff_wp = Ref{Bool}(true)
        function bearoff_lookup_with_capture(game)
            if !BearoffK6.is_bearoff_position(game.p0, game.p1)
                return nothing
            end
            bo = bearoff_table_equity(game)
            if first_bearoff_bo[] === nothing
                first_bearoff_bo[] = bo
                first_bearoff_wp[] = (game.current_player == 0)
            end
            return bo
        end

        result = GameLoop.play_game(az_agent, az_agent, env;
            record_trace=true,
            bearoff_truncation=BEAROFF_TRUNCATION,
            bearoff_lookup=bearoff_lookup_with_capture,
            rng=sub_rng,
            temperature_fn=get_temperature)

        # Extract trace arrays for convert_trace_to_samples
        tr = _extract_trace_arrays(result)

        if result.bearoff_truncated
            n_bearoff_truncated += 1
            bo = first_bearoff_bo[]
            wp = first_bearoff_wp[]
            final_reward = Float32(wp ? bo.value : -bo.value)
            samples = convert_trace_to_samples(
                gspec, tr.states, tr.policies, tr.actions, tr.rewards, tr.is_chance,
                final_reward, nothing; rng=sub_rng,
                bearoff_equity=bo.equity, bearoff_wp=wp)
        elseif first_bearoff_bo[] !== nothing
            bo = first_bearoff_bo[]
            wp = first_bearoff_wp[]
            final_reward = Float32(wp ? bo.value : -bo.value)
            outcome = GI.game_outcome(env)
            samples = convert_trace_to_samples(
                gspec, tr.states, tr.policies, tr.actions, tr.rewards, tr.is_chance,
                final_reward, outcome; rng=sub_rng,
                first_bearoff_equity=bo.equity, first_bearoff_wp=wp)
        else
            final_reward = Float32(result.reward)
            outcome = GI.game_outcome(env)
            samples = convert_trace_to_samples(
                gspec, tr.states, tr.policies, tr.actions, tr.rewards, tr.is_chance,
                final_reward, outcome; rng=sub_rng)
        end
        append!(all_samples, samples)
    end

    return all_samples
end

# GPU background workers — run continuously, push samples into shared channel
const GPU_SAMPLE_CHANNEL = Channel{Any}(1000)  # buffered channel for GPU-produced samples
const GPU_GAMES_COUNT = Threads.Atomic{Int}(0)

if USE_GPU
    for w in 1:GPU_WORKERS
        wid = NUM_WORKERS + w
        rng = MersenneTwister(isnothing(MAIN_SEED) ? rand(UInt) : MAIN_SEED + wid * 104729)
        Threads.@spawn begin
            # Infinite game loop — GPU workers never stop
            games_claimed_inf = Threads.Atomic{Int}(0)
            while true
                try
                    samples = worker_play_games_gpu(wid, games_claimed_inf, 1, rng)
                    for s in samples
                        put!(GPU_SAMPLE_CHANNEL, s)
                    end
                    Threads.atomic_add!(GPU_GAMES_COUNT, 1)
                    # Reset counter for next game
                    Threads.atomic_xchg!(games_claimed_inf, 0)
                catch e
                    println("GPU worker $wid error: $e")
                    sleep(1)
                end
            end
        end
    end
    println("Started $GPU_WORKERS GPU background workers")
end

"""Drain all available GPU samples from the channel (non-blocking)."""
function drain_gpu_samples!()
    samples = []
    while isready(GPU_SAMPLE_CHANNEL)
        push!(samples, take!(GPU_SAMPLE_CHANNEL))
    end
    return samples
end

"""Spawn CPU worker threads for self-play with work-stealing.
GPU workers run in background and contribute via channel."""
function parallel_self_play(num_games::Int)
    games_claimed = Threads.Atomic{Int}(0)

    tasks = Task[]
    for w in 1:NUM_WORKERS
        rng = MersenneTwister(isnothing(MAIN_SEED) ? rand(UInt) : MAIN_SEED + w * 104729)
        t = Threads.@spawn worker_play_games(w, games_claimed, num_games, rng)
        push!(tasks, t)
    end

    all_samples = reduce(vcat, [fetch(t) for t in tasks])

    # Also collect any GPU samples that have accumulated
    gpu_samples = drain_gpu_samples!()
    if !isempty(gpu_samples)
        append!(all_samples, gpu_samples)
    end

    return all_samples
end

#####
##### Main self-play loop
#####

#####
##### System stats collection
#####

"""Collect system-wide CPU and memory stats. Returns a Dict with metrics."""
function collect_system_stats(; games_per_sec::Float64=0.0, samples_per_sec::Float64=0.0)
    stats = Dict{String, Any}(
        "cpu_percent" => 0.0,
        "memory_used_gb" => 0.0,
        "memory_total_gb" => 0.0,
        "games_per_sec" => games_per_sec,
        "samples_per_sec" => samples_per_sec,
    )

    try
        if Sys.isapple()
            # macOS: sum per-process CPU% and divide by number of cores
            cpu_output = strip(read(`ps -A -o %cpu`, String))
            lines = split(cpu_output, '\n')
            total_cpu = 0.0
            for line in lines[2:end]  # skip header
                s = strip(line)
                isempty(s) && continue
                total_cpu += parse(Float64, s)
            end
            ncpu = parse(Int, strip(read(`sysctl -n hw.ncpu`, String)))
            stats["cpu_percent"] = round(total_cpu / ncpu, digits=1)

            # Memory: use vm_stat
            vm_output = read(`vm_stat`, String)
            page_size = 16384  # default on Apple Silicon
            m = match(r"page size of (\d+) bytes", vm_output)
            if m !== nothing
                page_size = parse(Int, m.captures[1])
            end
            # Parse pages
            free = 0; active = 0; inactive = 0; speculative = 0; wired = 0; compressed = 0
            for line in split(vm_output, '\n')
                m = match(r"Pages free:\s+(\d+)", line)
                m !== nothing && (free = parse(Int, m.captures[1]))
                m = match(r"Pages active:\s+(\d+)", line)
                m !== nothing && (active = parse(Int, m.captures[1]))
                m = match(r"Pages inactive:\s+(\d+)", line)
                m !== nothing && (inactive = parse(Int, m.captures[1]))
                m = match(r"Pages speculative:\s+(\d+)", line)
                m !== nothing && (speculative = parse(Int, m.captures[1]))
                m = match(r"Pages wired down:\s+(\d+)", line)
                m !== nothing && (wired = parse(Int, m.captures[1]))
                m = match(r"Pages occupied by compressor:\s+(\d+)", line)
                m !== nothing && (compressed = parse(Int, m.captures[1]))
            end
            total_pages = free + active + inactive + speculative + wired + compressed
            used_pages = active + wired + compressed
            stats["memory_total_gb"] = round(total_pages * page_size / 1e9, digits=2)
            stats["memory_used_gb"] = round(used_pages * page_size / 1e9, digits=2)
        elseif Sys.islinux()
            # Linux: /proc/stat for CPU
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

            # Memory: /proc/meminfo
            meminfo = read("/proc/meminfo", String)
            total_kb = 0; avail_kb = 0
            for line in split(meminfo, '\n')
                m = match(r"MemTotal:\s+(\d+)", line)
                m !== nothing && (total_kb = parse(Int, m.captures[1]))
                m = match(r"MemAvailable:\s+(\d+)", line)
                m !== nothing && (avail_kb = parse(Int, m.captures[1]))
            end
            stats["memory_total_gb"] = round(total_kb / 1e6, digits=2)
            stats["memory_used_gb"] = round((total_kb - avail_kb) / 1e6, digits=2)
        end
    catch e
        @debug "Failed to collect system stats" exception=e
    end

    return stats
end

"""Send client stats to server (fire-and-forget)."""
# send_client_stats disabled — HTTP.jl deadlocks with multiple threads doing HTTP.
# Stats are sent via upload batch metadata instead.
function send_client_stats(client::SelfPlayClient, stats::Dict)
    # no-op: avoid spawning HTTP threads
end

flush(stdout)
println("\n" * "=" ^ 60)
println("Starting self-play...")
println("=" ^ 60)
flush(stdout)

const UPLOAD_INTERVAL = ARGS["upload_interval"]

# Single background network thread handles uploads AND weight sync.
# HTTP.jl deadlocks with multiple concurrent spawned threads doing HTTP.
const UPLOAD_CHANNEL = Channel{Vector{UInt8}}(8)
Threads.@spawn begin
    while true
        # Block waiting for upload data
        bytes = take!(UPLOAD_CHANNEL)

        # Upload samples
        try
            headers = vcat(auth_headers(client),
                           ["Content-Type" => "application/msgpack"])
            t0 = time()
            resp = HTTP.post("$(client.server_url)/api/samples",
                             headers, bytes; status_exception=false,
                             connect_timeout=10, readtimeout=60)
            t_upload = time() - t0
            if resp.status == 200
                result = JSON.parse(String(resp.body))
                println("  Uploaded $(result["accepted"]) samples ($(round(length(bytes)/1024, digits=1)) KB, $(round(t_upload, digits=2))s), buffer=$(result["buffer_size"]))")
            else
                println("  Upload failed: $(resp.status)")
            end
        catch e
            println("  Upload error: $e")
        end

        # Weight sync — runs on same thread as upload, after each upload
        try
            version = check_weight_version(client)
            if version !== nothing
                CURRENT_ITERATION[] = get(version, "iteration", 0)
                needs_contact = version["contact_version"] > client.contact_version
                needs_race = version["race_version"] > client.race_version
                if needs_contact || needs_race
                    updated = sync_weights!(client, contact_network, race_network)
                    if updated
                        refresh_fast_weights!()
                        println("  Weights updated! contact=v$(client.contact_version), race=v$(client.race_version) (server iter=$(CURRENT_ITERATION[]))")
                    end
                end
            end
        catch e
            println("  Weight sync error: $e")
        end

        flush(stdout)
    end
end

# Shared sample channel: workers push completed game samples, main thread drains and uploads
const SAMPLE_CHANNEL = Channel{Vector{Any}}(NUM_WORKERS * 2)

"""Continuous worker: plays games forever, pushing samples into SAMPLE_CHANNEL."""
function continuous_worker(worker_id::Int, rng::MersenneTwister)
    println("  Worker $worker_id starting on thread $(Threads.threadid())")
    flush(stdout)
    sub_rng = MersenneTwister(rand(rng, UInt))

    # Play games forever — one at a time, push samples immediately
    games_played = 0
    while true
        # Check if self-play is paused (eval in progress)
        if PAUSE_SELFPLAY[]
            sleep(0.1)
            continue
        end
        Threads.atomic_add!(ACTIVE_SELFPLAY_GAMES, 1)
        try
            # Use _play_games_loop with total_games=1 via atomic counter
            games_claimed = Threads.Atomic{Int}(0)
            result = _play_games_loop(worker_id, games_claimed, 1, sub_rng)
            if !isempty(result.samples)
                put!(SAMPLE_CHANNEL, result.samples)
            end
            games_played += 1
            if games_played <= 3
                println("  Worker $worker_id: game $games_played done, $(length(result.samples)) samples")
                flush(stdout)
            end
        finally
            Threads.atomic_sub!(ACTIVE_SELFPLAY_GAMES, 1)
        end
    end
end

#####
##### Client-side eval mode
#####

# Eval weight state — separate from self-play weights
mutable struct EvalWeightState
    iter::Int
    contact_fast_weights::Any
    race_fast_weights::Any
end
const EVAL_WEIGHTS = EvalWeightState(0, nothing, nothing)

# Load eval positions if eval-capable
const EVAL_POSITIONS = if EVAL_CAPABLE
    pos_file = ARGS["eval_positions_file"]
    if isfile(pos_file)
        pos = Serialization.deserialize(pos_file)
        println("Eval: loaded $(length(pos)) positions from $pos_file")
        pos
    else
        println("WARNING: Eval positions file not found: $pos_file — eval disabled")
        nothing
    end
else
    nothing
end

# Find wildbg library for eval
function find_wildbg_lib_eval()
    lib = ARGS["wildbg_lib"]
    if !isempty(lib) && isfile(lib)
        return lib
    end
    candidates = [
        joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg_main.dylib"),
        joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg_main.so"),
        joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg.dylib"),
        joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg.so"),
    ]
    for c in candidates
        isfile(c) && return c
    end
    return ""
end

const WILDBG_LIB_EVAL = EVAL_CAPABLE ? find_wildbg_lib_eval() : ""
if EVAL_CAPABLE
    if isempty(WILDBG_LIB_EVAL)
        println("WARNING: wildbg library not found — eval disabled")
    else
        println("Eval: wildbg lib = $WILDBG_LIB_EVAL")
    end
end

# Eval agent struct (same pattern as training_server.jl / eval_race.jl)
struct EvalAlphaZeroAgent <: BackgammonNet.AbstractAgent
    single_oracle::Any
    batch_oracle::Any
    mcts_params::MctsParams
    batch_size::Int
    gspec_::Any
end

function BackgammonNet.agent_move(agent::EvalAlphaZeroAgent, g::BackgammonNet.BackgammonGame)
    env = GI.init(agent.gspec_)
    env.game = BackgammonNet.clone(g)
    player = BatchedMCTS.BatchedMctsPlayer(
        agent.gspec_, agent.single_oracle, agent.mcts_params;
        batch_size=agent.batch_size, batch_oracle=agent.batch_oracle)
    actions, policy = BatchedMCTS.think(player, env)
    BatchedMCTS.reset_player!(player)
    return actions[argmax(policy)]
end

struct PositionValueSample
    nn_val::Float64
    wb_val::Float64
end

"""Play a single eval game from a fixed position. Returns (reward, value_samples)."""
function eval_game_from_position(az_agent::EvalAlphaZeroAgent,
                                  wildbg_agent::BackgammonNet.BackendAgent,
                                  position_data::Tuple,
                                  value_batch_oracle;
                                  seed::Int=1, az_is_white::Bool=true)
    rng = MersenneTwister(seed)
    p0, p1, cp = position_data
    g = BackgammonNet.BackgammonGame(p0, p1, SVector{2,Int8}(0, 0), Int8(0), cp, false, 0.0f0;
                                      obs_type=:minimal_flat)
    value_samples = PositionValueSample[]

    while !BackgammonNet.game_terminated(g)
        if BackgammonNet.is_chance_node(g)
            BackgammonNet.sample_chance!(g, rng)
        else
            is_p0_turn = g.current_player == 0
            is_az_turn = is_p0_turn == az_is_white
            if is_az_turn
                nn_v = Float64(value_batch_oracle([g])[1][2])
                wb_v = Float64(BackgammonNet.evaluate(wildbg_agent.backend, g))
                push!(value_samples, PositionValueSample(nn_v, wb_v))
            end
            agent = is_az_turn ? az_agent : wildbg_agent
            action = BackgammonNet.agent_move(agent, g)
            BackgammonNet.apply_action!(g, action)
        end
    end

    white_reward = Float64(g.reward)
    az_reward = az_is_white ? white_reward : -white_reward
    return (reward=az_reward, value_samples=value_samples)
end

"""Check for eval work from server. If available, pause self-play and run eval."""
function check_and_do_eval!()
    EVAL_POSITIONS === nothing && return false
    isempty(WILDBG_LIB_EVAL) && return false

    # Check eval status
    local eval_status
    try
        headers = auth_headers(client)
        resp = HTTP.get("$(SERVER_URL)/api/eval/status", headers;
                        status_exception=false, connect_timeout=10, readtimeout=30)
        resp.status != 200 && (println("[EVAL] Status check returned $(resp.status)"); return false)
        eval_status = JSON3.read(String(resp.body), Dict{String,Any})
    catch e
        println("[EVAL] Status check failed: $e")
        return false
    end

    eval_iter = get(eval_status, "eval_iter", 0)
    eval_iter == 0 && return false
    available = get(eval_status, "available", 0)
    available == 0 && return false
    expected_weights_version = get(eval_status, "weights_version", 0)

    println("\n[EVAL] Server has eval work: iter=$eval_iter, available=$available chunks, weights_version=$expected_weights_version")
    flush(stdout)

    # Pause self-play first, then download weights (avoids contention with self-play inference)
    PAUSE_SELFPLAY[] = true
    println("[EVAL] Pausing self-play, waiting for active games to finish...")
    flush(stdout)
    t_wait_start = time()
    while ACTIVE_SELFPLAY_GAMES[] > 0
        sleep(0.1)
        if time() - t_wait_start > 60.0
            println("[EVAL] WARNING: Timed out waiting for active games ($(ACTIVE_SELFPLAY_GAMES[]) still running)")
            break
        end
    end
    println("[EVAL] Self-play paused ($(round(time() - t_wait_start, digits=1))s wait)")
    flush(stdout)

    # Create eval networks and download weights (separate from self-play weights)
    println("[EVAL] Setting up eval networks for iter $eval_iter...")
    eval_contact_net = FluxLib.FCResNetMultiHead(
        gspec, FluxLib.FCResNetMultiHeadHP(width=CONTACT_WIDTH, num_blocks=CONTACT_BLOCKS))
    eval_race_net = FluxLib.FCResNetMultiHead(
        gspec, FluxLib.FCResNetMultiHeadHP(width=RACE_WIDTH, num_blocks=RACE_BLOCKS))

    result_c = download_weights(client, :contact)
    result_r = download_weights(client, :race)
    if result_c !== nothing
        FluxLib.load_weights!(eval_contact_net, result_c[2])
    end
    if result_r !== nothing
        FluxLib.load_weights!(eval_race_net, result_r[2])
    end

    # Create FastWeights for eval if using fast backend
    eval_contact_fw = CPU_INFERENCE_BACKEND == :fast ?
        AlphaZero.FastInference.extract_fast_weights(eval_contact_net) : nothing
    eval_race_fw = CPU_INFERENCE_BACKEND == :fast ?
        AlphaZero.FastInference.extract_fast_weights(eval_race_net) : nothing
    EVAL_WEIGHTS.iter = eval_iter
    EVAL_WEIGHTS.contact_fast_weights = eval_contact_fw
    EVAL_WEIGHTS.race_fast_weights = eval_race_fw
    println("[EVAL] Eval weights ready for iter $eval_iter")

    # Set up eval oracles
    eval_oracle_cfg = AlphaZero.BackgammonInference.OracleConfig(
        _state_dim, NUM_ACTIONS, gspec;
        vectorize_state! = vectorize_state_into!,
        route_state = s -> (s isa BackgammonNet.BackgammonGame && !BackgammonNet.is_contact_position(s) ? 2 : 1))

    eval_oracles = if CPU_INFERENCE_BACKEND == :fast && eval_contact_fw !== nothing
        AlphaZero.BackgammonInference.make_cpu_oracles(
            CPU_INFERENCE_BACKEND, eval_contact_net, eval_oracle_cfg;
            secondary_net=eval_race_net, batch_size=INFERENCE_BATCH_SIZE,
            primary_fw=eval_contact_fw, secondary_fw=eval_race_fw)
    else
        AlphaZero.BackgammonInference.make_cpu_oracles(
            CPU_INFERENCE_BACKEND, eval_contact_net, eval_oracle_cfg;
            secondary_net=eval_race_net, batch_size=INFERENCE_BATCH_SIZE)
    end
    eval_single_oracle = eval_oracles[1]
    eval_batch_oracle = eval_oracles[2]

    # Also create a value-only oracle for value comparison (Flux-based, batch=1)
    value_oracles = AlphaZero.BackgammonInference.make_cpu_oracles(
        :flux, eval_contact_net, eval_oracle_cfg;
        secondary_net=eval_race_net, batch_size=1)
    value_batch_oracle = value_oracles[2]

    eval_mcts_params = MctsParams(
        num_iters_per_turn=EVAL_MCTS_ITERS,
        cpuct=1.5,
        temperature=ConstSchedule(0.0),
        dirichlet_noise_ϵ=0.0,
        dirichlet_noise_α=1.0)

    az_agent = EvalAlphaZeroAgent(eval_single_oracle, eval_batch_oracle,
                                   eval_mcts_params, INFERENCE_BATCH_SIZE, gspec)

    # Open wildbg backends (per-worker, not shared)
    lib_size = filesize(WILDBG_LIB_EVAL)
    nets_variant = lib_size > 10_000_000 ? :large : :small
    if nets_variant == :large
        BackgammonNet.wildbg_set_lib_path!(large=WILDBG_LIB_EVAL)
    else
        BackgammonNet.wildbg_set_lib_path!(small=WILDBG_LIB_EVAL)
    end
    println("[EVAL] wildbg: $nets_variant ($(round(lib_size/1e6, digits=1))MB)")

    wildbg_backends = [begin
        wb = BackgammonNet.WildbgBackend(nets=nets_variant)
        BackgammonNet.open!(wb)
        wb
    end for _ in 1:NUM_WORKERS]

    # Process eval chunks until none available
    chunks_done = 0
    t_eval_start = time()

    try
        while true
            # Checkout a chunk
            local chunk
            try
                headers = vcat(auth_headers(client), ["Content-Type" => "application/msgpack"])
                body = MsgPack.pack(Dict("client_name" => client_name))
                resp = HTTP.post("$(SERVER_URL)/api/eval/checkout", headers, body;
                                 status_exception=false, connect_timeout=10, readtimeout=30)
                resp.status != 200 && break
                # Parse JSON response
                chunk = JSON3.read(String(resp.body), Dict{String,Any})
            catch e
                println("[EVAL] Checkout error: $e")
                break
            end

            chunk_id = get(chunk, "chunk_id", 0)
            chunk_id == 0 && break

            # Validate weights version matches what we downloaded
            chunk_weights_version = get(chunk, "weights_version", 0)
            if expected_weights_version > 0 && chunk_weights_version > 0 && chunk_weights_version != expected_weights_version
                println("[EVAL] WARNING: weights version mismatch (expected $expected_weights_version, got $chunk_weights_version). Aborting eval.")
                break
            end

            pos_start = Int(chunk["position_range_start"])
            pos_end = Int(chunk["position_range_end"])
            az_is_white = Bool(chunk["az_is_white"])
            n_games = pos_end - pos_start + 1

            println("[EVAL] Chunk $chunk_id: positions $pos_start-$pos_end, az_is_white=$az_is_white ($n_games games)")
            flush(stdout)

            # Start heartbeat in background
            chunk_done = Ref(false)
            heartbeat_task = Threads.@spawn begin
                while !chunk_done[]
                    try
                        hb_headers = vcat(auth_headers(client), ["Content-Type" => "application/msgpack"])
                        hb_body = MsgPack.pack(Dict("chunk_id" => chunk_id, "client_name" => client_name))
                        resp = HTTP.post("$(SERVER_URL)/api/eval/heartbeat", hb_headers, hb_body;
                                         status_exception=false, connect_timeout=10, readtimeout=30)
                        if resp.status == 200
                            hb_result = JSON3.read(String(resp.body), Dict{String,Any})
                            if !get(hb_result, "lease_extended", false)
                                println("[EVAL] WARNING: Chunk $chunk_id lease lost")
                                break
                            end
                        end
                    catch; end
                    sleep(60)
                end
            end

            # Play games in parallel using work-stealing
            rewards = Vector{Float64}(undef, n_games)
            value_data = Vector{Vector{PositionValueSample}}(undef, n_games)
            claimed = Threads.Atomic{Int}(0)
            t_chunk_start = time()

            Threads.@threads for tid in 1:min(NUM_WORKERS, n_games)
                wb_agent = BackgammonNet.BackendAgent(wildbg_backends[tid])
                while true
                    job = Threads.atomic_add!(claimed, 1) + 1
                    job > n_games && break
                    pos_idx = pos_start + job - 1
                    if pos_idx > length(EVAL_POSITIONS)
                        rewards[job] = 0.0
                        value_data[job] = PositionValueSample[]
                        continue
                    end
                    result = eval_game_from_position(
                        az_agent, wb_agent, EVAL_POSITIONS[pos_idx], value_batch_oracle;
                        seed=chunk_id * 10000 + job, az_is_white=az_is_white)
                    rewards[job] = result.reward
                    value_data[job] = result.value_samples
                end
            end

            chunk_done[] = true
            t_chunk = time() - t_chunk_start

            # Extract value samples into separate arrays for server
            val_nn = Float64[]
            val_opp = Float64[]
            val_is_contact = Bool[]
            for vs in value_data
                for s in vs
                    push!(val_nn, s.nn_val)
                    push!(val_opp, s.wb_val)
                    push!(val_is_contact, false)  # Race-only model — no contact positions
                end
            end

            # Submit results with exponential backoff
            submitted = false
            backoff = 1.0
            for attempt in 1:5
                try
                    sub_headers = vcat(auth_headers(client), ["Content-Type" => "application/msgpack"])
                    sub_body = MsgPack.pack(Dict(
                        "chunk_id" => chunk_id,
                        "client_name" => client_name,
                        "rewards" => collect(rewards),
                        "value_nn" => val_nn,
                        "value_opp" => val_opp,
                        "value_is_contact" => val_is_contact))
                    resp = HTTP.post("$(SERVER_URL)/api/eval/submit", sub_headers, sub_body;
                                     status_exception=false, connect_timeout=10, readtimeout=60)
                    if resp.status == 200
                        sub_result = JSON3.read(String(resp.body), Dict{String,Any})
                        eval_complete = get(sub_result, "eval_complete", false)
                        chunks_done += 1
                        avg_reward = mean(rewards)
                        win_pct = 100 * count(r -> r > 0, rewards) / n_games
                        println("[EVAL] Chunk $chunk_id done: equity=$(round(avg_reward, digits=3)), win=$(round(win_pct, digits=1))%, $(round(t_chunk, digits=1))s" *
                                (eval_complete ? " [EVAL COMPLETE]" : ""))
                        flush(stdout)
                        submitted = true
                        if eval_complete
                            @goto eval_finished
                        end
                        break
                    else
                        println("[EVAL] Submit failed (HTTP $(resp.status)), retry $attempt...")
                    end
                catch e
                    println("[EVAL] Submit error: $e, retry $attempt...")
                end
                sleep(min(backoff, 30.0))
                backoff *= 2
            end
            if !submitted
                println("[EVAL] WARNING: Failed to submit chunk $chunk_id after 5 attempts")
            end
        end
    catch e
        println("[EVAL] Error during eval: $e")
        @show e
    end

    @label eval_finished
    t_eval = time() - t_eval_start
    println("[EVAL] Eval session complete: $chunks_done chunks in $(round(t_eval, digits=1))s")
    flush(stdout)

    # Close wildbg backends
    for wb in wildbg_backends
        try
            BackgammonNet.close(wb)
        catch; end
    end

    # Resume self-play and sync weights
    PAUSE_SELFPLAY[] = false
    println("[EVAL] Self-play resumed")
    flush(stdout)

    # Force weight sync after eval to ensure self-play uses latest training weights
    try
        updated = sync_weights!(client, contact_network, race_network)
        if updated
            refresh_fast_weights!()
            println("[EVAL] Weights re-synced after eval")
        end
    catch e
        println("[EVAL] Weight sync after eval failed: $e")
    end

    return true
end

function main_loop()
    # Start all workers as continuous background threads
    tasks = Task[]
    for w in 1:NUM_WORKERS
        rng = MersenneTwister(isnothing(MAIN_SEED) ? rand(UInt) : MAIN_SEED + w * 104729)
        t = Threads.@spawn continuous_worker(w, rng)
        push!(tasks, t)
    end
    println("Started $NUM_WORKERS continuous CPU workers")
    flush(stdout)

    # Wait briefly so workers can start and JIT-compile before we block on channel
    sleep(1)

    games_played = 0
    total_samples_collected = 0
    batch_num = 0
    batch_samples = []
    batch_games = 0
    t_batch_start = time()
    last_eval_check = time()

    while true
        # Drain completed games from workers (blocking wait for first game)
        game_samples = take!(SAMPLE_CHANNEL)
        append!(batch_samples, game_samples)
        batch_games += 1
        games_played += 1

        # Drain any other ready games
        while batch_games < UPLOAD_INTERVAL && isready(SAMPLE_CHANNEL)
            game_samples = take!(SAMPLE_CHANNEL)
            append!(batch_samples, game_samples)
            batch_games += 1
            games_played += 1
        end

        if batch_games < UPLOAD_INTERVAL
            continue
        end

        batch_num += 1
        n_samples = length(batch_samples)
        total_samples_collected += n_samples
        t_play = time() - t_batch_start
        gps = batch_games / t_play
        sps = n_samples / t_play
        println("Batch $batch_num: $batch_games games, $n_samples samples, $(round(gps, digits=1)) games/sec")

        # Queue upload + weight sync on background network thread (non-blocking)
        batch = samples_to_batch(batch_samples)
        bytes = pack_samples(batch)
        put!(UPLOAD_CHANNEL, bytes)

        # Periodic eval check (every 30 seconds)
        if EVAL_CAPABLE && time() - last_eval_check > 30.0
            last_eval_check = time()
            try
                println("[EVAL] Checking for eval work...")
                flush(stdout)
                check_and_do_eval!()
            catch e
                println("[EVAL] Check error: $e")
                for (exc, bt) in Base.catch_stack()
                    showerror(stdout, exc, bt)
                    println()
                end
            end
            last_eval_check = time()  # reset after eval completes (may take a while)
        end

        # Reset batch
        batch_samples = []
        batch_games = 0
        t_batch_start = time()
        flush(stdout)
    end
end

main_loop()
