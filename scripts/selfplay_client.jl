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
    end

    return ArgParse.parse_args(s)
end

const ARGS = parse_args()
const SERVER_URL = ARGS["server"]
const NUM_WORKERS = ARGS["num_workers"]
const GPU_WORKERS = ARGS["gpu_workers"]
const USE_GPU = GPU_WORKERS > 0

println("=" ^ 60)
println("AlphaZero Self-Play Client")
println("=" ^ 60)

# Load packages
using AlphaZero
using AlphaZero: GI, MCTS, Network, FluxLib, MctsParams, LearningParams
using AlphaZero: BatchedMCTS, Util
using AlphaZero: ConstSchedule
import Flux
import BackgammonNet
const CPU_INFERENCE_BACKEND = AlphaZero.BackgammonInference.resolve_cpu_backend(ARGS["inference_backend"])

println("Server: $SERVER_URL")
println("Workers: $NUM_WORKERS CPU" * (GPU_WORKERS > 0 ? " + $GPU_WORKERS GPU" : ""))
println("GPU: $(GPU_WORKERS > 0 ? "Metal ($GPU_WORKERS workers)" : "disabled")")
println("CPU inference: $(AlphaZero.BackgammonInference.cpu_backend_summary(CPU_INFERENCE_BACKEND))")
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
    player = BatchedMCTS.BatchedMctsPlayer(
        gspec, CPU_SINGLE_ORACLE, mcts_params;
        batch_size=INFERENCE_BATCH_SIZE, batch_oracle=CPU_BATCH_ORACLE,
        bearoff_evaluator=BEAROFF_EVALUATOR)

    all_samples = []
    while Threads.atomic_add!(games_claimed, 1) < total_games
        env = init_game(rng)

        trace_states = []
        trace_policies = []
        trace_actions = Vector{Int}[]
        trace_rewards = Float32[]
        trace_is_chance = Bool[]
        bearoff_truncated = false
        bearoff_bo = nothing
        bearoff_wp_at_trunc = true
        first_bearoff_bo = nothing
        decision_move_num = 0
        first_bearoff_wp = true

        while !GI.game_terminated(env)
            if GI.is_chance_node(env)
                bg = env.game
                if BearoffK6.is_bearoff_position(bg.p0, bg.p1)
                    if first_bearoff_bo === nothing
                        first_bearoff_bo = bearoff_table_equity(bg)
                        first_bearoff_wp = (bg.current_player == 0)
                    end
                    if BEAROFF_TRUNCATION
                        bearoff_truncated = true
                        bearoff_bo = first_bearoff_bo
                        bearoff_wp_at_trunc = first_bearoff_wp
                        break
                    end
                end
                outcomes = GI.chance_outcomes(env)
                idx = _sample_chance(rng, outcomes)
                GI.apply_chance!(env, outcomes[idx][1])
                continue
            end

            avail = GI.available_actions(env)
            if length(avail) == 1
                GI.play!(env, avail[1])
                continue
            end

            state = GI.current_state(env)
            push!(trace_states, state)
            actions, policy = BatchedMCTS.think(player, env)
            push!(trace_policies, Float32.(policy))
            push!(trace_actions, actions)
            push!(trace_is_chance, false)

            decision_move_num += 1
            τ = get_temperature(decision_move_num)
            if isone(τ)
                action = actions[sample_from_policy(policy, rng)]
            elseif iszero(τ)
                action = actions[argmax(policy)]
            else
                temp_policy = Util.apply_temperature(policy, τ)
                action = actions[sample_from_policy(temp_policy, rng)]
            end
            GI.play!(env, action)
            push!(trace_rewards, 0.0f0)
        end

        BatchedMCTS.reset_player!(player)
        if bearoff_truncated
            n_bearoff_truncated += 1
            final_reward = Float32(bearoff_wp_at_trunc ? bearoff_bo.value : -bearoff_bo.value)
            samples = convert_trace_to_samples(
                gspec, trace_states, trace_policies, trace_actions, trace_rewards, trace_is_chance,
                final_reward, nothing; rng=rng,
                bearoff_equity=bearoff_bo.equity, bearoff_wp=bearoff_wp_at_trunc)
        elseif first_bearoff_bo !== nothing
            final_reward = Float32(first_bearoff_wp ? first_bearoff_bo.value : -first_bearoff_bo.value)
            outcome = GI.game_outcome(env)
            samples = convert_trace_to_samples(
                gspec, trace_states, trace_policies, trace_actions, trace_rewards, trace_is_chance,
                final_reward, outcome; rng=rng,
                first_bearoff_equity=first_bearoff_bo.equity, first_bearoff_wp=first_bearoff_wp)
        else
            final_reward = Float32(GI.white_reward(env))
            outcome = GI.game_outcome(env)
            samples = convert_trace_to_samples(
                gspec, trace_states, trace_policies, trace_actions, trace_rewards, trace_is_chance,
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
    gpu_array_fn = x -> Metal.MtlArray(x)

    max_batch = INFERENCE_BATCH_SIZE + 1
    X_buf = zeros(Float32, _state_dim, max_batch)
    A_buf = zeros(Float32, NUM_ACTIONS, max_batch)
    _contact_idxs = Vector{Int}(undef, max_batch)
    _race_idxs = Vector{Int}(undef, max_batch)
    _results = Vector{Tuple{Vector{Float32}, Float32}}(undef, max_batch)

    n_oracle_calls = 0
    n_bearoff_truncated = 0

    function batch_oracle(states::Vector)
        n = length(states)
        n == 0 && return Tuple{Vector{Float32}, Float32}[]

        # Split into contact/race
        n_contact = 0
        n_race = 0
        for (i, s) in enumerate(states)
            is_contact = s isa BackgammonNet.BackgammonGame ? BackgammonNet.is_contact_position(s) : true
            if is_contact
                n_contact += 1
                _contact_idxs[n_contact] = i
            else
                n_race += 1
                _race_idxs[n_race] = i
            end
        end

        # Build feature matrices for each model
        if n_contact > 0
            X_c = zeros(Float32, _state_dim, n_contact)
            A_c = zeros(Float32, NUM_ACTIONS, n_contact)
            for j in 1:n_contact
                s = states[_contact_idxs[j]]
                v = GI.vectorize_state(gspec, s)
                X_c[:, j] .= vec(v)
                if !BackgammonNet.game_terminated(s)
                    for a in BackgammonNet.legal_actions(s)
                        1 <= a <= NUM_ACTIONS && (A_c[a, j] = 1f0)
                    end
                end
            end

            local Pr_c, V_c
            lock(GPU_LOCK) do
                X_g = gpu_array_fn(X_c)
                A_g = gpu_array_fn(A_c)
                result = Network.forward_normalized(contact_network_gpu, X_g, A_g)
                Metal.synchronize()
                Pr_c = Array(result[1])
                V_c = Array(result[2])
            end

            A_bool = A_c .> 0
            for j in 1:n_contact
                idx = _contact_idxs[j]
                _results[idx] = (Pr_c[@view(A_bool[:, j]), j], V_c[1, j])
            end
        end

        if n_race > 0
            X_r = zeros(Float32, _state_dim, n_race)
            A_r = zeros(Float32, NUM_ACTIONS, n_race)
            for j in 1:n_race
                s = states[_race_idxs[j]]
                v = GI.vectorize_state(gspec, s)
                X_r[:, j] .= vec(v)
                if !BackgammonNet.game_terminated(s)
                    for a in BackgammonNet.legal_actions(s)
                        1 <= a <= NUM_ACTIONS && (A_r[a, j] = 1f0)
                    end
                end
            end

            local Pr_r, V_r
            lock(GPU_LOCK) do
                X_g = gpu_array_fn(X_r)
                A_g = gpu_array_fn(A_r)
                result = Network.forward_normalized(race_network_gpu, X_g, A_g)
                Metal.synchronize()
                Pr_r = Array(result[1])
                V_r = Array(result[2])
            end

            A_bool = A_r .> 0
            for j in 1:n_race
                idx = _race_idxs[j]
                _results[idx] = (Pr_r[@view(A_bool[:, j]), j], V_r[1, j])
            end
        end

        n_oracle_calls += 1
        return @view(_results[1:n])
    end
    single_oracle(state) = batch_oracle([state])[1]

    mcts_params = MctsParams(
        num_iters_per_turn=MCTS_ITERS,
        cpuct=Float64(config["cpuct"]),
        temperature=ConstSchedule(1.0),
        dirichlet_noise_ϵ=Float64(config["dirichlet_epsilon"]),
        dirichlet_noise_α=Float64(config["dirichlet_alpha"]))
    player = BatchedMCTS.BatchedMctsPlayer(
        gspec, single_oracle, mcts_params;
        batch_size=INFERENCE_BATCH_SIZE, batch_oracle=batch_oracle,
        bearoff_evaluator=BEAROFF_EVALUATOR)

    all_samples = []
    while Threads.atomic_add!(games_claimed, 1) < total_games
        env = init_game(sub_rng)

        trace_states = []
        trace_policies = []
        trace_actions = Vector{Int}[]
        trace_rewards = Float32[]
        trace_is_chance = Bool[]
        bearoff_truncated = false
        bearoff_bo = nothing
        bearoff_wp_at_trunc = true
        first_bearoff_bo = nothing
        decision_move_num = 0
        first_bearoff_wp = true

        while !GI.game_terminated(env)
            if GI.is_chance_node(env)
                bg = env.game
                if BearoffK6.is_bearoff_position(bg.p0, bg.p1)
                    if first_bearoff_bo === nothing
                        first_bearoff_bo = bearoff_table_equity(bg)
                        first_bearoff_wp = (bg.current_player == 0)
                    end
                    if BEAROFF_TRUNCATION
                        bearoff_truncated = true
                        bearoff_bo = first_bearoff_bo
                        bearoff_wp_at_trunc = first_bearoff_wp
                        break
                    end
                end
                outcomes = GI.chance_outcomes(env)
                idx = _sample_chance(sub_rng, outcomes)
                GI.apply_chance!(env, outcomes[idx][1])
                continue
            end

            avail = GI.available_actions(env)
            if length(avail) == 1
                GI.play!(env, avail[1])
                continue
            end

            state = GI.current_state(env)
            push!(trace_states, state)
            actions, policy = BatchedMCTS.think(player, env)
            push!(trace_policies, Float32.(policy))
            push!(trace_actions, actions)
            push!(trace_is_chance, false)

            decision_move_num += 1
            τ = get_temperature(decision_move_num)
            if isone(τ)
                action = actions[sample_from_policy(policy, sub_rng)]
            elseif iszero(τ)
                action = actions[argmax(policy)]
            else
                temp_policy = Util.apply_temperature(policy, τ)
                action = actions[sample_from_policy(temp_policy, sub_rng)]
            end
            GI.play!(env, action)
            push!(trace_rewards, 0.0f0)
        end

        BatchedMCTS.reset_player!(player)
        if bearoff_truncated
            n_bearoff_truncated += 1
            final_reward = Float32(bearoff_wp_at_trunc ? bearoff_bo.value : -bearoff_bo.value)
            samples = convert_trace_to_samples(
                gspec, trace_states, trace_policies, trace_actions, trace_rewards, trace_is_chance,
                final_reward, nothing; rng=sub_rng,
                bearoff_equity=bearoff_bo.equity, bearoff_wp=bearoff_wp_at_trunc)
        elseif first_bearoff_bo !== nothing
            final_reward = Float32(first_bearoff_wp ? first_bearoff_bo.value : -first_bearoff_bo.value)
            outcome = GI.game_outcome(env)
            samples = convert_trace_to_samples(
                gspec, trace_states, trace_policies, trace_actions, trace_rewards, trace_is_chance,
                final_reward, outcome; rng=sub_rng,
                first_bearoff_equity=first_bearoff_bo.equity, first_bearoff_wp=first_bearoff_wp)
        else
            final_reward = Float32(GI.white_reward(env))
            outcome = GI.game_outcome(env)
            samples = convert_trace_to_samples(
                gspec, trace_states, trace_policies, trace_actions, trace_rewards, trace_is_chance,
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
    end
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

        # Reset batch
        batch_samples = []
        batch_games = 0
        t_batch_start = time()
        flush(stdout)
    end
end

main_loop()
