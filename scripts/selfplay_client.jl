#!/usr/bin/env julia

# Auto-install dependencies on first run
import Pkg
let manifest = joinpath(dirname(@__DIR__), "Manifest.toml")
    if !isfile(manifest) || filesize(manifest) < 100
        println("First run — installing dependencies...")
        # BackgammonNet.jl is not in the public registry — dev-link from sibling dir or clone
        bgnet_dir = joinpath(dirname(@__DIR__), "..", "BackgammonNet.jl")
        if !isdir(bgnet_dir)
            println("Cloning BackgammonNet.jl...")
            run(`git clone https://github.com/sile16/BackgammonNet.jl.git $bgnet_dir`)
        end
        Pkg.develop(path=bgnet_dir)
        Pkg.resolve()
        Pkg.instantiate()
        println("Dependencies installed. Precompiling (this takes ~10 min on first run)...")
    end
end

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

const SelfPlayRNG = Random.Xoshiro

new_selfplay_rng(seed::Integer) = SelfPlayRNG(seed)
new_selfplay_rng() = SelfPlayRNG(rand(UInt))

function parse_args()
    s = ArgParseSettings(
        description="Self-play client for distributed AlphaZero training",
        autofix_names=true
    )

    @add_arg_table! s begin
        "--server"
            help = "Training server URL"
            arg_type = String
            default = "http://jarvis:9090"
        "--api-key"
            help = "API key for server authentication"
            arg_type = String
            default = "alphazero-dev-key"
        "--client-name"
            help = "Client name (default: hostname)"
            arg_type = String
            default = ""
        "--num-workers"
            help = "Number of CPU self-play workers (default: num threads)"
            arg_type = Int
            default = 0
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
            help = "DEPRECATED — now controlled by server config"
            arg_type = Int
            default = 0
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
const NUM_WORKERS = ARGS["num_workers"] > 0 ? ARGS["num_workers"] : Threads.nthreads()
const GPU_WORKERS = ARGS["gpu_workers"]
const USE_GPU = GPU_WORKERS > 0
const EVAL_CAPABLE = ARGS["eval_capable"]

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
# Fail-fast: the :flux CPU oracle reads the MUTABLE network object that the
# weight-sync thread mutates in place (FluxLib.load_weights!), a live data race /
# torn-read hazard under threading. The :fast path (FastWeights + atomic Ref-swap)
# is both correct and faster on every platform, so we refuse to run :flux here
# rather than let a subtle race corrupt self-play silently.
CPU_INFERENCE_BACKEND == :flux && error(
    "--inference-backend=flux is unsafe for the self-play client (mutable-weight " *
    "data race with the sync thread). Use :fast (default) or :auto.")

println("Server: $SERVER_URL")
println("Workers: $NUM_WORKERS CPU" * (GPU_WORKERS > 0 ? " + $GPU_WORKERS GPU" : ""))
println("GPU: $(GPU_WORKERS > 0 ? "Metal ($GPU_WORKERS workers)" : "disabled")")
println("CPU inference: $(AlphaZero.BackgammonInference.cpu_backend_summary(CPU_INFERENCE_BACKEND))")
println("Eval capable: $EVAL_CAPABLE")
println("=" ^ 60)
flush(stdout)

# Include shared modules
include(joinpath(@__DIR__, "..", "src", "distributed", "buffer.jl"))
include(joinpath(@__DIR__, "..", "src", "distributed", "protocol.jl"))
include(joinpath(@__DIR__, "..", "src", "distributed", "client.jl"))

# Connect to server and get config
client_name = isempty(ARGS["client_name"]) ? lowercase(gethostname()) : ARGS["client_name"]
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
const EVAL_MCTS_ITERS = Int(config["eval_mcts_iters"])
if EVAL_CAPABLE
    println("Eval MCTS iters: $EVAL_MCTS_ITERS")
end

# Temperature scheduling
const TEMP_MOVE_CUTOFF = Int(config["temp_move_cutoff"])
const TEMP_FINAL = Float64(config["temp_final"])
const TEMP_ITER_DECAY = Bool(config["temp_iter_decay"])
const TEMP_ITER_FINAL = Float64(config["temp_iter_final"])
const TOTAL_ITERS = Int(config["total_iterations"])
const MCTS_BUDGET_MODE = Symbol(config["mcts_budget_mode"])
const PROGRESSIVE_SIM_MIN = Int(config["progressive_sim_min"])
const PROGRESSIVE_SIM_MAX = Int(config["progressive_sim_max"])
const TURN_SIM_MIN = Int(config["turn_sim_min"])
const TURN_SIM_TARGET = Int(config["turn_sim_target"])
const RAMP_TURNS_INITIAL = Int(config["ramp_turns_initial"])
const RAMP_TURNS_FINAL = Int(config["ramp_turns_final"])
MCTS_BUDGET_MODE in (:constant, :progressive, :turn_progressive) ||
    error("Unsupported mcts_budget_mode: $MCTS_BUDGET_MODE")
if MCTS_BUDGET_MODE == :progressive
    PROGRESSIVE_SIM_MIN > 0 && PROGRESSIVE_SIM_MAX > 0 ||
        error("progressive mode requires positive progressive_sim_min and progressive_sim_max")
elseif MCTS_BUDGET_MODE == :turn_progressive
    TURN_SIM_MIN > 0 && TURN_SIM_TARGET > 0 && RAMP_TURNS_INITIAL > 0 && RAMP_TURNS_FINAL > 0 ||
        error("turn_progressive mode requires positive turn simulation/ramp settings")
end

# Bear-off config (always enabled — mandatory for backgammon training)
const BEAROFF_HARD_TARGETS = Bool(config["bearoff_hard_targets"])
const BEAROFF_TRUNCATION = Bool(config["bearoff_truncation"])

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
    local_path = joinpath(homedir(), "github", "BackgammonNet.jl", "src", "bearoff_k7.jl")
    pkg_path = joinpath(dirname(pathof(BackgammonNet)), "bearoff_k7.jl")
    if isfile(local_path)
        dirname(local_path)
    elseif isfile(pkg_path)
        dirname(pkg_path)
    else
        error("Cannot find bearoff_k7.jl. Expected at:\n  $local_path\n  $pkg_path")
    end
end

include(joinpath(BEAROFF_SRC_DIR, "bearoff_k7.jl"))
using .BearoffK7
include(joinpath(@__DIR__, "bearoff_eval_common.jl"))

# F1/F3: bearoff is a CLUSTER-WIDE decision made by the server (config["use_bearoff"],
# derived from its --no-bearoff). When ON, EVERY client must have the LOCAL k=7 table:
# a client without it would silently produce DIFFERENT training targets (game-outcome
# fallback) than table-equipped clients. We fail-fast rather than allow that
# divergence — there is NO remote-table fallback.
# No default — a server that doesn't send use_bearoff is a protocol mismatch and must
# fail immediately rather than silently assuming a bearoff mode.
const USE_BEAROFF = Bool(config["use_bearoff"])

const BEAROFF_TABLE = let
    if !USE_BEAROFF
        println("Bearoff DISABLED by server (use_bearoff=false) — no bearoff eval or exact targets.")
        flush(stdout)
        nothing
    else
        # Prefer local copy (mmap over NFS can cause bus errors on large files)
        candidates = [
            joinpath(BEAROFF_SRC_DIR, "..", "tools", "bearoff_twosided", "bearoff_k7_twosided"),
            joinpath(homedir(), "bearoff_k7_twosided"),
            "/homeshare/projects/AlphaZero.jl/eval_data/bearoff_k7_twosided",
        ]
        table_dir = nothing
        for dir in candidates
            if isdir(dir) && isfile(joinpath(dir, "bearoff_k7_c14.bin"))
                table_dir = dir
                break
            end
        end
        table_dir === nothing && error(
            "Bearoff is ENABLED cluster-wide (server use_bearoff=true) but NO local k=7 table " *
            "was found.\n  Searched: $(join(candidates, ", "))\n" *
            "  Fail-fast: a client without the table would produce different training targets than " *
            "table-equipped clients.\n  Fix: copy bearoff_k7_twosided/ to ~/bearoff_k7_twosided/, " *
            "or run the server with --no-bearoff to disable bearoff for the whole cluster.")
        println("Loading k=7 bear-off table from $table_dir ...")
        t = BearoffTable(table_dir)
        println("  c14: $(round(length(t.c14_data)/1e9, digits=1)) GB")
        println("  c15: $(round(length(t.c15_data)/1e9, digits=1)) GB")
        flush(stdout)
        t
    end
end

# Fail-fast: bearoff TRUNCATION and hard targets read the LOCAL table's exact value
# (first_bearoff_bo.value). A remote-only config leaves BEAROFF_TABLE=nothing, which
# would make truncation dereference `nothing.value` and crash the worker. We never
# silently run truncation without exact local values — require the table or turn
# truncation off.
if BEAROFF_TRUNCATION && BEAROFF_TABLE === nothing
    error("Bearoff truncation is enabled but no LOCAL bearoff table is loaded " *
          "(BEAROFF_TABLE is nothing — remote bearoff cannot supply truncation/hard-target " *
          "values). Install a local k=7 table or disable bearoff truncation.")
end

"""
Look up exact bear-off equity from the precomputed table.

The k=7 bearoff table stores joint non-cumulative values directly:
- `pW`  = P(win)
- `pWG` = P(win AND gammon) — joint, not conditional
- `pLG` = P(lose AND gammon) — joint, not conditional
- bg = 0 in bearoff (not stored)

Maps to the NN's 5-head joint convention:
- `[P(win), P(win∧gammon+), P(win∧bg), P(lose∧gammon+), P(lose∧bg)]`
"""
function bearoff_table_equity(game::BackgammonNet.BackgammonGame)
    BEAROFF_TABLE === nothing && return nothing
    r = BearoffK7.lookup(BEAROFF_TABLE, game)
    eq = Float32[r.pW, r.pWG, 0.0f0, r.pLG, 0.0f0]
    value = BearoffK7.compute_equity(r)
    return (value=value, equity=eq)
end

"""
Compute exact post-dice bear-off value via move enumeration (k=7).

At a decision node (specific dice rolled), enumerate all legal moves, look up each
resulting position in the bear-off table, and return the best (max) value.
This gives the exact Q(board, dice) = max_move V(result(board, dice, move)),
where V is the opponent's pre-dice table value (negated for perspective flip).

Returns `(value, equity)` where `value` is white-relative scalar equity and
`equity` is the 5-element joint cumulative vector, or `nothing` if not a
bear-off position.
"""
function bearoff_post_dice_equity(game::BackgammonNet.BackgammonGame, table)
    if table === nothing
        return nothing
    end
    if !BearoffK7.is_bearoff_position(game.p0, game.p1)
        return nothing
    end
    if BackgammonNet.is_chance_node(game)
        # Pre-dice: table value is exact. lookup() returns mover-relative;
        # convert to white-relative for consistency with the function's contract.
        r = BearoffK7.lookup(table, game)
        mover_val = BearoffK7.compute_equity(r)
        mover_eq = Float32[r.pW, r.pWG, 0.0f0, r.pLG, 0.0f0]
        if game.current_player == 0
            return (value=mover_val, equity=mover_eq)
        else
            return (value=-mover_val,
                    equity=AlphaZero.flip_equity_perspective(mover_eq))
        end
    end

    # Decision node: enumerate all legal moves, find the best resulting position
    actions = BackgammonNet.legal_actions(game)
    if isempty(actions)
        return nothing
    end

    # Compute in mover's perspective (maximize), then convert to white-relative.
    # bearoff_turn_value_equity handles terminal (reward carries gammon multiplier),
    # completed-turn chance nodes (table lookup), AND doubles mid-turn states
    # (recursion) — see scripts/bearoff_eval_common.jl for the doubles pitfall.
    best_mover_value = -Inf
    best_mover_equity = nothing  # 5-elem joint vector from mover's perspective
    bg_copy = BackgammonNet.clone(game)
    mover = game.current_player  # 0 (P0/white) or 1 (P1/black)

    for action in actions
        BackgammonNet.copy_state!(bg_copy, game)
        BackgammonNet.apply_action!(bg_copy, action)
        mover_val, mover_eq = bearoff_turn_value_equity(table, bg_copy, mover)
        if mover_val > best_mover_value
            best_mover_value = mover_val
            best_mover_equity = mover_eq
        end
    end

    if best_mover_equity === nothing
        return nothing
    end
    best_mover_value = Float32(best_mover_value)

    # Convert mover-relative to white-relative
    if mover == 0
        # Mover is white — already white-relative
        return (value=best_mover_value, equity=best_mover_equity)
    else
        # Mover is black — flip to white perspective
        white_value = -best_mover_value
        white_eq = AlphaZero.flip_equity_perspective(best_mover_equity)
        return (value=white_value, equity=white_eq)
    end
end

"""
Create bear-off evaluator for MCTS (k=7).

At chance nodes (pre-dice): returns the pre-dice table value directly (exact).
At decision nodes (post-dice): enumerates all legal moves, looks up each resulting
position in the bear-off table, and returns the best value (exact post-dice Q-value).

Returns white-relative equity NORMALIZED to [-1,1] (points/3) so tree values are
on the same scale as NN value output (equity/3), or nothing if not a bear-off position.
"""
function make_bearoff_evaluator(table)
    # Single normalization point: raw bear-off points [-reward_scale, reward_scale]
    # → MCTS/NN value scale [-1,1]. Tied to GI.reward_scale so it can't drift.
    rs = Float64(GI.reward_scale(gspec))
    return function(game_env)
        # No local table → skip bearoff eval entirely (NN handles it).
        # Remote HTTP is far too slow for per-MCTS-iteration lookups.
        table === nothing && return nothing

        bg = game_env.game
        if !BearoffK7.is_bearoff_position(bg.p0, bg.p1)
            return nothing
        end

        if BackgammonNet.is_chance_node(bg)
            # Pre-dice: table value is exact (mover-relative → white-relative),
            # normalized to the NN value scale [-1,1] via reward_scale.
            r = BearoffK7.lookup(table, bg)
            mover_equity = bearoff_value_to_nn_scale(BearoffK7.compute_equity(r), rs)
            return bg.current_player == 0 ? mover_equity : -mover_equity
        end

        # Decision node (post-dice): enumerate moves to get exact Q(board, dice).
        # bearoff_best_move_value handles terminal rewards (gammon multiplier) and
        # doubles mid-turn recursion — see scripts/bearoff_eval_common.jl.
        actions = BackgammonNet.legal_actions(bg)
        if isempty(actions)
            return nothing
        end

        best_value = bearoff_best_move_value(table, bg)
        if best_value == -Inf
            return nothing
        end

        # Convert from current-player-relative to white-relative,
        # normalized to the NN value scale [-1,1] via reward_scale.
        best_value = bearoff_value_to_nn_scale(best_value, rs)
        return bg.current_player == 0 ? best_value : -best_value
    end
end

const BEAROFF_EVALUATOR = make_bearoff_evaluator(BEAROFF_TABLE)

#####
##### Custom starting positions (loaded from NFS file if configured)
#####

using Serialization
using StaticArrays

const TRAINING_MODE = config["training_mode"]
const START_POSITIONS_FILE = config["start_positions_file"]

"""Find a data file locally or download from server. Returns local path."""
function find_or_download(filename::String; required::Bool=true)
    isempty(filename) && (required ? error("Data file not configured") : return "")
    name = basename(filename)
    # Search local paths
    candidates = [
        filename,  # full path (if provided)
        joinpath(dirname(@__DIR__), "eval_data", name),
        joinpath(homedir(), "eval_data", name),
    ]
    for p in candidates
        isfile(p) && return p
    end
    # Download from server
    local_path = joinpath(dirname(@__DIR__), "eval_data", name)
    mkpath(dirname(local_path))
    println("Downloading $name from server...")
    try
        resp = HTTP.get("$(SERVER_URL)/api/file/$name";
            headers=auth_headers(client), status_exception=false,
            connect_timeout=10, readtimeout=120)
        if resp.status == 200
            open(local_path, "w") do f; write(f, resp.body); end
            println("  Downloaded $(round(length(resp.body)/1e6, digits=1)) MB → $local_path")
            return local_path
        end
    catch e
        required || return ""
        error("Data file $name not available locally or from server: $e")
    end
    required ? error("Data file $name not found") : ""
end

const START_POSITIONS = let
    path = find_or_download(START_POSITIONS_FILE; required=true)
    tuples = Serialization.deserialize(path)
    println("Loaded $(length(tuples)) starting positions from $(basename(path))")
    flush(stdout)
    tuples
end

"""Initialize a game environment from configured starting positions or default opening."""
function init_game(rng::AbstractRNG)
    env = GI.init(gspec)
    if hasproperty(env, :rng) && rng isa typeof(env.rng)
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

# Extract FastWeights wrapped in Ref for lock-free atomic swap on weight updates.
# Oracle closures capture the Ref and dereference each call, so workers always
# see the latest weights after a swap with zero synchronization overhead.
const CONTACT_FAST_WEIGHTS = CPU_INFERENCE_BACKEND == :fast ? Ref(AlphaZero.FastInference.extract_fast_weights(contact_network)) : nothing
const RACE_FAST_WEIGHTS = CPU_INFERENCE_BACKEND == :fast ? Ref(AlphaZero.FastInference.extract_fast_weights(race_network)) : nothing
if CPU_INFERENCE_BACKEND == :fast
    println("Fast forward (contact): $(CONTACT_FAST_WEIGHTS[].num_blocks) res blocks, $(CONTACT_FAST_WEIGHTS[].num_policy_layers) policy layers")
    println("Fast forward (race): $(RACE_FAST_WEIGHTS[].num_blocks) res blocks, $(RACE_FAST_WEIGHTS[].num_policy_layers) policy layers")
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
        # Atomic swap: create new FastWeights and swap the Ref contents.
        # Workers hold the old FastWeights for their current batch and pick up
        # the new one on their next oracle call. No locks, no data races.
        CONTACT_FAST_WEIGHTS[] = AlphaZero.FastInference.extract_fast_weights(contact_network)
        RACE_FAST_WEIGHTS[] = AlphaZero.FastInference.extract_fast_weights(race_network)
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

const SELFPLAY_SIM_BUDGET_FN = if MCTS_BUDGET_MODE == :constant
    nothing
elseif MCTS_BUDGET_MODE == :progressive
    let params = AlphaZero.ProgressiveSimParams(
            sim_min=PROGRESSIVE_SIM_MIN,
            sim_max=PROGRESSIVE_SIM_MAX)
        _turn -> AlphaZero.compute_sim_budget(params, CURRENT_ITERATION[], TOTAL_ITERS)
    end
else
    let params = AlphaZero.TurnProgressiveSimParams(
            turn_sim_min=TURN_SIM_MIN,
            turn_sim_target=TURN_SIM_TARGET,
            ramp_turns_initial=RAMP_TURNS_INITIAL,
            ramp_turns_final=RAMP_TURNS_FINAL)
        turn -> AlphaZero.compute_turn_sim_budget(params, turn, CURRENT_ITERATION[], TOTAL_ITERS)
    end
end

println("MCTS budget mode: $MCTS_BUDGET_MODE" *
        (MCTS_BUDGET_MODE == :constant ? " ($MCTS_ITERS sims)" :
         MCTS_BUDGET_MODE == :progressive ? " ($PROGRESSIVE_SIM_MIN->$PROGRESSIVE_SIM_MAX sims over iterations)" :
         " ($TURN_SIM_MIN->$TURN_SIM_TARGET sims over turns/iterations)"))

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

"""
Convert a recorded rollout into per-position training samples.

This is the main junction where self-play rewards, exact bear-off overrides, and
multi-head equity targets are aligned.

Per sample:
- `value` is always stored from the side-to-move perspective at that sample
- `equity` uses joint cumulative 5-head convention:
  `[P(win), P(win∧gammon+), P(win∧bg), P(lose∧gammon+), P(lose∧bg)]`

Target precedence is intentionally:
1. exact bear-off truncation target captured earlier in the game, if present
2. exact post-dice bear-off value for the current state, if available
3. final game outcome as a hard 0/1 target
"""
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
            AlphaZero.flip_equity_perspective(bearoff_equity)
        end
    else
        nothing
    end

    first_bo_probs_white = if first_bearoff_equity !== nothing
        if first_bearoff_wp
            copy(first_bearoff_equity)
        else
            AlphaZero.flip_equity_perspective(first_bearoff_equity)
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

        # `final_reward` is white-relative at the game level. Convert it to the
        # side-to-move view for this sample so training targets match the policy
        # and MCTS convention used elsewhere in the stack.
        z = wp ? final_reward : -final_reward
        eq = zeros(Float32, 5)
        has_eq = false

        if bearoff_equity !== nothing
            # Truncated rollouts reuse the exact first bear-off target for every
            # earlier position in the prefix. This keeps the scalar bootstrap
            # value and five-head equity target synchronized.
            has_eq = true
            if wp
                eq = copy(probs_white)
            else
                eq = AlphaZero.flip_equity_perspective(probs_white)
            end
        elseif !isnothing(outcome)
            # Hard terminal supervision. Zeros on the non-applicable side are
            # valid joint probabilities (e.g., P(win∧gammon)=0 when you lost).
            has_eq = true
            eq = AlphaZero.equity_vector_from_outcome(outcome, wp)
        end

        is_bearoff_pos = false
        if bearoff_equity === nothing && state isa BackgammonNet.BackgammonGame &&
                BearoffK7.is_bearoff_position(state.p0, state.p1)
            # Use post-dice move enumeration for exact Q(board, dice) values
            # and the matching five-head joint target for this specific
            # state. Override both together so `value` and `equity` stay aligned.
            bo = bearoff_post_dice_equity(state, BEAROFF_TABLE)
            if bo !== nothing
                z = wp ? bo.value : -bo.value
                eq = copy(bo.equity)
                if !wp
                    eq = AlphaZero.flip_equity_perspective(bo.equity)
                end
                has_eq = true
                is_bearoff_pos = true
            end
        end

        if !is_bearoff_pos && first_bo_probs_white !== nothing
            # Earlier prefix states in a truncated rollout inherit the same exact
            # first-bearoff target. This is a TD-style bootstrap target, not a
            # terminal outcome label.
            has_eq = true
            if wp
                eq = copy(first_bo_probs_white)
            else
                eq = AlphaZero.flip_equity_perspective(first_bo_probs_white)
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

Sample an action index from a probability distribution."""
function _sample_from_policy(policy::AbstractVector{<:Real}, rng)
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

"""Extract trace arrays from a GameResult.
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

"""Core game-playing loop — direct BatchedMCTS calls for minimal allocation.

Bypasses GameLoop.play_game() to avoid per-move TraceEntry allocations and
contact detection overhead that cause GC pressure under 32+ threads.
This matches the v6 inline game loop that achieved 200-400 games/min."""
function _play_games_loop(vworker_id::Int, games_claimed::Threads.Atomic{Int}, total_games::Int,
                          rng::AbstractRNG;
                          player=nothing)
    n_bearoff_truncated = 0

    if player === nothing
        mcts_params = MctsParams(
            num_iters_per_turn=MCTS_ITERS,
            cpuct=Float64(config["cpuct"]),
            temperature=ConstSchedule(1.0),
            dirichlet_noise_ϵ=Float64(config["dirichlet_epsilon"]),
            dirichlet_noise_α=Float64(config["dirichlet_alpha"]))
        az_agent = GameLoop.MctsAgent(
            CPU_SINGLE_ORACLE, CPU_BATCH_ORACLE,
            mcts_params, INFERENCE_BATCH_SIZE, gspec;
            bearoff_eval=BEAROFF_EVALUATOR,
            batch_oracle_with_actions=CPU_BATCH_ORACLE,
            sim_budget_fn=SELFPLAY_SIM_BUDGET_FN)
        player = GameLoop.create_player(az_agent; rng=rng)
    end

    all_samples = []
    while Threads.atomic_add!(games_claimed, 1) < total_games
        env = init_game(rng)

        trace_states = []
        trace_policies = Vector{Float32}[]
        trace_actions = Vector{Int}[]
        trace_rewards = Float32[]
        trace_is_chance = Bool[]
        bearoff_truncated = false
        first_bearoff_bo = nothing
        first_bearoff_wp = true
        decision_move_num = 0

        while !GI.game_terminated(env)
            if GI.is_chance_node(env)
                bg = env.game
                if BearoffK7.is_bearoff_position(bg.p0, bg.p1)
                    if first_bearoff_bo === nothing
                        first_bearoff_bo = bearoff_table_equity(bg)
                        first_bearoff_wp = (bg.current_player == 0)
                    end
                    if BEAROFF_TRUNCATION
                        bearoff_truncated = true
                        break
                    end
                end
                outcomes = GI.chance_outcomes(env)
                r = rand(rng)
                acc = 0.0
                @inbounds for (o, p) in outcomes
                    acc += p
                    if r <= acc; GI.apply_chance!(env, o); break; end
                end
                if !GI.is_chance_node(env) || GI.game_terminated(env)
                    continue
                end
                GI.apply_chance!(env, outcomes[end][1])
                continue
            end

            avail = GI.available_actions(env)
            if length(avail) == 1
                # Record forced move for value training (no MCTS needed)
                state = GI.current_state(env)
                push!(trace_states, state)
                push!(trace_policies, Float32[1.0])
                push!(trace_actions, [avail[1]])
                push!(trace_is_chance, false)
                push!(trace_rewards, 0.0f0)
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
                action = actions[_sample_from_policy(policy, rng)]
            elseif iszero(τ)
                action = actions[argmax(policy)]
            else
                temp_policy = Util.apply_temperature(policy, τ)
                action = actions[_sample_from_policy(temp_policy, rng)]
            end
            GI.play!(env, action)
            push!(trace_rewards, 0.0f0)
        end

        BatchedMCTS.reset_player!(player)

        if bearoff_truncated
            n_bearoff_truncated += 1
            final_reward = Float32(first_bearoff_wp ? first_bearoff_bo.value : -first_bearoff_bo.value)
            samples = convert_trace_to_samples(
                gspec, trace_states, trace_policies, trace_actions, trace_rewards, trace_is_chance,
                final_reward, nothing; rng=rng,
                bearoff_equity=first_bearoff_bo.equity, bearoff_wp=first_bearoff_wp)
        elseif first_bearoff_bo !== nothing
            final_reward = Float32(first_bearoff_wp ? first_bearoff_bo.value : -first_bearoff_bo.value)
            outcome = GI.game_outcome(env)
            samples = convert_trace_to_samples(
                gspec, trace_states, trace_policies, trace_actions, trace_rewards, trace_is_chance,
                final_reward, outcome; rng=rng,
                first_bearoff_equity=first_bearoff_bo.equity, first_bearoff_wp=first_bearoff_wp)
        else
            final_reward = Float32(GI.game_terminated(env) ? Float64(GI.white_reward(env)) : 0.0)
            outcome = GI.game_terminated(env) ? GI.game_outcome(env) : nothing
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
                           rng::AbstractRNG)
    sub_rng = new_selfplay_rng(rand(rng, UInt))

    result = _play_games_loop(worker_id, games_claimed, total_games, sub_rng)
    return result.samples
end

"""Play games on a worker thread with GPU inference (Metal, serialized via lock)."""
function worker_play_games_gpu(worker_id::Int, games_claimed::Threads.Atomic{Int}, total_games::Int,
                                rng::AbstractRNG)
    sub_rng = new_selfplay_rng(rand(rng, UInt))

    n_bearoff_truncated = 0

    mcts_params = MctsParams(
        num_iters_per_turn=MCTS_ITERS,
        cpuct=Float64(config["cpuct"]),
        temperature=ConstSchedule(1.0),
        dirichlet_noise_ϵ=Float64(config["dirichlet_epsilon"]),
        dirichlet_noise_α=Float64(config["dirichlet_alpha"]))

    # Eval uses NN+MCTS only (no bearoff table) — measures actual NN strength
    az_agent = GameLoop.MctsAgent(
        GPU_SINGLE_ORACLE, GPU_BATCH_ORACLE,
        mcts_params, INFERENCE_BATCH_SIZE, gspec;
        bearoff_eval=nothing,
        batch_oracle_with_actions=GPU_BATCH_ORACLE,
        sim_budget_fn=SELFPLAY_SIM_BUDGET_FN)

    # Create player ONCE and reuse across all games
    player = GameLoop.create_player(az_agent; rng=sub_rng)

    all_samples = []
    while Threads.atomic_add!(games_claimed, 1) < total_games
        env = init_game(sub_rng)

        # Capture full bearoff equity (5-element vector) via closure
        first_bearoff_bo = Ref{Any}(nothing)
        first_bearoff_wp = Ref{Bool}(true)
        function bearoff_lookup_with_capture(game)
            if !BearoffK7.is_bearoff_position(game.p0, game.p1)
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
            white_player=player, black_player=player,
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
        rng = isnothing(MAIN_SEED) ? new_selfplay_rng() : new_selfplay_rng(MAIN_SEED + wid * 104729)
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
        rng = isnothing(MAIN_SEED) ? new_selfplay_rng() : new_selfplay_rng(MAIN_SEED + w * 104729)
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

flush(stdout)
println("\n" * "=" ^ 60)
println("Starting self-play...")
println("=" ^ 60)
flush(stdout)

const UPLOAD_INTERVAL = ARGS["upload_interval"]

# Single background network thread handles uploads AND weight sync.
# HTTP.jl deadlocks with multiple concurrent spawned threads doing HTTP.
const UPLOAD_CHANNEL = Channel{Vector{UInt8}}(8)
const EVAL_CHANNEL = Channel{Dict{String,Any}}(40)  # eval chunks from server upload response
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
                if get(result, "restart", false)
                    println("\n*** Server requested restart — exiting for update ***")
                    flush(stdout)
                    exit(0)
                end
                # Queue eval work if server assigned a chunk
                eval_chunk = get(result, "eval_chunk", nothing)
                if eval_chunk !== nothing
                    put!(EVAL_CHANNEL, Dict{String,Any}(eval_chunk))
                    println("  [EVAL] Chunk $(eval_chunk["chunk_id"]) queued (iter=$(eval_chunk["eval_iter"]))")
                end
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
function continuous_worker(worker_id::Int, rng::AbstractRNG)
    println("  Worker $worker_id starting on thread $(Threads.threadid())")
    flush(stdout)
    sub_rng = new_selfplay_rng(rand(rng, UInt))

    # Create agent and player ONCE per worker — reused across all games
    mcts_params = MctsParams(
        num_iters_per_turn=MCTS_ITERS,
        cpuct=Float64(config["cpuct"]),
        temperature=ConstSchedule(1.0),
        dirichlet_noise_ϵ=Float64(config["dirichlet_epsilon"]),
        dirichlet_noise_α=Float64(config["dirichlet_alpha"]))
    az_agent = GameLoop.MctsAgent(
        CPU_SINGLE_ORACLE, CPU_BATCH_ORACLE,
        mcts_params, INFERENCE_BATCH_SIZE, gspec;
        bearoff_eval=BEAROFF_EVALUATOR,
        batch_oracle_with_actions=CPU_BATCH_ORACLE,
        sim_budget_fn=SELFPLAY_SIM_BUDGET_FN)
    player = GameLoop.create_player(az_agent; rng=sub_rng)

    # Play games forever — one at a time, push samples immediately
    games_played = 0
    while true
        games_claimed = Threads.Atomic{Int}(0)
        result = _play_games_loop(worker_id, games_claimed, 1, sub_rng;
                                  player=player)
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

#####
##### Client-side eval mode
#####

# Eval session — caches weights/config AND per-thread resources across chunks.
# Created once per eval iteration, reused across all chunks.
mutable struct EvalSession
    iter::Int
    weights_version::Int
    # Config
    eval_mcts_params::Any
    eval_cfg::Any
    eval_contact_fw::Any
    eval_race_fw::Any
    wildbg_nets_variant::Symbol
    # Per-thread resources (created once in setup_eval_session!, reused across chunks)
    agents::Vector{Any}             # EvalAlphaZeroAgent per thread
    wb_agents::Vector{Any}          # BackendAgent per thread
    value_batch_oracles::Vector{Any} # value oracle per thread
    n_threads::Int
end
const EVAL_SESSION = EvalSession(0, 0, nothing, nothing, nothing, nothing, :large,
                                  Any[], Any[], Any[], 0)

# Load eval positions if eval-capable
const EVAL_POSITIONS = if EVAL_CAPABLE
    eval_file = config["eval_positions_file"]
    path = find_or_download(eval_file; required=false)
    if !isempty(path)
        pos = Serialization.deserialize(path)
        println("Eval: loaded $(length(pos)) positions from $(basename(path))")
        pos
    else
        println("WARNING: Eval positions not available — eval disabled")
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

# Re-register with eval capability now that WILDBG_LIB_EVAL is known
if EVAL_CAPABLE
    register!(client; name=client_name,
              eval_capable=true, has_wildbg=!isempty(WILDBG_LIB_EVAL))
end

# Eval agent struct — holds a reusable MCTS player to avoid per-move allocation
mutable struct EvalAlphaZeroAgent <: BackgammonNet.AbstractAgent
    single_oracle::Any
    batch_oracle::Any
    mcts_params::MctsParams
    batch_size::Int
    gspec_::Any
    player::Any  # BatchedMctsPlayer, created once and reused
end

function EvalAlphaZeroAgent(single_oracle, batch_oracle, mcts_params, batch_size, gspec)
    player = BatchedMCTS.BatchedMctsPlayer(
        gspec, single_oracle, mcts_params;
        batch_size=batch_size, batch_oracle=batch_oracle,
        batch_oracle_with_actions=batch_oracle)
    EvalAlphaZeroAgent(single_oracle, batch_oracle, mcts_params, batch_size, gspec, player)
end

function BackgammonNet.agent_move(agent::EvalAlphaZeroAgent, g::BackgammonNet.BackgammonGame)
    env = GI.init(agent.gspec_)
    env.game = BackgammonNet.clone(g)
    actions, policy = BatchedMCTS.think(agent.player, env)
    BatchedMCTS.reset_player!(agent.player)
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
    rng = new_selfplay_rng(seed)
    az_agent.player.benv.env.rng = rng
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
                # NN V is normalized equity/reward_scale ∈ [-1,1]; wildbg returns
                # raw points. Scale NN back ×reward_scale so value MSE/corr compare
                # on the same (points) scale.
                nn_v = Float64(value_batch_oracle([g])[1][2]) * Float64(GI.reward_scale(gspec))
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

"""Set up eval session for a new iteration (download weights, build shared oracles).
Per-thread agents/wildbg are created in process_eval_chunk! — not here."""
function setup_eval_session!(eval_iter::Int, weights_version::Int)
    println("[EVAL] Setting up eval session for iter $eval_iter (weights v$weights_version)")
    flush(stdout)

    # A3/F2: capture the PREVIOUS session's wildbg backends but do NOT close them yet.
    # Each owns a native handle that must be freed — but only AFTER the new session is
    # fully built and swapped in. Closing first would leave EVAL_SESSION pointing at
    # closed handles if setup then fails (e.g. weights unavailable / backend open error).
    old_wb_agents = EVAL_SESSION.wb_agents

    # A1: download eval weights PINNED + VERSION-VERIFIED. Fail-fast — we must NOT
    # fall through to the self-play weight copy, or a chunk stamped weights_version=N
    # would submit metrics computed from DIFFERENT weights (silent mis-attribution).
    # download_weights(expected_version=...) throws on a version mismatch; a nothing
    # result (download failed / version not yet published) is fatal here too. The
    # enclosing chunk try/catch then skips the chunk (re-leased later) rather than
    # crediting the wrong iteration.
    eval_contact_net = Network.copy(contact_network; on_gpu=false, test_mode=true)
    eval_race_net = Network.copy(race_network; on_gpu=false, test_mode=true)
    result_c = download_weights(client, :contact; pinned_version=weights_version, expected_version=weights_version)
    result_r = download_weights(client, :race; pinned_version=weights_version, expected_version=weights_version)
    if result_c === nothing || result_r === nothing
        error("[EVAL] pinned weights v$weights_version unavailable (contact loaded=$(result_c !== nothing), " *
              "race loaded=$(result_r !== nothing)); refusing to eval with mismatched/self-play weights (fail-fast)")
    end
    FluxLib.load_weights!(eval_contact_net, result_c[2])
    FluxLib.load_weights!(eval_race_net, result_r[2])
    println("[EVAL] Loaded pinned+verified contact+race weights v$weights_version")

    # Create FastWeights for eval (immutable after creation — thread-safe to read)
    eval_contact_fw = CPU_INFERENCE_BACKEND == :fast ?
        AlphaZero.FastInference.extract_fast_weights(eval_contact_net) : nothing
    eval_race_fw = CPU_INFERENCE_BACKEND == :fast ?
        AlphaZero.FastInference.extract_fast_weights(eval_race_net) : nothing

    eval_cfg = AlphaZero.BackgammonInference.OracleConfig(
        _state_dim, NUM_ACTIONS, gspec;
        route_state=ORACLE_CFG.route_state)

    eval_mcts_params = MctsParams(
        num_iters_per_turn=EVAL_MCTS_ITERS,
        cpuct=1.5,
        temperature=ConstSchedule(0.0),
        dirichlet_noise_ϵ=0.0,
        dirichlet_noise_α=0.3)

    # Auto-detect wildbg nets variant
    lib_size = filesize(WILDBG_LIB_EVAL)
    nets_variant = lib_size > 16_000_000 ? :large : :small

    # One call for both backends: in :flux mode eval_contact_fw/eval_race_fw are
    # nothing and make_cpu_oracles ignores them (it routes on the net); in :fast
    # mode they carry the extracted FastWeights. make_cpu_oracles now guards
    # against a nothing net+fw, so we never silently build a broken oracle.
    make_eval_oracles(batch_size) = AlphaZero.BackgammonInference.make_cpu_oracles(
        CPU_INFERENCE_BACKEND, eval_contact_net, eval_cfg;
        secondary_net=eval_race_net, batch_size=batch_size,
        primary_fw=eval_contact_fw, secondary_fw=eval_race_fw, nslots=1)

    # Create ALL per-thread resources ONCE (reused across all chunks in this eval).
    # F2: if a backend fails to open partway, close the NEW backends already opened
    # before rethrowing — a partial setup must not leak handles, and the old session
    # (still referenced by EVAL_SESSION) stays intact because we have not swapped yet.
    n_threads = Threads.maxthreadid()
    agents = Vector{Any}(undef, n_threads)
    wb_agents = Vector{Any}(undef, n_threads)
    value_batch_oracles = Vector{Any}(undef, n_threads)
    # Track EVERY constructed wb, pushed BEFORE open!, so a failure inside open!
    # (which can acquire the native handle before a later dlsym/wildbg_new throws) or
    # inside BackendAgent() still closes it — closing an unopened/partially-opened
    # backend is a safe no-op. wb_agents-only cleanup would miss the in-flight wb.
    opened_wbs = Any[]
    try
        for tid in 1:n_threads
            oracles = make_eval_oracles(INFERENCE_BATCH_SIZE)
            agents[tid] = EvalAlphaZeroAgent(oracles[1], oracles[2],
                eval_mcts_params, INFERENCE_BATCH_SIZE, gspec)
            vo = make_eval_oracles(1)
            value_batch_oracles[tid] = vo[2]
            wb = BackgammonNet.WildbgBackend(; nets=nets_variant, lib_path=WILDBG_LIB_EVAL)
            push!(opened_wbs, wb)
            BackgammonNet.open!(wb)
            wb_agents[tid] = BackgammonNet.BackendAgent(wb)
        end
    catch
        for wb in opened_wbs
            try; close(wb); catch; end
        end
        rethrow()
    end

    # Store everything in session — reused across chunks
    EVAL_SESSION.iter = eval_iter
    EVAL_SESSION.weights_version = weights_version
    EVAL_SESSION.eval_mcts_params = eval_mcts_params
    EVAL_SESSION.eval_cfg = eval_cfg
    EVAL_SESSION.eval_contact_fw = eval_contact_fw
    EVAL_SESSION.eval_race_fw = eval_race_fw
    EVAL_SESSION.wildbg_nets_variant = nets_variant
    EVAL_SESSION.agents = agents
    EVAL_SESSION.wb_agents = wb_agents
    EVAL_SESSION.value_batch_oracles = value_batch_oracles
    EVAL_SESSION.n_threads = n_threads

    # F2: the new session is fully swapped in — NOW it is safe to close the previous
    # session's backends (freeing their native handles). A failure above rethrew
    # before reaching here, leaving the old session intact and its handles open.
    if old_wb_agents !== nothing && !isempty(old_wb_agents)
        for a in old_wb_agents
            a === nothing && continue
            try
                close(a.backend)
            catch e
                println("[EVAL] warning: failed to close old wildbg backend: $e")
            end
        end
    end

    println("[EVAL] Session ready (iter=$eval_iter, $n_threads eval slots)")
    flush(stdout)
end

"""Send heartbeat for an active eval chunk."""
function send_eval_heartbeat(chunk_id::Int)
    try
        body = MsgPack.pack(Dict("chunk_id" => chunk_id, "client_name" => client.client_id))
        HTTP.post("$(client.server_url)/api/eval/heartbeat",
                  vcat(auth_headers(client), ["Content-Type" => "application/msgpack"]),
                  body; status_exception=false, connect_timeout=5, readtimeout=10)
    catch
    end
end

"""Submit eval results for a completed chunk."""
function submit_eval_results(chunk_id::Int, rewards, val_nn, val_opp, val_is_contact)
    result = Dict(
        "chunk_id" => chunk_id,
        "client_name" => client.client_id,
        "rewards" => rewards,
        "value_nn" => val_nn,
        "value_opp" => val_opp,
        "value_is_contact" => val_is_contact,
    )
    body = MsgPack.pack(result)
    for attempt in 1:3
        try
            resp = HTTP.post("$(client.server_url)/api/eval/submit",
                             vcat(auth_headers(client), ["Content-Type" => "application/msgpack"]),
                             body; status_exception=false, connect_timeout=10, readtimeout=30)
            if resp.status == 200
                data = JSON.parse(String(resp.body))
                if get(data, "eval_complete", false)
                    println("[EVAL] *** Eval complete for iter $(EVAL_SESSION.iter) ***")
                end
                return true
            else
                println("[EVAL] Submit failed (attempt $attempt): HTTP $(resp.status)")
            end
        catch e
            println("[EVAL] Submit error (attempt $attempt): $e")
        end
        sleep(2^attempt)
    end
    return false
end

"""Process eval chunk with all worker threads in parallel.

Pre-creates per-thread agents (mutable MCTS tree) and wildbg backends
(mutable cached target) on the MAIN thread, then dispatches games via
Threads.@threads. Shared oracles are thread-safe (per-task buffers)."""
function process_eval_chunk!(chunk_data::Dict)
    EVAL_POSITIONS === nothing && return
    isempty(WILDBG_LIB_EVAL) && return

    chunk_id = Int(chunk_data["chunk_id"])
    eval_iter = Int(chunk_data["eval_iter"])
    weights_version = Int(chunk_data["weights_version"])
    pos_start = Int(chunk_data["position_range_start"])
    pos_end = Int(chunk_data["position_range_end"])
    az_is_white = Bool(chunk_data["az_is_white"])
    n_games = pos_end - pos_start + 1

    println("[EVAL] Chunk $chunk_id: $n_games games ($(az_is_white ? "white" : "black")), iter=$eval_iter, $NUM_WORKERS workers")
    flush(stdout)

    # Setup or refresh eval session if iter or weights version changed
    if EVAL_SESSION.n_threads == 0 || EVAL_SESSION.n_threads < Threads.maxthreadid() ||
       EVAL_SESSION.iter != eval_iter ||
       EVAL_SESSION.weights_version != weights_version
        setup_eval_session!(eval_iter, weights_version)
    end

    # Use per-thread resources from session (created once, reused across chunks)
    agents = EVAL_SESSION.agents
    wb_agents = EVAL_SESSION.wb_agents
    value_batch_oracles = EVAL_SESSION.value_batch_oracles
    n_threads = EVAL_SESSION.n_threads

    rewards = Vector{Float64}(undef, n_games)
    # Per-thread value sample collection (no locks needed)
    thread_val_nn = [Float64[] for _ in 1:n_threads]
    thread_val_opp = [Float64[] for _ in 1:n_threads]
    t0 = time()

    # Heartbeat task
    heartbeat_done = Threads.Atomic{Bool}(false)
    heartbeat_task = Threads.@spawn begin
        next_heartbeat = time() + 10
        while !heartbeat_done[]
            if time() >= next_heartbeat
                send_eval_heartbeat(chunk_id)
                next_heartbeat = time() + 10
            end
            sleep(1)
        end
    end

    # Parallel eval via an explicit WORKER POOL. Spawn min(n_threads, n_games)
    # tasks; each OWNS a fixed resource slot `wid` for its whole lifetime and pulls
    # jobs from a shared atomic counter. This removes all reliance on
    # Threads.threadid() (brittle under task migration / Julia's interactive pool):
    # a task's slot is fixed at spawn, so agents[wid]/wb_agents[wid]/oracle[wid] and
    # the thread_val_* accumulators are exclusively owned no matter which OS thread
    # runs the task, and each rewards[job] is written exactly once.
    n_workers = min(n_threads, max(n_games, 1))
    next_job = Threads.Atomic{Int}(0)
    try
        @sync for wid in 1:n_workers
            Threads.@spawn begin
                agent = agents[wid]; wb = wb_agents[wid]; vo = value_batch_oracles[wid]
                while true
                    job = Threads.atomic_add!(next_job, 1) + 1
                    job > n_games && break
                    pos_idx = pos_start + job - 1
                    if pos_idx > length(EVAL_POSITIONS)
                        rewards[job] = 0.0
                    else
                        result = eval_game_from_position(
                            agent, wb, EVAL_POSITIONS[pos_idx], vo;
                            seed=chunk_id * 10000 + job, az_is_white=az_is_white)
                        rewards[job] = result.reward
                        for s in result.value_samples
                            push!(thread_val_nn[wid], s.nn_val)
                            push!(thread_val_opp[wid], s.wb_val)
                        end
                    end
                end
            end
        end
    finally
        Threads.atomic_xchg!(heartbeat_done, true)
        wait(heartbeat_task)
    end

    # Merge per-thread results
    val_nn = reduce(vcat, thread_val_nn)
    val_opp = reduce(vcat, thread_val_opp)
    val_is_contact = fill(false, length(val_nn))

    t_chunk = time() - t0
    gpm = n_games / t_chunk * 60
    avg_reward = length(rewards) > 0 ? sum(rewards) / length(rewards) : 0.0
    win_pct = length(rewards) > 0 ? count(r -> r > 0, rewards) / length(rewards) * 100 : 0.0
    println("[EVAL] Chunk $chunk_id done: equity=$(round(avg_reward, digits=3)), win=$(round(win_pct, digits=1))%, $(round(t_chunk, digits=1))s ($(round(gpm, digits=0)) games/min)")
    flush(stdout)

    submit_eval_results(chunk_id, rewards, val_nn, val_opp, val_is_contact)
end

#####
function main_loop()
    # Start all workers as continuous background threads
    tasks = Task[]
    for w in 1:NUM_WORKERS
        rng = isnothing(MAIN_SEED) ? new_selfplay_rng() : new_selfplay_rng(MAIN_SEED + w * 104729)
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
    t_session_start = time()
    t_batch_start = time()

    while true
        # Process eval chunks (prioritize eval over selfplay)
        while isready(EVAL_CHANNEL)
            chunk_data = take!(EVAL_CHANNEL)
            try
                process_eval_chunk!(chunk_data)
            catch e
                println("[EVAL] Chunk error: $e")
                Base.showerror(stdout, e, catch_backtrace())
                println()
            end
        end

        # Drain selfplay games — poll with timeout so we can check eval between waits
        if batch_games == 0
            t_batch_start = time()
        end
        while batch_games < UPLOAD_INTERVAL
            if isready(SAMPLE_CHANNEL)
                game_samples = take!(SAMPLE_CHANNEL)
                append!(batch_samples, game_samples)
                batch_games += 1
                games_played += 1
            elseif isready(EVAL_CHANNEL)
                # Eval work arrived while waiting for selfplay — process it
                chunk_data = take!(EVAL_CHANNEL)
                try
                    process_eval_chunk!(chunk_data)
                catch e
                    println("[EVAL] Chunk error: $e")
                    Base.showerror(stdout, e, catch_backtrace())
                    println()
                end
            else
                sleep(0.01)  # Brief sleep to avoid busy-wait
            end
        end

        batch_num += 1
        n_samples = length(batch_samples)
        total_samples_collected += n_samples
        t_total = time() - t_session_start
        avg_gps = games_played / max(t_total, 1.0)
        println("Batch $batch_num: $batch_games games, $n_samples samples (avg $(round(avg_gps, digits=1)) games/sec)")

        # Queue upload + weight sync on background network thread (non-blocking)
        batch = samples_to_batch(batch_samples)
        bytes = pack_samples(batch)
        put!(UPLOAD_CHANNEL, bytes)

        # Reset batch
        batch_samples = []
        batch_games = 0
        flush(stdout)
    end
end

main_loop()
