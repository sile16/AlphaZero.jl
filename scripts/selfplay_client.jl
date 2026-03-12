#!/usr/bin/env julia
"""
Self-play client for distributed training.

Connects to the training server, downloads model weights,
runs MCTS self-play locally, and uploads samples.

This script reuses the self-play infrastructure from train_distributed.jl
(FastWeights, batched MCTS, inference server) but replaces the training
loop with HTTP client calls.

Usage:
    julia --threads 30 --project scripts/selfplay_client.jl \\
        --server http://jarvis:9090 \\
        --api-key my-secret-key \\
        --num-workers 22
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
            help = "Training server URL"
            arg_type = String
            default = "http://jarvis:9090"
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
            default = 50
        "--weight-sync-interval"
            help = "Check for weight updates every N seconds"
            arg_type = Float64
            default = 30.0
        "--seed"
            arg_type = Int
            default = 0
    end

    return ArgParse.parse_args(s)
end

const ARGS = parse_args()
const SERVER_URL = ARGS["server"]
const NUM_WORKERS = ARGS["num_workers"]

# Set seed (0 = random)
if ARGS["seed"] > 0
    Random.seed!(ARGS["seed"])
end

println("=" ^ 60)
println("AlphaZero Self-Play Client")
println("=" ^ 60)
println("Server: $SERVER_URL")
println("Workers: $NUM_WORKERS")
println("=" ^ 60)
flush(stdout)

# Load packages
using AlphaZero
using AlphaZero: GI, MCTS, Network, FluxLib, MctsParams, LearningParams
using AlphaZero: BatchedMCTS
using AlphaZero.NetLib
import Flux

# Include shared distributed code
include(joinpath(@__DIR__, "..", "src", "distributed", "buffer.jl"))
include(joinpath(@__DIR__, "..", "src", "distributed", "protocol.jl"))
include(joinpath(@__DIR__, "..", "src", "distributed", "client.jl"))

# Connect to server and get config
client_name = isempty(ARGS["client_name"]) ? "julia-$(gethostname())-$(getpid())" : ARGS["client_name"]
client = SelfPlayClient(SERVER_URL, ARGS["api_key"];
                        client_id=client_name, upload_threshold=ARGS["upload_interval"] * 200)

println("\nConnecting to server...")
if !register!(client; name=client_name)
    error("Failed to register with server")
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
const CONTACT_WIDTH = config["contact_width"]
const CONTACT_BLOCKS = config["contact_blocks"]
const RACE_WIDTH = config["race_width"]
const RACE_BLOCKS = config["race_blocks"]
const MCTS_ITERS = config["mcts_iters"]
const INFERENCE_BATCH_SIZE = config["inference_batch_size"]
const _state_dim = config["state_dim"]
const NUM_ACTIONS = config["num_actions"]

# Game setup
push!(LOAD_PATH, joinpath(@__DIR__, "..", "games"))
include(joinpath(@__DIR__, "..", "games", GAME_NAME, "main.jl"))
const gspec = Training.GameSpec()

# Create networks (CPU for self-play inference)
println("\nCreating networks...")
contact_network = NetLib.FCResNetMultiHead(gspec;
    width=CONTACT_WIDTH, num_blocks=CONTACT_BLOCKS,
    depth_phead=2, depth_vhead=2, share_value_trunk=true)
race_network = NetLib.FCResNetMultiHead(gspec;
    width=RACE_WIDTH, num_blocks=RACE_BLOCKS,
    depth_phead=2, depth_vhead=2, share_value_trunk=true)

println("Contact model: $(CONTACT_WIDTH)w×$(CONTACT_BLOCKS)b ($(sum(length(p) for p in Flux.params(contact_network))) params)")
println("Race model: $(RACE_WIDTH)w×$(RACE_BLOCKS)b ($(sum(length(p) for p in Flux.params(race_network))) params)")

# Download initial weights from server
println("\nDownloading initial weights from server...")
if !sync_weights!(client, contact_network, race_network)
    println("No weights available yet (new run). Starting with random weights.")
end
flush(stdout)

# Set up BLAS for single-threaded per-worker inference
import LinearAlgebra; LinearAlgebra.BLAS.set_num_threads(1)

# ============================================================
# Self-play infrastructure
# (This section would include FastWeights, batched MCTS, etc.
#  from train_distributed.jl. For now, we include the essentials.)
# ============================================================

# TODO: The full self-play infrastructure (FastWeights, _gemm_bias!,
# inference server, batched MCTS workers) needs to be extracted from
# train_distributed.jl into a shared module. For now, we include the
# key parts inline.
#
# The self-play loop structure:
# 1. Initialize FastWeights from current network
# 2. Start worker threads (each runs batched MCTS)
# 3. Collect game samples
# 4. Upload samples to server
# 5. Periodically sync weights

println("\n" * "=" ^ 60)
println("Starting self-play...")
println("=" ^ 60)
flush(stdout)

# Placeholder for the full self-play loop.
# The actual implementation requires extracting ~1000 lines of self-play
# code from train_distributed.jl (FastWeights, batched MCTS, game loop).
#
# For now, print what would happen:
println("""
TODO: Self-play loop implementation.

This script needs the following from train_distributed.jl:
1. FastWeights struct + extract_fast_weights()     (~lines 596-750)
2. _gemm_bias! and matmul helpers                  (~lines 756-870)
3. _fast_forward() allocation-free inference        (~lines 870-1050)
4. BatchedMCTS integration                          (~lines 1080-1250)
5. play_game() with sample collection               (~lines 1250-1520)
6. parallel_self_play() worker coordination          (~lines 1520-1600)

Each iteration:
- Play $(ARGS["upload_interval"]) games across $NUM_WORKERS workers
- Upload ~$(ARGS["upload_interval"] * 200) samples to $SERVER_URL
- Check for weight updates every $(ARGS["weight_sync_interval"])s
""")

# Simple demo loop showing the client protocol
println("Running client protocol demo...")
flush(stdout)

games_played = 0
last_weight_check = time()

while true
    # Check for weight updates periodically
    if time() - last_weight_check > ARGS["weight_sync_interval"]
        updated = sync_weights!(client, contact_network, race_network)
        if updated
            println("  Weights updated! New version: contact=$(client.contact_version), race=$(client.race_version)")
            # TODO: Refresh FastWeights from updated networks
        end
        last_weight_check = time()
    end

    # Show server status
    status = server_status(client)
    if status !== nothing
        println("  Server: iter=$(status["iteration"]), buffer=$(status["buffer_size"]), clients=$(status["total_clients"])")
    end

    sleep(10.0)
end
