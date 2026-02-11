#!/usr/bin/env julia
"""
Threaded AlphaZero training with GPU inference server.

Architecture:
- Inference server thread: Aggregates GPU requests from all workers
- Worker threads: Self-play game generation with batched MCTS (zero-copy channels)
- Main thread: Training loop, replay buffer, weight updates

Usage:
    julia --project --threads=16 scripts/train_distributed.jl \\
        --num-workers=14 \\
        --total-iterations=50
"""

using ArgParse
using Dates
using Random
using StaticArrays: SVector
using Statistics
using TensorBoardLogger
using Logging: with_logger

# Use Intel MKL for ~3.8x faster BLAS on Intel CPUs (vs OpenBLAS default)
try
    using MKL
catch e
    @warn "MKL not available, using OpenBLAS. Install with: Pkg.add(\"MKL\")"
end

function parse_args()
    s = ArgParseSettings(
        description="Threaded AlphaZero training with GPU inference server",
        autofix_names=true
    )

    @add_arg_table! s begin
        "--game"
            help = "Game to train"
            arg_type = String
            default = "backgammon-deterministic"
        "--network-type"
            help = "Network type"
            arg_type = String
            default = "fcresnet-multihead"
        "--network-width"
            help = "Network width"
            arg_type = Int
            default = 128
        "--network-blocks"
            help = "Number of residual blocks"
            arg_type = Int
            default = 3
        "--num-workers"
            help = "Number of worker threads for self-play"
            arg_type = Int
            default = 14
        "--total-iterations"
            help = "Total training iterations"
            arg_type = Int
            default = 50
        "--games-per-iteration"
            help = "Games to collect per iteration"
            arg_type = Int
            default = 50
        "--batch-size"
            help = "Training batch size"
            arg_type = Int
            default = 256
        "--buffer-capacity"
            help = "Replay buffer capacity"
            arg_type = Int
            default = 100000
        "--mcts-iters"
            help = "MCTS iterations per move"
            arg_type = Int
            default = 100
        "--learning-rate"
            help = "Learning rate"
            arg_type = Float64
            default = 0.001
        "--l2-reg"
            help = "L2 regularization"
            arg_type = Float64
            default = 0.0001
        "--eval-interval"
            help = "Evaluation interval"
            arg_type = Int
            default = 10
        "--eval-games"
            help = "Games per evaluation"
            arg_type = Int
            default = 50
        "--final-eval-games"
            help = "Games for final evaluation (0 to disable)"
            arg_type = Int
            default = 1000
        "--checkpoint-interval"
            help = "Checkpoint save interval"
            arg_type = Int
            default = 10
        "--seed"
            help = "Random seed"
            arg_type = Int
            default = nothing
        "--inference-batch-size"
            help = "Batch size for GPU inference during MCTS (leaves per batch)"
            arg_type = Int
            default = 100
        "--session-dir"
            help = "Session directory (auto-generated if not specified)"
            arg_type = String
            default = ""
        "--resume"
            help = "Resume from session directory (loads weights + continues iteration count)"
            arg_type = String
            default = ""
        "--use-per"
            help = "Enable Prioritized Experience Replay"
            action = :store_true
        "--per-alpha"
            help = "PER prioritization exponent (0=uniform, 1=full priority)"
            arg_type = Float32
            default = 0.6f0
        "--per-beta"
            help = "PER importance sampling initial beta (anneals to 1.0)"
            arg_type = Float32
            default = 0.4f0
        "--per-epsilon"
            help = "PER small constant for priority stability"
            arg_type = Float32
            default = 0.01f0
        "--use-reanalyze"
            help = "Enable buffer reanalysis with latest network"
            action = :store_true
        "--reanalyze-fraction"
            help = "Fraction of buffer to reanalyze per iteration"
            arg_type = Float64
            default = 0.25
        "--use-bearoff"
            help = "Enable exact bear-off table value targets (k=6 two-sided database)"
            action = :store_true
    end

    return ArgParse.parse_args(s)
end

const ARGS = parse_args()

# Constants
const GAME_NAME = ARGS["game"]
const NET_WIDTH = ARGS["network_width"]
const NET_BLOCKS = ARGS["network_blocks"]
const NUM_WORKERS = ARGS["num_workers"]
const MCTS_ITERS = ARGS["mcts_iters"]
const INFERENCE_BATCH_SIZE = ARGS["inference_batch_size"]
const MAIN_SEED = ARGS["seed"]
const LEARNING_RATE = Float32(ARGS["learning_rate"])
const L2_REG = Float32(ARGS["l2_reg"])
const BUFFER_CAPACITY = ARGS["buffer_capacity"]
const BATCH_SIZE = ARGS["batch_size"]

# Setup session directory
const RESUME_DIR = ARGS["resume"]
const RESUMING = !isempty(RESUME_DIR)
const SESSION_DIR = if RESUMING
    RESUME_DIR
elseif isempty(ARGS["session_dir"])
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    # Build descriptive suffix
    suffixes = String[]
    ARGS["use_per"] && push!(suffixes, "per")
    ARGS["use_reanalyze"] && push!(suffixes, "reanalyze")
    ARGS["use_bearoff"] && push!(suffixes, "bearoff")
    suffix = isempty(suffixes) ? "" : "_" * join(suffixes, "_")
    joinpath("sessions", "distributed_$(timestamp)$(suffix)")
else
    ARGS["session_dir"]
end
mkpath(SESSION_DIR)
mkpath(joinpath(SESSION_DIR, "checkpoints"))

# TensorBoard logging
const TB_DIR = joinpath(SESSION_DIR, "tb")
mkpath(TB_DIR)
const TB_LOGGER = TBLogger(TB_DIR, tb_append)

# Log reproducibility info to TensorBoard
with_logger(TB_LOGGER) do
    # Git commit
    git_commit = try strip(read(`git rev-parse HEAD`, String)) catch; "unknown" end
    git_dirty = try strip(read(`git diff --stat`, String)) catch; "" end
    git_status = isempty(git_dirty) ? "clean" : "dirty"

    # Command line
    cmd = "julia " * join(Base.ARGS, " ")

    # All hyperparameters
    params_lines = ["## Hyperparameters\n"]
    for (k, v) in sort(collect(ARGS), by=first)
        push!(params_lines, "- **$k**: `$v`")
    end

    # Build features description
    features = String[]
    ARGS["use_per"] && push!(features, "PER (α=$(ARGS["per_alpha"]), β=$(ARGS["per_beta"]))")
    ARGS["use_reanalyze"] && push!(features, "Reanalyze ($(ARGS["reanalyze_fraction"]*100)%)")
    ARGS["use_bearoff"] && push!(features, "Bear-off table (k=6 exact, MCTS+TD-bootstrap)")
    features_str = isempty(features) ? "Baseline (no enhancements)" : join(features, ", ")

    repro_text = """
    ## Goal
    Train world-class SOTA backgammon model using AlphaZero self-play with multi-head equity network.

    ## Experiment
    - **Features**: $(features_str)
    - **Session**: `$(SESSION_DIR)`

    ## Reproducibility
    - **Git commit**: `$(git_commit)` ($(git_status))
    - **Command**: `$(cmd)`
    - **Julia version**: `$(VERSION)`
    - **Threads**: $(Threads.nthreads())
    - **Date**: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))

    $(join(params_lines, "\n"))
    """
    @info "config" text=repro_text log_step_increment=0
end

println("=" ^ 60)
println(RESUMING ? "Threaded AlphaZero Training (RESUMING)" : "Threaded AlphaZero Training")
println("=" ^ 60)
println("Session: $SESSION_DIR")
println("Game: $GAME_NAME")
println("Network: $(ARGS["network_type"]) ($(NET_WIDTH)x$(NET_BLOCKS))")
println("Workers: $NUM_WORKERS threads ($(Threads.nthreads()) Julia threads)")
println("Iterations: $(ARGS["total_iterations"])")
println("Games/iteration: $(ARGS["games_per_iteration"])")
println("MCTS iterations: $MCTS_ITERS")
println("Eval interval: $(ARGS["eval_interval"]) iterations")
println("Eval games: $(ARGS["eval_games"])")
println("Final eval games: $(ARGS["final_eval_games"])")
println("Inference batch size: $INFERENCE_BATCH_SIZE")
println("=" ^ 60)
flush(stdout)

# Check thread count
if Threads.nthreads() < NUM_WORKERS + 2
    @warn "Julia has $(Threads.nthreads()) threads but $NUM_WORKERS workers requested. " *
          "For best performance: julia --threads=$(NUM_WORKERS + 2) --project ..."
end

# Load packages (single process — all threads share memory)
using AlphaZero
using AlphaZero: GI, MCTS, Network, FluxLib, MctsParams, LearningParams, Adam
using AlphaZero: CONSTANT_WEIGHT, losses, ConstSchedule
using AlphaZero: BatchedMCTS
using AlphaZero.NetLib
import Flux
import CUDA

# Check GPU
const USE_GPU = CUDA.functional()
if USE_GPU
    CUDA.allowscalar(false)
    println("\nGPU: $(CUDA.name(CUDA.device()))")
else
    println("\nGPU: Not available, using CPU")
end
flush(stdout)

# Load game
if GAME_NAME == "backgammon-deterministic"
    ENV["BACKGAMMON_OBS_TYPE"] = "minimal"
    include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
elseif GAME_NAME == "backgammon"
    include(joinpath(@__DIR__, "..", "games", "backgammon", "game.jl"))
else
    error("Unknown game: $GAME_NAME")
end

# Create network
const gspec = GameSpec()
const NUM_ACTIONS = GI.num_actions(gspec)

network = FluxLib.FCResNetMultiHead(
    gspec, FluxLib.FCResNetMultiHeadHP(width=NET_WIDTH, num_blocks=NET_BLOCKS))
if USE_GPU
    network = Network.to_gpu(network)
end

# Resume: load weights from checkpoint
START_ITER = 0
if RESUMING
    ckpt_path = joinpath(RESUME_DIR, "checkpoints", "latest.data")
    iter_path = joinpath(RESUME_DIR, "checkpoints", "iter.txt")
    if isfile(ckpt_path)
        FluxLib.load_weights(ckpt_path, network)
        println("Resumed weights from: $ckpt_path")
    else
        error("Resume checkpoint not found: $ckpt_path")
    end
    if isfile(iter_path)
        START_ITER = parse(Int, strip(read(iter_path, String)))
        println("Resuming from iteration: $START_ITER")
    else
        @warn "iter.txt not found, starting from iteration 0"
    end
end

println("Network parameters: $(sum(length, Flux.params(network)))")

# CPU copy for local inference during self-play (avoids channel/queue overhead)
const cpu_network = Flux.cpu(deepcopy(network))
import LinearAlgebra; LinearAlgebra.BLAS.set_num_threads(1)  # Prevent BLAS thread contention
println("CPU inference: BLAS threads=1, lightweight clones, shared RNG")
flush(stdout)

#####
##### Allocation-free forward pass
#####

"""Pre-extracted weights from FCResNetMultiHead for allocation-free forward."""
struct FastWeights
    # Input layer: Dense(indim, width)
    W_in::Matrix{Float32}     # (width, indim)
    b_in::Vector{Float32}     # (width,)
    ln_in_s::Vector{Float32}  # LayerNorm scale
    ln_in_b::Vector{Float32}  # LayerNorm bias

    # Residual blocks: each has 2 Dense + 2 LayerNorm
    res_W1::Vector{Matrix{Float32}}   # Dense 1 weight per block
    res_b1::Vector{Vector{Float32}}   # Dense 1 bias
    res_ln1_s::Vector{Vector{Float32}}  # LN 1 scale
    res_ln1_b::Vector{Vector{Float32}}  # LN 1 bias
    res_W2::Vector{Matrix{Float32}}   # Dense 2 weight
    res_b2::Vector{Vector{Float32}}   # Dense 2 bias
    res_ln2_s::Vector{Vector{Float32}}  # LN 2 scale
    res_ln2_b::Vector{Vector{Float32}}  # LN 2 bias

    # Post-tower LayerNorm
    ln_post_s::Vector{Float32}
    ln_post_b::Vector{Float32}

    # Value trunk: Dense(width, width) + LN
    W_vt::Matrix{Float32}
    b_vt::Vector{Float32}
    ln_vt_s::Vector{Float32}
    ln_vt_b::Vector{Float32}

    # 5 value heads: Dense(width, 1) each
    W_vh::Vector{Vector{Float32}}  # (width,) per head (since output=1)
    b_vh::Vector{Float32}          # scalar bias per head

    # Policy head layers
    W_p::Vector{Matrix{Float32}}   # Dense weights
    b_p::Vector{Vector{Float32}}   # Dense biases
    ln_p_s::Vector{Vector{Float32}}  # LN scales
    ln_p_b::Vector{Vector{Float32}}  # LN biases
    W_pout::Matrix{Float32}        # Final Dense(width, nactions)
    b_pout::Vector{Float32}        # Final bias

    num_blocks::Int
    num_policy_layers::Int
end

"""Extract layer norm parameters from a Flux LayerNorm layer."""
function extract_ln(ln)
    # Flux LayerNorm stores diag.scale and diag.bias
    return (Vector{Float32}(ln.diag.scale), Vector{Float32}(ln.diag.bias))
end

"""Extract Dense parameters."""
function extract_dense(d)
    return (Matrix{Float32}(d.weight), Vector{Float32}(d.bias))
end

function FastWeights(nn::FluxLib.FCResNetMultiHead)
    # Common trunk: Chain(input_layer_chain, ResBlock1, ..., ResBlockN, LN, relu)
    # where input_layer_chain = Chain(flatten, Dense(342,128), LN(128), relu)
    common = nn.common
    layers = common.layers

    # Input Dense and LayerNorm are inside the nested input_layer Chain
    input_chain = layers[1]  # Chain(flatten, Dense, LN, relu)
    W_in, b_in = extract_dense(input_chain.layers[2])   # Dense(342, 128)
    ln_in_s, ln_in_b = extract_ln(input_chain.layers[3])  # LayerNorm(128)

    # Residual blocks: layers[2], layers[3], ..., layers[1+num_blocks]
    num_blocks = nn.hyper.num_blocks
    res_W1 = Matrix{Float32}[]
    res_b1 = Vector{Float32}[]
    res_ln1_s = Vector{Float32}[]
    res_ln1_b = Vector{Float32}[]
    res_W2 = Matrix{Float32}[]
    res_b2 = Vector{Float32}[]
    res_ln2_s = Vector{Float32}[]
    res_ln2_b = Vector{Float32}[]

    for i in 1:num_blocks
        block = layers[1 + i]  # PreActResBlock
        w1, b1_ = extract_dense(block.dense1)
        s1, b1_ln = extract_ln(block.ln1)
        w2, b2_ = extract_dense(block.dense2)
        s2, b2_ln = extract_ln(block.ln2)
        push!(res_W1, w1); push!(res_b1, b1_)
        push!(res_ln1_s, s1); push!(res_ln1_b, b1_ln)
        push!(res_W2, w2); push!(res_b2, b2_)
        push!(res_ln2_s, s2); push!(res_ln2_b, b2_ln)
    end

    # Post-tower LN: layers[1 + num_blocks + 1]
    ln_post_s, ln_post_b = extract_ln(layers[1 + num_blocks + 1])

    # Value trunk
    vt = nn.vhead_trunk
    W_vt, b_vt = extract_dense(vt.layers[1])
    ln_vt_s, ln_vt_b = extract_ln(vt.layers[2])

    # 5 value heads: each Chain(Dense(w,1), sigmoid)
    W_vh = Vector{Float32}[]
    b_vh = Float32[]
    for head in (nn.vhead_win, nn.vhead_gw, nn.vhead_bgw, nn.vhead_gl, nn.vhead_bgl)
        d = head.layers[1]  # Dense(width, 1)
        push!(W_vh, vec(d.weight))  # (width,) vector
        push!(b_vh, d.bias[1])
    end

    # Policy head: Chain(Dense, LN, relu, Dense, LN, relu, Dense(w, nactions), softmax)
    phead = nn.phead
    n_policy = nn.hyper.depth_phead  # typically 2
    W_p = Matrix{Float32}[]
    b_p = Vector{Float32}[]
    ln_p_s = Vector{Float32}[]
    ln_p_b = Vector{Float32}[]
    for i in 1:n_policy
        base = (i - 1) * 3  # Dense, LN, relu groups
        w, b = extract_dense(phead.layers[base + 1])
        s, lb = extract_ln(phead.layers[base + 2])
        push!(W_p, w); push!(b_p, b)
        push!(ln_p_s, s); push!(ln_p_b, lb)
    end
    # Final policy Dense (before softmax)
    W_pout, b_pout = extract_dense(phead.layers[n_policy * 3 + 1])

    FastWeights(W_in, b_in, ln_in_s, ln_in_b,
                res_W1, res_b1, res_ln1_s, res_ln1_b,
                res_W2, res_b2, res_ln2_s, res_ln2_b,
                ln_post_s, ln_post_b,
                W_vt, b_vt, ln_vt_s, ln_vt_b,
                W_vh, b_vh,
                W_p, b_p, ln_p_s, ln_p_b, W_pout, b_pout,
                num_blocks, n_policy)
end

"""Pre-allocated buffers for allocation-free forward pass."""
struct FastBuffers
    h1::Matrix{Float32}    # (width, max_batch) — primary hidden state
    h2::Matrix{Float32}    # (width, max_batch) — temp for residual/ops
    skip::Matrix{Float32}  # (width, max_batch) — skip connection storage
    vt::Matrix{Float32}    # (width, max_batch) — value trunk output
    p::Matrix{Float32}     # (nactions, max_batch) — policy output
    ln_mean::Vector{Float32}  # (max_batch,) — LayerNorm mean
    ln_rstd::Vector{Float32}  # (max_batch,) — LayerNorm 1/std
    # Pre-allocated result pool (avoids ~440K allocs/worker/iter)
    result_vecs::Vector{Vector{Float32}}  # max_batch policy vectors (pre-sized)
    results::Vector{Tuple{Vector{Float32}, Float32}}  # reusable result tuple vector
end

function FastBuffers(width::Int, nactions::Int, max_batch::Int)
    FastBuffers(
        zeros(Float32, width, max_batch),
        zeros(Float32, width, max_batch),
        zeros(Float32, width, max_batch),
        zeros(Float32, width, max_batch),
        zeros(Float32, nactions, max_batch),
        zeros(Float32, max_batch),
        zeros(Float32, max_batch),
        [Vector{Float32}(undef, nactions) for _ in 1:max_batch],
        Vector{Tuple{Vector{Float32}, Float32}}(undef, max_batch))
end

import LinearAlgebra: BLAS, mul!

"""In-place LayerNorm: out[:, j] = scale .* (x[:, j] - mean) / std + bias"""
function layernorm!(out::AbstractMatrix, x::AbstractMatrix, scale::Vector, bias::Vector,
                    mean_buf::Vector, rstd_buf::Vector, n::Int)
    d = size(x, 1)
    inv_d = 1.0f0 / d
    @inbounds for j in 1:n
        # Compute mean
        m = 0.0f0
        @simd for i in 1:d
            m += x[i, j]
        end
        m *= inv_d

        # Compute variance
        v = 0.0f0
        @simd for i in 1:d
            diff = x[i, j] - m
            v += diff * diff
        end
        rstd_buf[j] = 1.0f0 / sqrt(v * inv_d + 1.0f-5)
        mean_buf[j] = m
    end

    # Normalize, scale, shift
    @inbounds for j in 1:n
        m = mean_buf[j]
        rs = rstd_buf[j]
        @simd for i in 1:d
            out[i, j] = scale[i] * (x[i, j] - m) * rs + bias[i]
        end
    end
end

"""Fused LayerNorm + relu: out[:, j] = max(0, scale .* (x[:, j] - mean) / std + bias)"""
function layernorm_relu!(out::AbstractMatrix, x::AbstractMatrix, scale::Vector, bias::Vector,
                         mean_buf::Vector, rstd_buf::Vector, n::Int)
    d = size(x, 1)
    inv_d = 1.0f0 / d
    @inbounds for j in 1:n
        m = 0.0f0
        @simd for i in 1:d
            m += x[i, j]
        end
        m *= inv_d

        v = 0.0f0
        @simd for i in 1:d
            diff = x[i, j] - m
            v += diff * diff
        end
        rs = 1.0f0 / sqrt(v * inv_d + 1.0f-5)

        # Fused normalize + relu (single memory write)
        @simd for i in 1:d
            val = scale[i] * (x[i, j] - m) * rs + bias[i]
            out[i, j] = max(0.0f0, val)
        end
    end
end

"""In-place Dense + bias: out = W * x .+ b (uses BLAS gemm!)"""
function dense!(out::AbstractMatrix, W::Matrix, x::AbstractMatrix, b::Vector, n::Int)
    d_out = size(W, 1)
    # out = W * x
    @views mul!(out[1:d_out, 1:n], W, x[1:size(W, 2), 1:n])
    # out .+= b
    @inbounds for j in 1:n
        @simd for i in 1:d_out
            out[i, j] += b[i]
        end
    end
end

"""In-place Dense + bias + relu fused: out = max(0, W * x .+ b)"""
function dense_relu!(out::AbstractMatrix, W::Matrix, x::AbstractMatrix, b::Vector, n::Int)
    d_out = size(W, 1)
    @views mul!(out[1:d_out, 1:n], W, x[1:size(W, 2), 1:n])
    @inbounds for j in 1:n
        @simd for i in 1:d_out
            out[i, j] = max(0.0f0, out[i, j] + b[i])
        end
    end
end

"""Allocation-free forward pass returning (P_masked, V_equity, P_invalid).
All computation done in pre-allocated buffers. Only allocates the result tuple."""
function fast_forward_normalized!(fw::FastWeights, fb::FastBuffers,
                                   X::AbstractMatrix, A::AbstractMatrix, n::Int)
    w = size(fw.W_in, 1)  # width (128)

    # Input: Dense(indim, width) → LN+relu (fused)
    dense!(fb.h1, fw.W_in, X, fw.b_in, n)
    layernorm_relu!(fb.h2, fb.h1, fw.ln_in_s, fw.ln_in_b, fb.ln_mean, fb.ln_rstd, n)

    # Residual blocks (pre-activation): LN+relu → Dense → LN+relu → Dense → skip
    for blk in 1:fw.num_blocks
        # Save input for skip connection
        @inbounds for j in 1:n
            @simd for i in 1:w
                fb.skip[i, j] = fb.h2[i, j]
            end
        end

        # LN1+relu (fused) → Dense1
        layernorm_relu!(fb.h1, fb.h2, fw.res_ln1_s[blk], fw.res_ln1_b[blk], fb.ln_mean, fb.ln_rstd, n)
        dense!(fb.h2, fw.res_W1[blk], fb.h1, fw.res_b1[blk], n)

        # LN2+relu (fused) → Dense2
        layernorm_relu!(fb.h1, fb.h2, fw.res_ln2_s[blk], fw.res_ln2_b[blk], fb.ln_mean, fb.ln_rstd, n)
        dense!(fb.h2, fw.res_W2[blk], fb.h1, fw.res_b2[blk], n)

        # Skip connection: h2 += skip
        @inbounds for j in 1:n
            @simd for i in 1:w
                fb.h2[i, j] += fb.skip[i, j]
            end
        end
    end

    # Post-tower LN+relu (fused)
    layernorm_relu!(fb.h1, fb.h2, fw.ln_post_s, fw.ln_post_b, fb.ln_mean, fb.ln_rstd, n)
    # h1 now has the common trunk output

    # Value trunk: Dense → LN+relu (fused)
    dense!(fb.vt, fw.W_vt, fb.h1, fw.b_vt, n)
    layernorm_relu!(fb.h2, fb.vt, fw.ln_vt_s, fw.ln_vt_b, fb.ln_mean, fb.ln_rstd, n)
    # h2 now has value trunk output

    # 5 value heads: dot products → sigmoid → equity (all inline, no allocs)
    local V_equity = fb.ln_mean  # Reuse buffer for V output (1 × n)
    wvh1 = fw.W_vh[1]; wvh2 = fw.W_vh[2]; wvh3 = fw.W_vh[3]
    wvh4 = fw.W_vh[4]; wvh5 = fw.W_vh[5]
    bvh1 = fw.b_vh[1]; bvh2 = fw.b_vh[2]; bvh3 = fw.b_vh[3]
    bvh4 = fw.b_vh[4]; bvh5 = fw.b_vh[5]
    @inbounds for j in 1:n
        p_win = bvh1; p_gw = bvh2; p_bgw = bvh3; p_gl = bvh4; p_bgl = bvh5
        @simd for i in 1:w
            v = fb.h2[i, j]
            p_win += wvh1[i] * v
            p_gw += wvh2[i] * v
            p_bgw += wvh3[i] * v
            p_gl += wvh4[i] * v
            p_bgl += wvh5[i] * v
        end
        # Sigmoid + equity
        p_win = 1.0f0 / (1.0f0 + exp(-p_win))
        p_gw = 1.0f0 / (1.0f0 + exp(-p_gw))
        p_bgw = 1.0f0 / (1.0f0 + exp(-p_bgw))
        p_gl = 1.0f0 / (1.0f0 + exp(-p_gl))
        p_bgl = 1.0f0 / (1.0f0 + exp(-p_bgl))
        p_loss = 1.0f0 - p_win
        V_equity[j] = (p_win * (1.0f0 + p_gw + p_bgw) - p_loss * (1.0f0 + p_gl + p_bgl)) / 3.0f0
    end

    # Policy head: Dense → LN+relu (fused) layers
    @views for i in 1:fw.num_policy_layers
        dense!(fb.vt, fw.W_p[i], fb.h1, fw.b_p[i], n)
        layernorm_relu!(fb.h1, fb.vt, fw.ln_p_s[i], fw.ln_p_b[i], fb.ln_mean, fb.ln_rstd, n)
    end
    # Final policy Dense: (nactions, width) × (width, n) → (nactions, n)
    nact = size(fw.W_pout, 1)
    dense!(fb.p, fw.W_pout, fb.h1, fw.b_pout, n)

    # Softmax + action masking (in-place in fb.p)
    @inbounds for j in 1:n
        max_val = -Inf32
        for i in 1:nact
            if A[i, j] > 0.0f0 && fb.p[i, j] > max_val
                max_val = fb.p[i, j]
            end
        end
        s = 0.0f0
        for i in 1:nact
            if A[i, j] > 0.0f0
                fb.p[i, j] = exp(fb.p[i, j] - max_val)
                s += fb.p[i, j]
            else
                fb.p[i, j] = 0.0f0
            end
        end
        inv_s = 1.0f0 / (s + 1.0f-7)
        @simd for i in 1:nact
            fb.p[i, j] *= inv_s
        end
    end

    return fb.p, V_equity, n  # Policy matrix, value vector, batch size
end

# Extract weights from CPU network and create per-worker buffers
const FAST_WEIGHTS = FastWeights(cpu_network)
const USE_FAST_FORWARD = get(ENV, "FAST_FORWARD", "1") == "1"
if USE_FAST_FORWARD
    println("Fast forward: allocation-free ($(FAST_WEIGHTS.num_blocks) res blocks, $(FAST_WEIGHTS.num_policy_layers) policy layers)")
else
    println("Fast forward: disabled (using Flux forward_normalized)")
end
flush(stdout)

"""Refresh FastWeights from current cpu_network (after weight sync)."""
function refresh_fast_weights!()
    fw_new = FastWeights(cpu_network)
    # Copy all weight data in-place
    copyto!(FAST_WEIGHTS.W_in, fw_new.W_in)
    copyto!(FAST_WEIGHTS.b_in, fw_new.b_in)
    copyto!(FAST_WEIGHTS.ln_in_s, fw_new.ln_in_s)
    copyto!(FAST_WEIGHTS.ln_in_b, fw_new.ln_in_b)
    for i in 1:FAST_WEIGHTS.num_blocks
        copyto!(FAST_WEIGHTS.res_W1[i], fw_new.res_W1[i])
        copyto!(FAST_WEIGHTS.res_b1[i], fw_new.res_b1[i])
        copyto!(FAST_WEIGHTS.res_ln1_s[i], fw_new.res_ln1_s[i])
        copyto!(FAST_WEIGHTS.res_ln1_b[i], fw_new.res_ln1_b[i])
        copyto!(FAST_WEIGHTS.res_W2[i], fw_new.res_W2[i])
        copyto!(FAST_WEIGHTS.res_b2[i], fw_new.res_b2[i])
        copyto!(FAST_WEIGHTS.res_ln2_s[i], fw_new.res_ln2_s[i])
        copyto!(FAST_WEIGHTS.res_ln2_b[i], fw_new.res_ln2_b[i])
    end
    copyto!(FAST_WEIGHTS.ln_post_s, fw_new.ln_post_s)
    copyto!(FAST_WEIGHTS.ln_post_b, fw_new.ln_post_b)
    copyto!(FAST_WEIGHTS.W_vt, fw_new.W_vt)
    copyto!(FAST_WEIGHTS.b_vt, fw_new.b_vt)
    copyto!(FAST_WEIGHTS.ln_vt_s, fw_new.ln_vt_s)
    copyto!(FAST_WEIGHTS.ln_vt_b, fw_new.ln_vt_b)
    for i in 1:5
        copyto!(FAST_WEIGHTS.W_vh[i], fw_new.W_vh[i])
    end
    copyto!(FAST_WEIGHTS.b_vh, fw_new.b_vh)
    for i in 1:FAST_WEIGHTS.num_policy_layers
        copyto!(FAST_WEIGHTS.W_p[i], fw_new.W_p[i])
        copyto!(FAST_WEIGHTS.b_p[i], fw_new.b_p[i])
        copyto!(FAST_WEIGHTS.ln_p_s[i], fw_new.ln_p_s[i])
        copyto!(FAST_WEIGHTS.ln_p_b[i], fw_new.ln_p_b[i])
    end
    copyto!(FAST_WEIGHTS.W_pout, fw_new.W_pout)
    copyto!(FAST_WEIGHTS.b_pout, fw_new.b_pout)
end

#####
##### Bear-off exact table lookup (k=6 two-sided database)
#####

const USE_BEAROFF = ARGS["use_bearoff"]

# Load BearoffK6 module from local BackgammonNet.jl dev repo
# (bearoff_k6.jl and 5.5GB data files are not in the published package)
const BEAROFF_SRC_DIR = let
    # Try local dev repo first, fall back to installed package
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

const BEAROFF_TABLE = if USE_BEAROFF
    table_dir = joinpath(BEAROFF_SRC_DIR, "..", "tools", "bearoff_twosided", "bearoff_k6_twosided")
    if !isdir(table_dir)
        error("Bear-off table not found at: $table_dir\nDownload or generate the k=6 two-sided database first.")
    end
    println("Loading k=6 bear-off table from $table_dir ...")
    t = BearoffTable(table_dir)
    println("  c14: $(t.c14_pairs) pairs ($(round(length(t.c14_data)/1e9, digits=1)) GB)")
    println("  c15: $(t.c15_pairs) pairs ($(round(length(t.c15_data)/1e9, digits=1)) GB)")
    flush(stdout)
    t
else
    nothing
end

"""Look up exact bear-off equity from precomputed table.
Returns (value, equity_5tuple) from current player's perspective."""
function bearoff_table_equity(game::BackgammonNet.BackgammonGame)
    r = BearoffK6.lookup(BEAROFF_TABLE, game)
    eq = Float32[r.p_win, r.p_gammon_win, r.p_bg_win, r.p_gammon_loss, r.p_bg_loss]
    value = BearoffK6.compute_equity(r)
    return (value=value, equity=eq)
end

"""Create bear-off evaluator for MCTS chance nodes.
Returns exact pre-dice value at bear-off chance nodes, nothing otherwise.
Value is from current player's perspective (matches MCTS terminal_value convention)."""
function make_bearoff_evaluator(table)
    table === nothing && return nothing
    return function(game_env)
        bg = game_env.game  # Unwrap GameEnv → BackgammonGame
        if BearoffK6.is_bearoff_position(bg.p0, bg.p1)
            r = BearoffK6.lookup(table, bg)
            return Float64(BearoffK6.compute_equity(r))
        end
        return nothing
    end
end

const BEAROFF_EVALUATOR = USE_BEAROFF ? make_bearoff_evaluator(BEAROFF_TABLE) : nothing

#####
##### Bear-off accuracy benchmark (fixed 10K positions, evaluated each iteration)
#####

"""Generate a random bear-off position (both players in home board, no contact)."""
function _random_bearoff_board(rng)
    # P0: distribute up to 15 checkers across points 19-24, rest are off (index 25)
    p0 = UInt128(0)
    remaining = rand(rng, 1:15)  # at least 1 checker on board
    off_p0 = 15 - remaining
    p0 = p0 | (UInt128(off_p0) << (25 << 2))  # P0 off
    for pt in 19:23
        c = rand(rng, 0:remaining)
        if c > 0
            p0 = p0 | (UInt128(c) << (pt << 2))
        end
        remaining -= c
    end
    if remaining > 0
        p0 = p0 | (UInt128(remaining) << (24 << 2))
    end

    # P1: distribute up to 15 checkers across points 1-6, rest are off (index 0)
    p1 = UInt128(0)
    remaining = rand(rng, 1:15)
    off_p1 = 15 - remaining
    p1 = p1 | (UInt128(off_p1) << (0 << 2))  # P1 off
    for pt in 1:5
        c = rand(rng, 0:remaining)
        if c > 0
            p1 = p1 | (UInt128(c) << (pt << 2))
        end
        remaining -= c
    end
    if remaining > 0
        p1 = p1 | (UInt128(remaining) << (6 << 2))
    end

    cp = Int8(rand(rng, 0:1))
    dice = SVector{2, Int8}(Int8(rand(rng, 1:6)), Int8(rand(rng, 1:6)))
    game = BackgammonNet.BackgammonGame(p0, p1, dice, Int8(1), cp, false, 0.0f0;
                                         obs_type=:minimal_flat)
    return game
end

# State dimension for pre-allocated buffers
const _state_dim = let env = GI.init(gspec); length(vec(GI.vectorize_state(gspec, GI.current_state(env)))); end

# Pre-generate fixed 10K bear-off positions for benchmark
const BEAROFF_EVAL_N = 10_000
const BEAROFF_EVAL_DATA = if USE_BEAROFF
    println("Generating $BEAROFF_EVAL_N fixed bear-off benchmark positions ...")
    bo_rng = MersenneTwister(12345)  # Fixed seed for reproducibility
    bo_games = [_random_bearoff_board(bo_rng) for _ in 1:BEAROFF_EVAL_N]

    # Pre-vectorize into batch matrix and compute exact table values
    bo_X = zeros(Float32, _state_dim, BEAROFF_EVAL_N)
    bo_A = zeros(Float32, NUM_ACTIONS, BEAROFF_EVAL_N)
    bo_exact_equity = zeros(Float32, 5, BEAROFF_EVAL_N)  # 5-way equity
    bo_exact_value = zeros(Float32, BEAROFF_EVAL_N)       # scalar equity

    for (i, g) in enumerate(bo_games)
        vectorize_state_into!(@view(bo_X[:, i]), gspec, g)
        if !BackgammonNet.game_terminated(g)
            legal = BackgammonNet.legal_actions(g)
            for a in legal
                if 1 <= a <= NUM_ACTIONS
                    bo_A[a, i] = 1.0f0
                end
            end
        end
        r = BearoffK6.lookup(BEAROFF_TABLE, g)
        bo_exact_value[i] = BearoffK6.compute_equity(r)
        bo_exact_equity[1, i] = r.p_win
        bo_exact_equity[2, i] = r.p_gammon_win
        bo_exact_equity[3, i] = r.p_bg_win
        bo_exact_equity[4, i] = r.p_gammon_loss
        bo_exact_equity[5, i] = r.p_bg_loss
    end
    println("  Bear-off benchmark ready ($(BEAROFF_EVAL_N) positions)")
    flush(stdout)
    (X=bo_X, A=bo_A, exact_equity=bo_exact_equity, exact_value=bo_exact_value)
else
    nothing
end

"""Evaluate NN value head accuracy on fixed bear-off positions.
Returns (mae_equity, mae_per_head) — all mean absolute errors."""
function eval_bearoff_accuracy(net)
    data = BEAROFF_EVAL_DATA
    data === nothing && return nothing

    head_names = ["P(win)", "P(gw|w)", "P(bgw|w)", "P(gl|l)", "P(bgl|l)"]

    # Move data to same device as network (GPU if USE_GPU)
    X_eval = USE_GPU ? Flux.gpu(data.X) : data.X
    A_eval = USE_GPU ? Flux.gpu(data.A) : data.A

    if net isa FluxLib.FCResNetMultiHead
        # Multihead: get all 5 equity heads
        _, V̂_win, V̂_gw, V̂_bgw, V̂_gl, V̂_bgl, _ =
            FluxLib.forward_normalized_multihead(net, X_eval, A_eval)

        # Move results back to CPU for comparison
        V̂_win = Array(V̂_win); V̂_gw = Array(V̂_gw); V̂_bgw = Array(V̂_bgw)
        V̂_gl = Array(V̂_gl); V̂_bgl = Array(V̂_bgl)

        # Overall equity MAE (scalar, range [-3, +3])
        equity_pred = FluxLib.compute_equity(V̂_win, V̂_gw, V̂_bgw, V̂_gl, V̂_bgl)
        mae_value = Float32(mean(abs.(vec(equity_pred) .- data.exact_value)))

        # Per-head MAE
        preds = [V̂_win, V̂_gw, V̂_bgw, V̂_gl, V̂_bgl]
        mae_heads = Float32[Float32(mean(abs.(vec(preds[h]) .- data.exact_equity[h, :]))) for h in 1:5]
    else
        # Single value head fallback
        _, V̂, _ = Network.forward_normalized(net, X_eval, A_eval)
        V̂ = Array(V̂)
        mae_value = Float32(mean(abs.(vec(V̂) .- data.exact_value)))
        mae_heads = Float32[]
    end

    return (mae_value=mae_value, mae_heads=mae_heads, head_names=head_names)
end

#####
##### Helper functions
#####

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

"""Sample a hard game outcome from bear-off probability distribution.
Returns (value, equity_5tuple) with hard 0/1 targets, sampled from exact probs."""
function sample_bearoff_outcome(probs::Vector{Float32}, wp::Bool, rng)
    # probs = [p_win, p_gw|w, p_bgw|w, p_gl|l, p_bgl|l] from white's perspective
    p_win_white = probs[1]
    p_gw = probs[2]; p_bgw = probs[3]
    p_gl = probs[4]; p_bgl = probs[5]

    # Sample: did white win?
    white_won = rand(rng) < p_win_white
    is_gammon = false
    is_backgammon = false

    if white_won
        # Sample gammon/backgammon conditional on win
        r = rand(rng)
        if r < p_bgw
            is_backgammon = true; is_gammon = true
        elseif r < p_gw
            is_gammon = true
        end
    else
        # Sample gammon/backgammon conditional on loss
        r = rand(rng)
        if r < p_bgl
            is_backgammon = true; is_gammon = true
        elseif r < p_gl
            is_gammon = true
        end
    end

    # Build hard equity tuple from current player's perspective
    won = white_won == wp
    eq = zeros(Float32, 5)
    if won
        eq[1] = 1.0f0  # P(win) = 1
        eq[2] = is_gammon ? 1.0f0 : 0.0f0
        eq[3] = is_backgammon ? 1.0f0 : 0.0f0
    else
        eq[4] = is_gammon ? 1.0f0 : 0.0f0
        eq[5] = is_backgammon ? 1.0f0 : 0.0f0
    end

    # Compute value: points won/lost
    multiplier = is_backgammon ? 3.0f0 : (is_gammon ? 2.0f0 : 1.0f0)
    z = won ? multiplier : -multiplier

    return (value=z, equity=eq)
end

function convert_trace_to_samples(gspec, states, policies, trace_actions, rewards, is_chance, final_reward, outcome; rng=nothing)
    n = length(states)
    samples = []
    num_actions = GI.num_actions(gspec)

    for i in 1:n
        state = states[i]
        policy = policies[i]
        actions = trace_actions[i]
        is_ch = is_chance[i]
        wp = GI.white_playing(gspec, state)

        # Default: use actual game outcome
        z = wp ? final_reward : -final_reward
        eq = zeros(Float32, 5)
        has_eq = false
        if !isnothing(outcome)
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

        # Bear-off positions: replace with exact table equity (training targets only)
        # For chance nodes (pre-dice), table is mathematically exact.
        # For decision nodes (post-dice), table is noisy but unbiased.
        if USE_BEAROFF && state isa BackgammonNet.BackgammonGame &&
                BearoffK6.is_bearoff_position(state.p0, state.p1)
            bo = bearoff_table_equity(state)
            z = wp ? bo.value : -bo.value
            eq = copy(bo.equity)
            if !wp
                # Flip equity to current player's perspective
                eq = Float32[1.0f0 - bo.equity[1], bo.equity[4], bo.equity[5], bo.equity[2], bo.equity[3]]
            end
            has_eq = true
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

        push!(samples, (
            state=state_vec,
            policy=full_policy,
            value=z,
            equity=eq,
            has_equity=has_eq,
            is_chance=is_ch,
        ))
    end

    return samples
end

#####
##### Inference server (thread-safe channels, zero-copy)
#####

const REQ_CHAN = Channel{Any}(256)
const RESP_CHANS = Dict{Int, Channel{Any}}()

# _state_dim defined earlier (before bear-off eval section)

"""
Inference server loop: drains request queue, batches all pending requests,
runs single GPU forward pass, distributes results to worker response channels.

Uses blocking take! (no polling). Send `nothing` on REQ_CHAN to shut down.
"""
function inference_server_loop!()
    while true
        pending = []

        # Block until first request (kernel-level wait, zero latency)
        first = take!(REQ_CHAN)
        first === nothing && return  # Shutdown sentinel
        push!(pending, first)

        # Immediate non-blocking drain: grab everything already queued (no wait window)
        shutdown = false
        while isready(REQ_CHAN) && length(pending) < 1024
            req = take!(REQ_CHAN)
            if req === nothing
                shutdown = true
                break
            end
            push!(pending, req)
        end

        # Concatenate all requests (hcat handles SubArray views natively)
        X_all = hcat([r.X for r in pending]...)
        A_all = hcat([r.A for r in pending]...)

        # Single GPU forward pass for ALL workers' requests
        Xnet, Anet = Network.convert_input_tuple(network, (X_all, A_all))
        P_raw, V, _ = Network.convert_output_tuple(
            network, Network.forward_normalized(network, Xnet, Anet))

        # Distribute results to each worker's response channel
        offset = 0
        for req in pending
            n = size(req.X, 2)
            results = Vector{Tuple{Vector{Float32}, Float32}}(undef, n)
            for i in 1:n
                col = offset + i
                legal = @view(A_all[:, col]) .> 0
                results[i] = (P_raw[legal, col], V[1, col])
            end
            put!(RESP_CHANS[req.wid], results)
            offset += n
        end

        shutdown && return
    end
end

function start_inference_server!()
    return Threads.@spawn inference_server_loop!()
end

function stop_inference_server!(task)
    # Drain any pending requests, then send shutdown sentinel
    while isready(REQ_CHAN)
        take!(REQ_CHAN)
    end
    put!(REQ_CHAN, nothing)
    wait(task)
end

#####
##### Lock-free GPU inference mailbox
#####

"""Pre-allocated per-worker buffers for lock-free GPU inference.
Workers write states/masks, server writes policy/values. Atomic flags for sync."""
struct InferenceSlot
    X::Matrix{Float32}        # (state_dim, max_batch) worker writes states
    A::Matrix{Float32}        # (num_actions, max_batch) worker writes masks
    P_out::Matrix{Float32}    # (num_actions, max_batch) server writes policy
    V_out::Vector{Float32}    # (max_batch,) server writes values
    n::Threads.Atomic{Int}    # batch size for this request
    status::Threads.Atomic{Int}  # 0=idle, 1=request_ready, 2=result_ready
end

function InferenceSlot(state_dim::Int, num_actions::Int, max_batch::Int)
    InferenceSlot(
        zeros(Float32, state_dim, max_batch),
        zeros(Float32, num_actions, max_batch),
        zeros(Float32, num_actions, max_batch),
        zeros(Float32, max_batch),
        Threads.Atomic{Int}(0),
        Threads.Atomic{Int}(0))
end

"""Lock-free inference mailbox. One slot per worker, plus combined buffers for batching."""
mutable struct InferenceMailbox
    slots::Vector{InferenceSlot}
    X_combined::Matrix{Float32}  # Pre-allocated combined input buffer
    A_combined::Matrix{Float32}
    running::Threads.Atomic{Int}  # 0=stopped, 1=running
end

function InferenceMailbox(num_workers::Int, state_dim::Int, num_actions::Int, max_batch::Int)
    slots = [InferenceSlot(state_dim, num_actions, max_batch) for _ in 1:num_workers]
    total = num_workers * max_batch
    InferenceMailbox(slots,
        zeros(Float32, state_dim, total),
        zeros(Float32, num_actions, total),
        Threads.Atomic{Int}(0))
end

"""Lock-free GPU inference server loop.
Polls worker slots for requests, batches them, runs single GPU forward, distributes results.
No channels or locks — uses atomic flags for synchronization (safe on x86 TSO)."""
function mailbox_server_loop!(mailbox::InferenceMailbox, net;
                              min_ready::Int=0, agg_wait_ns::Int=0)
    nslots = length(mailbox.slots)
    ready_buf = Vector{Int}(undef, nslots)
    sizes_buf = Vector{Int}(undef, nslots)
    min_slots = min_ready > 0 ? min_ready : max(1, nslots ÷ 2)
    max_wait_ns = agg_wait_ns > 0 ? agg_wait_ns : 2_000_000  # default 2ms
    batch_count = 0
    total_states = 0
    total_slots_served = 0
    t_server_start = time()

    while mailbox.running[] == 1
        # Scan for ready requests
        nready = 0
        total_n = 0
        for i in 1:nslots
            if mailbox.slots[i].status[] == 1
                nready += 1
                ready_buf[nready] = i
                sizes_buf[nready] = mailbox.slots[i].n[]
                total_n += sizes_buf[nready]
            end
        end

        if nready == 0
            yield()
            continue
        end

        # Aggregation window: wait for more workers (up to 2ms) to improve batching
        t_start = time_ns()
        while nready < min_slots && (time_ns() - t_start) < max_wait_ns
            yield()
            for i in 1:nslots
                if mailbox.slots[i].status[] == 1
                    already = false
                    for j in 1:nready
                        if ready_buf[j] == i; already = true; break; end
                    end
                    if !already
                        nready += 1
                        ready_buf[nready] = i
                        sizes_buf[nready] = mailbox.slots[i].n[]
                        total_n += sizes_buf[nready]
                    end
                end
            end
        end

        # Copy data into combined buffer
        offset = 0
        for k in 1:nready
            slot = mailbox.slots[ready_buf[k]]
            n = sizes_buf[k]
            @views mailbox.X_combined[:, offset+1:offset+n] .= slot.X[:, 1:n]
            @views mailbox.A_combined[:, offset+1:offset+n] .= slot.A[:, 1:n]
            offset += n
        end

        # Single GPU forward pass for ALL workers' requests
        X_batch = mailbox.X_combined[:, 1:total_n]
        A_batch = mailbox.A_combined[:, 1:total_n]
        Xnet, Anet = Network.convert_input_tuple(net, (X_batch, A_batch))
        P_raw, V, _ = Network.convert_output_tuple(
            net, Network.forward_normalized(net, Xnet, Anet))

        # Distribute results back to slots
        offset = 0
        for k in 1:nready
            slot = mailbox.slots[ready_buf[k]]
            n = sizes_buf[k]
            @views slot.P_out[:, 1:n] .= P_raw[:, offset+1:offset+n]
            for j in 1:n
                slot.V_out[j] = V[1, offset+j]
            end
            offset += n
            slot.status[] = 2  # Signal result ready
        end

        batch_count += 1
        total_states += total_n
        total_slots_served += nready
        if batch_count == 10
            elapsed = time() - t_server_start
            @info "Server stats (first 10 batches)" avg_slots=round(total_slots_served/batch_count, digits=1) avg_states=round(total_states/batch_count, digits=0) batches_per_sec=round(batch_count/elapsed, digits=1)
        end
    end

    if batch_count > 0
        elapsed = time() - t_server_start
        @info "Server final" total_batches=batch_count avg_slots=round(total_slots_served/batch_count, digits=1) avg_states=round(total_states/batch_count, digits=0) throughput_states_per_sec=round(total_states/elapsed, digits=0)
    end
end

function start_mailbox_server!(mailbox::InferenceMailbox, net;
                               min_ready::Int=0, agg_wait_ns::Int=0)
    mailbox.running[] = 1
    for slot in mailbox.slots
        slot.status[] = 0
    end
    return Threads.@spawn mailbox_server_loop!(mailbox, net;
                                                min_ready=min_ready, agg_wait_ns=agg_wait_ns)
end

function stop_mailbox_server!(mailbox::InferenceMailbox, task)
    mailbox.running[] = 0
    wait(task)
end

# GPU inference disabled by default: for 283K model, GPU launch+transfer overhead > CPU BLAS time.
# Enable with GPU_INFERENCE=1 for larger models where GPU forward pass dominates overhead.
const USE_GPU_INFERENCE = get(ENV, "GPU_INFERENCE", "0") == "1"

# CPU inference server: centralizes all BLAS calls on one thread.
# Disabled by default (slower than per-worker BLAS for 283K model due to serialization).
# Enable with CPU_SERVER=1 for testing or models where contention dominates.
const USE_CPU_SERVER = !USE_GPU_INFERENCE && get(ENV, "CPU_SERVER", "0") == "1"

const MAILBOX = (USE_GPU_INFERENCE || USE_CPU_SERVER) ?
    InferenceMailbox(NUM_WORKERS, _state_dim, NUM_ACTIONS, INFERENCE_BATCH_SIZE + 1) : nothing

const INFERENCE_MODE = if USE_CPU_SERVER
    "CPU (centralized server, lock-free mailbox)"
elseif USE_GPU_INFERENCE
    "GPU (lock-free mailbox)"
else
    "CPU (per-worker BLAS)"
end
println("Inference mode: $INFERENCE_MODE")
flush(stdout)

#####
##### Worker functions (self-contained, run on worker threads)
#####

"""Sample a chance outcome index from (outcome, probability) pairs."""
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

"""Core game-playing loop. Uses GPU mailbox inference, channel server, or local CPU inference."""
function _play_games_loop(vworker_id::Int, games_claimed::Threads.Atomic{Int}, total_games::Int,
                          req_chan::Channel, resp_chan::Channel, rng::MersenneTwister;
                          mb_slot::Union{InferenceSlot, Nothing}=nothing,
                          fast_bufs::Union{FastBuffers, Nothing}=nothing)
    # Use mailbox slot buffers for GPU inference, else allocate local
    max_batch = INFERENCE_BATCH_SIZE + 1
    X_buf = mb_slot !== nothing ? mb_slot.X : zeros(Float32, _state_dim, max_batch)
    A_buf = mb_slot !== nothing ? mb_slot.A : zeros(Float32, NUM_ACTIONS, max_batch)

    # Timing accumulators
    t_vectorize = 0.0
    t_gpu_wait = 0.0
    t_mcts_cpu = 0.0
    n_oracle_calls = 0

    function batch_oracle(states::Vector)
        n = length(states)
        n == 0 && return Tuple{Vector{Float32}, Float32}[]
        t0 = time_ns()

        # Vectorize all states for NN inference (self-play always uses NN, no bear-off table)
        nn_count = 0
        for (i, s) in enumerate(states)
            nn_count += 1
            slot = nn_count
            vectorize_state_into!(@view(X_buf[:, slot]), gspec, s)
            a_col = @view(A_buf[:, slot])
            fill!(a_col, 0.0f0)
            if !BackgammonNet.game_terminated(s)
                legal = BackgammonNet.legal_actions(s)
                @inbounds for action in legal
                    if 1 <= action <= NUM_ACTIONS
                        a_col[action] = 1.0f0
                    end
                end
            end
        end
        t1 = time_ns()

        # NN inference for non-bearoff states
        local nn_results
        if nn_count == 0
            nn_results = Tuple{Vector{Float32}, Float32}[]
        elseif fast_bufs !== nothing
            # Allocation-free CPU inference + result pooling
            P_buf, V_buf, _ = fast_forward_normalized!(FAST_WEIGHTS, fast_bufs, X_buf, A_buf, nn_count)
            nn_results = Vector{Tuple{Vector{Float32}, Float32}}(undef, nn_count)
            @inbounds for i in 1:nn_count
                pv = fast_bufs.result_vecs[i]
                k = 0
                for a in 1:NUM_ACTIONS
                    if A_buf[a, i] > 0.0f0
                        k += 1
                        pv[k] = P_buf[a, i]
                    end
                end
                resize!(pv, k)
                nn_results[i] = (pv, V_buf[i])
            end
        elseif mb_slot !== nothing
            # GPU mailbox inference
            mb_slot.n[] = nn_count
            mb_slot.status[] = 1
            while mb_slot.status[] != 2; yield(); end
            nn_results = Vector{Tuple{Vector{Float32}, Float32}}(undef, nn_count)
            for i in 1:nn_count
                legal = @view(A_buf[:, i]) .> 0
                nn_results[i] = (mb_slot.P_out[legal, i], mb_slot.V_out[i])
            end
            mb_slot.status[] = 0
        elseif USE_GPU_INFERENCE
            put!(req_chan, (wid=vworker_id, X=@view(X_buf[:, 1:nn_count]), A=@view(A_buf[:, 1:nn_count])))
            nn_results = take!(resp_chan)
        else
            # Local CPU inference (Flux)
            X = @view(X_buf[:, 1:nn_count])
            A = @view(A_buf[:, 1:nn_count])
            P_raw, V, _ = Network.convert_output_tuple(
                cpu_network, Network.forward_normalized(cpu_network, X, A))
            nn_results = Vector{Tuple{Vector{Float32}, Float32}}(undef, nn_count)
            for i in 1:nn_count
                legal = @view(A[:, i]) .> 0
                nn_results[i] = (P_raw[legal, i], V[1, i])
            end
        end
        t2 = time_ns()

        results = nn_results

        t_vectorize += (t1 - t0)
        t_gpu_wait += (t2 - t1)
        n_oracle_calls += 1
        return results
    end
    single_oracle(state) = batch_oracle([state])[1]

    mcts_params = MctsParams(
        num_iters_per_turn=MCTS_ITERS,
        cpuct=2.0,
        temperature=ConstSchedule(1.0),
        dirichlet_noise_ϵ=0.25,
        dirichlet_noise_α=0.3)
    player = BatchedMCTS.BatchedMctsPlayer(
        gspec, single_oracle, mcts_params;
        batch_size=INFERENCE_BATCH_SIZE, batch_oracle=batch_oracle,
        bearoff_evaluator=BEAROFF_EVALUATOR)

    all_samples = []
    while Threads.atomic_add!(games_claimed, 1) < total_games
        env = GI.init(gspec)
        if hasproperty(env, :rng)
            env.rng = rng
        end

        trace_states = []
        trace_policies = []
        trace_actions = Vector{Int}[]  # Store actions list to avoid GI.init in trace conversion
        trace_rewards = Float32[]
        trace_is_chance = Bool[]

        while !GI.game_terminated(env)
            # Handle chance nodes: record state for value-head training, then sample dice
            if GI.is_chance_node(env)
                state = GI.current_state(env)
                push!(trace_states, state)
                push!(trace_policies, Float32[])
                push!(trace_actions, Int[])
                push!(trace_is_chance, true)
                push!(trace_rewards, 0.0f0)
                outcomes = GI.chance_outcomes(env)
                idx = _sample_chance(rng, outcomes)
                GI.apply_chance!(env, outcomes[idx][1])
                continue
            end

            # Decision node: MCTS think + record trace
            state = GI.current_state(env)
            push!(trace_states, state)

            t_m0 = time_ns()
            actions, policy = BatchedMCTS.think(player, env)
            t_m1 = time_ns()
            t_mcts_cpu += (t_m1 - t_m0)
            push!(trace_policies, Float32.(policy))
            push!(trace_actions, actions)
            push!(trace_is_chance, false)
            action = actions[sample_from_policy(policy, rng)]
            GI.play!(env, action)

            push!(trace_rewards, 0.0f0)
        end

        BatchedMCTS.reset_player!(player)
        final_reward = Float32(GI.white_reward(env))
        outcome = GI.game_outcome(env)
        samples = convert_trace_to_samples(
            gspec, trace_states, trace_policies, trace_actions, trace_rewards, trace_is_chance,
            final_reward, outcome; rng=rng)
        append!(all_samples, samples)
    end

    return (samples=all_samples, t_vectorize=t_vectorize, t_gpu_wait=t_gpu_wait,
            t_mcts_cpu=t_mcts_cpu, n_oracle_calls=n_oracle_calls)
end

const ASYNC_GAMES_PER_THREAD = 1  # Green threads per OS thread (1 = disabled)

"""Play games using green-thread latency hiding. Each OS thread runs multiple @async
game loops. mb_slot: mailbox slot for lock-free GPU inference (nothing = CPU or channel)."""
function worker_play_games(worker_id::Int, games_claimed::Threads.Atomic{Int}, total_games::Int,
                           req_chan::Channel, rng::MersenneTwister;
                           mb_slot::Union{InferenceSlot, Nothing}=nothing)
    sub_tasks = Task[]
    for k in 1:ASYNC_GAMES_PER_THREAD
        vid = (worker_id - 1) * ASYNC_GAMES_PER_THREAD + k
        if mb_slot !== nothing
            ch = Channel{Any}(1)  # Dummy, not used with mailbox
        elseif USE_GPU_INFERENCE
            ch = RESP_CHANS[vid]  # Pre-created by parallel_self_play
        else
            ch = Channel{Any}(4)  # Local only (CPU inference doesn't use channels)
        end
        sub_rng = MersenneTwister(rand(rng, UInt))
        # Create per-green-thread fast forward buffers (avoids cross-thread sharing)
        fb = USE_FAST_FORWARD ? FastBuffers(NET_WIDTH, NUM_ACTIONS, INFERENCE_BATCH_SIZE + 1) : nothing
        t = @async _play_games_loop(vid, games_claimed, total_games, req_chan, ch, sub_rng;
                                     mb_slot=mb_slot, fast_bufs=fb)
        push!(sub_tasks, t)
    end

    all_samples = []
    total_vectorize = 0.0
    total_gpu_wait = 0.0
    total_mcts = 0.0
    total_oracle = 0
    for t in sub_tasks
        result = fetch(t)
        append!(all_samples, result.samples)
        total_vectorize += result.t_vectorize
        total_gpu_wait += result.t_gpu_wait
        total_mcts += result.t_mcts_cpu
        total_oracle += result.n_oracle_calls
    end

    if worker_id <= 3
        t_mcts_pure = total_mcts - total_vectorize - total_gpu_wait
        @info "Worker $worker_id timing ($(ASYNC_GAMES_PER_THREAD) green threads)" vectorize_s=round(total_vectorize/1e9, digits=2) gpu_wait_s=round(total_gpu_wait/1e9, digits=2) mcts_total_s=round(total_mcts/1e9, digits=2) mcts_pure_s=round(t_mcts_pure/1e9, digits=2) oracle_calls=total_oracle
    end

    return all_samples
end

"""Core eval game loop for a single green thread."""
function _eval_games_loop(vworker_id::Int, games_claimed::Threads.Atomic{Int}, total_games::Int,
                          play_as_white::Bool,
                          req_chan::Channel, resp_chan::Channel, rng::MersenneTwister)
    max_batch = INFERENCE_BATCH_SIZE + 1
    X_buf = zeros(Float32, _state_dim, max_batch)
    A_buf = zeros(Float32, NUM_ACTIONS, max_batch)

    function batch_oracle(states::Vector)
        n = length(states)
        n == 0 && return Tuple{Vector{Float32}, Float32}[]
        for (i, s) in enumerate(states)
            vectorize_state_into!(@view(X_buf[:, i]), gspec, s)
            a_col = @view(A_buf[:, i])
            fill!(a_col, 0.0f0)
            if !BackgammonNet.game_terminated(s)
                legal = BackgammonNet.legal_actions(s)
                @inbounds for action in legal
                    if 1 <= action <= NUM_ACTIONS
                        a_col[action] = 1.0f0
                    end
                end
            end
        end
        put!(req_chan, (wid=vworker_id, X=@view(X_buf[:, 1:n]), A=@view(A_buf[:, 1:n])))
        return take!(resp_chan)
    end
    single_oracle(state) = batch_oracle([state])[1]

    eval_mcts_params = MctsParams(
        num_iters_per_turn=MCTS_ITERS,
        cpuct=1.5,
        temperature=ConstSchedule(0.0),
        dirichlet_noise_ϵ=0.0,
        dirichlet_noise_α=1.0)
    eval_player = BatchedMCTS.BatchedMctsPlayer(
        gspec, single_oracle, eval_mcts_params;
        batch_size=INFERENCE_BATCH_SIZE, batch_oracle=batch_oracle,
        bearoff_evaluator=BEAROFF_EVALUATOR)

    rewards = Float64[]
    while Threads.atomic_add!(games_claimed, 1) < total_games
        env = GI.init(gspec)
        if hasproperty(env, :rng)
            env.rng = rng
        end

        while !GI.game_terminated(env)
            if GI.is_chance_node(env)
                outcomes = GI.chance_outcomes(env)
                idx = _sample_chance(rng, outcomes)
                GI.apply_chance!(env, outcomes[idx][1])
            elseif play_as_white == GI.white_playing(env)
                actions, policy = BatchedMCTS.think(eval_player, env)
                GI.play!(env, actions[argmax(policy)])
            else
                GI.play!(env, rand(rng, GI.available_actions(env)))
            end
        end

        reward = GI.white_reward(env)
        if !play_as_white
            reward = -reward
        end
        push!(rewards, reward)
        BatchedMCTS.reset_player!(eval_player)
    end

    return rewards
end

"""Play eval games with green-thread latency hiding.
RESP_CHANS entries must be pre-created before calling this (thread safety)."""
function worker_eval_games(worker_id::Int, games_claimed::Threads.Atomic{Int}, total_games::Int,
                           play_as_white::Bool, req_chan::Channel, rng::MersenneTwister)
    sub_tasks = Task[]
    for k in 1:ASYNC_GAMES_PER_THREAD
        vid = (worker_id - 1) * ASYNC_GAMES_PER_THREAD + k + 1000  # offset to avoid self-play IDs
        ch = RESP_CHANS[vid]  # Pre-created by parallel_eval
        sub_rng = MersenneTwister(rand(rng, UInt))
        t = @async _eval_games_loop(vid, games_claimed, total_games, play_as_white, req_chan, ch, sub_rng)
        push!(sub_tasks, t)
    end
    all_rewards = Float64[]
    for t in sub_tasks
        append!(all_rewards, fetch(t))
    end
    return all_rewards
end

#####
##### Parallel self-play and evaluation
#####

"""Spawn worker threads for self-play with work-stealing.
Uses lock-free mailbox for GPU inference when available, else CPU per-worker BLAS."""
function parallel_self_play(num_games::Int)
    games_claimed = Threads.Atomic{Int}(0)

    # Start inference server (CPU centralized or GPU mailbox/channel)
    local server_task
    if MAILBOX !== nothing
        server_net = USE_CPU_SERVER ? cpu_network : network
        # CPU server: moderate wait (BLAS time scales linearly, batching doesn't help throughput)
        # GPU server: aggressive wait (GPU kernel overhead is fixed, larger batches amortize it)
        mr = USE_CPU_SERVER ? max(1, NUM_WORKERS ÷ 2) : max(1, NUM_WORKERS - 1)
        mw = USE_CPU_SERVER ? 1_000_000 : 5_000_000  # 1ms CPU, 5ms GPU
        server_task = start_mailbox_server!(MAILBOX, server_net; min_ready=mr, agg_wait_ns=mw)
    elseif USE_GPU_INFERENCE
        for w in 1:NUM_WORKERS
            for k in 1:ASYNC_GAMES_PER_THREAD
                vid = (w - 1) * ASYNC_GAMES_PER_THREAD + k
                RESP_CHANS[vid] = Channel{Any}(4)
            end
        end
        server_task = start_inference_server!()
    end

    tasks = Task[]
    for w in 1:NUM_WORKERS
        rng = MersenneTwister(isnothing(MAIN_SEED) ? rand(UInt) : MAIN_SEED + w * 104729)
        slot = MAILBOX !== nothing ? MAILBOX.slots[w] : nothing
        t = Threads.@spawn worker_play_games(w, games_claimed, num_games, REQ_CHAN, rng;
                                              mb_slot=slot)
        push!(tasks, t)
    end

    all_samples = reduce(vcat, [fetch(t) for t in tasks])

    # Stop GPU inference server
    if MAILBOX !== nothing
        stop_mailbox_server!(MAILBOX, server_task)
    elseif USE_GPU_INFERENCE
        stop_inference_server!(server_task)
    end

    return all_samples, num_games
end

"""Spawn worker threads for evaluation with work-stealing, return results."""
function parallel_eval(num_games::Int; verbose::Bool=true)
    games_per_side = num_games ÷ 2

    # Pre-create all RESP_CHANS entries (thread-safe: done before spawning)
    for w in 1:NUM_WORKERS
        for k in 1:ASYNC_GAMES_PER_THREAD
            vid = (w - 1) * ASYNC_GAMES_PER_THREAD + k + 1000
            RESP_CHANS[vid] = Channel{Any}(4)
        end
    end

    # White games
    if verbose
        println("  Playing $games_per_side games as white...")
        flush(stdout)
    end
    white_claimed = Threads.Atomic{Int}(0)
    white_tasks = Task[]
    for w in 1:NUM_WORKERS
        rng = MersenneTwister(rand(UInt))
        t = Threads.@spawn worker_eval_games(w, white_claimed, games_per_side, true, REQ_CHAN, rng)
        push!(white_tasks, t)
    end
    white_rewards = reduce(vcat, [fetch(t) for t in white_tasks])

    # Black games
    if verbose
        println("  Playing $games_per_side games as black...")
        flush(stdout)
    end
    black_claimed = Threads.Atomic{Int}(0)
    black_tasks = Task[]
    for w in 1:NUM_WORKERS
        rng = MersenneTwister(rand(UInt))
        t = Threads.@spawn worker_eval_games(w, black_claimed, games_per_side, false, REQ_CHAN, rng)
        push!(black_tasks, t)
    end
    black_rewards = reduce(vcat, [fetch(t) for t in black_tasks])

    white_avg = isempty(white_rewards) ? 0.0 : mean(white_rewards)
    black_avg = isempty(black_rewards) ? 0.0 : mean(black_rewards)
    combined = (white_avg + black_avg) / 2
    actual_games = length(white_rewards) + length(black_rewards)

    return (white_avg=white_avg, black_avg=black_avg, combined=combined, actual_games=actual_games)
end

#####
##### Training setup
#####

# AdamW: decoupled weight decay (L2 through optimizer, not loss gradient)
# This prevents weight norm explosion that occurs with L2 reg + Adam
opt = Flux.AdamW(LEARNING_RATE, (0.9f0, 0.999f0), L2_REG)
opt_state = Flux.setup(opt, network)

const LEARNING_PARAMS = LearningParams(
    use_gpu=USE_GPU,
    use_position_averaging=false,
    samples_weighing_policy=CONSTANT_WEIGHT,
    optimiser=Adam(lr=LEARNING_RATE),
    l2_regularization=0f0,  # Weight decay handled by AdamW, not loss
    rewards_renormalization=1f0,
    nonvalidity_penalty=1f0,
    batch_size=BATCH_SIZE,
    loss_computation_batch_size=BATCH_SIZE,
    min_checkpoints_per_epoch=1,
    max_batches_per_checkpoint=100,
    num_checkpoints=1
)

function prepare_batch(batch, num_actions, use_gpu_flag, net)
    n = length(batch)

    W = ones(Float32, 1, n)
    X = hcat([s.state for s in batch]...)
    P = hcat([s.policy for s in batch]...)
    V = reshape(Float32[s.value for s in batch], 1, n)

    A = zeros(Float32, num_actions, n)
    IsChance = zeros(Float32, 1, n)
    for i in 1:n
        if batch[i].is_chance
            A[:, i] .= 1.0f0
            IsChance[1, i] = 1.0f0
        else
            A[:, i] .= Float32.(batch[i].policy .> 0)
        end
    end

    EqWin = zeros(Float32, 1, n)
    EqGW = zeros(Float32, 1, n)
    EqBGW = zeros(Float32, 1, n)
    EqGL = zeros(Float32, 1, n)
    EqBGL = zeros(Float32, 1, n)
    HasEquity = zeros(Float32, 1, n)
    for i in 1:n
        if batch[i].has_equity
            eq = batch[i].equity
            EqWin[1, i] = eq[1]
            EqGW[1, i] = eq[2]
            EqBGW[1, i] = eq[3]
            EqGL[1, i] = eq[4]
            EqBGL[1, i] = eq[5]
            HasEquity[1, i] = 1.0f0
        end
    end

    batch_data = (; W, X, A, P, V, IsChance, EqWin, EqGW, EqBGW, EqGL, EqBGL, HasEquity)

    if use_gpu_flag
        batch_data = Network.convert_input_tuple(net, batch_data)
    end

    return batch_data
end

#####
##### PER (Prioritized Experience Replay)
#####

const USE_PER = ARGS["use_per"]
const PER_ALPHA = ARGS["per_alpha"]
const PER_BETA_INIT = ARGS["per_beta"]
const PER_EPSILON = ARGS["per_epsilon"]

"""Prioritized Experience Replay buffer.
Wraps a flat buffer with per-sample priorities for proportional sampling."""
mutable struct PERBuffer
    samples::Vector{Any}       # The actual samples
    priorities::Vector{Float32} # Priority for each sample (|TD-error| + ε)
    capacity::Int
    beta::Float32              # Current IS beta (anneals to 1.0)
    beta_init::Float32
    beta_annealing_iters::Int
    current_iter::Int
end

function PERBuffer(capacity::Int; beta_init=0.4f0, annealing_iters=200)
    PERBuffer(Any[], Float32[], capacity, beta_init, beta_init, annealing_iters, 0)
end

function per_add!(buf::PERBuffer, samples, initial_priority::Float32=1.0f0)
    append!(buf.samples, samples)
    append!(buf.priorities, fill(initial_priority, length(samples)))
    if length(buf.samples) > buf.capacity
        excess = length(buf.samples) - buf.capacity
        deleteat!(buf.samples, 1:excess)
        deleteat!(buf.priorities, 1:excess)
    end
end

function per_anneal_beta!(buf::PERBuffer)
    buf.current_iter += 1
    frac = min(1.0f0, Float32(buf.current_iter) / Float32(buf.beta_annealing_iters))
    buf.beta = buf.beta_init + (1.0f0 - buf.beta_init) * frac
end

"""Sample from PER buffer. Returns (indices, samples, importance_weights)."""
function per_sample(buf::PERBuffer, batch_size::Int, alpha::Float32, epsilon::Float32)
    n = length(buf.samples)
    n < batch_size && error("Buffer too small")

    # Compute priorities^alpha
    pα = similar(buf.priorities)
    @inbounds for i in 1:n
        pα[i] = (buf.priorities[i] + epsilon) ^ alpha
    end
    total = sum(pα)

    # Proportional sampling
    probs = pα ./ total
    indices = Vector{Int}(undef, batch_size)
    for j in 1:batch_size
        r = rand(Float32) * total
        cumsum = 0.0f0
        idx = n  # fallback
        @inbounds for i in 1:n
            cumsum += pα[i]
            if cumsum >= r
                idx = i
                break
            end
        end
        indices[j] = idx
    end

    # Importance sampling weights: w_i = (N * P(i))^(-beta) / max(w)
    weights = Vector{Float32}(undef, batch_size)
    @inbounds for j in 1:batch_size
        weights[j] = (Float32(n) * probs[indices[j]]) ^ (-buf.beta)
    end
    max_w = maximum(weights)
    weights ./= max_w  # Normalize so max weight = 1.0

    samples = [buf.samples[i] for i in indices]
    return (indices, samples, weights)
end

"""Update priorities for given indices with new TD-errors."""
function per_update_priorities!(buf::PERBuffer, indices::Vector{Int}, td_errors::Vector{Float32})
    @inbounds for (idx, td) in zip(indices, td_errors)
        if 1 <= idx <= length(buf.priorities)
            buf.priorities[idx] = abs(td)
        end
    end
end

# Replay buffer (PER or uniform)
replay_buffer = USE_PER ? PERBuffer(BUFFER_CAPACITY; beta_init=PER_BETA_INIT, annealing_iters=ARGS["total_iterations"]) : Any[]

# Training metrics
total_games = START_ITER * ARGS["games_per_iteration"]  # Approximate for resume
total_samples = 0
start_time = time()

# Save run info
open(joinpath(SESSION_DIR, "run_info.txt"), "w") do f
    println(f, "# Threaded AlphaZero Training")
    println(f, "timestamp: $(Dates.format(now(), "yyyymmdd_HHMMSS"))")
    println(f, "num_workers: $NUM_WORKERS")
    println(f, "julia_threads: $(Threads.nthreads())")
    println(f, "game: $GAME_NAME")
    println(f, "network: $(ARGS["network_type"]) $(NET_WIDTH)x$(NET_BLOCKS)")
    println(f, "mcts_iters: $MCTS_ITERS")
    println(f, "inference_batch_size: $INFERENCE_BATCH_SIZE")
    println(f, "seed: $(isnothing(MAIN_SEED) ? "none" : MAIN_SEED)")
end

println("\n" * "=" ^ 60)
println("Starting training...")
println("=" ^ 60)
flush(stdout)

# Pre-compile training path (prevents ~70s JIT penalty on iteration 2)
print("Pre-compiling training path...")
flush(stdout)
let
    warmup_net = deepcopy(network)
    warmup_opt = Flux.setup(Flux.AdamW(LEARNING_RATE, (0.9f0, 0.999f0), L2_REG), warmup_net)
    dummy_state = zeros(Float32, _state_dim)
    dummy_policy = zeros(Float32, NUM_ACTIONS); dummy_policy[1] = 1.0f0
    dummy_sample = (state=dummy_state, policy=dummy_policy, value=0.0f0,
                    equity=zeros(Float32, 5), has_equity=false, is_chance=false)
    dummy_buffer = fill(dummy_sample, BATCH_SIZE)
    bd = prepare_batch(dummy_buffer, NUM_ACTIONS, USE_GPU, warmup_net)
    Wmean = 1.0f0; Hp = 0.0f0
    loss, grads = Flux.withgradient(
        nn -> losses(nn, LEARNING_PARAMS, Wmean, Hp, bd)[1], warmup_net)
    Flux.update!(warmup_opt, warmup_net, grads[1])
end
GC.gc()  # Clean up warmup allocations
println(" done.")
flush(stdout)

#####
##### Training loop
#####

"""Buffer length helper (works for both PER and plain array)."""
buf_length(buf::PERBuffer) = length(buf.samples)
buf_length(buf::Vector) = length(buf)

"""Train on replay buffer (GPU). Returns avg loss.
Caps training at min(buffer_size, games_per_iter * avg_moves) / batch_size batches
to prevent training from exceeding self-play time at large buffer sizes."""
function train_on_buffer!(replay_buffer, network, opt_state)
    n_buf = buf_length(replay_buffer)
    n_buf < BATCH_SIZE && return (avg_loss=0.0, avg_Lp=0.0, avg_Lv=0.0, avg_Linv=0.0)
    # Cap at ~1 epoch of new data (avg ~200 samples/game × games_per_iter)
    max_batches = max(1, ARGS["games_per_iteration"] * 200 ÷ BATCH_SIZE)
    num_batches = min(max(1, n_buf ÷ BATCH_SIZE), max_batches)
    total_loss = 0.0
    total_Lp = 0.0
    total_Lv = 0.0
    total_Linv = 0.0

    if USE_PER && replay_buffer isa PERBuffer
        per_anneal_beta!(replay_buffer)
    end

    for _ in 1:num_batches
        local indices, batch, is_weights

        if USE_PER && replay_buffer isa PERBuffer
            indices, batch, is_weights = per_sample(replay_buffer, BATCH_SIZE, PER_ALPHA, PER_EPSILON)
        else
            indices = rand(1:n_buf, BATCH_SIZE)
            batch = USE_PER ? [replay_buffer.samples[i] for i in indices] : [replay_buffer[i] for i in indices]
            is_weights = ones(Float32, BATCH_SIZE)
        end

        batch_data = prepare_batch(batch, NUM_ACTIONS, USE_GPU, network)

        # Apply importance sampling weights to sample weights
        if USE_PER
            W_is = reshape(Float32.(is_weights), 1, BATCH_SIZE)
            if USE_GPU
                W_is = Network.convert_input_tuple(network, (; W=W_is)).W
            end
            batch_data = merge(batch_data, (; W=batch_data.W .* W_is))
        end

        Wmean = mean(batch_data.W)
        Hp = 0.0f0

        loss_fn(nn) = losses(nn, LEARNING_PARAMS, Wmean, Hp, batch_data)[1]

        loss, grads = Flux.withgradient(loss_fn, network)
        Flux.update!(opt_state, network, grads[1])

        # Get component losses (outside gradient computation)
        L, Lp, Lv, _, Linv = losses(network, LEARNING_PARAMS, Wmean, Hp, batch_data)
        total_loss += Float64(L)
        total_Lp += Float64(Lp)
        total_Lv += Float64(Lv)
        total_Linv += Float64(Linv)

        # Update PER priorities with TD-errors
        if USE_PER && replay_buffer isa PERBuffer
            # Compute per-sample value prediction vs target for TD-error
            td_errors = compute_td_errors(network, batch_data)
            per_update_priorities!(replay_buffer, indices, td_errors)
        end
    end

    return (avg_loss=total_loss / num_batches, avg_Lp=total_Lp / num_batches,
            avg_Lv=total_Lv / num_batches, avg_Linv=total_Linv / num_batches)
end

"""Compute per-sample TD-errors for PER priority updates."""
function compute_td_errors(nn, batch_data)
    X, A, V = batch_data.X, batch_data.A, batch_data.V

    is_multihead = nn isa FluxLib.FCResNetMultiHead
    if is_multihead
        _, V̂_win, V̂_gw, V̂_bgw, V̂_gl, V̂_bgl, _ =
            FluxLib.forward_normalized_multihead(nn, X, A)
        # Compute equity: (P(win) + P(gammon|win) + P(bg|win)) - (P(gammon|loss) + P(bg|loss))
        equity = FluxLib.compute_equity(V̂_win, V̂_gw, V̂_bgw, V̂_gl, V̂_bgl)
        V̂_combined = equity ./ 3f0  # Scale to [-1, 1]
        td = abs.(Flux.cpu(V̂_combined) .- Flux.cpu(V))
    else
        _, V̂, _ = Network.forward_normalized(nn, X, A)
        td = abs.(Flux.cpu(V̂) .- Flux.cpu(V))
    end

    return Float32.(vec(td))
end

#####
##### Reanalyze
#####

const USE_REANALYZE = ARGS["use_reanalyze"]
const REANALYZE_FRACTION = ARGS["reanalyze_fraction"]

"""Reanalyze a fraction of buffer positions with the latest network.
Updates value targets with blended old/new values (EMA with α=0.5).
If PER is enabled, also updates priorities."""
function reanalyze_buffer!(replay_buffer, network)
    samples = replay_buffer isa PERBuffer ? replay_buffer.samples : replay_buffer
    n = length(samples)
    n == 0 && return 0

    num_to_reanalyze = max(1, round(Int, n * REANALYZE_FRACTION))
    # Random subset (or prioritize stale samples if PER)
    reanalyze_indices = randperm(n)[1:min(num_to_reanalyze, n)]

    batch_size = min(512, length(reanalyze_indices))
    total_updated = 0

    for batch_start in 1:batch_size:length(reanalyze_indices)
        batch_end = min(batch_start + batch_size - 1, length(reanalyze_indices))
        batch_indices = reanalyze_indices[batch_start:batch_end]
        batch = [samples[i] for i in batch_indices]

        batch_data = prepare_batch(batch, NUM_ACTIONS, USE_GPU, network)

        # Get current network's value predictions
        is_multihead = network isa FluxLib.FCResNetMultiHead
        if is_multihead
            _, V̂_win, V̂_gw, V̂_bgw, V̂_gl, V̂_bgl, _ =
                FluxLib.forward_normalized_multihead(network, batch_data.X, batch_data.A)
            equity = FluxLib.compute_equity(V̂_win, V̂_gw, V̂_bgw, V̂_gl, V̂_bgl)
            new_values = Float32.(vec(Flux.cpu(equity ./ 3f0)))

            # Also update equity targets with EMA blend
            new_eq_win = Float32.(vec(Flux.cpu(V̂_win)))
            new_eq_gw = Float32.(vec(Flux.cpu(V̂_gw)))
            new_eq_bgw = Float32.(vec(Flux.cpu(V̂_bgw)))
            new_eq_gl = Float32.(vec(Flux.cpu(V̂_gl)))
            new_eq_bgl = Float32.(vec(Flux.cpu(V̂_bgl)))
        else
            _, V̂, _ = Network.forward_normalized(network, batch_data.X, batch_data.A)
            new_values = Float32.(vec(Flux.cpu(V̂)))
        end

        # Update each sample with blended value (EMA α=0.5)
        α_blend = 0.5f0
        for (j, buf_idx) in enumerate(batch_indices)
            old = samples[buf_idx]
            old_val = old.value
            blended_val = (1f0 - α_blend) * old_val + α_blend * new_values[j]

            if is_multihead && old.has_equity
                old_eq = old.equity
                new_eq = Float32[
                    (1f0 - α_blend) * old_eq[1] + α_blend * new_eq_win[j],
                    (1f0 - α_blend) * old_eq[2] + α_blend * new_eq_gw[j],
                    (1f0 - α_blend) * old_eq[3] + α_blend * new_eq_bgw[j],
                    (1f0 - α_blend) * old_eq[4] + α_blend * new_eq_gl[j],
                    (1f0 - α_blend) * old_eq[5] + α_blend * new_eq_bgl[j],
                ]
                samples[buf_idx] = merge(old, (value=blended_val, equity=new_eq))
            else
                samples[buf_idx] = merge(old, (value=blended_val,))
            end

            # Update PER priorities
            if replay_buffer isa PERBuffer
                td_error = abs(new_values[j] - old_val)
                replay_buffer.priorities[buf_idx] = td_error
            end

            total_updated += 1
        end
    end

    return total_updated
end

for iter in (START_ITER + 1):ARGS["total_iterations"]
    iter_start = time()

    local new_samples, games_this_iter, avg_loss, train_result

    if USE_GPU_INFERENCE
        # Sequential: GPU can't do training + inference simultaneously
        # 1. Self-play (GPU inference server)
        new_samples, games_this_iter = parallel_self_play(ARGS["games_per_iteration"])
        # 2. Train on buffer (GPU training)
        train_result = train_on_buffer!(replay_buffer, network, opt_state)
        avg_loss = train_result.avg_loss
    else
        # Overlapped: self-play (CPU) runs concurrently with training (GPU)
        # Self-play uses cpu_network (read-only), training uses network (GPU, write)
        sp_task = Threads.@spawn parallel_self_play(ARGS["games_per_iteration"])

        # Train on existing buffer while self-play runs (iter 1: buffer empty, no-op)
        train_result = train_on_buffer!(replay_buffer, network, opt_state)
        avg_loss = train_result.avg_loss

        # Wait for self-play to complete
        new_samples, games_this_iter = fetch(sp_task)

        # Sync GPU→CPU weights AFTER self-play done (safe: no concurrent reads)
        if avg_loss > 0
            Flux.loadmodel!(cpu_network, Flux.cpu(network))
            if USE_FAST_FORWARD
                refresh_fast_weights!()
            end
        end
    end

    # Update buffer
    if USE_PER && replay_buffer isa PERBuffer
        per_add!(replay_buffer, new_samples)
    else
        append!(replay_buffer, new_samples)
        if length(replay_buffer) > BUFFER_CAPACITY
            deleteat!(replay_buffer, 1:(length(replay_buffer) - BUFFER_CAPACITY))
        end
    end
    global total_games += games_this_iter
    global total_samples += length(new_samples)

    # Reanalyze buffer positions with latest network
    reanalyzed_count = 0
    if USE_REANALYZE && buf_length(replay_buffer) >= BATCH_SIZE
        reanalyzed_count = reanalyze_buffer!(replay_buffer, network)
    end

    cur_buf_size = buf_length(replay_buffer)
    iter_time = time() - iter_start
    elapsed = time() - start_time
    games_per_min = total_games / (elapsed / 60)

    @info "Iteration $iter" avg_loss buffer_size=cur_buf_size total_games games_per_min iter_time
    flush(stdout)
    flush(stderr)

    # TensorBoard logging
    with_logger(TB_LOGGER) do
        set_step!(TB_LOGGER, iter)
        # Only log loss metrics when actual training happened (skip iter 1 with empty buffer)
        if avg_loss > 0
            @info "train/loss" value=avg_loss log_step_increment=0
            @info "train/loss_policy" value=train_result.avg_Lp log_step_increment=0
            @info "train/loss_value" value=train_result.avg_Lv log_step_increment=0
            @info "train/loss_invalid" value=train_result.avg_Linv log_step_increment=0
        end
        @info "train/buffer_size" value=cur_buf_size log_step_increment=0
        @info "perf/games_per_min" value=games_per_min log_step_increment=0
        @info "perf/iter_time_s" value=iter_time log_step_increment=0
        @info "train/total_games" value=total_games log_step_increment=0

        # Bear-off accuracy benchmark (fast, runs every iteration)
        if USE_BEAROFF
            bo_t0 = time()
            bo_result = eval_bearoff_accuracy(network)
            bo_time = time() - bo_t0
            if bo_result !== nothing
                @info "bearoff/mae_equity" value=bo_result.mae_value log_step_increment=0
                for (h, name) in enumerate(bo_result.head_names)
                    @info "bearoff/mae_$(name)" value=bo_result.mae_heads[h] log_step_increment=0
                end
                @info "bearoff/eval_time_s" value=Float32(bo_time) log_step_increment=0
                if iter == 1 || iter % 10 == 0
                    @info "Bear-off MAE: equity=$(round(bo_result.mae_value, digits=4)), " *
                          "P(win)=$(round(bo_result.mae_heads[1], digits=4)) " *
                          "($(round(bo_time*1000, digits=1))ms)"
                    flush(stderr)
                end
            end
        end
    end

    # Evaluation
    if ARGS["eval_interval"] > 0 && iter % ARGS["eval_interval"] == 0
        eval_games = ARGS["eval_games"]
        @info "Running threaded evaluation ($eval_games games)..."
        flush(stdout)

        eval_server = start_inference_server!()
        eval_start = time()
        eval_results = parallel_eval(eval_games; verbose=true)
        eval_time = time() - eval_start
        stop_inference_server!(eval_server)

        @info "Eval results: white=$(round(eval_results.white_avg, digits=3)), " *
              "black=$(round(eval_results.black_avg, digits=3)), " *
              "combined=$(round(eval_results.combined, digits=3)) " *
              "($(eval_results.actual_games) games in $(round(eval_time, digits=1))s)"

        # TensorBoard eval metrics
        with_logger(TB_LOGGER) do
            set_step!(TB_LOGGER, iter)
            @info "eval/vs_random_combined" value=eval_results.combined log_step_increment=0
            @info "eval/vs_random_white" value=eval_results.white_avg log_step_increment=0
            @info "eval/vs_random_black" value=eval_results.black_avg log_step_increment=0
        end
    end

    # Checkpoint
    if iter % ARGS["checkpoint_interval"] == 0
        checkpoint_path = joinpath(SESSION_DIR, "checkpoints", "iter_$iter.data")
        FluxLib.save_weights(checkpoint_path, network)

        latest_path = joinpath(SESSION_DIR, "checkpoints", "latest.data")
        FluxLib.save_weights(latest_path, network)

        open(joinpath(SESSION_DIR, "checkpoints", "iter.txt"), "w") do f
            println(f, iter)
        end

        @info "Saved checkpoint at iteration $iter"
    end
end

# Final evaluation
final_eval_games = ARGS["final_eval_games"]
if final_eval_games > 0
    println("\n" * "=" ^ 60)
    println("Running Final Evaluation")
    println("=" ^ 60)
    println("Games: $final_eval_games")
    println("Workers: $NUM_WORKERS threads")
    println("MCTS iterations: $MCTS_ITERS")
    flush(stdout)

    final_server = start_inference_server!()
    final_eval_start = time()
    final_results = parallel_eval(final_eval_games; verbose=true)
    final_eval_time = time() - final_eval_start
    stop_inference_server!(final_server)

    println("=" ^ 60)
    println("Final Evaluation Results:")
    println("  White:    $(round(final_results.white_avg, digits=3))")
    println("  Black:    $(round(final_results.black_avg, digits=3))")
    println("  Combined: $(round(final_results.combined, digits=3))")
    println("  Games:    $(final_results.actual_games)")
    println("  Time:     $(round(final_eval_time / 60, digits=2)) minutes")
    println("  Speed:    $(round(final_results.actual_games / final_eval_time * 60, digits=1)) games/min")
    println("=" ^ 60)
    flush(stdout)

    open(joinpath(SESSION_DIR, "final_eval_results.txt"), "w") do f
        println(f, "# Final Evaluation Results")
        println(f, "timestamp: $(Dates.format(now(), "yyyymmdd_HHMMSS"))")
        println(f, "games: $(final_results.actual_games)")
        println(f, "white_avg: $(final_results.white_avg)")
        println(f, "black_avg: $(final_results.black_avg)")
        println(f, "combined: $(final_results.combined)")
        println(f, "time_seconds: $final_eval_time")
        println(f, "workers: $NUM_WORKERS")
        println(f, "mcts_iters: $MCTS_ITERS")
    end
    println("Results saved to: $(joinpath(SESSION_DIR, "final_eval_results.txt"))")
end

# Close TensorBoard logger
close(TB_LOGGER)

# Training complete
elapsed = time() - start_time
println("\n" * "=" ^ 60)
println("Training Complete!")
println("=" ^ 60)
println("Total iterations: $(ARGS["total_iterations"])")
println("Total games: $total_games")
println("Total samples: $total_samples")
println("Total time: $(round(elapsed/60, digits=1)) minutes")
println("Session: $SESSION_DIR")
println("TensorBoard logs: $TB_DIR")
println("=" ^ 60)
