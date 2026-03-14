#!/usr/bin/env julia
"""Benchmark reanalyze batch sizes on GPU to find optimal throughput for RTX 4090."""

using Pkg; Pkg.activate(".")
using AlphaZero
import CUDA
using Flux
using Statistics

const FluxLib = AlphaZero.FluxLib

# Load game + network
println("Loading network...")
ENV["BACKGAMMON_OBS_TYPE"] = "minimal"
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
gspec = GameSpec()
hyper = FluxLib.FCResNetMultiHeadHP(width=128, num_blocks=3)
network = FluxLib.FCResNetMultiHead(gspec, hyper)
num_actions = GI.num_actions(gspec)
state_dim = GI.state_dim(gspec)
if CUDA.functional()
    println("GPU: $(CUDA.name(CUDA.device()))")
    network = Flux.gpu(network)
    CUDA.allowscalar(false)
else
    error("No GPU available")
end

num_params = sum(length, Flux.params(network))
println("Network: 128w×3b, $(num_params) params")
println("Input: $(state_dim) features, $(num_actions) actions")
println()

# Batch sizes to test
batch_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

# Total samples to process (simulate 25% of 600K buffer)
total_samples = 150_000

println("=" ^ 70)
println("Benchmarking forward pass (reanalyze) — $(total_samples) samples total")
println("=" ^ 70)
println()
println(rpad("Batch Size", 12), rpad("Batches", 10), rpad("Time (s)", 12),
        rpad("Samples/s", 12), rpad("GPU Util%", 12), "GPU Mem%")
println("-" ^ 70)

for bs in batch_sizes
    # Pre-allocate GPU tensors
    X = CUDA.rand(Float32, state_dim..., bs)
    A = CUDA.ones(Float32, num_actions, bs)  # All actions valid for benchmark

    num_batches = ceil(Int, total_samples / bs)

    # Warmup (2 iterations)
    for _ in 1:2
        FluxLib.forward_normalized_multihead(network, X, A)
    end
    CUDA.synchronize()

    # Benchmark
    t0 = time()
    for _ in 1:num_batches
        _, V̂_win, V̂_gw, V̂_bgw, V̂_gl, V̂_bgl, _ =
            FluxLib.forward_normalized_multihead(network, X, A)
        equity = FluxLib.compute_equity(V̂_win, V̂_gw, V̂_bgw, V̂_gl, V̂_bgl)
        # Simulate CPU transfer (like real reanalyze)
        Flux.cpu(equity)
    end
    CUDA.synchronize()
    elapsed = time() - t0

    throughput = total_samples / elapsed

    # Sample GPU utilization mid-benchmark (run a quick burst and measure)
    gpu_util = "N/A"
    gpu_mem = "N/A"
    try
        # Run another burst while measuring
        util_task = @async begin
            sleep(0.5)
            result = read(`nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits`, String)
            strip(result)
        end
        for _ in 1:max(10, num_batches ÷ 10)
            FluxLib.forward_normalized_multihead(network, X, A)
        end
        CUDA.synchronize()
        util_str = fetch(util_task)
        parts = split(util_str, ",")
        gpu_util = strip(parts[1]) * "%"
        gpu_mem = strip(parts[2]) * "%"
    catch; end

    println(rpad(bs, 12), rpad(num_batches, 10), rpad(round(elapsed, digits=3), 12),
            rpad(round(Int, throughput), 12), rpad(gpu_util, 12), gpu_mem)

    GC.gc(false)
end

println()
println("=" ^ 70)

# Also benchmark the full reanalyze pipeline (prepare_batch overhead)
println()
println("Note: Real reanalyze also includes prepare_batch() CPU overhead")
println("(data gathering, GPU transfer). Optimal batch size balances")
println("GPU kernel efficiency vs. CPU data prep latency.")
