#!/usr/bin/env julia
"""
Deep performance profiling for inference and MCTS.
Breaks down time by component to identify optimization targets.

Usage:
  julia --threads N --project scripts/bench_perf_deep.jl [--width 256] [--blocks 5] [--batch 50] [--workers N]
"""

using ArgParse
s = ArgParseSettings()
@add_arg_table! s begin
    "--width"; arg_type=Int; default=256
    "--blocks"; arg_type=Int; default=5
    "--batch"; arg_type=Int; default=50
    "--mcts-iters"; arg_type=Int; default=600
    "--workers"; arg_type=Int; default=0
    "--games"; arg_type=Int; default=100
end
ARGS_PARSED = parse_args(s)
WIDTH = ARGS_PARSED["width"]
BLOCKS = ARGS_PARSED["blocks"]
BATCH = ARGS_PARSED["batch"]
MCTS_ITERS = ARGS_PARSED["mcts-iters"]
NUM_WORKERS = ARGS_PARSED["workers"] > 0 ? ARGS_PARSED["workers"] : Threads.nthreads()
NUM_GAMES = ARGS_PARSED["games"]

using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

using AlphaZero
using AlphaZero: GI, Network, FluxLib, MctsParams, BatchedMCTS, ConstSchedule
using AlphaZero.FastInference: FastWeights, FastBuffers, extract_fast_weights,
    fast_forward_normalized!, _gemm_bias!, dense!, layernorm_relu!
using AlphaZero.BackgammonInference: OracleConfig, make_cpu_oracles
using Random, Statistics, Printf

# Load game
const GAMES_DIR = joinpath(@__DIR__, "..", "games")
include(joinpath(GAMES_DIR, "backgammon-deterministic", "main.jl"))
const BGD = Main.BackgammonDeterministic
const gspec = BGD.GameSpec()
const state_dim = let env = GI.init(gspec); length(vec(GI.vectorize_state(gspec, GI.current_state(env)))); end
const num_actions = GI.num_actions(gspec)

println("=" ^ 70)
println("Deep Performance Profile")
println("=" ^ 70)
println("  Model: $(WIDTH)w × $(BLOCKS)b")
println("  Batch size: $BATCH")
println("  MCTS iters: $MCTS_ITERS")
println("  Workers: $NUM_WORKERS (threads: $(Threads.nthreads()))")
println("  Games: $NUM_GAMES")
println("  Platform: $(Sys.ARCH) $(Sys.isapple() ? "macOS" : "Linux")")
println("  Julia: $(VERSION)")
println("=" ^ 70)

# Create network and extract weights
net = FluxLib.FCResNetMultiHead(gspec, FluxLib.FCResNetMultiHeadHP(
    width=WIDTH, num_blocks=BLOCKS, depth_phead=1))
fw = extract_fast_weights(net)

n_params = length(fw.W_in) + length(fw.b_in) + sum(length.(fw.res_W1)) + sum(length.(fw.res_W2)) + sum(length.(fw.res_b1)) + sum(length.(fw.res_b2)) + length(fw.W_vt) + length(fw.b_vt) + length(fw.W_pout) + length(fw.b_pout)
println("\nModel size: $(round(n_params * 4 / 1024 / 1024, digits=1)) MB ($(n_params) params)")
println("W_in: $(size(fw.W_in)) = $(length(fw.W_in) * 4 / 1024) KB")
println("Res W1/W2: $(size(fw.res_W1[1])) = $(length(fw.res_W1[1]) * 4 / 1024) KB each × $(BLOCKS) blocks")

#####
##### Section 1: Micro-benchmarks (single-thread)
#####

println("\n" * "=" ^ 70)
println("SECTION 1: Micro-benchmarks (single thread)")
println("=" ^ 70)

# Create buffers for the forward pass
fb = FastBuffers(WIDTH, state_dim, num_actions, BATCH; num_policy_layers=1)

# Random input
X = rand(Float32, state_dim, BATCH)
A = ones(Float32, num_actions, BATCH)  # all actions legal

# Warmup
fast_forward_normalized!(fw, fb, X, A, BATCH)

# --- GEMM benchmark ---
println("\n--- GEMM (_gemm_bias!) ---")
C = zeros(Float32, WIDTH, BATCH)
for (label, mat_A, mat_B, bias) in [
    ("Input ($(state_dim)×$(WIDTH) × $(state_dim)×$BATCH)", fw.W_in, X, fw.b_in),
    ("ResBlock ($(WIDTH)×$(WIDTH) × $(WIDTH)×$BATCH)", fw.res_W1[1], fb.h1, fw.res_b1[1]),
]
    # Warmup
    _gemm_bias!(C, mat_A, mat_B, bias, BATCH)

    n_iters = 1000
    t = @elapsed for _ in 1:n_iters
        _gemm_bias!(C, mat_A, mat_B, bias, BATCH)
    end
    m, k = size(mat_A)
    gflops = 2.0 * m * k * BATCH * n_iters / t / 1e9
    μs = t / n_iters * 1e6
    @printf("  %s: %.1f μs (%.1f GFLOPS)\n", label, μs, gflops)
end

# --- BLAS comparison ---
println("\n--- BLAS mul! comparison ---")
A_blas = fw.res_W1[1]
B_blas = fb.h1[:, 1:BATCH]
C_blas = zeros(Float32, WIDTH, BATCH)
mul!(C_blas, A_blas, B_blas)  # warmup
n_iters = 1000
t_blas = @elapsed for _ in 1:n_iters
    mul!(C_blas, A_blas, B_blas)
end
m, k = size(A_blas)
gflops_blas = 2.0 * m * k * BATCH * n_iters / t_blas / 1e9
μs_blas = t_blas / n_iters * 1e6
@printf("  BLAS mul! (%d×%d × %d×%d): %.1f μs (%.1f GFLOPS)\n", m, k, k, BATCH, μs_blas, gflops_blas)

# --- LayerNorm+ReLU ---
println("\n--- LayerNorm + ReLU ---")
ln_mean = zeros(Float32, BATCH)
ln_rstd = zeros(Float32, BATCH)
layernorm_relu!(fb.h2, fb.h1, fw.ln_in_s, fw.ln_in_b, ln_mean, ln_rstd, BATCH)  # warmup
n_iters = 5000
t_ln = @elapsed for _ in 1:n_iters
    layernorm_relu!(fb.h2, fb.h1, fw.ln_in_s, fw.ln_in_b, ln_mean, ln_rstd, BATCH)
end
@printf("  LayerNorm+ReLU (%d×%d): %.1f μs\n", WIDTH, BATCH, t_ln / n_iters * 1e6)

# --- Full forward pass ---
println("\n--- Full forward pass ---")
fast_forward_normalized!(fw, fb, X, A, BATCH)  # warmup
n_iters = 200
t_fwd = @elapsed for _ in 1:n_iters
    fast_forward_normalized!(fw, fb, X, A, BATCH)
end
@printf("  Forward pass (%dw×%db, batch=%d): %.1f μs\n", WIDTH, BLOCKS, BATCH, t_fwd / n_iters * 1e6)

# --- Forward pass breakdown ---
println("\n--- Forward pass breakdown ---")
# Count operations
n_gemm_calls = 1 + 2 * BLOCKS + 1 + 1  # input + 2*blocks + value_trunk + policy_out
n_ln_calls = 1 + 2 * BLOCKS + 1 + 1     # input + 2*blocks + post + value_trunk
gemm_time = t_fwd / n_iters * 1e6  # approximate
@printf("  ~%d GEMM calls, ~%d LN calls per forward\n", n_gemm_calls, n_ln_calls)
@printf("  Estimated GEMM fraction: %.0f%%\n", 100.0)  # placeholder, will compute below

# Detailed breakdown
t_gemm_total = 0.0
t_ln_total = 0.0
# Input GEMM
n_rep = 500
t = @elapsed for _ in 1:n_rep; dense!(fb.h1, fw.W_in, X, fw.b_in, BATCH); end
t_gemm_total += t / n_rep
# Res blocks
for blk in 1:BLOCKS
    t = @elapsed for _ in 1:n_rep; dense!(fb.h2, fw.res_W1[blk], fb.h1, fw.res_b1[blk], BATCH); end
    t_gemm_total += t / n_rep
    t = @elapsed for _ in 1:n_rep; dense!(fb.h2, fw.res_W2[blk], fb.h1, fw.res_b2[blk], BATCH); end
    t_gemm_total += t / n_rep
end
# Value trunk + policy
t = @elapsed for _ in 1:n_rep; dense!(fb.vt, fw.W_vt, fb.h1, fw.b_vt, BATCH); end
t_gemm_total += t / n_rep
t = @elapsed for _ in 1:n_rep; dense!(fb.p, fw.W_pout, fb.h1, fw.b_pout, BATCH); end
t_gemm_total += t / n_rep
# Policy layers
for i in 1:fw.num_policy_layers
    t = @elapsed for _ in 1:n_rep; dense!(fb.vt, fw.W_p[i], fb.h1, fw.b_p[i], BATCH); end
    t_gemm_total += t / n_rep
end

# LN calls
t = @elapsed for _ in 1:n_rep; layernorm_relu!(fb.h2, fb.h1, fw.ln_in_s, fw.ln_in_b, ln_mean, ln_rstd, BATCH); end
t_ln_total += t / n_rep
for blk in 1:BLOCKS
    t = @elapsed for _ in 1:n_rep; layernorm_relu!(fb.h1, fb.h2, fw.res_ln1_s[blk], fw.res_ln1_b[blk], ln_mean, ln_rstd, BATCH); end
    t_ln_total += t / n_rep
    t = @elapsed for _ in 1:n_rep; layernorm_relu!(fb.h1, fb.h2, fw.res_ln2_s[blk], fw.res_ln2_b[blk], ln_mean, ln_rstd, BATCH); end
    t_ln_total += t / n_rep
end
t = @elapsed for _ in 1:n_rep; layernorm_relu!(fb.h1, fb.h2, fw.ln_post_s, fw.ln_post_b, ln_mean, ln_rstd, BATCH); end
t_ln_total += t / n_rep
t = @elapsed for _ in 1:n_rep; layernorm_relu!(fb.h2, fb.vt, fw.ln_vt_s, fw.ln_vt_b, ln_mean, ln_rstd, BATCH); end
t_ln_total += t / n_rep
for i in 1:fw.num_policy_layers
    t = @elapsed for _ in 1:n_rep; layernorm_relu!(fb.h1, fb.vt, fw.ln_p_s[i], fw.ln_p_b[i], ln_mean, ln_rstd, BATCH); end
    t_ln_total += t / n_rep
end

fwd_total = t_fwd / n_iters
other = fwd_total - t_gemm_total - t_ln_total
@printf("  GEMM total:      %7.1f μs (%4.1f%%)\n", t_gemm_total * 1e6, t_gemm_total / fwd_total * 100)
@printf("  LayerNorm total: %7.1f μs (%4.1f%%)\n", t_ln_total * 1e6, t_ln_total / fwd_total * 100)
@printf("  Other (value/softmax/skip): %7.1f μs (%4.1f%%)\n", other * 1e6, other / fwd_total * 100)
@printf("  Total forward:   %7.1f μs\n", fwd_total * 1e6)

# --- Memory bandwidth estimate ---
println("\n--- Memory bandwidth estimate ---")
# GEMM reads A (m×k) + B (k×n) + writes C (m×n), all Float32
bytes_per_gemm = (WIDTH * WIDTH + WIDTH * BATCH + WIDTH * BATCH) * 4  # res block gemm
total_bytes = bytes_per_gemm * (2 * BLOCKS + 3)  # approx
bw = total_bytes / fwd_total / 1e9
@printf("  Approx data moved per forward: %.1f MB\n", total_bytes / 1e6)
@printf("  Effective bandwidth: %.1f GB/s (single thread)\n", bw)

#####
##### Section 2: Multi-threaded scaling
#####

println("\n" * "=" ^ 70)
println("SECTION 2: Multi-threaded forward pass scaling")
println("=" ^ 70)

cfg = OracleConfig(state_dim, num_actions, gspec)

for nw in [1, 2, 4, min(8, NUM_WORKERS), min(16, NUM_WORKERS), NUM_WORKERS]
    nw > Threads.nthreads() && continue
    nw < 1 && continue

    # Create per-thread oracles
    _, batch_oracle = make_cpu_oracles(:fast, net, cfg; batch_size=BATCH, nslots=nw)

    env = GI.init(gspec)
    states = [GI.current_state(env) for _ in 1:BATCH]

    # Warmup
    batch_oracle(states)

    # Benchmark: all threads doing forward passes concurrently
    n_calls = max(100, 500 ÷ nw)
    t = @elapsed begin
        Threads.@threads for w in 1:nw
            for _ in 1:n_calls
                batch_oracle(states)
            end
        end
    end
    total_calls = nw * n_calls
    calls_per_sec = total_calls / t
    μs_per_call = t / total_calls * 1e6
    @printf("  %2d workers: %7.1f μs/call, %7.0f calls/sec (%.1fx vs 1 worker)\n",
            nw, μs_per_call, calls_per_sec, nw == 1 ? 1.0 : calls_per_sec / (nw == 1 ? calls_per_sec : 0))
end

# Redo with tracking for proper speedup
println("\n  Speedup relative to 1 worker:")
baseline_cps = 0.0
for nw in [1, 2, 4, min(8, NUM_WORKERS), min(16, NUM_WORKERS), NUM_WORKERS]
    nw > Threads.nthreads() && continue
    nw < 1 && continue

    _, batch_oracle = make_cpu_oracles(:fast, net, cfg; batch_size=BATCH, nslots=nw)
    env = GI.init(gspec)
    states = [GI.current_state(env) for _ in 1:BATCH]
    batch_oracle(states)  # warmup

    n_calls = max(100, 500 ÷ nw)
    t = @elapsed begin
        Threads.@threads for w in 1:nw
            for _ in 1:n_calls
                batch_oracle(states)
            end
        end
    end
    cps = nw * n_calls / t
    if nw == 1
        baseline_cps = cps
    end
    efficiency = cps / (baseline_cps * nw) * 100
    @printf("    %2d workers: %.0f calls/sec, %.1fx speedup (%.0f%% efficiency)\n",
            nw, cps, cps / baseline_cps, efficiency)
end

#####
##### Section 3: MCTS + Game loop profiling
#####

println("\n" * "=" ^ 70)
println("SECTION 3: MCTS game profiling ($NUM_GAMES games, $MCTS_ITERS iters)")
println("=" ^ 70)

single_oracle, batch_oracle = make_cpu_oracles(:fast, net, cfg; batch_size=BATCH)

mcts_params = MctsParams(
    num_iters_per_turn=MCTS_ITERS,
    cpuct=2.0,
    temperature=ConstSchedule(0.0),
    dirichlet_noise_ϵ=0.0,
    dirichlet_noise_α=0.3)

player = BatchedMCTS.BatchedMctsPlayer(gspec, single_oracle, mcts_params;
    batch_size=BATCH, batch_oracle=batch_oracle)

# Play games and time them
println("\n--- Single-thread game timing ---")
env = GI.init(gspec)
rng = MersenneTwister(42)

# Warmup: play 2 games
for _ in 1:2
    GI.set_state!(env, GI.init(gspec))
    while !GI.game_terminated(env)
        if GI.is_chance_node(env)
            outcomes = GI.chance_outcomes(env)
            GI.apply_chance!(env, outcomes[rand(rng, eachindex(outcomes))][1])
        else
            actions, policy = BatchedMCTS.think(player, env)
            BatchedMCTS.reset_player!(player)
            GI.play!(env, actions[argmax(policy)])
        end
    end
end

# Timed games
game_times = Float64[]
move_counts = Int[]
for g in 1:min(10, NUM_GAMES)
    GI.set_state!(env, GI.init(gspec))
    moves = 0
    t0 = time()
    while !GI.game_terminated(env)
        if GI.is_chance_node(env)
            outcomes = GI.chance_outcomes(env)
            GI.apply_chance!(env, outcomes[rand(rng, eachindex(outcomes))][1])
        else
            actions, policy = BatchedMCTS.think(player, env)
            BatchedMCTS.reset_player!(player)
            GI.play!(env, actions[argmax(policy)])
            moves += 1
        end
    end
    push!(game_times, time() - t0)
    push!(move_counts, moves)
end

avg_time = mean(game_times)
avg_moves = mean(move_counts)
time_per_move = avg_time / avg_moves
fwd_passes_per_move = MCTS_ITERS / BATCH  # approximate
@printf("  Avg game: %.2f s, %.1f moves\n", avg_time, avg_moves)
@printf("  Per move: %.1f ms (%d MCTS iters, ~%d forward passes)\n",
        time_per_move * 1000, MCTS_ITERS, fwd_passes_per_move)
@printf("  Forward pass time: ~%.1f μs (from section 1)\n", t_fwd / n_iters * 1e6)
@printf("  Expected NN time/move: %.1f ms (%.0f%% of actual)\n",
        fwd_passes_per_move * t_fwd / n_iters * 1000,
        fwd_passes_per_move * t_fwd / n_iters / time_per_move * 100)
overhead_pct = (1.0 - fwd_passes_per_move * t_fwd / n_iters / time_per_move) * 100
@printf("  MCTS overhead (traverse+backprop+game): %.0f%%\n", overhead_pct)

# --- Multi-threaded game throughput ---
println("\n--- Multi-threaded game throughput ---")
for nw in [1, min(4, NUM_WORKERS), min(8, NUM_WORKERS), min(16, NUM_WORKERS), NUM_WORKERS]
    nw > Threads.nthreads() && continue
    nw < 1 && continue

    games_per_worker = max(2, NUM_GAMES ÷ nw)

    t = @elapsed begin
        Threads.@threads for w in 1:nw
            w_rng = MersenneTwister(42 + w)
            _, w_batch_oracle = make_cpu_oracles(:fast, net, cfg; batch_size=BATCH, nslots=1)
            w_player = BatchedMCTS.BatchedMctsPlayer(gspec, single_oracle, mcts_params;
                batch_size=BATCH, batch_oracle=w_batch_oracle)
            w_env = GI.init(gspec)
            for _ in 1:games_per_worker
                GI.set_state!(w_env, GI.init(gspec))
                while !GI.game_terminated(w_env)
                    if GI.is_chance_node(w_env)
                        outcomes = GI.chance_outcomes(w_env)
                        GI.apply_chance!(w_env, outcomes[rand(w_rng, eachindex(outcomes))][1])
                    else
                        actions, policy = BatchedMCTS.think(w_player, w_env)
                        BatchedMCTS.reset_player!(w_player)
                        GI.play!(w_env, actions[argmax(policy)])
                    end
                end
            end
        end
    end
    total_games = nw * games_per_worker
    gps = total_games / t
    gpm = gps * 60
    @printf("  %2d workers: %5.1f games/sec (%6.0f games/min), %.1f sec for %d games\n",
            nw, gps, gpm, t, total_games)
end

#####
##### Section 4: Julia native code inspection
#####

println("\n" * "=" ^ 70)
println("SECTION 4: Code quality check")
println("=" ^ 70)

# Check if SIMD is actually being used
println("\n--- LLVM vectorization check ---")
C_check = zeros(Float32, WIDTH, BATCH)
A_check = fw.res_W1[1]
B_check = fb.h1
bias_check = fw.res_b1[1]

code = sprint(io -> code_llvm(io, _gemm_bias!,
    Tuple{typeof(C_check), typeof(A_check), typeof(B_check), typeof(bias_check), Int};
    debuginfo=:none, optimize=true))
has_simd = occursin("vector", lowercase(code)) || occursin("x4", code) || occursin("x8", code)
has_fma = occursin("fma", lowercase(code)) || occursin("fmuladd", lowercase(code))
println("  SIMD vectorized: $has_simd")
println("  FMA instructions: $has_fma")
println("  LLVM IR length: $(length(code)) chars")

# Check for allocations in hot path
println("\n--- Allocation check ---")
allocs = @allocated fast_forward_normalized!(fw, fb, X, A, BATCH)
@printf("  Forward pass allocations: %d bytes\n", allocs)

allocs2 = @allocated begin
    actions, policy = BatchedMCTS.think(player, env)
    BatchedMCTS.reset_player!(player)
end
@printf("  MCTS think allocations: %d bytes\n", allocs2)

println("\n" * "=" ^ 70)
println("DONE")
println("=" ^ 70)
