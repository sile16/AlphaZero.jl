# Inference Performance Assessment (2026-03-19)

Comprehensive benchmark results and analysis for AlphaZero.jl neural network
inference across Neo (M3 Max) and Jarvis (i7-10700K). Goal: identify the fastest
inference path for eval scripts and self-play on each machine.

## Hardware

| | Neo | Jarvis |
|---|---|---|
| CPU | M3 Max (14P + 8E = 32 threads) | i7-10700K (8C/16T) |
| GPU | 40-core Apple GPU, 128-bit NEON | RTX 4090 24GB |
| Memory | 128GB unified, 400 GB/s | 64GB DDR4, ~50 GB/s |
| BLAS | Apple Accelerate | Intel MKL |
| Julia | 1.12.5 | 1.12.4 |

## Model Under Test

FCResNetMultiHead 256w×5b (~1.1M parameters):
- Input: 344 (minimal_flat obs) → 256 hidden
- 5 residual blocks (each: LN+ReLU → Dense → LN+ReLU → Dense + skip)
- 5 value heads (sigmoid each) + policy head (2 layers + softmax)
- ~12 sequential matmuls in the forward pass
- Action space: 676 actions

## Inference Backends Tested

1. **Flux** — Standard Julia deep learning. Uses BLAS internally.
2. **FastWeights** — Pure Julia GEMM (`_gemm_bias!` in `src/inference/fast_weights.jl`). Tile_k=64, 4-column unrolling, no BLAS. Thread-safe by design.
3. **ONNX Runtime CPU** — Python, uses oneDNN/MKL internally.
4. **ONNX Runtime CoreML EP** — Python, dispatches to Apple Neural Engine.
5. **Metal GPU** — Apple GPU via Metal.jl (`Flux.gpu` path).
6. **Octavian.jl** — Julia BLAS replacement using LoopVectorization `@turbo`.

---

## 1. Raw GEMM Benchmarks (256×256 × 256×batch)

### Neo (M3 Max)

Single isolated GEMM (not full forward pass), single thread, `BLAS.set_num_threads(1)`:

| Batch | BLAS (Accelerate) | FastWeights | BLAS/Fast |
|-------|------------------|-------------|-----------|
| 1 | 302,569 | 227,181 | 1.33x |
| 10 | 64,083 | 318,173 | 0.20x |
| 25 | 122,851 | 332,039 | 0.37x |
| 50 | 234,486 | 333,314 | 0.70x |

Multi-threaded (22 workers, batch=50, each doing independent GEMMs):

| Backend | Total st/s | Per Worker | Ratio |
|---------|-----------|------------|-------|
| BLAS (Accelerate) | 283,014 | 12,864 | 1.0x |
| **FastWeights** | **5,902,551** | **268,298** | **20.9x** |

**Apple Accelerate BLAS collapses under thread contention.** Even with
`set_num_threads(1)`, 22 concurrent callers cause catastrophic slowdown.
Each worker drops from 234K→12.8K st/s (18x degradation). FastWeights
maintains near-linear scaling because it uses zero shared state.

### Jarvis (i7-10700K)

Single GEMM, single thread:

| Batch | BLAS (MKL) | FastWeights | BLAS/Fast |
|-------|-----------|-------------|-----------|
| 1 | 27,652 | 263,494 | 0.10x |
| 10 | 217,207 | 366,446 | 0.59x |
| 25 | 429,528 | 376,165 | 1.14x |
| 50 | 621,480 | 379,124 | 1.64x |

Multi-threaded (22 workers via 16 threads, batch=50):

| Backend | Total st/s | Per Worker | Ratio |
|---------|-----------|------------|-------|
| **BLAS (MKL)** | **8,487,254** | **385,784** | **1.78x** |
| FastWeights | 4,763,775 | 216,535 | 1.0x |

**MKL BLAS has no contention problem.** It handles 22 concurrent callers
gracefully and outperforms FastWeights at batch≥25 due to AVX2-optimized
microkernels. FastWeights' pure Julia GEMM cannot match MKL's hand-tuned
assembly. The situation is exactly reversed vs Neo.

---

## 2. Full Forward Pass (single thread, batch=50)

12 sequential GEMMs simulating the full network forward pass:

| Machine | BLAS | FastWeights | Winner |
|---------|------|-------------|--------|
| Neo | 20,965 st/s | 26,315 st/s | FastWeights 1.26x |
| Jarvis | 47,470 st/s | 29,269 st/s | BLAS 1.62x |

**Jarvis single-thread is 1.8x faster than Neo** for full forward pass
despite M3 Max having more theoretical compute. This is because MKL BLAS
at batch=50 exploits AVX2 FMA (256-bit) better than our pure Julia GEMM
exploits NEON (128-bit).

---

## 3. End-to-End Eval Game Throughput

50 games vs wildbg, 100 MCTS iterations per move, 256w×5b dual-model:

### FastWeights (current, post-optimization)

| Machine | Workers | Games/min | Time for 100 games |
|---------|---------|-----------|-------------------|
| Neo | 24 | ~4,987 | 0.1 min (6s play + 8s startup) |
| Jarvis | 14 | ~2,810 | 0.2 min (12s total) |

### Flux (previous, pre-optimization)

| Machine | Workers | Games/min | Time for 100 games |
|---------|---------|-----------|-------------------|
| Neo | 24 | ~274 | ~22 min |
| Jarvis | 14 | ~889 | ~6.7 min |

### Speedup from Flux → FastWeights

| Machine | Speedup |
|---------|---------|
| **Neo** | **18.2x** |
| **Jarvis** | **3.2x** |

Neo benefits enormously because Accelerate BLAS contention is eliminated.
Jarvis benefits less because MKL handles contention better, and FastWeights
GEMM is slower than MKL.

---

## 4. Metal GPU Benchmarks (Neo M3 Max only)

### Single GEMM: CPU FastWeights vs Metal GPU

| Batch | CPU (st/s) | GPU (st/s) | GPU/CPU |
|-------|-----------|-----------|---------|
| 1 | 237,675 | 10,416 | 0.04x |
| 10 | 317,955 | 78,581 | 0.25x |
| 50 | 321,906 | 147,708 | 0.46x |
| 200 | 328,510 | 340,191 | 1.04x |
| 500 | 329,444 | 846,285 | 2.57x |

GPU crossover for single GEMM at batch ≈ 200.

### Full Forward Pass (12 sequential GEMMs): CPU vs GPU

| Batch | CPU (st/s) | GPU (st/s) | GPU/CPU |
|-------|-----------|-----------|---------|
| 50 | 20,897 | 77,410 | **3.7x** |
| 100 | 25,958 | 155,213 | **6.0x** |
| 200 | 26,738 | 304,331 | **11.4x** |
| 500 | 26,893 | 759,273 | **28.2x** |

**For the full forward pass, GPU wins even at batch=50** because the ~0.1ms
kernel launch overhead is amortized across 12 layers. At batch=500, GPU is
28x faster than a single CPU thread.

### Batched GPU Forward (multiple eval positions aggregated)

| Total Batch | GPU st/s |
|-------------|----------|
| 50 | 76,527 |
| 200 | 222,385 |
| 500 | 770,328 |
| 1,000 | 1,464,128 |
| 2,000 | 2,422,423 |

At batch=2000, the GPU delivers 2.4M st/s — approaching the 5.9M total
from 22 CPU workers, but from a single thread. This could free all CPU
cores for MCTS tree operations.

### GPU Constraints

- **Metal is NOT thread-safe** — cannot call from multiple threads concurrently
- Previous GPU-Lock approach (6 GPU workers): 2.36x speedup over CPU-only eval
- Previous producer-consumer server: 0.49x (slower, channel/wait overhead)
- Previous speculative prefetch: 0.09x (13M positions generated, 400 used)
- Kernel launch overhead: ~0.1ms per dispatch (constant regardless of batch)

---

## 5. ONNX Runtime Benchmarks (Neo only)

Model exported via: Julia .data → binary weights → PyTorch → ONNX (opset 17).
Tested with onnxruntime 1.24.4.

### Single-thread Forward Pass

| Batch | CoreML EP (st/s) | CPU EP (st/s) | FastWeights Julia (st/s) |
|-------|-----------------|---------------|------------------------|
| 1 | 797 | 16,486 | 7,313 |
| 10 | 7,448 | 39,241 | ~16,000 |
| 25 | 16,430 | 60,731 | ~17,000 |
| 50 | 31,353 | 91,188 | 17,230 |
| 100 | 49,346 | 131,099 | 17,240 |

**ONNX Runtime CPU is 5.3x faster than FastWeights single-threaded** at
batch=50. It uses oneDNN internally with optimized NEON kernels. However,
FastWeights' constant ~17K regardless of batch suggests our GEMM is
memory-bound, not compute-bound, for these matrix sizes.

**CoreML EP is slower than CPU EP** for this model size. The ANE dispatch
overhead (~1.2ms) dominates actual compute for a 1.1M parameter MLP.
CoreML/ANE targets large vision/language models, not small MLPs.

### Multi-thread ONNX Runtime CPU (Python threads)

| Workers | intra_threads=1 (st/s) | intra_threads=4 (st/s) |
|---------|----------------------|----------------------|
| 1 | 39,526 | 102,082 |
| 4 | 145,505 | 202,789 |
| 8 | 268,567 | 292,002 |
| 16 | 337,050 | 328,453 |
| 24 | 306,040 | 300,597 |

Peak at ~337K st/s (16 workers, intra=1), similar to FastWeights multi-thread
(~324K). Both converge because memory bandwidth becomes the bottleneck at
scale, not compute.

---

## 6. Octavian.jl / LoopVectorization

- **Status**: LoopVectorization is deprecated for Julia 1.11+. Successor
  (LoopModels) is in development but not usable yet.
- Both packages install on Julia 1.12 with deprecation warnings.
- **Octavian segfaults in multi-threaded mode** on Julia 1.12 + Apple Silicon
  (SIGSEGV in `_vstore!` during `packamul!`). Unusable for our workload.
- Single-threaded Octavian was not benchmarked due to the segfault blocking
  the test script.

---

## 7. Apple Neural Engine (ANE) Direct Access

Investigated the reverse-engineered ANE project ([maderix/ANE](https://github.com/maderix/ANE))
which bypasses CoreML to access ANE hardware directly via private `_ANEClient`/`_ANECompiler` APIs.

### Key Findings

- ANE is a **graph execution engine** — submits entire computation graphs atomically
  (no per-layer kernel launch overhead unlike Metal/CoreML EP)
- Peak: 18.6 TOPS FP16 on M4 (M3 Max similar or better)
- Inference for 109M-param model: 8.8ms. Our 1.1M-param model would be sub-millisecond
- Written in Objective-C, would need C bridge + `ccall` for Julia
- Private APIs, no stability guarantee across macOS versions
- ~119 compile limit per process (workaround: `exec()` restart)
- Actual utilization: only 5-9% of peak TOPS achieved

### Assessment for Our Use Case

ANE dispatch overhead makes it unsuitable for our tiny 1.1M-param MLP. The
overhead-to-compute ratio is unfavorable. ANE shines at 100M+ parameter models.
Our CPU ONNX Runtime already does a full forward in 0.55ms (batch=50) — ANE
cannot meaningfully beat this for models this small. Would become relevant if
model scales to 256w×10b+ (~10M params).

---

## 8. Locking Analysis

### Current Locking in Eval Hot Path

The FastWeights oracle in both `eval_vs_wildbg.jl` and `eval_race.jl` uses a
**global `ReentrantLock`** to manage per-task buffer allocation:

```julia
bufs_lock = ReentrantLock()

function get_bufs()
    tid = objectid(current_task())
    lock(bufs_lock) do          # <-- CONTENDED BY ALL 22+ WORKERS
        if !haskey(bufs, tid)
            bufs[tid] = (FastBuffers(...), FastInputBuffers(...), ...)
        end
        return bufs[tid]
    end
end
```

This lock is taken on **every batch oracle call** (hundreds of times per game,
thousands per eval run). After initialization it's just `haskey` + return, but
`ReentrantLock` still involves atomic CAS operations that cause cache line
bouncing across 22 cores.

### Fix: Pre-allocate Per-Worker Buffers

The selfplay_client already does this correctly — it allocates one `FastBuffers`
per worker thread at startup (lines 824-825) and passes them directly. No locking
in the hot path.

**Recommended fix for eval scripts**: Pre-allocate a `Vector{Tuple{FastBuffers, FastInputBuffers}}`
of size `num_workers` before the game loop, and index by worker thread ID:

```julia
# Before game loop:
worker_bufs = [(FastBuffers(width, NUM_ACTIONS, max_batch),
                FastInputBuffers(max_batch)) for _ in 1:num_workers]

# In game loop (tid is 1:num_workers):
fb, fib = worker_bufs[tid]
# No lock needed — each worker owns its buffers
```

### Other Locking

- `GPU_LOCK` (ReentrantLock) — Only used when `--gpu-workers > 0`. Required because
  Metal is not thread-safe. Not in default code path.
- `Threads.Atomic{Int}` for game claiming — Minimal overhead, single atomic_add per game.
  Not a bottleneck.
- **BatchedMCTS has no locking** — Single-threaded per game.

### Estimated Impact

The lock itself is ~50-100ns per acquisition under low contention, but with 22
threads all calling `get_bufs()` in tight loops, cache line invalidation from the
atomic CAS can add 500ns-1μs per call. With ~100 oracle calls per game and 100
games, that's ~10ms total — small relative to game time but unnecessary.

---

## 9. Recommendations

### Immediate (low effort, high impact)

1. **Keep the shared backend/oracle refactor as the only CPU inference path**.
   Eval and self-play now share one implementation in
   `src/inference/backgammon_oracles.jl`. Do not reintroduce script-specific
   CPU inference paths.

2. **Use production eval throughput, not just raw oracle throughput, to choose
   the x86 default**:
   - Neo (macOS/ARM): `FastWeights` remains the expected default.
   - Jarvis (Linux/x86): still unresolved after refactor. Raw shared-oracle
     throughput favored `FastWeights` at batch=16, but tiny production eval
     still favored `Flux/BLAS`.
   - Keep `auto` as-is for now, then validate it with a larger Jarvis matrix.

3. **Treat the old `bufs_lock` item as completed**.
   The eval hot-path lock was removed by moving to preallocated per-thread
   buffers in the shared backend layer.

### Medium-term (moderate effort)

4. **GPU inference server for Neo eval** — Batch inference requests from
   all worker threads into large GPU batches via a Channel. At batch=500,
   GPU delivers 759K st/s (28x single-CPU-thread). Even with channel
   overhead, should beat 22-worker CPU for full forward pass. Key: batch
   across games, not within single MCTS (previous failed attempt batched
   within MCTS iterations).

5. **Optimize FastWeights GEMM for NEON** — Current GEMM is generic Julia.
   ONNX Runtime CPU (which uses optimized NEON kernels) is 5.3x faster
   single-threaded. Options:
   - Use `@simd` with explicit 4-wide Float32 operations matching NEON
   - Call Apple Accelerate `cblas_sgemm` with process-wide BLAS lock
     (1 lock per forward pass vs per-GEMM — amortizes contention)
   - Use Appleʼs `vDSP_mmul` for small matrix specialization

6. **CUDA inference on Jarvis** — RTX 4090 should massively outperform
   i7-10700K CPU for batched inference. Jarvis eval could use Flux GPU
   path with CUDA. Not yet tested.

### Long-term (high effort)

7. **ONNX Runtime integration** — Call libonnxruntime from Julia via `ccall`
   for both CPU and CoreML/CUDA execution providers. Gets best-of-platform
   optimized kernels without maintaining custom GEMM code. Single-threaded
   is 5x faster; multi-threaded converges. Main value: simpler code.

8. **ANE direct access** — Only worthwhile if model scales to 10M+ params.
   For current 1.1M param model, dispatch overhead dominates.

---

## 10. Key Takeaways

1. **The #1 performance issue was BLAS contention on Neo** — switching to
   FastWeights gave 18.2x speedup. This is now fixed.

2. **Before the refactor, Neo and Jarvis appeared to have opposite optimal
   CPU backends** — FastWeights for Neo, BLAS for Jarvis. After moving both
   eval and self-play to one shared backend/oracle layer, Neo still points to
   FastWeights, but Jarvis now needs production-path revalidation before we
   lock in the x86 default.

3. **The M3 Max GPU is underutilized** — At batch=500, it's 28x faster than
   a single CPU thread for full forward. A GPU inference server batching
   across eval games could be the next big win on Neo.

4. **Our GEMM is 5x slower than ONNX Runtime CPU** single-threaded on Neo.
   The pure Julia GEMM doesn't fully exploit NEON SIMD. This gap is hidden
   in multi-threaded mode because memory bandwidth becomes the bottleneck.

5. **For our tiny 1.1M param model**, dispatch overhead kills GPU/ANE/CoreML
   for single forward passes. Batching is essential to amortize overhead.

6. **Jarvis is faster than Neo for single-thread inference** despite lower
   theoretical compute, because MKL BLAS is better optimized than our Julia
   GEMM for the M3 Max NEON architecture.

---

## 11. Post-Refactor Validation (2026-03-19)

After consolidating eval/self-play CPU inference onto the shared
`src/inference/backgammon_oracles.jl` path with `--inference-backend=auto|fast|flux`,
we re-ran small sanity checks on both Neo and Jarvis.

### Implementation Status

- Eval and self-play now share one CPU oracle/backend layer.
- `auto` currently resolves to:
  - `FastWeights` on Apple
  - `Flux/BLAS` on non-Apple
- Eval hot-path `bufs_lock` was removed as part of the consolidation.
- Shared oracle fix: **chance nodes now produce an empty action mask**. This was
  a correctness issue in the refactored packer and is now fixed.

### Neo (M3 Max) Post-Refactor

Tiny production eval (`eval_vs_wildbg.jl`, 4 workers, 2 games/side, 20 MCTS iters,
batch=16):

- `auto` → `FastWeights`: **0.1 min**
- `flux`: **0.1 min**

Both complete successfully through the production path, with `auto` resolving to
`FastWeights` as intended. At this tiny scale the wall-clock difference is too small
to be decisive, but the raw oracle benchmark still strongly favors `FastWeights`.

### Jarvis (i7-10700K) Post-Refactor

Shared raw oracle benchmark (`bench_eval_backends.jl`, 4 workers, batch=16):

- `auto` → `Flux/BLAS`: 29.1K st/s total
- `fast`: **40.6K st/s total**
- `flux`: 31.7K st/s total

Tiny production eval (`eval_vs_wildbg.jl`, 4 workers, 2 games/side, 20 MCTS iters,
batch=16):

- `fast`: **0.2 min**
- `flux`: **0.1 min**

### Revised Interpretation

The raw shared-oracle benchmark and end-to-end eval disagree on Jarvis:

- **Raw oracle throughput favors `FastWeights`** at batch=16 in the refactored path.
- **End-to-end eval still favors `Flux/BLAS`** for this tiny production test.

This means the current `auto` heuristic is **good enough to keep for now**, but it
should be treated as provisional rather than proven. We should not choose the x86
default based on raw oracle throughput alone.

### Practical Insights / Caveats

- **The benchmark that now matters most is the production path**:
  `eval_vs_wildbg.jl` and `eval_race.jl` with `--inference-backend=...`.
  Raw oracle throughput is useful for diagnosis, but it is not sufficient for
  choosing the default on x86.
- **The shared raw benchmark uncovered two real integration issues**:
  - chance nodes must produce an empty action mask in the shared packer
  - multithreaded synthetic benchmarks must clone per-worker states, because
    backgammon observation is not safe to run concurrently on the same state object
- **Do not overfit to tiny runs**. The post-refactor production checks were only
  2 games/side and 20 MCTS iterations. They are useful for smoke testing and
  directionality, not for final policy.

### Updated Priority

1. Keep the shared backend/oracle refactor as the base path.
2. Use **production eval throughput** (`eval_vs_wildbg.jl`, `eval_race.jl`) as the
   deciding signal for the x86 default, not just raw oracle throughput.
3. Run a larger Jarvis comparison matrix before changing `auto`:
   - batch sizes: 16, 32, 50
   - workers: 4, 8, 14
   - backends: `fast`, `flux`
4. Only after that matrix should we decide whether Jarvis stays on `Flux/BLAS`
   by default or flips to `FastWeights` for some regimes.
5. SIMD work remains Apple-focused. x86 should only get custom GEMM effort if
   end-to-end testing shows `FastWeights` consistently beats MKL in production.

### Immediate Next Step Matrix

This matrix was completed on Jarvis on 2026-03-19 before changing `auto`:

1. `eval_vs_wildbg.jl` with `--inference-backend=fast`
2. `eval_vs_wildbg.jl` with `--inference-backend=flux`
3. Repeat for:
   - `--inference-batch-size=16,32,50`
   - `--num-workers=4,8,14`
   - `--mcts-iters=50` first, then `100`
4. Record:
   - wall time
   - games/min
   - any instability / contention symptoms
5. Only after that choose whether x86 `auto` should stay on `Flux/BLAS` or move
   to `FastWeights` for some or all regimes.

### Jarvis Production Matrix Results (2026-03-19)

We ran the full production-path Jarvis matrix from the shared refactored code:

- Script: `scripts/eval_vs_wildbg.jl`
- Checkpoint: `contact_latest.data`
- Common args: `256w×5b + 128w×3b`, `--num-games=10`
- Backends: `fast`, `flux`
- Workers: `4`, `8`, `14`
- Batch sizes: `16`, `32`, `50`
- MCTS iterations: `50`, `100`
- Total runs: **36**
- Failures: **0**

#### Reported Script Results

Every run completed successfully and the script-reported summary was identical:

| Backend | Workers | Batch | MCTS iters | Games | Script Time | Derived games/min | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| `fast` | 4, 8, 14 | 16, 32, 50 | 50, 100 | 20 | 0.2 min | 100.0 | pass |
| `flux` | 4, 8, 14 | 16, 32, 50 | 50, 100 | 20 | 0.2 min | 100.0 | pass |

#### What This Means

This matrix did **not** separate `fast` from `flux` on Jarvis, but the reason is
measurement precision rather than confirmed equivalence:

- `eval_vs_wildbg.jl` currently prints time as `round(eval_time / 60, digits=1)`.
- With only 20 total games per run, many distinct wall-clock durations collapse
  to the same printed `0.2 min`.
- External wrapper timing showed small jitter, with `fast` often around `25s`
  and `flux` around `26-27s`, but that gap is too small and noisy to treat as
  a reliable backend winner.

#### Updated Interpretation

- The Jarvis matrix is still useful as a **stability check**: both backends ran
  cleanly across workers `4-14`, batch sizes `16-50`, and MCTS iterations
  `50-100`.
- It is **not sufficient** to choose the x86 default.
- `auto` should therefore remain **provisional** on x86 until we rerun with
  higher-resolution timing or larger runs.

#### Revised Next Step

Before changing Jarvis defaults, rerun a smaller focused comparison with one of:

1. **Higher-resolution timing**:
   - print `eval_time` with at least 3 decimal places in minutes or directly in seconds
   - optionally wrap each run in `/usr/bin/time -p`
2. **Larger workloads**:
   - increase `--num-games` enough that backend differences exceed the script's
     current rounding granularity
   - keep the same production path and compare `fast` vs `flux`

Until then, the safest policy is:

- **Neo**: keep `FastWeights` as default
- **Jarvis/x86**: keep `Flux/BLAS` as the current `auto` default, but treat it as
  unproven rather than final

---

## Appendix: Reproduction Commands

```bash
# GEMM benchmark (both machines)
julia --threads 28 --project scripts/bench_gemm.jl

# Shared raw backend comparison (Neo or Jarvis)
julia --threads 8 --project scripts/bench_eval_backends.jl \
    /homeshare/projects/AlphaZero.jl/sessions/distributed_20260213_031243_per_reanalyze/checkpoints/contact_latest.data \
    --width=256 --blocks=5 --race-width=128 --race-blocks=3 \
    --num-workers=4 --batch-size=16 --backends=auto,fast,flux --raw-positions=32

# Metal GPU benchmark (Neo only)
julia --threads 4 --project scripts/bench_metal_gemm.jl

# ONNX export pipeline
julia --project scripts/quick_export_onnx.jl <checkpoint> /tmp/model.bin 256 5
python3 /tmp/convert_to_onnx.py /tmp/model.bin /tmp/model.onnx
python3 /tmp/bench_coreml.py /tmp/model.onnx       # single-thread
python3 /tmp/bench_coreml_mt.py /tmp/model.onnx     # multi-thread

# Full production eval through the shared backend layer
julia --threads 28 --project scripts/eval_vs_wildbg.jl <checkpoint> \
    --width=256 --blocks=5 --num-workers=24 --mcts-iters=100 --num-games=500 \
    --inference-backend=auto
```

## Appendix: File Locations

| File | Purpose |
|------|---------|
| `src/inference/fast_weights.jl` | FastWeights/FastBuffers, pure Julia GEMM |
| `src/inference/backgammon_oracles.jl` | Shared CPU backend/oracle layer for eval + self-play |
| `scripts/eval_vs_wildbg.jl` | Full-game eval via shared backend (`auto|fast|flux`) |
| `scripts/eval_race.jl` | Race eval via shared backend (`auto|fast|flux`) |
| `scripts/selfplay_client.jl` | Self-play via shared backend (`auto|fast|flux`) |
| `scripts/bench_eval_backends.jl` | Shared raw CPU backend benchmark (`auto|fast|flux`) |
| `scripts/bench_gemm.jl` | Raw GEMM benchmark (BLAS vs FastWeights) |
| `scripts/bench_metal_gemm.jl` | Metal GPU vs CPU forward pass benchmark |
| `scripts/bench_inference.jl` | Forward pass scaling benchmark |
| `/tmp/bench_coreml.py` | ONNX Runtime single-thread benchmark |
| `/tmp/bench_coreml_mt.py` | ONNX Runtime multi-thread benchmark |
