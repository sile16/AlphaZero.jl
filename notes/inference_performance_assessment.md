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

2. **Use `FastWeights` as the production CPU default on both Neo and Jarvis**:
   - Neo (macOS/ARM): `FastWeights` wins both eval and self-play by clear margins.
   - Jarvis (Linux/x86): focused production benchmarks now also favor
     `FastWeights` for self-play, and for eval it is both faster and more stable
     at representative worker counts.
   - `Flux/BLAS` should remain available as a fallback/diagnostic backend, not
     the default.

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

2. **Production-path benchmarks matter more than raw GEMM or raw oracle speed**.
   After moving eval and self-play to one shared backend/oracle layer, the
   current production winner is `FastWeights` on both Neo and Jarvis, even
   though older low-level x86 measurements suggested MKL/BLAS might win.

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
- `auto` now resolves to:
  - `FastWeights` on Apple
  - `FastWeights` on non-Apple
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

The initial post-refactor Jarvis read was inconclusive because tiny eval runs and
coarse timer rounding hid the backend difference. Focused production benchmarks
with exact timing changed that conclusion:

- **Neo**: `FastWeights` clearly wins eval and self-play.
- **Jarvis self-play**: `FastWeights` clearly wins.
- **Jarvis eval**: `FastWeights` wins at representative worker counts, and the
  `Flux/BLAS` path is currently unstable at the 14-worker eval setting.

We should therefore choose the x86 default from the production path, not from raw
oracle throughput or historical GEMM results.

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
2. Keep `FastWeights` as the default CPU backend on both Neo and Jarvis.
3. Treat `Flux/BLAS` as a fallback/diagnostic path until the Jarvis eval
   instability is understood.
4. SIMD work remains Apple-focused. That is still the best place to improve the
   remaining CPU inference gap.
5. Only after the Apple CPU path is tighter should we spend more time on x86
   backend tuning.

### Immediate Next Step Matrix

This matrix was completed on Jarvis on 2026-03-19 before changing `auto`.
It was useful as a stability check, but not decisive for backend selection
because the timer rounded everything to `0.2 min`.

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

This matrix did **not** separate `fast` from `flux` on Jarvis, but the reason was
measurement precision rather than confirmed equivalence:

- `eval_vs_wildbg.jl` currently prints time as `round(eval_time / 60, digits=1)`.
- With only 20 total games per run, many distinct wall-clock durations collapse
  to the same printed `0.2 min`.
- External wrapper timing showed small jitter, with `fast` often around `25s`
  and `flux` around `26-27s`, but that gap is too small and noisy to treat as
  a reliable backend winner.

#### Updated Interpretation

- The Jarvis matrix was still useful as a **stability check**.
- It was **not sufficient** to choose the x86 default.
- We followed it with focused production runs using higher-resolution timing and
  representative worker counts. Those runs are the basis for the current default.

#### Revised Next Step

Before changing Jarvis defaults, rerun a smaller focused comparison with one of:

1. **Higher-resolution timing**:
   - print `eval_time` with at least 3 decimal places in minutes or directly in seconds
   - optionally wrap each run in `/usr/bin/time -p`
2. **Larger workloads**:
   - increase `--num-games` enough that backend differences exceed the script's
     current rounding granularity
   - keep the same production path and compare `fast` vs `flux`

That measurement issue is now resolved for the production recommendation below.

## 12. Focused Production Benchmarks (2026-03-19)

To answer the real deployment question directly, we ran focused production-path
benchmarks for eval and self-play on both target machines.

### Neo (M3 Max)

#### Eval (`eval_vs_wildbg.jl`)

Representative config:

- 24 workers
- 100 MCTS iterations
- inference batch size 50
- dual-model checkpoint (`256w×5b + 128w×3b`)
- 5 games/side (10 total)

| Backend | Seconds | Games/min | Result |
|---|---:|---:|---|
| `fast` | **5.21** | **115.2** | winner |
| `flux` | 8.31 | 72.2 | slower |

`FastWeights` is **1.60x faster** than `Flux/BLAS` on Neo eval.

#### Self-play (`bench_selfplay_backends.jl`)

Representative config:

- 22 workers
- 400 MCTS iterations
- inference batch size 50
- dual-model checkpoint
- 20 self-play games

| Backend | Seconds | Games/min | Result |
|---|---:|---:|---|
| `fast` | **6.23** | **192.7** | winner |
| `flux` | 120.40 | 10.0 | much slower |

`FastWeights` is **19.3x faster** than `Flux/BLAS` on Neo self-play.

### Jarvis (i7-10700K)

#### Eval (`eval_vs_wildbg.jl`)

Representative production config:

- 14 workers
- 100 MCTS iterations
- inference batch size 50
- dual-model checkpoint
- 20 games/side (40 total)

| Backend | Seconds | Games/min | Result |
|---|---:|---:|---|
| `fast` | **9.89** | **242.6** | winner |
| `flux` | failed | failed | `BoundsError` in `BatchedMCTS.traverse_to_leaf!` at 14 workers |

To check whether this was only a high-worker issue, we reran Jarvis eval at
8 workers:

| Backend | Seconds | Games/min | Result |
|---|---:|---:|---|
| `fast` | **9.84** | **243.8** | winner |
| `flux` | 11.73 | 204.6 | slower |

So on Jarvis eval, `FastWeights` is:

- **faster** than `Flux/BLAS` at 8 workers (`1.19x`)
- **stable** at both 8 and 14 workers
- currently the only backend that completed the representative 14-worker run

#### Self-play (`bench_selfplay_backends.jl`)

Representative config:

- 8 workers
- 400 MCTS iterations
- inference batch size 50
- dual-model checkpoint
- 20 self-play games

| Backend | Seconds | Games/min | Result |
|---|---:|---:|---|
| `fast` | **14.18** | **84.6** | winner |
| `flux` | 26.99 | 44.5 | slower |

`FastWeights` is **1.90x faster** than `Flux/BLAS` on Jarvis self-play.

### Final Current Recommendation

For the current codebase, model size, and production settings:

- **Neo eval**: use `FastWeights`
- **Neo self-play**: use `FastWeights`
- **Jarvis eval**: use `FastWeights`
- **Jarvis self-play**: use `FastWeights`

So the shared `auto` backend should now resolve to `FastWeights` on both target
machines.

### Remaining Caveats

- Jarvis raw GEMM / raw oracle benchmarks and older notes suggested MKL/BLAS
  might be best on x86. The current production-path data does not support using
  that as the default.
- The old `fast` eval mismatches were real correctness bugs in the shared
  scratch path, not harmless timing noise.
- Apple-specific SIMD work remains the most promising next CPU optimization.

## 13. Lessons Learned

These investigations surfaced several classes of problems that matter as much as
raw kernel speed:

### 1. Production-path correctness beats low-level intuition

- Older GEMM and raw-oracle measurements suggested one backend choice.
- Production eval/self-play sometimes pointed the other way.
- The right decision signal is the real workload, not isolated math kernels.

### 2. Coarse measurement can send us in the wrong direction

- The original Jarvis matrix rounded time to `0.1 min`, flattening distinct runs
  into identical `0.2 min` outputs.
- This made a real backend comparison look inconclusive when the instrumentation
  was simply too coarse.

### 3. State aliasing inside MCTS is extremely dangerous

- The lightweight `GI.current_state` optimization returned states that aliased
  pooled game buffers.
- BatchedMCTS stores those states in the tree while reusing pooled game objects.
- That allowed later simulations to mutate data that earlier tree entries still
  depended on, producing hard-to-reason-about action mismatches and crashes.

### 4. Oracle/action contracts must come from one canonical source

- The shared oracle built action masks manually from `legal_actions(state)`.
- The network interface and MCTS semantics are defined in terms of
  `GI.actions_mask(GI.init(gspec, state))` and `GI.available_actions(game)`.
- Reconstructing that logic by hand created silent drift between the oracle's
  policy vector and MCTS's action list.

### 5. Preallocated buffers are easy to misuse

- The shared `flux` path accidentally forwarded the full preallocated batch
  matrix instead of only the active `1:n` columns.
- That likely distorted both speed and stress behavior.
- Reusing buffers is good, but only if the active slice is explicit everywhere.

### 6. Thread-local is not task-local in modern Julia

- The shared `fast` oracle originally keyed scratch buffers by `threadid()`.
- In Julia 1.9+, multiple runnable tasks can share a thread, and a task can
  yield while still "owning" a logical worker.
- That made `threadid()` an unsafe key for mutable scratch state and produced
  policy/action mismatches like `23 vs 21` and `8 vs 19`.
- The fix was to give each concurrent task its own `FastWorkerBuffers`.

### 7. Assertions pay for themselves

- Adding `length(P) == length(actions)` in `init_state_info` turned an obscure
  `BoundsError` deep in traversal into an immediate, actionable contract failure.
- Once the true invariant was checked, the debugging path got much shorter.

## 14. How To Avoid This Class of Issue

### A. Make the oracle contract executable

Add tests that, for random decision states:

- compare `make_cpu_oracles(...).batch_oracle([s])[1][1]` length against
  `length(GI.available_actions(GI.init(gspec, s)))`
- compare the shared backend outputs against `Network.evaluate_batch`
- run this for both `fast` and `flux`

This should be a required invariant test, not an ad hoc debugging step.

### B. Ban aliasing optimizations in tree-stored states unless proven safe

Any optimization that returns a "lightweight clone" of a state should be assumed
unsafe until it passes explicit tests showing:

- no shared mutable buffers
- stable hashing/equality after later environment mutation
- stable legal action sets after pool reuse

If we cannot prove those properties, use an owning clone.

### C. Use the game interface as the single source of truth

Do not hand-roll action masks or action ordering in backend-specific code.

Preferred rule:

- action masks come from `GI.actions_mask(GI.init(gspec, state))`
- action lists come from `GI.available_actions(game)`

Backend code should consume that contract, not reinterpret it.

### D. Keep safety assertions in hot-path integration points

The following assertions are worth keeping even in production or at least behind
an easy debug flag:

- `length(P) == length(actions)` when creating tree nodes
- batched oracle result count equals number of queried states
- no chance-node policies are created

These catch structural bugs before they become meaningless throughput numbers.

### E. Separate "safe baseline" from "optimized path"

Whenever we add a specialized path:

1. first make it match the generic `Network.evaluate_batch` behavior exactly
2. then optimize allocations / pooling / routing
3. then benchmark

That ordering would likely have prevented both the mask mismatch and the aliased
state bug.

### F. Benchmark with enough precision and at the right level

Going forward:

- print seconds and games/min, not just rounded minutes
- keep raw kernel/oracle benchmarks, but treat them as diagnostic only
- make production eval/self-play the final arbiter for backend choice

### G. Stress test concurrency explicitly

For each backend and machine class, add a short stress suite that runs:

- low worker count
- representative worker count
- max expected worker count
- enough games to flush out intermittent contract violations

This is especially important for batched MCTS plus pooled state reuse.

---

## Post-Fix Backend Revalidation

The remaining `fast` mismatch was traced to the shared oracle scratch path:

- reused policy vectors could be mutated by later calls
- scratch buffers were keyed by `threadid()`, which is not safe ownership for
  concurrent Julia tasks
- a `num_workers=1` run passed cleanly, while multi-worker runs reproduced the
  failure, which strongly supported the concurrency diagnosis

The fix set was:

- return owning policy vectors from `fast` oracle results
- replace `threadid()` scratch ownership with per-task scratch ownership
- add regressions for cross-call policy mutation and many-tasks-per-thread load

### Neo Eval vs Wildbg

Representative run:

- `fast`, `8 workers`, `100 MCTS`, `batch=50`, `10 games/side`:
  `5.59 s`, `214.7 games/min`
- `flux`, same settings:
  `9.23 s`, `130.1 games/min`

Result:

- `fast` is `1.65x` faster than `flux` on the production eval path after the
  concurrency fix

### Jarvis Eval vs Wildbg

Representative run:

- `fast`, `14 workers`, `100 MCTS`, `batch=50`, `20 games/side`:
  `11.07 s`, `216.8 games/min`
- `flux`, same settings:
  `13.92 s`, `172.4 games/min`

Result:

- `fast` is `1.26x` faster than `flux` on the production eval path after the
  concurrency fix

### Self-play Benchmark

Shared benchmark script (`scripts/bench_selfplay_backends.jl`) using the same
backend layer:

- Neo, `8 workers`, `100 MCTS`, `batch=50`, `12 games`
  - `fast`: `4.46 s`, `161.4 games/min`
  - `flux`: `16.42 s`, `43.9 games/min`
  - `fast` advantage: `3.68x`

- Jarvis, `14 workers`, `100 MCTS`, `batch=50`, `12 games`
  - `fast`: `4.21 s`, `171.0 games/min`
  - `flux`: `6.11 s`, `117.9 games/min`
  - `fast` advantage: `1.45x`

### Neo Mixed CPU+GPU Eval Matrix

After introducing the shared GPU inference server, we ran a small production
eval matrix on Neo (`50 MCTS`, `batch=16`, `2 games/side`) to check whether
mixed CPU+GPU workers actually beat CPU-only throughput yet.

| CPU | GPU | Time (s) | Games/min |
|---:|---:|---:|---:|
| 1 | 0 | 5.66 | 42.4 |
| 2 | 0 | 4.89 | 49.1 |
| 4 | 0 | 4.81 | 49.9 |
| 1 | 1 | 49.67 | 4.8 |
| 2 | 2 | 33.07 | 7.3 |
| 4 | 2 | 46.78 | 5.1 |

Result:

- The new server path is correct, but it is **not yet a throughput win**.
- At these representative small-batch eval settings, every mixed CPU+GPU
  configuration was much slower than CPU-only `fast`.
- That means the remaining bottleneck is not worker orchestration anymore, but
  per-batch GPU overhead: host packing, `MtlArray(...)` creation, sync, and
  result copies back to CPU.

### Takeaway

- The other engineer's `threadid()` race diagnosis was correct.
- After fixing task ownership of scratch buffers, `fast` is the best validated
  CPU backend on both machines for both eval and self-play.
- The shared GPU server is the right architecture direction, but the current
  Metal implementation still does not justify mixed CPU+GPU production use.
- Current practical default:
  - Neo eval: `fast`
  - Neo self-play: `fast`
  - Jarvis eval: `fast`
  - Jarvis self-play: `fast`
- Backend-selection policy should stay unified unless a future benchmark with
  the corrected code clearly overturns these results.

---

## Legacy Checkpoint Spot-Check

To estimate whether the old state-aliasing issue may have distorted our view of
earlier training runs, we re-ran a small set of older 128w×3b checkpoints under
the fixed code (`8 workers`, `100 MCTS`, `10 games/side`).

| Session | Backend | Outcome | Combined Equity | Games/min | Note |
|---|---|---|---:|---:|---|
| `20260206_baseline_v2_50iter` | `fast` | failed | NA | NA | `Oracle policy/action mismatch` |
| `20260206_baseline_v2_50iter` | `flux` | passed | -2.100 | 181.9 | usable control run |
| `20260206_baseline_50iter` | `fast` | passed | -1.950 | 200.6 | completed cleanly |
| `20260207_per_50iter` | `fast` | passed | -1.950 | 205.2 | completed cleanly |

### Legacy Takeaway

- We do **not** have evidence that all old checkpoints are invalid under the
  fixed code.
- But we also do **not** have evidence that all old checkpoints are cleanly
  comparable under the current `fast` eval path.
- At least one early checkpoint (`baseline_v2_50iter`) still reproduces the
  remaining `fast` contract failure, while nearby checkpoints do not.
- That means the old v1-v5-era question should be treated cautiously:
  the unsafe optimization was present in that era, but its practical impact was
  likely **checkpoint-dependent**, not uniformly catastrophic.

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
