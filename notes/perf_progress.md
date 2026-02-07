# Performance Optimization Progress

## Goal
Maximize simulations/s and games/s on this system at ~80% utilization.
System: 8 cores / 16 HT (i7-10700K @ 3.8GHz) + NVIDIA RTX 4090, 283K param FCResNetMultiHead.

## Progress Log

| # | Architecture | Workers | Games/min (steady) | Notes |
|---|-------------|---------|-------------------|-------|
| 1 | Distributed (GPU_LOCK) | 6 proc | 8.4 | Lock contention destroyed parallelism |
| 2 | Distributed (RemoteChannel, batch=32) | 6 proc | 47 | IPC overhead ~10ms/req |
| 3 | Distributed (RemoteChannel, batch=100) | 6 proc | 72 | Fewer RPC calls/move |
| 4 | Threaded (Channel GPU server, batch=100) | 14 thr | 97 | Zero-copy IPC |
| 5b | + Work-stealing + pre-alloc buffers | 14 thr | 100 | Marginal improvement |
| 7 | + Zero aggregation window | 14 thr | 120 | Removed 0.5ms delay |
| 8 | Green threads (2 per worker) | 14 thr | 97 | WORSE: doubled queue depth |
| 9 | CPU inference (BLAS=1) | 14 thr | 102 | Eliminated GPU queue wait but contention |
| 10 | + Direct actions mask (no GI.init) | 14 thr | ~150 | Eliminated env allocs in oracle |
| 10+overlap | + Training/self-play overlap | 14 thr | ~148 | Training hidden behind self-play |
| 11 | + Lightweight state clones | 14 thr | ~150 | Shared vectors in GI.current_state |
| 11b | + Fast GI.clone (no double-clone) | 14 thr | ~155 | 1 clone instead of 2 per GI.clone |
| **11c** | **+ Shared RNG in clone** | **14 thr** | **~185** | **Eliminated 1.1GB MersenneTwister allocs/iter** |
| 12 | GPU inference (channel server) | 14 thr | ~150 | WORSE: channel overhead > GPU speed gain |
| 13 | + Cached actions + inline UCT + train cap | 14 thr | ~226 | Eliminated ~1.15GB/worker traversal allocs + capped training batches |
| 13b | + sizehint + fast available_actions | 14 thr | ~249 | Pre-sized vectors, direct BackgammonNet.legal_actions, cached Dirichlet |
| 14 | + Game pool + in-place vectorize + trace actions | 14 thr | ~310 | Zero-alloc GI.clone_into!, observe_minimal_flat! into buffer, skip GI.init in trace |
| **14b** | **+ Pre-alloc PendingSimulation vectors** | **14 thr** | **~316** | **Reuse path/rewards/player_switches across sims via sim_pool** |
| 15a | CPU inference server (centralized BLAS=1) | 14 thr | ~85 | WORSE: serialization kills 13.9x parallelism |
| 15b | GPU mailbox (aggressive aggregation) | 14 thr | ~150* | WORSE: GPU overhead > CPU BLAS for 283K model |
| **15c** | **+ Allocation-free forward pass** | **14 thr** | **~800** | **Zero-alloc Dense/LN/relu/softmax via BLAS mul! + manual ops** |
| **16** | **+ SIMD + fused LN+relu + fill! removal** | **14 thr** | **~835** | **@simd loops, fused layernorm_relu!, skip redundant fill!** |
| 17 | + ActionStats vector pool | 14 thr | ~790 | WORSE: variable-size resize! fragments memory |
| **18** | **+ Dict get() + inline VL + training warmup** | **14 thr** | **~849** | **7→2 Dict lookups/node, JIT warmup: iter 2 87.5→18s** |

\* Estimated from 77K states/s GPU throughput vs 107K CPU baseline.

## Key Milestones
- **v16 → v18**: 835 → 849 games/min (**+1.7%** steady-state, **+50%** for short runs via JIT warmup)
- **v15c → v18**: 800 → 849 games/min (**+6%**)
- **v14b → v18**: 316 → 849 games/min (**+169%**)
- **v7 → v18**: 120 → 849 games/min (**+608%**)
- **v1 → v18**: 8.4 → 849 games/min (**101x improvement**)

## Per-Worker Timing Breakdown (v14b, 200 games/iter, iter 4-5 avg)

| Component | Time | % | Description |
|-----------|------|---|-------------|
| vectorize | 2.5s | 7% | In-place observe_minimal_flat! (no intermediate alloc) |
| cpu_forward | 27.5s | 72% | Network.forward_normalized on CPU (Flux, alloc contention) |
| mcts_pure | 7.9s | 21% | Tree traversal, pool clone, backprop |
| **Total** | **38.0s** | | ~4400 oracle calls, 14 workers |

## Per-Worker Timing Breakdown (v15c, 200 games/iter, iter 4-5 avg)

| Component | Time | % | Description |
|-----------|------|---|-------------|
| vectorize | 1.6s | 13% | In-place observe_minimal_flat! |
| cpu_forward | 6.3s | 53% | Allocation-free forward (BLAS mul! + manual LN/relu) |
| mcts_pure | 4.0s | 34% | Tree traversal, pool clone, backprop |
| **Total** | **11.9s** | | ~4400 oracle calls, 14 workers |

## Per-Worker Timing Breakdown (v18, 200 games/iter, iter 4-5 avg)

| Component | Time | % | Description |
|-----------|------|---|-------------|
| vectorize | 1.5s | 14% | In-place observe_minimal_flat! (no fill!) |
| cpu_forward | 5.5s | 51% | Alloc-free forward (BLAS mul! + fused LN+relu + @simd) |
| mcts_pure | 3.8s | 36% | Tree traversal with get() + inline VL, backprop |
| **Total** | **10.8s** | | ~4400 oracle calls, 14 workers |

Key insight: Dict lookup reduction from 7→2 per node gave ~5% mcts_pure improvement.
Training JIT warmup eliminated 70s compilation overhead (iter 2: 87.5→18.1s).

Key insight: Flux forward allocates ~50 arrays (49.7KB) per call. Under 14-thread concurrency,
GC stop-the-world triples from 6.3% to 20.5%. Eliminating ALL allocations via manual BLAS/LN/relu
drops per-call time from 6.25ms (Flux) to 1.48ms (4.2x faster). GC reduction also improves
mcts_pure (7.9→4.0s) and vectorize (2.5→1.6s).

Training batch cap: `min(buffer/batch_size, games*200/batch_size)` = 156 batches (vs 390 uncapped).
Training time ~6s (fully hidden behind ~12s self-play).

## Optimization Summary

### What worked (in order of impact)
1. **Allocation-free forward pass** (v15c): Extract weights to plain arrays, use BLAS mul! + manual LN/relu/sigmoid/softmax — zero allocations. Eliminates GC stop-the-world scaling (6.3%→20.5% under 14 threads). Forward: 6.25ms→1.48ms (4.2x). Overall: 316→800 g/m (2.53x).
2. **Shared RNG** (v11c): Eliminated 430K MersenneTwister allocations/iter (1.1GB/worker, 15GB total). mcts_pure **halved** (34→17s). Biggest single optimization.
3. **Direct actions mask** (v10): Bypassing GI.init → BackgammonNet.legal_actions directly. Saved ~92K env allocs/iter. vectorize: 28→4s.
4. **Cached actions in MCTS tree** (v13): Store available_actions in StateInfo; avoid recomputing during traversal (was allocating ~5.5KB/call × 210K calls/worker = 1.15GB/worker). mcts_pure: 17.2→12.3s (-28%).
5. **Game clone pool + in-place vectorize** (v14): Pre-allocated game objects via GI.clone_into! (zero-alloc per sim). observe_minimal_flat! writes directly into buffer (no intermediate). Reduced GC pressure → cpu_forward 33.7→27.5s (-18%), vectorize 3.7→2.5s (-32%), mcts_pure 9.4→7.9s (-16%).
6. **CPU inference** (v9): Eliminated channel queue wait (12ms→0ms per call). Tied with GPU approach on this hardware.
7. **Training overlap**: GPU training concurrent with CPU self-play. Saves 15-35s/iter.
8. **Lightweight clones** (v11): Shared vectors in GI.current_state for MCTS tree keys. Saved ~7GB alloc/iter.
9. **Fast GI.clone** (v11b): 1 full clone instead of 2 per GI.clone call.
10. **Inline UCT argmax** (v13): best_uct_action avoids map+argmax vector allocation.
11. **Training batch cap** (v13): Cap training at ~1 epoch of new data per iteration. Prevents training (15s at 390 batches) from exceeding self-play time (51s). Now ~6s, fully hidden.
12. **Zero aggregation** (v7): Removing 0.5ms delay = 19% improvement.
13. **Trace actions caching** (v14): Store MCTS actions in trace to skip GI.init + actions_mask in sample conversion.
14. **Dict lookup reduction** (v18): Use `get()` and inline virtual loss in batched MCTS. Reduced 7 Dict lookups per node visit to 2 (1 traversal + 1 backprop). mcts_pure: 4.0→3.8s (-5%).
15. **Training JIT warmup** (v18): Pre-compile training path with dummy batch before main loop. Eliminates 70s Flux/Zygote JIT overhead on first training iteration. Iter 2: 87.5→18.1s. 5-iter training loop 50% faster.

### What didn't work
- Green threads: doubled queue depth, cancelled throughput gain
- Worker oversubscription (28 or 15): 8 physical cores = hard limit
- BLAS threads=2: cross-worker contention > single-thread overhead
- Fewer workers (7): ~same throughput (memory bandwidth limited, not CPU limited)
- GPU inference (v12): channel overhead (41s gpu_wait) > CPU forward (34s). 4090 is overkill for 283K model
- GC.enable(false) during self-play: massive GC spike when re-enabled (75s pause) because training runs concurrently and also accumulates garbage
- In-place forward_normalized: calling Network.forward directly with manual normalization was 2x slower (66s vs 27s). Suspected cause: API/dimension mismatch or threading interaction with Flux
- **CPU inference server** (v15a): Centralizing BLAS on 1 thread serializes all forward passes. 85 g/m — 3.7x worse than 14 concurrent workers (316 g/m). The 14 HT threads achieve ~13.9x effective parallelism via workload interleaving.
- **GPU aggressive aggregation** (v15b): Even with batch=1400 states, GPU kernel launch + CUDA sync + CPU↔GPU transfer overhead exceeds per-worker CPU BLAS for 283K model

### Current bottleneck (v18)
- CPU forward pass (51%, 5.5s) — allocation-free BLAS mul! + fused LN/relu, hardware-limited
- MCTS pure (36%, 3.8s) — tree traversal + backprop with optimized Dict lookups
- Vectorize (14%, 1.5s) — in-place observe_minimal_flat! (no fill!)
- GPU idle during self-play, used only for training (overlapped) and eval
- 14 workers, ~4400 oracle calls/iter, ~50-100 states/call
- Self-play at hardware limit: BLAS throughput and memory bandwidth saturated

### Remaining optimization opportunities
- **ONNX Runtime / oneDNN**: Optimized CPU kernels might be 1.5-2x faster than manual BLAS for small matmuls
- **State hash keys**: UInt64 hash-indexed tree to skip GI.current_state allocation + cheaper Dict ops
- **Smaller model**: Fewer params → faster forward, but affects learning quality
- **Algorithmic**: Fewer MCTS sims with better exploration, or tree reuse between moves
