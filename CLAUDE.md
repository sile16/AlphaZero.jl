# AlphaZero.jl - Claude Code Context

## Project Goal

**Build a world-class SOTA backgammon AI** using AlphaZero self-play. Always use the best known approach — never settle for "good enough" when a better method exists. Document all experiments for eventual paper publication.

## Project Overview

Julia implementation of AlphaZero for backgammon with:
- **Distributed training**: Jarvis (GPU training server) + Neo (self-play client) via HTTP API
- Multi-head equity network (5 value heads: P(win), P(gammon|win), P(bg|win), P(gammon|loss), P(bg|loss))
- GnuBG/Wildbg evaluation for meaningful benchmarks
- Exact k=6 bear-off table (c14: 3.0GB + c15: 5.8GB from BackgammonNet.jl) for training targets, with gammon conditionals for c15 positions

## Best Practices

- **Always git commit before training runs** — store the commit hash in TensorBoard for reproducibility
- **TensorBoard logs**: reproducibility info (git commit, command, all hyperparams) auto-logged at session start
- **Bear-off benchmark**: 10K fixed positions evaluated every iteration — tracks value head accuracy vs exact table
- **One change at a time** for experiments — isolate variables to know what works
- **2000+ GnuBG eval games** (500 per matchup × 4) for reliable comparisons

## Training (Distributed Only)

**Distributed is the ONLY training path.** No single-machine training scripts. To run on one machine, spin up both server and client locally.

### Training Server (Jarvis)
```bash
julia --threads 16 --project scripts/training_server.jl --port 9090 --data-dir /home/sile/alphazero-server
```

### Self-Play Clients
```bash
# Neo (M3 Max) — selfplay + eval
./start_client.sh --threads 18 --num-workers 16 --eval-capable

# Jarvis (co-located with server) — selfplay only, reduced workers to prevent OOM
./start_client.sh --threads 8 --num-workers 4
```

**Note**: Use `--threads N` (with space) before `--project`. The `--threads=N` form after `--project` leaks into ARGS.
**Auto-restart**: `start_client.sh` loops with git pull between restarts. Server triggers via `POST /api/restart-clients`.

### Architecture
- **Server** (Jarvis): HTTP API, PER buffer, GPU training (RTX 4090), weight serving, eval job management, TensorBoard
- **Client** (Neo, 16 workers): self-play + eval with batched MCTS, CPU inference, sample upload, weight sync
- **Client** (Jarvis, 4 workers): self-play only (no eval — OOM risk with 6+ workers alongside server)
- Same HTTP API for Julia clients and future web clients (tavlatalk)
- Per-component loss logging to TensorBoard (policy, value, invalid)
- Training uses `AlphaZero.losses()` from `src/learning.jl` (multi-head equity BCE + policy KL)

### Game
- `games/backgammon-deterministic/` with `SHORT_GAME=true` (BackgammonNet built-in short positions)
- BackgammonNet v0.3.2+ has symmetric initial positions (previous versions had asymmetric P0/P1)
- Observation: `BACKGAMMON_OBS_TYPE=minimal_flat` (344 features, flat vector for MLP)
- Action space: ~680 actions (2-checker move encoding)

## Evaluation

### Standard: Race Eval (effective 2026-03-14)

**Race models MUST be evaluated from race starting positions**, not opening positions. The race model is untrained on contact positions and will play terribly from the opening.

```bash
# Race eval: 2000 fixed positions, both sides, 600 MCTS iters, wildbg large nets
julia --threads 28 --project scripts/eval_race.jl <checkpoint> \
    --width=128 --blocks=3 --num-workers=24 --mcts-iters=600 --num-games=0 \
    --wildbg-lib=/Users/sile/github/wildbg/target/release/libwildbg.dylib
```

- `--num-games=0` means all 2000 positions (each played from both sides = 4000 games)
- `--mcts-iters=600` is the standard MCTS budget
- Uses wildbg large nets (not small/main branch)
- Fixed positions: `/homeshare/projects/AlphaZero.jl/eval_data/race_eval_2000.jls`

### Full-Game Eval (contact models)

```bash
# Wildbg evaluation (primary for contact models)
julia --threads 28 --project scripts/eval_vs_wildbg.jl <checkpoint> --width=256 --blocks=5 --num-workers=24 --mcts-iters=600
```

**Always evaluate against wildbg**, not random. Random baseline is misleading.

## Performance Baselines

**CRITICAL**: All GnuBG eval results before 2026-02-14 were invalid due to board encoding bugs. See `notes/corrected_eval_results_20260214.md`.

### vs Wildbg (2026-03-10, 1000 games, 100 MCTS iters, small nets on main branch)

| Rank | Experiment | Architecture | Iters | Equity | Win% |
|------|-----------|-------------|-------|--------|------|
| 1 | **PER+Reanalyze** | **256w×5b** | **200** | **-1.468** | **4.4%** |
| 2 | PER+Reanalyze | 128w×3b | 200 | -1.652 | 3.2% |
| 3 | PER+Reanalyze | 128w×3b | 200 | -1.694 | 2.8% |
| 4 | PER | 128w×3b | 200 | -1.714 | 2.4% |
| 5 | PER+Reanalyze | 256w×5b | 50 | -1.780 | 2.2% |
| 6 | Baseline | 128w×3b | 200 | -1.878 | 1.4% |
| 7 | PER | 128w×3b | 200 | -1.896 | 1.1% |

Full results (28 checkpoints): `/homeshare/projects/AlphaZero.jl/sessions/wildbg_eval_results_20260310_012327.txt`

### vs GnuBG 0-ply (2026-02-14, 1000 games, 100 MCTS iters)

| Rank | Experiment | Architecture | Iters | Equity | Win% |
|------|-----------|-------------|-------|--------|------|
| 1 | **PER+Reanalyze** | **256w×5b** | **200** | **-1.361** | **9.6%** |
| 2 | PER | 128w×3b | 200 | -1.558 | 7.8% |
| 3 | PER+Reanalyze | 256w×5b | 50 | -1.573 | 7.0% |
| 4 | Baseline | 128w×3b | 200 | -1.746 | 4.6% |

**All models are genuinely weak** — best is 4.4% wins vs wildbg, 9.6% vs GnuBG 0-ply. Wildbg is harder than GnuBG 0-ply but rankings are consistent across both opponents.

**Note**: Pre-v0.3.2 results used asymmetric initial positions and are NOT comparable.

### Key Training Parameters
- **Optimizer**: AdamW (lr=0.001, weight_decay=1e-4) — decoupled weight decay prevents loss explosion
- **MCTS**: 400 sims, inference_batch_size=50 (must be << mcts_iters)
- **Buffer**: 600K capacity (3-5 iterations worth of data)
- **Dirichlet noise**: fixed α=0.3, ε=0.25
- **Throughput**: ~350 games/min on i7-10700K (14 workers, CPU inference)

## Key Files

### Core
- `src/learning.jl` - `losses()` function: multi-head BCE + KL policy + L2 reg
- `src/networks/architectures/fc_resnet_multihead.jl` - Multi-head equity network
- `src/batched_mcts.jl` - Batched MCTS for training (passthrough chance sampling — fast, narrow)
- `src/mcts.jl` - Standard MCTS with full chance node support (4 modes: full/sampling/stratified/progressive)
- `src/game.jl` - Game interface (`game_outcome()` for win types)
- `src/params.jl` - MctsParams, LearningParams

### Game Loop (v7, 2026-03-22)
- `src/game_loop.jl` - GameLoop module: `play_game()`, MctsAgent, ExternalAgent, GameResult, TraceEntry
  - Used by eval scripts (sequential, per-move allocations OK)
  - NOT used by selfplay — `selfplay_client.jl` uses direct `BatchedMCTS.think()` for zero-alloc threading
  - Bear-off truncation, chance node passthrough, temperature scheduling

### Distributed Training + Eval
- `scripts/training_server.jl` - **Training server** (Jarvis, port 9090) — training loop, PER sampling, loss, checkpoints
- `scripts/selfplay_client.jl` - **Self-play client** (Neo) — MCTS self-play, sample upload, weight sync + **distributed eval** (--eval-capable)
- `src/distributed/server.jl` - HTTP server + routes + client tracking + **4 eval endpoints** (status/checkout/submit/heartbeat)
- `src/distributed/eval_manager.jl` - **Eval job manager**: chunked work queue (50 games/chunk), lease-based checkout, expiry, result aggregation
- `src/distributed/client.jl` - HTTP client for self-play workers
- `src/distributed/buffer.jl` - Thread-safe PER buffer with `per_sample_partition()` for dual-model
- `src/distributed/protocol.jl` - Wire format (MsgPack + JSON)

### Evaluation & Utility Scripts
- `scripts/eval_vs_wildbg.jl` - Full-game wildbg evaluation (contact models)
- `scripts/eval_race.jl` - Race model evaluation on 2000 fixed positions (standard: 600 MCTS, wildbg large)
- `scripts/diagnose_loss.jl` - Per-component loss analysis across checkpoints
- `scripts/generate_race_positions.jl` - Generate fixed race eval positions

### PER + Reanalyze + Bear-off
- PER: per-model partition sampling with IS weight correction (α=0.6, β=0.4→1.0, ε=0.01)
  - `per_sample_partition()` samples from contact/race subsets independently — avoids cross-model priority scale mismatch
  - IS weights passed to `losses()` via batch `W` field
- Reanalyze: buffer reanalysis with EMA value blending (25% per iter)
- Bear-off: exact k=6 table values for race/bearoff positions. Post-dice move enumeration gives exact Q(board, dice) at decision nodes; pre-dice table lookup at chance nodes. Training targets use the same post-dice enumeration.

### Shared Data (NFS)
- `/homeshare/projects/AlphaZero.jl/eval_data/race_starts_tuples.jls` — 98,516 beginning-of-race positions
- `/homeshare/projects/AlphaZero.jl/eval_data/race_eval_2000.jls` — 2000 fixed eval positions

### Archived (2026-03-14 cleanup)
- `scripts/archive/` - Old single-machine training scripts, completed benchmarks, quick_eval
- `src/archive/distributed/` - Old ZMQ-based distributed module (unused)
- `src/archive/cluster_v0/` - Old thread-based cluster module (replaced by HTTP distributed)
- `/homeshare/projects/AlphaZero.jl/sessions/archive/` - Pre-v0.3.2 and experimental sessions

### Inference
- `src/inference/fast_weights.jl` - FastInference module: allocation-free CPU forward pass with pure Julia GEMM + LayerNorm (thread-safe, zero-allocation). Used by selfplay_client.jl.
  - Pure Julia `_gemm_bias!` beats BLAS for multi-threaded selfplay on both platforms (BLAS `@view` allocations cause GC pressure under threading that negates single-thread GEMM speedup)
  - **ARM (Neo)**: 43 GFLOPS (triggers Apple AMX via LLVM `@simd`, 1.3x faster than Apple BLAS)
  - **x86 (Jarvis)**: 47 GFLOPS (BLAS is 1.6x faster single-thread but slower multi-threaded due to 96 bytes/call view allocations)
  - GEMM is 85-93% of forward pass time. LayerNorm and softmax are negligible.

### Legacy modules (still compiled by AlphaZero.jl, not used by active training/eval)
- `src/batchifier.jl`, `src/async_batchifier.jl` - Old batchifiers (used by simulations.jl → session.jl)
- `src/simulations.jl`, `src/training.jl`, `src/benchmark.jl` - Legacy single-machine training (used by session.jl)
- `src/async_mcts.jl`, `src/gumbel_mcts.jl`, `src/pingpong_batchifier.jl` - Removed from compilation (2026-03-19)

## Testing

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

## Session Directories

All training artifacts (sessions, logs, results, wandb) are stored on the NFS-shared directory
`/homeshare/projects/AlphaZero.jl/` so they are accessible from all machines. Symlinks exist
in the git repo for backward compatibility.

Sessions saved to `/homeshare/projects/AlphaZero.jl/sessions/distributed_YYYYMMDD_HHMMSS/` containing:
- `checkpoints/latest.data` - Latest network weights
- `checkpoints/iter_N.data` - Checkpoint at iteration N
- `tb/` - TensorBoard logs
- `final_eval_results.txt` - Final evaluation results

**TensorBoard**: `tensorboard --logdir /homeshare/projects/AlphaZero.jl/sessions`

### Active Sessions
All under `/homeshare/projects/AlphaZero.jl/sessions/`:

**Full-game (contact) models** (corrected equity vs GnuBG 0-ply):
- `distributed_20260213_031243_per_reanalyze` - **Best overall** (256w×5b, 200 iter PER+Reanalyze, -1.361 equity, 9.6% wins)
- `distributed_20260209_215824_per` - Best 128w (200 iter PER, -1.558 equity, 7.8% wins)
- `distributed_20260213_010615_per_reanalyze` - 256w×5b 50-iter (-1.573 equity, 7.0% wins)
- `distributed_20260206_223524` - 200 iter baseline (-1.746 equity, 4.6% wins)
- `distributed_20260207_030412_per` - PER 50-iter (-1.759 equity, 4.8% wins)
- `distributed_20260206_204548` - Original 50-iter baseline (-1.841 equity, 3.5% wins)

**Race-only models** (distributed training, 2026-03-14):
- `distributed_race_20260314` - Race 128w×3b, 50 iter, bootstrap-seeded, Reanalyze (NOTE: PER was broken — uniform sampling)
  - Standard eval (2000 pos × 2 sides = 4000 games, 600 MCTS iters, wildbg large):

  | Iter | Equity | Win% | As White | As Black | Value MSE | Value Corr |
  |------|--------|------|----------|----------|-----------|------------|
  | 5 | -0.128 | 46.0% | -0.120 | -0.135 | 0.595 | 0.959 |
  | 10 | -0.090 | 46.5% | -0.074 | -0.105 | 0.801 | 0.811 |
  | 20 | -0.065 | 47.4% | -0.054 | -0.075 | 0.796 | 0.824 |
  | 30 | **-0.051** | **47.8%** | -0.044 | -0.059 | 0.801 | 0.825 |
  | 40 | -0.053 | 47.7% | -0.045 | -0.061 | 0.821 | 0.808 |
  | 50 | -0.056 | 47.5% | -0.053 | -0.059 | 0.805 | 0.822 |

  - Peak at iter 30 (-0.051 equity). Plateaus after. AZ slightly weaker as black.
  - PER fix + more iterations should improve next run.

## Key Lessons

### Critical Bug (2026-02-14)
0. **ALWAYS verify board encoding against external reference** -- `_to_gnubg_board` had 3 bugs (off-by-one, bar position, opponent perspective) that made GnuBG evaluate wrong positions. All pre-fix results showed 65-92% win rates when reality was 3-10%. Self-consistency checks are NOT sufficient; verify against gnubg's known position IDs.

### Training Dynamics
1. **PER is the only reliable improvement** -- +0.082 equity at 50 iter, +0.188 at 200 iter over baseline. The only technique consistently validated post-fix.
2. **Bearoff rollouts and reanalyze HURT** -- both showed regressions vs baseline in corrected eval (-0.152 and -0.213 respectively). Previous apparent gains were artifacts of the encoding bug.
3. **Loss plateau != strength plateau** -- loss plateaus at ~3.95 by iter 50, but GnuBG strength improves steadily through iter 200
4. **AdamW + lr=0.001 is best optimizer config** -- decoupled weight decay prevents loss explosion
5. **inference_batch_size << mcts_iters** -- batch=50 with iters=400 -> depth ~8 (batch=400 -> depth-1 -> divergence)
6. **Buffer must hold 3-5 iterations** -- too small = buffer churn -> divergence

### Technique Insights
7. **256w×5b >> 128w×3b** -- larger model at 200 iter beats best 128w by +0.197 equity
8. **All models are genuinely weak** -- best gets 4.4% wins vs wildbg (small nets), 9.6% vs GnuBG 0-ply. Need training efficiency improvements before scaling.
9. **Bear-off table signal mismatch FIXED** (2026-03-19) -- table values are pre-dice, game states are post-dice. Fixed via move enumeration: at decision nodes, enumerate all legal moves, look up resulting positions in table, take max. Also fixed perspective bug (evaluator returned mover-relative but MCTS assumed white-relative). Previous "bearoff hurt" finding needs retesting with this fix. See `notes/bearoff_chance_node_issue.md`.
10. **Dual-model architecture** (2026-03-10) -- contact (256w×10b) + race (128w×3b) implemented in training_server.jl. Race-only training tested (50 iter).
11. **PER was broken in distributed training** (fixed 2026-03-14) -- training_server.jl used uniform sampling instead of `per_sample()`. IS weights were not passed to the loss function. Fixed: per-model partition PER sampling with IS weight correction via `per_sample_partition()`. Per-component loss (policy/value/invalid) now logged to TensorBoard.
12. **Jarvis OOM with server + client** (2026-03-23) -- Server 28GB + Client 34GB = 62GB on 64GB machine. Linux OOM killer terminates client. Reduce Jarvis workers or run client only on Neo.
13. **GameLoop.play_game() kills threading perf** (2026-03-23) -- Per-move TraceEntry allocations cause 28% more GC, amplified to 20-30x under 32 threads. Selfplay must use direct BatchedMCTS calls. GameLoop only for eval scripts.
14. **Multi-threaded scaling is memory-bandwidth limited** (2026-03-31) -- Neo M3 Max: peaks at 8-16 workers (6x speedup), **degrades at 24** (4.4x). Jarvis i7-10700K: scales linearly to 6 workers (5x, 84% efficiency). Workers share L2/L3 cache for 256KB weight matrices. Optimal: Neo 16 workers, Jarvis 6 workers. Use `--threads N+2 --num-workers N` to reserve threads for upload+eval.
15. **Multihead equity heads must be masked** (2026-03-23) -- Train P(gammon|win)/P(bg|win) only on won games, P(gammon|loss)/P(bg|loss) only on lost games. Without masking, network learns joint probabilities instead of conditionals, inconsistent with compute_equity().

### Evaluation
12. **Random baseline is misleading** -- always evaluate vs wildbg or GnuBG
13. **Wildbg is harder than GnuBG 0-ply** -- roughly halves win rates, but rankings are consistent across both opponents
14. **MINIMAL features generalize best** -- simpler obs beats larger feature sets (relative ranking valid post-fix)
15. **Race eval standard (2026-03-14)** -- 2000 fixed race positions, both sides (4000 games), 600 MCTS iters, wildbg large nets. Script: `scripts/eval_race.jl`
16. **Full-game eval**: 1000+ games for reliable comparisons. Script: `scripts/eval_vs_wildbg.jl`

### MCTS & Eval (2026-03-11)
17. **Passthrough chance sampling is best for eval** -- Benchmarked all 5 mcts.jl chance modes (passthrough, sampling, stratified, progressive, full) at 1600 iters. Passthrough wins: -1.050 equity, 16% wins vs wildbg. Best alternative (sampling v=1.0) was -1.180. At 1600 iters, alternatives waste iterations on chance coverage instead of decision-tree depth. With a weak network, depth > width.
18. **1600 MCTS iters >> 100 iters for eval** -- dual-model at 1600 iters: -1.277 equity, 9.7% wins vs wildbg small. At 100 iters: -1.828, 1.6% wins vs wildbg large. More MCTS budget at eval time dramatically improves play quality.
19. **EvalMCTS removed** -- Had 3 critical bugs, was unused. All eval uses BatchedMCTS with passthrough chance sampling.
20. **Re-test progressive widening when model is stronger** -- As NN values become more accurate, proper chance averaging should help. The crossover point is likely when model can consistently beat wildbg/GnuBG 0-ply. Progressive widening is the best candidate (probability-ordered expansion).
21. **GPU inference on M3 Max (Metal.jl)** -- 4.12x raw speedup at batch=500, but 12ms kernel launch overhead means crossover at batch≈20-30. Best end-to-end: GPU-Lock with 6 workers = 2.36x (30 games/min). Metal is NOT thread-safe. Adding CPU workers alongside GPU HURTS. Speculative prefetch failed (0.09x). GPU is useful only when batch sizes are guaranteed large.
22. **Value error vs wildbg not yet measured** -- Need to track NN value prediction accuracy against wildbg equity to understand when chance expansion will start helping. TODO: build eval script with per-position value comparison.

## Next Steps

### Eval Improvements (immediate)

1. **Value error tracking vs wildbg** -- Build eval script that compares NN value predictions against wildbg equity on positions from self-play games. Track MSE/correlation separately for contact vs race model. Script: `scripts/eval_value_accuracy.jl` (TODO).

2. **GPU eval (revisit after model is stronger)** -- GPU-Lock with 6 workers = 2.36x speedup on Neo. Worth revisiting when model strength justifies longer eval runs. Scripts: `scripts/bench_gpu_eval.jl`.

### Phase 1: Training Efficiency (before scaling)

4. **Bear-off MCTS value override at decision nodes** -- DONE (2026-03-19). Bear-off evaluator now fires at both chance nodes (pre-dice, exact table value) and decision nodes (post-dice, exact via move enumeration). Also fixed perspective bug and training target mismatch. Retest bear-off with this fix to see if previous regression reverses.

5. **Progressive MCTS budget** -- Early iterations (1-20): 100 sims. Mid (20-100): 200 sims. Late (100+): 400 sims. Early network is random so deep search is wasted compute. 4x more games/iter in early training. Trivial: parameterize `num_iters_per_turn` as a schedule.

6. **Bear-off backward equity propagation** -- For games that reach bear-off, use first bear-off position's exact equity as target for ALL preceding contact positions (not final game outcome). Already partially coded (lines 1425-1435 in train_distributed.jl). Previous "bearoff hurt" finding was pre-board-fix — needs retesting.

7. **CPUCT tuning** -- Try 1.0, 1.5, 3.0 (default 2.0, never tuned with correct eval). Cheap parallel experiment.

### Phase 2: Curriculum Learning (after Phase 1 results)

8. **Race-only pre-training** -- Train race network (128w×3b) exclusively on race positions for 50-100 iters. Use games starting from near-bearoff positions. Race phase is simpler, network learns faster.

9. **Contact training with frozen race evaluator** -- After race network is solid, train contact network (256w×10b). When MCTS reaches a race position, use trained race network as exact endgame evaluator instead of continuing rollout. Clean signal propagation from known endpoints.

### Phase 3: Advanced Techniques

10. **Re-test progressive widening** -- When model is strong enough to consistently beat wildbg/GnuBG 0-ply, re-test chance node expansion. Progressive widening (α=0.25) was best alternative at -1.220 equity. As NN accuracy improves, proper chance averaging should cross over passthrough.

11. **Gumbel MCTS** -- Replace standard MCTS with Gumbel-top-k for training. Code exists in `gumbel_mcts.jl` but isn't wired into training. Sequential halving focuses compute on top-k candidates instead of spreading across 50+ legal moves.

12. **External position bootstrapping** -- Generate 50-100K positions, evaluate with wildbg, pre-fill replay buffer before self-play starts. Solves cold-start problem (early games are random noise).

13. **Full game training** -- Switch from SHORT_GAME to standard opening. Short games skip opening/midgame strategic depth. May need bear-off truncation to compensate for longer games.

### Scaling (after efficiency gains proven)

14. **Scale up model + iterations** -- 256w×10b + PER for 500+ iterations, using Phase 1-2 efficiency improvements.

### v7 Distributed Eval (2026-03-22)

**Architecture**: Eval moved from server to clients via chunked work queue. Server creates eval jobs (non-blocking), clients claim 50-game chunks, play vs wildbg, submit results. Server aggregates and logs to TB.

**Key results**:
- Eval takes ~2min (vs 17-34min in v6) — 10x faster
- Server never blocks during eval — always responsive
- Play strength consistent: +0.007 equity, ~50% wins vs wildbg

**Active training**: `race_v7_distributed_eval` on Jarvis
- 256w×5b race, 4000 gradient steps/iter, PER, cosine LR, bootstrap
- Jarvis: 12 workers (eval-capable), Neo: 32 workers (self-play only)
- Data dir: `/home/sile/alphazero-server-race-v7/`
- Checkpoint interval: 5 iters

### v6 10x Replay Ratio Results (2026-03-20)

v6 tested `--training-steps 4000` (10x v5's ~390). Key finding: play strength holds steady at even with wildbg despite rising training loss.

| Iter | Equity | Win% |
|------|--------|------|
| 10 | +0.015 | 50.1% |
| 20 | +0.007 | 50.0% |
| 30 | +0.016 | 50.1% |

### v8 Multihead Fix + Performance (2026-03-23)

**Key fixes:**
- Conditional equity head masking (P(gammon|win) only trained on wins, etc.)
- GameLoop.play_game() caused 20-30x selfplay regression under threading (GC pressure from per-move allocations). Selfplay now uses direct BatchedMCTS calls.
- EvalAlphaZeroAgent created new MCTS player per MOVE — fixed to reuse.
- @inbounds audit on MCTS hot paths (~2-5% gain).
- Tailscale network extension on Neo blocked Julia LAN connections (EHOSTUNREACH). Removed.

**v8 eval results** (1000 positions × 2 sides, 600 MCTS, wildbg large):

| Iter | Loss | Equity | Win% |
|------|------|--------|------|
| 10 | 1.663 | +0.01 | 49.7% |
| 20 | 1.968 | +0.02 | 50.0% |

Play strength holds at ~even with wildbg despite loss rising from 1.5→2.1. Bootstrap→self-play transition artifact.

**Performance lesson:** Never allocate in per-move inner loops. Julia's stop-the-world GC amplifies small per-move overhead into catastrophic throughput loss under 32+ threads. Profile under realistic concurrency, not single-thread.

**Data:** Eval positions excluded from training set: `race_starts_tuples_no_eval.jls` (96,514 positions, 2000 eval positions removed).

### v10-v12 Infrastructure Fixes (2026-03-30 to 2026-04-02)

**Eval system rebuilt** — eval never completed in any prior run. Fixed:
- Server: don't replace running eval jobs, one chunk per client, weight pinning by version
- Client: parallel eval with per-thread resources (oracles+agents+wildbg), session-level caching
- TB: eval metrics logged at eval iteration step, not finalization step

**FastWeights data race fixed** — shared `FastWeights` mutated via `copyto!` during weight updates while workers read. Caused 18→1 games/sec collapse. Fixed with `Ref{FastWeights}` atomic swap.

**Performance profiled** — GEMM is 90%+ of forward pass. Pure Julia `_gemm_bias!` wins on both platforms (BLAS `@view` allocations kill multi-threaded). Neo peaks at 16 workers (degrades at 24, memory bandwidth saturation).

**Bootstrap phasing** — `--bootstrap-train-iters N` clears replay buffer after N iterations of bootstrap training, switching to pure self-play. Without this, bootstrap data anchors model to bootstrap source's level (~33% of buffer is still bootstrap at iter 20 with 3M buffer).

**v11 results** (20 iters, 4000 games, 600 MCTS, wildbg large):

| Iter | Equity | Win% | Value corr |
|------|--------|------|------------|
| 0 (bootstrap) | -0.160 | 43.9% | -0.195 |
| 5 | -0.046 | 49.0% | 0.976 |
| 10 | -0.025 | 49.5% | 0.962 |
| **15 (peak)** | **+0.013** | **49.5%** | 0.954 |
| 20 | -0.022 | 49.4% | 0.957 |

### Future
15. Web-based workers (WASM + WebGPU) via sibling project `/home/sile/github/tavlatalk/`
16. Cloud remote workers for distributed self-play
17. Match equity table (MET) for proper match play scoring
18. Exam eval (known tricky positions from GnuBG analysis)
