# AlphaZero.jl - Claude Code Context

## Project Overview

Julia implementation of AlphaZero for backgammon with:
- Multi-head equity network (5 value heads: P(win), P(gammon|win), P(bg|win), P(gammon|loss), P(bg|loss))
- Threaded training with CPU inference (14 workers on i7-10700K)
- GnuBG evaluation for meaningful benchmarks

## Training

```bash
julia --threads 16 --project scripts/train_distributed.jl \
    --num-workers=14 \
    --total-iterations=50 \
    --games-per-iteration=500 \
    --mcts-iters=400 \
    --inference-batch-size=50 \
    --buffer-capacity=600000 \
    --learning-rate=0.001 \
    --eval-interval=5 --eval-games=100 \
    --final-eval-games=200 \
    --seed=42
```

**Note**: Use `--threads 16` (with space) before `--project`. The `--threads=16` form after `--project` leaks into ARGS.

### Architecture
- Main thread: training loop, replay buffer, weight updates (AdamW optimizer)
- Worker threads (14): self-play with batched MCTS, CPU BLAS inference
- Per-component loss logging to TensorBoard (policy, value, invalid)
- Training uses `AlphaZero.losses()` from `src/learning.jl` (multi-head equity BCE + policy KL)

### Game
- `games/backgammon-deterministic/` with `SHORT_GAME=true` (BackgammonNet built-in short positions)
- BackgammonNet v0.3.2+ has symmetric initial positions (previous versions had asymmetric P0/P1)
- Observation: `BACKGAMMON_OBS_TYPE=minimal` (330 features, flat vector)
- Action space: ~680 actions (2-checker move encoding)

## Evaluation

```bash
# GnuBG evaluation (RECOMMENDED — use 8 workers for ~12 min runtime)
julia --threads 16 --project scripts/eval_vs_gnubg.jl <checkpoint> [obs_type] [num_games] [width] [blocks] [num_workers] [mcts_iters]
# Example:
julia --threads 16 --project scripts/eval_vs_gnubg.jl sessions/distributed_20260206_204548/checkpoints/latest.data minimal 500 128 3 8 100

# Quick eval vs random
julia --project scripts/quick_eval.jl
```

**Always evaluate against GnuBG**, not just random. Random baseline is misleading -- models that dominate random may not generalize.

## Performance Baselines (post-initial-position-fix, BackgammonNet v0.3.2+)

All use FCResNetMultiHead 128w x 3b (283K params), MINIMAL observations, 400 MCTS sims, AdamW lr=0.001.

| Experiment | Iters | Loss (final) | vs GnuBG 0-ply | vs GnuBG 1-ply | Time (min) |
|-----------|-------|-------------|----------------|----------------|------------|
| **Baseline** | 200 | 3.89 | +1.31 (89%) | **+1.05 (78%)** | 254.5 |
| Baseline | 50 | 3.97 | +0.94 (79%) | +0.55 (66%) | 75.0 |
| Bear-off rollouts | 50 | 3.98 | +1.00 (81%) | +0.83 (72%) | 67.1 |
| PER | 50 | 4.07 | +1.08 (83%) | +0.79 (72%) | 74.4 |
| Reanalyze | 50 | 3.98 | +0.97 (79%) | +0.69 (71%) | 80.0 |

See `notes/experiment_results_20260207.md` for full analysis and raw GnuBG evaluation details.

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
- `src/batched_mcts.jl` - Batched MCTS for training (samples through chance nodes)
- `src/mcts.jl` - Standard MCTS with full chance node support (4 modes)
- `src/game.jl` - Game interface (`game_outcome()` for win types)
- `src/params.jl` - MctsParams, LearningParams

### Training & Evaluation Scripts
- `scripts/train_distributed.jl` - **Primary training script** (threaded, CPU inference)
- `scripts/eval_vs_gnubg.jl` - GnuBG evaluation (parallel, multi-process)
- `scripts/GnubgPlayer.jl` / `GnubgPlayerFast.jl` - GnuBG integration
- `scripts/diagnose_loss.jl` - Per-component loss analysis across checkpoints

### PER + Reanalyze + Bear-off (implemented in train_distributed.jl)
- PER: `--use-per` — proportional priority sampling with IS weights (α=0.6, β=0.4→1.0)
- Reanalyze: `--use-reanalyze` — buffer reanalysis with EMA value blending (25% per iter)
- Bear-off: `--use-bearoff` — rollout equity for no-contact race positions (50 rollouts)
- Resume: `--resume <session_dir>` — load weights from checkpoint, continue iteration count

### Archived
- `scripts/archive/` - Archived scripts including train_cluster.jl
- `src/archive/distributed/` - Old ZMQ-based distributed module (unused)
- `sessions/archive/` - Pre-v0.3.2 and experimental sessions

## Testing

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

## Session Directories

Sessions saved to `sessions/distributed_YYYYMMDD_HHMMSS/` containing:
- `checkpoints/latest.data` - Latest network weights
- `checkpoints/iter_N.data` - Checkpoint at iteration N
- `tb/` - TensorBoard logs
- `final_eval_results.txt` - Final evaluation results

### Active Sessions
- `distributed_20260206_223524` - **Best overall** (200 iter baseline, +1.05 vs GnuBG 1-ply)
- `distributed_20260207_061713_bearoff` - Best 50-iter (bear-off rollouts, +0.83 vs GnuBG 1-ply)
- `distributed_20260207_030412_per` - PER experiment (+0.79 vs GnuBG 1-ply)
- `distributed_20260207_043500_reanalyze` - Reanalyze experiment (+0.69 vs GnuBG 1-ply)
- `distributed_20260206_204548` - Original 50-iter baseline (+0.55 vs GnuBG 1-ply)

## Key Lessons

### Training Dynamics
1. **Training longer >> any single technique** -- 200-iter baseline (+1.05) beats all 50-iter experiments. Scale compute first.
2. **Loss plateau ≠ strength plateau** -- loss plateaus at ~3.95 by iter 50, but GnuBG strength improves steadily through iter 200
3. **AdamW + lr=0.001 is best optimizer config** -- decoupled weight decay prevents loss explosion
4. **inference_batch_size << mcts_iters** -- batch=50 with iters=400 → depth ~8 (batch=400 → depth-1 → divergence)
5. **Buffer must hold 3-5 iterations** -- too small = buffer churn → divergence

### Technique Insights (2026-02-07 experiments)
6. **Bear-off rollouts = best single improvement** -- +51% equity gain at 50 iter, and actually faster (better endgame targets)
7. **PER raises loss but strengthens play** -- IS weights focus on hard positions → higher avg loss but better decisions
8. **Reanalyze gives moderate gains** -- +25% equity gain, refreshes stale value targets, slight overhead
9. **Bear-off + PER likely compound** -- they address orthogonal aspects (endgame accuracy vs sampling efficiency)

### Evaluation
10. **Random baseline is misleading** -- always evaluate vs GnuBG
11. **MINIMAL features generalize best** -- simpler obs beats larger feature sets against GnuBG
12. **2000+ eval games** for reliable comparisons (500-game matchups × 4 = 2000 total, both sides)

## Next Steps

### Priority (next experiments)
1. **200 iter with PER + bear-off combined** -- best two techniques, test if improvements compound
2. **Larger model (256w×10b)** -- loss plateau at ~3.95 suggests 128w×3b (283K) may be capacity-limited
3. **200 iter with all three (PER + bear-off + reanalyze)** -- full kitchen sink if #1 shows compounding

### Future
4. Web-based workers (WASM + WebGPU) via sibling project `/home/sile/github/tavlatalk/`
5. Cloud remote workers for distributed self-play
6. Match equity table (MET) for proper match play scoring
7. Bear-off database (exact lookup vs rollouts) for endgame
8. Exam eval (known tricky positions from GnuBG analysis)
