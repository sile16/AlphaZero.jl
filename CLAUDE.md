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

All use FCResNetMultiHead 128w x 3b (283K params), MINIMAL observations, 400 MCTS sims.

| Run | Config | Iters | Loss | vs Random | vs GnuBG 0-ply | vs GnuBG 1-ply |
|-----|--------|-------|------|-----------|----------------|----------------|
| **AdamW lr=0.001** | AdamW, wd=1e-4, α=0.3 | 50 | 5.01→3.97 | +2.19 | +0.94 (79%) | **+0.55 (66%)** |
| AdamW lr=0.0005 | AdamW, wd=1e-4, dyn-α | 50 | 5.26→4.08 | +2.29 | +0.97 (79%) | +0.36 (59%) |

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

### PER + Reanalyze Reference (in `src/cluster/`, to be ported to train_distributed.jl)
- `src/cluster/coordinator.jl` - PER sampling, TD-error priority updates
- `src/cluster/types.jl` - ClusterSample with priority/reanalyze fields
- `src/cluster/async_workers.jl` - Async reanalyze and eval workers

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
- `distributed_20260206_204548` - Best run (AdamW lr=0.001, +0.55 vs GnuBG 1-ply)
- `distributed_20260206_172527` - AdamW lr=0.0005 comparison run

## Key Lessons

1. **Random baseline is misleading** -- always evaluate vs GnuBG
2. **AdamW > Adam + L2 in loss** -- decoupled weight decay prevents loss explosion while maintaining play strength
3. **Learning rate matters most** -- lr=0.001 vs 0.0005 was the main factor in GnuBG strength
4. **inference_batch_size << mcts_iters** -- batch=400 with iters=400 → depth-1 trees → divergence
5. **Buffer must hold 3-5 iterations** -- too small = buffer churn → divergence
6. **Dynamic Dirichlet unnecessary** -- fixed α=0.3 works fine for ~680 action space
7. **MINIMAL features generalize best** -- simpler obs beats larger feature sets against GnuBG
8. **1000+ eval games** for reliable comparisons (100-game evals show ±0.2 variance)

## Next Steps

### Priority
1. **Scale up training** -- longer runs (200+ iter), larger model (256w×10b)
2. **Port PER + reanalyze to train_distributed.jl** -- reference implementation in `src/cluster/`

### Future
3. Web-based workers (WASM + WebGPU) via sibling project `/home/sile/github/tavlatalk/`
4. Cloud remote workers for distributed self-play
5. Match equity table (MET) for proper match play scoring
6. Bear-off database integration
7. Exam eval (known tricky positions)
