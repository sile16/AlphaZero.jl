# AlphaZero.jl - Claude Code Context

## Project Goal

**Build a world-class SOTA backgammon AI** using AlphaZero self-play. Always use the best known approach — never settle for "good enough" when a better method exists. Document all experiments for eventual paper publication.

## Project Overview

Julia implementation of AlphaZero for backgammon with:
- Multi-head equity network (5 value heads: P(win), P(gammon|win), P(bg|win), P(gammon|loss), P(bg|loss))
- Threaded training with CPU inference (14 workers on i7-10700K)
- GnuBG evaluation for meaningful benchmarks
- Exact k=6 bear-off table (c14: 3.0GB + c15: 5.8GB from BackgammonNet.jl) for training targets, with gammon conditionals for c15 positions

## Best Practices

- **Always git commit before training runs** — store the commit hash in TensorBoard for reproducibility
- **TensorBoard logs**: reproducibility info (git commit, command, all hyperparams) auto-logged at session start
- **Bear-off benchmark**: 10K fixed positions evaluated every iteration — tracks value head accuracy vs exact table
- **One change at a time** for experiments — isolate variables to know what works
- **2000+ GnuBG eval games** (500 per matchup × 4) for reliable comparisons

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

## Performance Baselines (corrected 2026-02-14, post-board-encoding-fix)

**CRITICAL**: All GnuBG eval results before 2026-02-14 were invalid due to a board encoding bug (commit e164a85). The bug caused GnuBG to evaluate wrong positions and play terribly, inflating our win rates from ~3-10% to 65-92%. See `notes/corrected_eval_results_20260214.md` for details.

Corrected results below (vs GnuBG 0-ply, 1000 games, 100 MCTS iters, 8 workers):

| Rank | Experiment | Architecture | Iters | Equity | Win% |
|------|-----------|-------------|-------|--------|------|
| 1 | **PER+Reanalyze** | **256w×5b** | **200** | **-1.361** | **9.6%** |
| 2 | PER | 128w×3b | 200 | -1.558 | 7.8% |
| 3 | PER+Reanalyze | 256w×5b | 50 | -1.573 | 7.0% |
| 4 | Baseline | 128w×3b | 200 | -1.746 | 4.6% |
| 5 | PER | 128w×3b | 50 | -1.759 | 4.8% |
| 6 | Baseline | 128w×3b | 50 | -1.841 | 3.5% |
| 7 | Bearoff rollouts | 128w×3b | 50 | -1.993 | 2.1% |
| 8 | Reanalyze | 128w×3b | 50 | -2.054 | 2.3% |

**All models are genuinely weak** — best is 9.6% wins vs GnuBG 0-ply (weakest setting).

**Note**: Pre-v0.3.2 results used asymmetric initial positions and are NOT comparable. Pre-2026-02-14 GnuBG results used buggy board encoding and are INVALID.

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

### Active Sessions (corrected equity vs GnuBG 0-ply)
- `distributed_20260213_031243_per_reanalyze` - **Best overall** (256w×5b, 200 iter PER+Reanalyze, -1.361 equity, 9.6% wins)
- `distributed_20260209_215824_per` - Best 128w (200 iter PER, -1.558 equity, 7.8% wins)
- `distributed_20260213_010615_per_reanalyze` - 256w×5b 50-iter (-1.573 equity, 7.0% wins)
- `distributed_20260206_223524` - 200 iter baseline (-1.746 equity, 4.6% wins)
- `distributed_20260207_030412_per` - PER 50-iter (-1.759 equity, 4.8% wins)
- `distributed_20260206_204548` - Original 50-iter baseline (-1.841 equity, 3.5% wins)

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
8. **All models are genuinely weak** -- best gets 9.6% wins vs GnuBG 0-ply. Self-play training alone is not producing competitive play.
9. **Bear-off table has signal mismatch** -- table values are pre-dice, game states are post-dice. See `notes/bearoff_chance_node_issue.md`

### Evaluation
10. **Random baseline is misleading** -- always evaluate vs GnuBG
11. **MINIMAL features generalize best** -- simpler obs beats larger feature sets against GnuBG (relative ranking still valid post-fix)
12. **2000+ eval games** for reliable comparisons (500-game matchups × 4 = 2000 total, both sides)

## Next Steps

### Priority (next experiments)
1. **Diagnose weak self-play** -- best model gets 9.6% wins vs GnuBG 0-ply. Need fundamental training improvements.
2. **Scale up model + training** -- 256w×5b at 200 iter is best; try 256w×10b or 500+ iterations
3. **PER is the key technique** -- only reliable improvement. Focus on PER + longer training.
4. **CPUCT tuning** -- try 1.0, 1.5, 3.0 (default 2.0, never tuned with correct eval)
5. **Fix GnubgPlayerFast move conversion** -- would enable 16x faster gnubg eval

### Future
6. Web-based workers (WASM + WebGPU) via sibling project `/home/sile/github/tavlatalk/`
7. Cloud remote workers for distributed self-play
8. Match equity table (MET) for proper match play scoring
9. Exam eval (known tricky positions from GnuBG analysis)
