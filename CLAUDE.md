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

### Self-Play Client (Neo)
```bash
julia --threads 30 --project scripts/selfplay_client.jl --server http://jarvis:9090 --api-key <key> --num-workers 22
```

**Note**: Use `--threads N` (with space) before `--project`. The `--threads=N` form after `--project` leaks into ARGS.

### Architecture
- **Server** (Jarvis): HTTP API, PER buffer, GPU training (RTX 4090), weight serving, eval, TensorBoard
- **Client** (Neo): self-play with batched MCTS, CPU inference (22 workers), sample upload, weight sync
- Same HTTP API for Julia clients and future web clients (tavlatalk)
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
julia --threads 16 --project scripts/eval_vs_gnubg.jl /homeshare/projects/AlphaZero.jl/sessions/distributed_20260206_204548/checkpoints/latest.data minimal 500 128 3 8 100

# Quick eval vs random
julia --project scripts/quick_eval.jl
```

**Always evaluate against GnuBG**, not just random. Random baseline is misleading -- models that dominate random may not generalize.

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

### Distributed Training
- `scripts/training_server.jl` - **Training server** (Jarvis, port 9090)
- `scripts/selfplay_client.jl` - **Self-play client** (Neo)
- `src/distributed/server.jl` - HTTP server + routes + client tracking
- `src/distributed/client.jl` - HTTP client for self-play workers
- `src/distributed/buffer.jl` - Thread-safe PER buffer
- `src/distributed/protocol.jl` - Wire format (MsgPack + JSON)
- `src/distributed/training.jl` - GPU training engine

### Evaluation Scripts
- `scripts/eval_vs_wildbg.jl` - Wildbg evaluation (primary)
- `scripts/eval_vs_gnubg.jl` - GnuBG evaluation (parallel, multi-process)
- `scripts/eval_race.jl` - Race model evaluation on fixed 2000 positions
- `scripts/diagnose_loss.jl` - Per-component loss analysis across checkpoints

### PER + Reanalyze + Bear-off
- PER: proportional priority sampling with IS weights (α=0.6, β=0.4→1.0)
- Reanalyze: buffer reanalysis with EMA value blending (25% per iter)
- Bear-off: exact k=6 table values for race/bearoff positions

### Shared Data (NFS)
- `/homeshare/projects/AlphaZero.jl/eval_data/race_starts_tuples.jls` — 98,516 beginning-of-race positions
- `/homeshare/projects/AlphaZero.jl/eval_data/race_eval_2000.jls` — 2000 fixed eval positions

### Archived
- `scripts/archive/` - Old single-machine training scripts (train_distributed.jl, train_race.jl, etc.)
- `src/archive/distributed/` - Old ZMQ-based distributed module (unused)
- `/homeshare/projects/AlphaZero.jl/sessions/archive/` - Pre-v0.3.2 and experimental sessions

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

### Active Sessions (corrected equity vs GnuBG 0-ply)
All under `/homeshare/projects/AlphaZero.jl/sessions/`:
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
8. **All models are genuinely weak** -- best gets 4.4% wins vs wildbg (small nets), 9.6% vs GnuBG 0-ply. Need training efficiency improvements before scaling.
9. **Bear-off table has signal mismatch** -- table values are pre-dice, game states are post-dice. See `notes/bearoff_chance_node_issue.md`. Previous "bearoff hurt" finding needs retesting post-board-fix.
10. **Dual-model architecture** (2026-03-10) -- contact (256w×10b) + race (128w×3b) implemented in train_distributed.jl. Testing in progress.

### Evaluation
11. **Random baseline is misleading** -- always evaluate vs wildbg or GnuBG
12. **Wildbg is harder than GnuBG 0-ply** -- roughly halves win rates, but rankings are consistent across both opponents
13. **MINIMAL features generalize best** -- simpler obs beats larger feature sets (relative ranking valid post-fix)
14. **1000+ eval games** for reliable comparisons (500 per side). Eval script: `scripts/eval_vs_wildbg.jl`
15. **Wildbg eval used small nets on main branch** (2026-03-10) -- not large/custom wildbg nets. Results may change with stronger wildbg build.

### MCTS & Eval (2026-03-11)
16. **Passthrough chance sampling is best for eval** -- Benchmarked all 5 mcts.jl chance modes (passthrough, sampling, stratified, progressive, full) at 1600 iters. Passthrough wins: -1.050 equity, 16% wins vs wildbg. Best alternative (sampling v=1.0) was -1.180. At 1600 iters, alternatives waste iterations on chance coverage instead of decision-tree depth. With a weak network, depth > width.
17. **1600 MCTS iters >> 100 iters for eval** -- dual-model at 1600 iters: -1.277 equity, 9.7% wins vs wildbg small. At 100 iters: -1.828, 1.6% wins vs wildbg large. More MCTS budget at eval time dramatically improves play quality.
18. **EvalMCTS module (`src/eval_mcts.jl`) has 3 critical bugs** -- terminal_value always 0.0 (no game outcome signal), player_switch always false at chance nodes, no virtual loss in batched traversal. These explain why EvalMCTS full expansion scored -2.304 equity (0% wins) vs passthrough -1.266 (10.8%). Fix bugs before retesting full expansion.
19. **Re-test progressive widening when model is stronger** -- As NN values become more accurate, proper chance averaging should help. The crossover point is likely when model can consistently beat wildbg/GnuBG 0-ply. Progressive widening is the best candidate (probability-ordered expansion).
20. **GPU inference on M3 Max (Metal.jl)** -- 4.12x raw speedup at batch=500, but 12ms kernel launch overhead means crossover at batch≈20-30. Best end-to-end: GPU-Lock with 6 workers = 2.36x (30 games/min). Metal is NOT thread-safe. Adding CPU workers alongside GPU HURTS. Speculative prefetch failed (0.09x). GPU is useful only when batch sizes are guaranteed large.
21. **Value error vs wildbg not yet measured** -- Need to track NN value prediction accuracy against wildbg equity to understand when chance expansion will start helping. TODO: build eval script with per-position value comparison.

## Next Steps

### Eval Improvements (immediate)

1. **Value error tracking vs wildbg** -- Build eval script that compares NN value predictions against wildbg equity on positions from self-play games. Track MSE/correlation separately for contact vs race model. Script: `scripts/eval_value_accuracy.jl` (TODO).

2. **Fix EvalMCTS bugs** -- 3 bugs in `src/eval_mcts.jl`: (a) terminal_value=0.0 → use `GI.white_reward`, (b) player_switch=false at chance nodes, (c) no virtual loss. After fixing, re-benchmark full chance expansion vs passthrough.

3. **GPU eval (revisit after model is stronger)** -- GPU-Lock with 6 workers = 2.36x speedup on Neo. Worth revisiting when model strength justifies longer eval runs. Scripts: `scripts/bench_gpu_eval.jl`.

### Phase 1: Training Efficiency (before scaling)

4. **Bear-off MCTS value override at decision nodes** -- Currently bear-off evaluator only fires at chance nodes. Add check at decision-node leaves too: if position is in bear-off table, return exact equity immediately (no oracle call). Massive variance reduction for ~50% of game positions. Modify `batched_mcts.jl` leaf evaluation.

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

### Future
15. Web-based workers (WASM + WebGPU) via sibling project `/home/sile/github/tavlatalk/`
16. Cloud remote workers for distributed self-play
17. Match equity table (MET) for proper match play scoring
18. Exam eval (known tricky positions from GnuBG analysis)
