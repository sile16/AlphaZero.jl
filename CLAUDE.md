# AlphaZero.jl - Claude Code Context

## Project Overview

This is a Julia implementation of AlphaZero with extensions for backgammon, including:
- Multi-head equity network (TD-Gammon style)
- Stochastic game support (dice rolling)
- TensorBoard integration for experiment tracking (pure Julia, no Python conflicts)

## Current Best Approach (2026-02-04)

### Training Options

**Option 1: Concurrent architecture with PER + Smart Reanalyze (RECOMMENDED)**
```bash
julia --project --threads=8 scripts/train_cluster.jl \
    --game=backgammon-deterministic \
    --network-type=fcresnet-multihead \
    --network-width=128 \
    --network-blocks=3 \
    --num-workers=6 \
    --total-iterations=250 \
    --games-per-iteration=50 \
    --mcts-iters=100 \
    --per --reanalyze --distributed \
    --final-eval-games=1000
```

**Option 2: Simple thread-based (no PER/reanalyze)**
```bash
julia --project --threads=8 scripts/train_cluster.jl \
    --game=backgammon-deterministic \
    --network-type=fcresnet-multihead \
    --network-width=128 \
    --network-blocks=3 \
    --num-workers=6 \
    --total-iterations=70 \
    --games-per-iteration=50 \
    --mcts-iters=100 \
    --final-eval-games=1000
```

TensorBoard is enabled by default. To disable: `--no-tensorboard`
View logs with: `tensorboard --logdir=sessions/<session>/tensorboard`

**Current Status**: See `notes/distributed_training_infrastructure.md` for TODOs and next steps.

**Features**:
- Git commit hash logged at start and saved to `run_info.txt`
- Parallel final evaluation (1000 games default, uses all threads)
- Progress logging every 10% during evaluation
- **PER** (Prioritized Experience Replay) for better sample efficiency
- **Smart Reanalyze** - MuZero-style, tracks model iteration per sample, stops when buffer is up-to-date
- **Concurrent architecture** (`--distributed`) - self-play, reanalyze, and eval run simultaneously
- **GnuBG evaluation during training** (use `--eval-vs-gnubg`)

**Option 3: Training with GnuBG evaluation** (more meaningful than random)
```bash
julia --project --threads=8 scripts/train_cluster.jl \
    --game=backgammon-deterministic \
    --network-type=fcresnet-multihead \
    --num-workers=6 \
    --total-iterations=70 \
    --eval-vs-gnubg \
    --gnubg-ply=0 \
    --gnubg-eval-games=50
```

**Note**: GnuBG uses PyCall which has threading issues with Julia's GC. For GnuBG eval, use single-threaded or run standalone eval after training.


### Multi-Head Equity Network

**Architecture**: FCResNetMultiHead with 5 value heads:
1. P(win) - probability of winning
2. P(gammon|win) - probability of gammon given win
3. P(bg|win) - probability of backgammon given win
4. P(gammon|loss) - probability of gammon given loss
5. P(bg|loss) - probability of backgammon given loss

**Results**: Outperforms single-head SimpleNet baseline by 11% in half the iterations.

### Evaluation

```bash
# Quick evaluation vs random player
julia --project scripts/quick_eval.jl

# Detailed evaluation with histograms
julia --project scripts/eval_current_iteration.jl sessions/<session_dir>

# GnuBG evaluation (RECOMMENDED - more meaningful than random)
julia --project scripts/eval_vs_gnubg.jl <checkpoint_path> [obs_type] [num_games]
# Example: julia --project scripts/eval_vs_gnubg.jl sessions/cluster_20260126_231628/checkpoints/latest.data minimal 500
```

**Important**: Always evaluate against GnuBG, not just random! See "Key Insight" below.

## Key Files

### Core Implementation
- `src/networks/architectures/fc_resnet_multihead.jl` - Multi-head network
- `src/game.jl` - Game interface including `game_outcome()` for win types
- `src/learning.jl` - Multi-head loss computation
- `src/trace.jl` - Trace with outcome storage
- `src/params.jl` - Parameters including `always_replace`

### Scripts (Active)
- `scripts/train_cluster.jl` - **Primary training script** (thread-based, TensorBoard, parallel final eval)
- `scripts/train_distributed.jl` - **Distributed training** (Julia Distributed, multi-machine, parallel worker eval)
- `scripts/eval_vs_gnubg.jl` - **GnuBG evaluation** (recommended for meaningful benchmarks)
- `scripts/quick_eval.jl` - Quick evaluation vs random
- `scripts/eval_current_iteration.jl` - Evaluation with histograms
- `scripts/backgammon_full_evaluation.jl` - Comprehensive evaluation
- `scripts/benchmark_gnubg.jl` - GnuBG benchmarking
- `scripts/GnubgPlayer.jl` - GnuBG integration module
- `scripts/alphazero.jl`, `mcts.jl`, `minmax.jl` - Core utilities

**Archived scripts**: `scripts/archive/` contains 55+ obsolete experimental scripts

### Games
- `games/backgammon-deterministic/` - Backgammon without exposed chance nodes
- `games/backgammon/` - Backgammon with stochastic MCTS support

### Documentation
- `notes/backgammon_improvement_roadmap.md` - Detailed experiment results and roadmap
- `notes/distributed_training_infrastructure.md` - Distributed training plan, browser workers, testing requirements

## TensorBoard Integration

TensorBoard logging is enabled by default. Logs are saved to `sessions/<session>/tensorboard/`.

### Viewing Logs
```bash
# Start TensorBoard server
tensorboard --logdir=sessions/<session>/tensorboard

# Or view all sessions
tensorboard --logdir=sessions
```

### Module: `src/ui/tensorboard.jl`

Pure Julia implementation using TensorBoardLogger.jl - no Python dependencies, works alongside PyCall (GnuBG).

Key functions:
- `tb_init(; logdir, run_name)` - Initialize TensorBoard logging
- `tb_log(metrics; step)` - Log metrics
- `tb_log_config(config)` - Log configuration as text
- `tb_finish()` - Finish logging
- `all_system_metrics(; host_id, cuda_module)` - Collect system/GPU metrics

### Metrics Tracked

**Training metrics** (every iteration):
- `train/loss` - Training loss
- `train/iteration` - Current iteration
- `train/iteration_time_s` - Time per iteration
- `buffer/size` - Replay buffer sample count
- `buffer/capacity_pct` - Buffer fullness %
- `games/total` - Total games played
- `games/per_minute` - Self-play throughput
- `samples/total` - Total samples generated
- `workers/active` - Number of workers

**System metrics** (every 5 iterations):
- `system/<host_id>/cpu_load_1m` - CPU load average
- `system/<host_id>/ram_used_gb` - RAM usage
- `system/<host_id>/ram_used_pct` - RAM usage %
- `system/<host_id>/gpu_mem_used_gb` - GPU memory usage
- `system/<host_id>/gpu_mem_used_pct` - GPU memory %
- `system/<host_id>/gpu_utilization_pct` - GPU utilization

**Evaluation metrics** (every N iterations, default 10):
- `eval/vs_random_white` - Avg reward as white vs random
- `eval/vs_random_black` - Avg reward as black vs random
- `eval/vs_random_combined` - Combined average
- `eval/games` - Games per evaluation
- `eval/time_s` - Evaluation duration

**GnuBG evaluation metrics** (when `--eval-vs-gnubg` is used):
- `eval/vs_gnubg{N}ply_white` - Avg reward as white vs GnuBG N-ply
- `eval/vs_gnubg{N}ply_black` - Avg reward as black vs GnuBG N-ply
- `eval/vs_gnubg{N}ply_combined` - Combined average
- `eval/vs_gnubg{N}ply_wr_white` - Win rate as white
- `eval/vs_gnubg{N}ply_wr_black` - Win rate as black
- `eval/vs_gnubg{N}ply_wr_combined` - Combined win rate
- `eval/gnubg_games` - Games per GnuBG evaluation
- `eval/gnubg_ply` - GnuBG lookahead depth used

**Summary metrics** (end of training):
- `summary/total_iterations`
- `summary/total_games`
- `summary/total_samples`
- `summary/total_time_min`
- `eval/final_vs_random_*` - Final evaluation results

### Multi-Machine Support (Future)

System metrics include `host_id` prefix to distinguish metrics from different machines.
When running distributed training across multiple hosts:
- Each host reports its own system metrics with unique prefix
- Training metrics come from coordinator only
- Use `--host-id=<name>` to set custom host identifier

## Testing

```bash
# Run multi-head tests
julia --project -e 'using Pkg; Pkg.test()'

# Or specifically
julia --project test/test_multihead.jl
```

## Session Directories

Training sessions are saved to `sessions/` with format:
- `cluster_YYYYMMDD_HHMMSS/`

Each contains:
- `checkpoints/latest.data` - Latest network weights
- `checkpoints/network_iterN.data` - Checkpoint at iteration N
- `tensorboard/` - TensorBoard logs
- `run_info.txt` - Git commit and run parameters

### Notable Sessions
- `cluster_20260201_211302` - 100-iter MINIMAL features (+2.01 vs random, +0.35 vs GnuBG 0-ply)
- `cluster_20260202_103525` - Sweep A: Baseline (128w×3b, 100 MCTS) → 1.73
- `cluster_20260202_112558` - Sweep B: Wider (256w×3b, 100 MCTS) → 1.68
- `cluster_20260202_130346` - Sweep C: Deeper (128w×6b, 100 MCTS) → 1.865
- `cluster_20260202_141047` - Sweep D: More MCTS (128w×3b, 200 MCTS) → 1.95
- `cluster_20260204_092136` - 100-iter PER + smart reanalyze test (+0.902 vs random, 48 min)

**Note**: Sessions before 2026-02-04 used the old asymmetric `short_game` initial position.
Results may not be directly comparable to newer runs with the fixed position.

## Performance Baselines

| Model | Iterations | vs Random | vs GnuBG 0-ply | vs GnuBG 1-ply | Notes |
|-------|------------|-----------|----------------|----------------|-------|
| SimpleNet (128, 6) | 128 | +1.11 | - | - | Single-process baseline |
| FCResNetMultiHead (128, 3) | 69 | +1.23 | - | - | Multi-head baseline |
| FCResNetMultiHead (cluster) | 70 | +1.21 | - | - | train_cluster.jl |
| **FCResNetMultiHead (100 iter)** | **100** | **+2.01** | **+0.35 (62%)** | **+0.20 (55%)** | **MINIMAL features, 2.6h training** |
| PER + Smart Reanalyze (100 iter) | 100 | +0.902 | - | - | 48 min, concurrent arch |

**Note on color asymmetry**: Earlier results showed severe white/black asymmetry (white 38-48% win vs
black 72-93%). This was caused by a bug in BackgammonNet's `short_game` initial position which was
asymmetric between players. The fix in BackgammonNet corrects the starting position. Previous results
measured with the old initial position may not be directly comparable to new results.

## Training (2026-01-26)

### Overview
Thread-based parallel training using `train_cluster.jl`:
- N self-play worker threads (single machine)
- Shared GPU for training
- Replay buffer with 100K sample capacity
- **TensorBoard integration** for real-time metrics tracking
- **Parallel final evaluation** using all threads

### Running Training

```bash
# Standard training (single machine, multi-threaded)
julia --project --threads=8 scripts/train_cluster.jl \
    --game=backgammon-deterministic \
    --network-type=fcresnet-multihead \
    --network-width=128 \
    --network-blocks=3 \
    --num-workers=6 \
    --total-iterations=70 \
    --games-per-iteration=50 \
    --mcts-iters=100 \
    --batch-size=256 \
    --eval-interval=10 \
    --eval-games=50 \
    --final-eval-games=1000

# View logs with TensorBoard
tensorboard --logdir=sessions/<session>/tensorboard
```

### Key Lessons Learned
1. **Argument syntax**: Use `--arg=value` not `--arg value` when game modules are loaded
2. **Julia 1.12 world age**: Include game modules at top-level, not in functions
3. **Loss metrics differ**: Distributed script reports raw loss; baseline reports decomposed (Lv+Lp+Lreg)
4. **Playing strength matches**: Despite different loss values, actual performance vs random is equivalent
5. **GPU sharing works**: Multiple components can share GPU with lazy memory allocation
6. **TensorBoard is pure Julia**: Uses TensorBoardLogger.jl, no Python conflicts with GnuBG
7. **Julia Distributed serialization is tricky**: Closures referencing complex types fail to serialize; thread-based approach more reliable
8. **Thread safety for parallel training**: ReentrantLock essential for sample buffer; version counters for weight sync
9. **Evaluation variance is high**: 50-game evals show ±0.2 variance; use 1000+ games for reliable comparisons
10. **Buffer capacity matters**: 100K samples prevents overfitting to recent games
11. **Reproducibility via --seed flag**: Use `--seed=12345` for reproducible runs; each worker gets a unique derived seed
12. **Git commit hash logged**: `train_cluster.jl` logs git commit at start and saves to `run_info.txt` for traceability
13. **Random baseline is misleading**: Models that beat random badly may not generalize to strong opponents (GnuBG)
14. **Simpler features may generalize better**: MINIMAL (780 features) beats BIASED (3172) against GnuBG despite worse vs random
15. **More MCTS iterations > larger networks**: 200 MCTS with baseline network (1.95) outperformed 256-wide network (1.68) at same iteration count
16. **Deeper networks help**: 6 blocks (1.865) outperformed 3 blocks (1.73) with only 35% more parameters
17. **Wider networks underperform**: 256-width (3x params) was slower AND scored worse than baseline - not worth the cost
18. **MCTS quality matters more than network size**: Higher quality self-play games lead to better learning
19. **Use clone() for game state copying**: Manual struct construction breaks when fields are added (e.g., `obs_type`). Always use `BackgammonNet.clone()` in `GI.current_state()`
20. **Hyperparameter sweep rankings reverse vs GnuBG**: More MCTS (1.95 vs random) performed WORST vs GnuBG 1-ply (-0.04), while baseline (1.73 vs random) performed BEST (+0.19). Overfitting to self-play distribution.

### Hyperparameter Sweep Results (2026-02-02)

30-iteration experiments with 200-game final evaluation vs random:

| Config | Network | MCTS | Final Score | Time | Throughput |
|--------|---------|------|-------------|------|------------|
| Baseline | 128w × 3b (281K) | 100 | 1.73 | 50 min | ~62 g/min |
| Wider | 256w × 3b (857K) | 100 | 1.68 | 95 min | ~20 g/min |
| Deeper | 128w × 6b (382K) | 100 | 1.865 | 66 min | ~47 g/min |
| **More MCTS** | 128w × 3b (281K) | 200 | **1.95** | 71 min | ~32 g/min |

**Winner: 200 MCTS iterations** - best score despite 2x slower per-game throughput.

**Recommended config for long training:**
- Network: 128w × 3b or 128w × 6b
- MCTS: 200 iterations
- Expected throughput: 25-32 games/min

### Training Infrastructure
- `scripts/train_cluster.jl` - **Primary training script** (thread-based, TensorBoard, parallel eval)
- `scripts/train_distributed.jl` - **Distributed training** (Julia Distributed, multi-machine capable)
- `src/cluster/` - Thread-based cluster module (4 files)
- `src/ui/tensorboard.jl` - TensorBoard integration with system metrics

### Distributed Training (2026-02-01)

For multi-machine training, use `train_distributed.jl`:

```bash
# Local distributed (spawns worker processes on same machine)
julia --project scripts/train_distributed.jl \
    --num-workers=6 \
    --total-iterations=50 \
    --games-per-iteration=50 \
    --mcts-iters=100

# Remote workers (future: add machines via --worker-hosts)
julia --project scripts/train_distributed.jl \
    --num-workers=0 \
    --worker-hosts="worker1,worker2,worker3"
```

**Performance vs Thread-based (single machine, 6 workers):**
| Metric | Thread-based | Distributed |
|--------|-------------|-------------|
| Throughput | ~59 games/min | ~47 games/min |
| Overhead | - | ~20% (IPC serialization) |

**Advantages of distributed:**
- Scales across multiple machines
- Isolated memory per worker (no GIL issues)
- Foundation for web-based workers (WASM/WebGPU)

### Browser Workers (Planned)

**Sibling project**: `/home/sile/github/tavlatalk/` - WebGPU-powered backgammon AI

**Architecture:**
- ONNX Runtime Web for neural network inference (WebGPU backend)
- Rust/WebAssembly for MCTS
- AssemblyScript for game logic

**Integration plan:**
1. WebSocket server in Julia for model distribution + sample collection
2. Browser workers run self-play and send samples back
3. Leaderboard and stats for community participation
4. User-adjustable resource controls

See `notes/distributed_training_infrastructure.md` for full plan.

## Observation Types (BackgammonNet v0.3.2+)

| obs_type | Size | Format | Description |
|----------|------|--------|-------------|
| `:minimal_flat` | **330** | Vector | Flat MLP input (RECOMMENDED) |
| `:full_flat` | **362** | Vector | Flat + extra features |
| `:minimal` | 30×1×26 | Tensor | For Conv1D networks |
| `:full` | 62×1×26 | Tensor | Conv + extra features |

Set via environment variable: `BACKGAMMON_OBS_TYPE=minimal` (or `full`, `minimal_flat`, `full_flat`)

**Note**: v0.3.2 simplified observation sizes significantly (330/362 vs previous 780/1612).

## Key Insight: Observation Features (2026-01-27)

**Critical finding from GnuBG evaluation (pre-v0.3.2 observation sizes):**

| Model | Features | vs Random | vs GnuBG 0-ply | vs GnuBG 1-ply |
|-------|----------|-----------|----------------|----------------|
| MINIMAL | 780 | +1.23 (3rd) | **+0.553 (1st)** | **+0.391 (1st)** |
| FULL | 1612 | +1.318 (2nd) | +0.466 (2nd) | +0.305 (2nd) |
| BIASED | 3172 | **+1.339 (1st)** | +0.424 (3rd) | +0.235 (3rd) |

**Rankings flip completely!** BIASED wins vs random but MINIMAL wins vs GnuBG.

**Implications:**
1. **Random baseline is misleading** - doesn't test generalization
2. **More features can hurt** - heuristic features may cause overfitting
3. **MINIMAL features recommended** for robust, generalizable play
4. **Always evaluate vs GnuBG** to catch overfitting to weak opponents

See `notes/stochastic_mcts_experiments.md` Experiment 6 for full analysis.

## Next Steps (from roadmap)

### Completed
1. ✅ Multi-head equity network - **DONE**
2. ✅ Thread-based parallel training - **DONE** (train_cluster.jl, 4-6x throughput)
3. ✅ TensorBoard integration - **DONE** (system + training metrics, pure Julia)
4. ✅ Parallel final evaluation - **DONE** (40-60x speedup)
5. ✅ `--seed` flag for reproducibility - **DONE** (thread-local derived seeds)
6. ✅ Observation feature comparison - **DONE** (2026-01-27)
7. ✅ GnuBG evaluation scripts - **DONE** (scripts/eval_vs_gnubg.jl)
8. ✅ Julia Distributed training - **DONE** (train_distributed.jl, ~80% of thread throughput, multi-machine capable)
9. ✅ GnuBG eval in training loop - **DONE** (use `--eval-vs-gnubg` in train_cluster.jl)
10. ✅ Replace WandB with TensorBoard - **DONE** (pure Julia, no Python conflicts)
11. ✅ Longer training with MINIMAL features - **DONE** (100 iter, +2.01 vs random, +0.35 vs GnuBG 0-ply)
12. ✅ PER (Prioritized Experience Replay) - **DONE** (Schaul et al. 2016, proportional prioritization)
13. ✅ Reanalyze Phase 1 (MuZero style) - **DONE** (basic value reanalysis with TD-error tracking)
14. ✅ Smart Reanalyze (Phase 2) - **DONE** (model iteration tracking, staleness-based prioritization)
15. ✅ Concurrent architecture - **DONE** (self-play, reanalyze, eval run simultaneously)
16. ✅ Color asymmetry resolved - **DONE** (was caused by asymmetric `short_game` initial position in BackgammonNet)

### Next Priority
17. **Scale up training** - 2+ hour runs with PER + smart reanalyze to validate at scale
18. **Match equity table (MET) integration** - Proper match play scoring
19. **GnuBG eval for new runs** - Benchmark new PER+reanalyze runs against GnuBG

### Future
20. Web-based workers using WASM and WebGPU - Leverage browser compute for distributed training
21. Curriculum learning - Progressive training difficulty
22. Pre-race-net and race-net - Specialized networks for racing positions
23. Exam eval - Known tricky positions for evaluation only
24. Gym - Training on known board positions with known targets
25. Precomputed endgame tables - Avoid running games to completion
26. **Reanalyze Phase 3** - Bear-off database and curated position library integration

## PER + Smart Reanalyze (2026-02-04)

### Prioritized Experience Replay (PER)
Based on Schaul et al. 2016 "Prioritized Experience Replay" (ICLR 2016).
- **Proportional prioritization**: P(i) = p_i^alpha / sum(p_k^alpha)
- **Importance sampling weights**: w_i = (N * P(i))^(-beta) for bias correction
- **Beta annealing**: beta linearly anneals from 0.4 to 1.0 over training
- Enable with `--per` flag

### Smart Reanalyze (MuZero-style)
Re-evaluates stored positions with the latest network. Smart strategy:
- **Model iteration tracking**: Each sample tracks which model iteration it was last reanalyzed with
- **Staleness-based priority**: Samples with oldest model iteration are reanalyzed first
- **Automatic stop/resume**: Stops when all samples are up-to-date, resumes when model updates
- **Concurrent**: Runs on a separate thread, doesn't block training
- Enable with `--reanalyze` flag

### Key Files
- `src/reanalyze.jl` - ReanalyzeConfig, ReanalyzeStats, smart sampling functions
- `src/cluster/types.jl` - ClusterSample with PER + reanalyze fields
- `src/cluster/async_workers.jl` - AsyncReanalyzeWorker, AsyncEvalWorker
- `src/cluster/Cluster.jl` - Concurrent training loop integration
- `src/cluster/coordinator.jl` - PER sampling, TD-error priority updates

### Usage
```bash
julia --project --threads=8 scripts/train_cluster.jl \
    --per --reanalyze --distributed \
    --game=backgammon-deterministic \
    --total-iterations=250 \
    ...
```

### ClusterSample Fields
| Field | Type | Description |
|-------|------|-------------|
| `priority` | Float32 | TD-error based priority for PER sampling |
| `added_step` | Int | Training iteration when sample was added |
| `last_reanalyze_step` | Int | Last reanalysis step (worker counter) |
| `reanalyze_count` | Int | Times this sample has been reanalyzed |
| `model_iter_reanalyzed` | Int | Model iteration sample was last reanalyzed with |

### Training Results (100 iter test)
- 605K smart reanalyze operations in 48 min
- Buffer correctly tracks stale samples per model iteration
- FIFO eviction (research-backed: priority eviction doesn't help)

### Future: Phase 3 - External Data
- **Bear-off database**: Exact equity for endgame positions
- **Curated position library**: High-quality positions from rollouts/experts

**Ideas/Notes:**
- For stochastic implementation: train a stochastic head that outputs priors for all 21 dice outcomes for V, so we know the prior for all 21 options. Optionally predict which stochastic options will have the highest absolute change in V, use that to sample top-k extreme outcomes for better value estimates.
- Another idea: use full stochastic node expansion in eval ONLY (doesn't seem to help training) 
