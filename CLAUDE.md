# AlphaZero.jl - Claude Code Context

## Project Overview

This is a Julia implementation of AlphaZero with extensions for backgammon, including:
- Multi-head equity network (TD-Gammon style)
- Stochastic game support (dice rolling)
- Wandb integration for experiment tracking

## Current Best Approach (2026-01-26)

### ALWAYS Use Cluster Training with WandB

**Standard training command** (single host, multi-threaded):
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

WandB is enabled by default (project: `alphazero-jl`). To disable: `--no-wandb`

**Features**:
- Git commit hash logged at start and saved to `run_info.txt`
- Parallel final evaluation (1000 games default, uses all threads)
- Progress logging every 10% during evaluation

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
```

## Key Files

### Core Implementation
- `src/networks/architectures/fc_resnet_multihead.jl` - Multi-head network
- `src/game.jl` - Game interface including `game_outcome()` for win types
- `src/learning.jl` - Multi-head loss computation
- `src/trace.jl` - Trace with outcome storage
- `src/params.jl` - Parameters including `always_replace`

### Scripts (Active)
- `scripts/train_cluster.jl` - **Primary training script** (thread-based, wandb, parallel final eval)
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

## WandB Integration

WandB is enabled by default in `train_single_server.jl`. All training runs log to project `alphazero-jl`.

### Setup (one-time)
```bash
# wandb is auto-installed via CondaPkg when you first run the script
# Just need to login once:
wandb login
```

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

**Summary metrics** (end of training):
- `summary/total_iterations`
- `summary/total_games`
- `summary/total_samples`
- `summary/total_time_min`
- `summary/avg_loss`
- `eval/final_vs_random_*` - Final evaluation results

### Multi-Machine Support (Future)

System metrics include `host_id` prefix to distinguish metrics from different machines.
When running distributed training across multiple hosts:
- Each host reports its own system metrics with unique prefix
- Training metrics come from coordinator only
- Use `--host-id=<name>` to set custom host identifier

### Module: `src/ui/wandb.jl`

Key functions:
- `wandb_init(; project, name, config)` - Initialize run
- `wandb_log(metrics; step)` - Log metrics
- `wandb_finish()` - Finish run
- `all_system_metrics(; host_id, cuda_module)` - Collect system/GPU metrics

**Important**: Scripts using wandb must `using PythonCall` before calling wandb functions.

## Testing

```bash
# Run multi-head tests
julia --project -e 'using Pkg; Pkg.test()'

# Or specifically
julia --project test/test_multihead.jl
```

## Session Directories

Training sessions are saved to `sessions/` with format:
- `bg-multihead-v2-YYYYMMDD_HHMMSS/`

Each contains:
- `bestnn.data` - Best network weights
- `iter.txt` - Current iteration count
- `log.txt` - Training log
- `params.json` - Training parameters

## Performance Baselines

| Model | Iterations | Combined Reward vs Random | Notes |
|-------|------------|---------------------------|-------|
| SimpleNet (128, 6) | 128 | +1.11 | Single-process baseline |
| FCResNetMultiHead (128, 3) | 69 | +1.23 | Multi-head baseline |
| **FCResNetMultiHead (cluster)** | **70** | **+1.21** | **train_cluster.jl (1000 games, 98.5% of baseline)** |

## Training (2026-01-26)

### Overview
Thread-based parallel training using `train_cluster.jl`:
- N self-play worker threads (single machine)
- Shared GPU for training
- Replay buffer with 100K sample capacity
- **WandB integration** for real-time metrics tracking
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
    --final-eval-games=1000 \
    --wandb-project=alphazero-jl

# Disable wandb if needed
julia --project --threads=8 scripts/train_cluster.jl --no-wandb ...
```

### Key Lessons Learned
1. **Argument syntax**: Use `--arg=value` not `--arg value` when game modules are loaded
2. **Julia 1.12 world age**: Include game modules at top-level, not in functions
3. **Loss metrics differ**: Distributed script reports raw loss; baseline reports decomposed (Lv+Lp+Lreg)
4. **Playing strength matches**: Despite different loss values, actual performance vs random is equivalent
5. **GPU sharing works**: Multiple components can share GPU with lazy memory allocation
6. **WandB requires PythonCall**: Scripts must `using PythonCall` before calling wandb functions
7. **Julia Distributed serialization is tricky**: Closures referencing complex types fail to serialize; thread-based approach more reliable
8. **Thread safety for parallel training**: ReentrantLock essential for sample buffer; version counters for weight sync
9. **Evaluation variance is high**: 50-game evals show ±0.2 variance; use 1000+ games for reliable comparisons
10. **Buffer capacity matters**: 100K samples prevents overfitting to recent games
11. **Reproducibility via --seed flag**: Use `--seed=12345` for reproducible runs; each worker gets a unique derived seed
12. **Git commit hash logged**: `train_cluster.jl` logs git commit at start and saves to `run_info.txt` for traceability

### Training Infrastructure
- `scripts/train_cluster.jl` - **Primary training script** (thread-based, wandb, parallel eval)
- `src/cluster/` - Thread-based cluster module (4 files)
- `src/ui/wandb.jl` - WandB integration with system metrics
- `src/distributed/` - Multi-server module (WIP, archived)

## Next Steps (from roadmap)

### Completed
1. ✅ Multi-head equity network - **DONE**
2. ✅ Thread-based parallel training - **DONE** (train_cluster.jl, 4-6x throughput)
3. ✅ WandB integration - **DONE** (system + training metrics)
4. ✅ Parallel final evaluation - **DONE** (40-60x speedup)
5. ✅ `--seed` flag for reproducibility - **DONE** (thread-local derived seeds)
6. ✅ Observation feature engineering comparison - **DONE** (2026-01-27)
   - BIASED (3172 features) best: +8.9% vs minimal
   - Recommend BIASED for production training

### Next Priority
7. **GnuBG evaluation integration** - Benchmark against real backgammon AI
8. **Match equity table (MET) integration** - Proper match play scoring
9. **Longer training runs with BIASED features** - Find performance ceiling

### Future
10. Multi-machine training using Julia Distributed
11. Reanalyze (MuZero style)
12. Curriculum learning - Progressive training difficulty
13. Pre-race-net and race-net - Specialized networks for racing positions
14. Exam eval - Known tricky positions for evaluation only
15. Gym - Training on known board positions with known targets
16. Precomputed endgame tables - Avoid running games to completion

**Ideas/Notes:**
- For stochastic implementation: train a stochastic head that outputs priors for all 21 dice outcomes for V, so we know the prior for all 21 options. Optionally predict which stochastic options will have the highest absolute change in V, use that to sample top-k extreme outcomes for better value estimates.
- Another idea: use full stochastic node expansion in eval ONLY (doesn't seem to help training) 
