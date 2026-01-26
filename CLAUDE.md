# AlphaZero.jl - Claude Code Context

## Project Overview

This is a Julia implementation of AlphaZero with extensions for backgammon, including:
- Multi-head equity network (TD-Gammon style)
- Stochastic game support (dice rolling)
- Wandb integration for experiment tracking

## Current Best Approach (2026-01-26)

### ALWAYS Use Distributed Training with WandB

**Standard training command** (single host, multi-threaded):
```bash
julia --project --threads=8 scripts/train_single_server.jl \
    --game=backgammon-deterministic \
    --network-type=fcresnet-multihead \
    --network-width=128 \
    --network-blocks=3 \
    --num-workers=4 \
    --total-iterations=300 \
    --mcts-iters=100
```

WandB is enabled by default (project: `alphazero-jl`). To disable: `--no-wandb`

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
- `scripts/train_single_server.jl` - **Primary training script** (distributed, wandb)
- `scripts/quick_eval.jl` - Quick evaluation vs random
- `scripts/eval_current_iteration.jl` - Evaluation with histograms
- `scripts/backgammon_full_evaluation.jl` - Comprehensive evaluation
- `scripts/benchmark_gnubg.jl` - GnuBG benchmarking
- `scripts/GnubgPlayer.jl` - GnuBG integration module
- `scripts/train_multihead_v2.jl` - Legacy single-process training
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

**Summary metrics** (end of training):
- `summary/total_iterations`
- `summary/total_games`
- `summary/total_samples`
- `summary/total_time_min`
- `summary/avg_loss`

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

| Model | Iterations | Combined Reward vs Random |
|-------|------------|---------------------------|
| SimpleNet (128, 6) | 128 | +1.11 |
| **FCResNetMultiHead (128, 3)** | **69** | **+1.23** |
| FCResNetMultiHead (distributed) | 300 | +1.24 |

## Distributed Training (2026-01-26)

### Overview
Distributed training system with ZMQ-based communication supporting:
- N self-play workers (local threads or remote servers with GPUs)
- Centralized inference server (optional, for CPU-only workers)
- Replay buffer manager
- Training process with GPU learning
- **WandB integration** for real-time metrics tracking

### Running Distributed Training

```bash
# Single-server with wandb (RECOMMENDED)
julia --project --threads=8 scripts/train_single_server.jl \
    --game=backgammon-deterministic \
    --network-type=fcresnet-multihead \
    --network-width=128 \
    --network-blocks=3 \
    --num-workers=4 \
    --total-iterations=300 \
    --mcts-iters=100

# Disable wandb if needed
julia --project --threads=8 scripts/train_single_server.jl --no-wandb ...

# Multi-server (run coordinator + remote workers) - TODO
julia --project scripts/train_distributed.jl --coordinator ...
julia --project scripts/run_worker.jl --coordinator <ip> ...
```

### Key Lessons Learned
1. **Argument syntax**: Use `--arg=value` not `--arg value` when game modules are loaded
2. **Julia 1.12 world age**: Include game modules at top-level, not in functions
3. **Loss metrics differ**: Distributed script reports raw loss; baseline reports decomposed (Lv+Lp+Lreg)
4. **Playing strength matches**: Despite different loss values, actual performance vs random is equivalent
5. **GPU sharing works**: Multiple components can share GPU with lazy memory allocation
6. **WandB requires PythonCall**: Scripts must `using PythonCall` before calling wandb functions

### Distributed Training Files
- `src/distributed/` - Core distributed module (11 files)
- `src/ui/wandb.jl` - WandB integration with system metrics
- `scripts/train_single_server.jl` - Single-machine training with wandb
- `scripts/train_distributed.jl` - Multi-server coordinator (WIP)
- `scripts/run_worker.jl` - Remote worker script (WIP)

## Next Steps (from roadmap)

1. ✅ Multi-head equity network - **DONE**
2. ✅ Distributed self-play - **DONE**
3. ✅ WandB integration - **DONE** (system + training metrics)
4. GnuBG evaluation integration
5. Match equity table (MET) integration
6. Multi-machine distributed training
7. Reanalyze (MuZero style)
