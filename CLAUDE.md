# AlphaZero.jl - Claude Code Context

## Project Overview

This is a Julia implementation of AlphaZero with extensions for backgammon, including:
- Multi-head equity network (TD-Gammon style)
- Stochastic game support (dice rolling)
- Wandb integration for experiment tracking

## Current Best Approach (2026-01-25)

### Multi-Head Equity Network

**Architecture**: FCResNetMultiHead with 5 value heads:
1. P(win) - probability of winning
2. P(gammon|win) - probability of gammon given win
3. P(bg|win) - probability of backgammon given win
4. P(gammon|loss) - probability of gammon given loss
5. P(bg|loss) - probability of backgammon given loss

**Results**: Outperforms single-head SimpleNet baseline by 11% in half the iterations.

### Recommended Training Configuration

```julia
# Use train_multihead_v2.jl as template
netparams = FluxLib.FCResNetMultiHeadHP(
    width = 128,
    num_blocks = 3,
    depth_phead = 1,
    depth_vhead = 1,
    share_value_trunk = true
)

# Key parameters
arena = ArenaParams(
    ...,
    always_replace = true  # Track eval metrics but always accept new network
)
```

### Running Training

```bash
# With wandb logging (recommended)
julia --project scripts/train_multihead_v2.jl

# Evaluation with reward histograms
julia --project scripts/eval_current_iteration.jl sessions/<session_dir>
```

## Key Files

### Core Implementation
- `src/networks/architectures/fc_resnet_multihead.jl` - Multi-head network
- `src/game.jl` - Game interface including `game_outcome()` for win types
- `src/learning.jl` - Multi-head loss computation
- `src/trace.jl` - Trace with outcome storage
- `src/params.jl` - Parameters including `always_replace`

### Scripts (Active - 9 total)
- `scripts/train_multihead_v2.jl` - **Primary training script**
- `scripts/train_multihead_baseline_match.jl` - Baseline comparison
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

## Wandb Integration

All training runs should use wandb for tracking. Ensure wandb is configured:

```bash
# Install wandb in Julia's Python environment
julia -e 'using CondaPkg; CondaPkg.add("wandb")'

# Or via pip in CondaPkg environment
pip install wandb
wandb login
```

### Current State
Training scripts log:
- Initial configuration at start
- Final metrics at completion

### TODO: Per-Iteration Logging
The `src/ui/wandb.jl` module has helpers for per-iteration metrics:
- `iteration_metrics(report, iteration)` - Full iteration stats
- `checkpoint_metrics(report)` - Checkpoint evaluation
- `learning_metrics(report)` - Learning phase stats

To add per-iteration logging, need to either:
1. Add a handler/callback system to the training loop
2. Or use custom training script that calls `wandb_log()` after each iteration

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

## Distributed Training (2026-01-25)

### Overview
Distributed training system with ZMQ-based communication supporting:
- N self-play workers (local threads or remote servers with GPUs)
- Centralized inference server (optional, for CPU-only workers)
- Replay buffer manager
- Training process with GPU learning

### Running Distributed Training

```bash
# Single-server (all components share GPU)
julia --project --threads=8 scripts/train_single_server.jl \
    --game=backgammon-deterministic \
    --network-type=fcresnet-multihead \
    --network-width=128 \
    --network-blocks=3 \
    --num-workers=4 \
    --total-iterations=300 \
    --mcts-iters=100

# Multi-server (run coordinator + remote workers)
julia --project scripts/train_distributed.jl --coordinator ...
julia --project scripts/run_worker.jl --coordinator <ip> ...
```

### Key Lessons Learned
1. **Argument syntax**: Use `--arg=value` not `--arg value` when game modules are loaded
2. **Julia 1.12 world age**: Include game modules at top-level, not in functions
3. **Loss metrics differ**: Distributed script reports raw loss; baseline reports decomposed (Lv+Lp+Lreg)
4. **Playing strength matches**: Despite different loss values, actual performance vs random is equivalent
5. **GPU sharing works**: Multiple components can share GPU with lazy memory allocation

### Distributed Training Files
- `src/distributed/` - Core distributed module (11 files)
- `scripts/train_single_server.jl` - Single-machine training
- `scripts/train_distributed.jl` - Multi-server coordinator
- `scripts/run_worker.jl` - Remote worker script

## Next Steps (from roadmap)

1. ✅ Multi-head equity network - **DONE**
2. GnuBG evaluation integration
3. Match equity table (MET) integration
4. ✅ Distributed self-play - **DONE**
5. Reanalyze (MuZero style)
