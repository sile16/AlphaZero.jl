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

## Next Steps (from roadmap)

1. ✅ Multi-head equity network - **DONE**
2. GnuBG evaluation integration
3. Match equity table (MET) integration
4. Distributed self-play
5. Reanalyze (MuZero style)
