# Session Notes - 2026-02-05

## RESOLVED: Asymmetric Short Game Position

### Problem (BackgammonNet v0.3.2)
The `short_game=true` position was **asymmetric**:

| Starting Player | White Pips | Black Pips | Advantage |
|-----------------|------------|------------|-----------|
| White starts | 237 | 267 | White +30 pips |
| Black starts | 267 | 237 | Black +30 pips |

This explained the color asymmetry in training/evaluation results.

### Fix (BackgammonNet v0.4.1)
Updated to v0.4.1 which has a **symmetric** position:
- Both players: **113 pips** (45.7% shorter than standard 208 pips)
- Position is now identical regardless of starting player

```
    ┌─────────────────────────────┬─────────────────────────────┐
    │  13   14   15   16   17   18│  19   20   21   22   23   24│ BLACK
    │ B 1   ·    ·    ·   B 2  B 3 │  ·   B 3  W 2   ·   B 3  B 1 │ HOME
    │            BAR              │                             │
    │ W 1   ·    ·    ·   W 2  W 3 │  ·   W 3  B 2   ·   W 3  W 1 │ HOME
    │  12   11   10    9    8    7│   6    5    4    3    2    1│ WHITE
    └─────────────────────────────┴─────────────────────────────┘
    White: 113 pips | Black: 113 pips ✓ SYMMETRIC
```

## Changes Made

1. **Project.toml**:
   - Added `JSON` to deps (required for BackgammonNet v0.4.1)
   - Updated BackgammonNet compat from `0.3.2` to `0.4`
   - Added `JSON = "1.4"` to compat

2. **Manifest.toml**: Regenerated with new dependencies

## Next TODOs

1. **Run validation training** with symmetric position
   - All previous runs used asymmetric position, results not comparable

2. **Port PER + reanalyze** from `src/cluster/` to `train_distributed.jl`

## Key Learnings

1. **Always verify game state symmetry** when using custom starting positions
2. **Pip count calculation**:
   - White moves toward point 0: pips = point_number × checkers
   - Black moves toward point 25: pips = (25 - point_number) × checkers
3. **Standard backgammon**: 208 pips each (symmetric)
4. **New short game**: 113 pips each (symmetric, 45.7% shorter)
