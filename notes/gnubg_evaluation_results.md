# GnuBG Evaluation Results

## Summary

Comprehensive evaluation of trained AlphaZero models against GnuBG (GNU Backgammon) at various ply levels.

## Best Models vs GnuBG

### vs GnuBG 1-ply (Most Meaningful Benchmark)

| Rank | Model | Obs Features | vs GnuBG 1-ply | Win Rate | Training |
|------|-------|--------------|----------------|----------|----------|
| **1** | MINIMAL (v0.3.1) | 780 | **+0.391** | **62.1%** | 69 iter |
| 2 | FULL (v0.3.1) | 1612 | +0.305 | 58.8% | 69 iter |
| 3 | full_flat (v0.3.2) | 362 | +0.290 | 57.6% | 69 iter |
| 4 | minimal_flat (v0.3.2) | 330 | +0.245 | 57.7% | 69 iter |
| 5 | BIASED (v0.3.1) | 3172 | +0.235 | 55.6% | 69 iter |
| 6 | 100-iter MINIMAL | 780 | +0.20 | 55% | 100 iter |

### vs GnuBG 0-ply (Neural Network Only)

| Rank | Model | Obs Features | vs GnuBG 0-ply | Win Rate |
|------|-------|--------------|----------------|----------|
| **1** | MINIMAL (v0.3.1) | 780 | **+0.553** | **70.2%** |
| 2 | FULL (v0.3.1) | 1612 | +0.466 | 66.8% |
| 3 | full_flat (v0.3.2) | 362 | +0.455 | 66.3% |
| 4 | BIASED (v0.3.1) | 3172 | +0.424 | 64.4% |
| 5 | 100-iter MINIMAL | 780 | +0.35 | 62% |

## Key Findings

### 1. Rankings Reverse vs Random

| Model | vs Random | vs GnuBG 0-ply | vs GnuBG 1-ply |
|-------|-----------|----------------|----------------|
| BIASED | **+1.339 (1st)** | +0.424 (3rd) | +0.235 (3rd) |
| FULL | +1.318 (2nd) | +0.466 (2nd) | +0.305 (2nd) |
| MINIMAL | +1.23 (3rd) | **+0.553 (1st)** | **+0.391 (1st)** |

**Critical insight**: Performance vs random does NOT predict generalization to strong opponents.

### 2. Fewer Features Can Be Better

- MINIMAL (780 features) beats BIASED (3172 features) by a large margin vs GnuBG
- full_flat (362 features) nearly matches FULL (1612 features) with 4.5x fewer features
- Heuristic features may cause overfitting to self-play distribution

### 3. Color Asymmetry Problem

All models show severe performance asymmetry:

| Role | Win Rate Range | Notes |
|------|----------------|-------|
| As Black (2nd player) | 72-93% | Strong |
| As White (1st player) | 38-48% | Weak |

This suggests models learned a reactive/defensive style.

## Training Configuration

All evaluations used models trained with:
- Game: `backgammon-deterministic` with `SHORT_GAME=true`
- Network: FCResNetMultiHead (128 width, 3 blocks, 281K params)
- MCTS: 100 iterations during self-play
- Evaluation MCTS: 100 iterations, temperature=0 (deterministic)

## Bug Fix: Game State Cloning (2026-01-27)

### The Problem

The `GI.current_state()` function in `games/backgammon-deterministic/game.jl` was manually constructing `BackgammonGame` objects for MCTS state copying:

```julia
# OLD (BUGGY) CODE:
function GI.current_state(g::GameEnv)
  game = g.game
  return BackgammonNet.BackgammonGame(
    game.p0, game.p1, game.dice, game.remaining_actions,
    game.current_player, game.terminated, game.reward,
    copy(game.history), game.doubles_only,
    Int[], Int[], Int[]  # Fresh buffers for the copy
  )
end
```

**Issues with this approach:**
1. **Missing `obs_type` field** - Added in BackgammonNet v0.3.2, not included in manual construction
2. **Missing `_actions_cached` field** - Added for legal actions caching
3. **Brittle to BackgammonNet changes** - Any new field breaks the copy

### The Fix

Use `BackgammonNet.clone()` which properly copies all fields:

```julia
# NEW (FIXED) CODE:
function GI.current_state(g::GameEnv)
  # Use clone() for safe deep copy with all fields including obs_type
  return BackgammonNet.clone(g.game)
end
```

### Impact

When `obs_type` was missing from cloned states:
- MCTS simulations used wrong observation encoding
- Network received inconsistent input dimensions
- Training may have been corrupted

### Related Changes in BackgammonNet.jl

1. **c8e873e**: Added `clone()` function and backwards-compatible 12-arg constructor
2. **50efcc0**: Added `obs_type` field to `BackgammonGame` struct
3. **c48ef94**: Updated test helpers to use `clone()` instead of direct construction

### Verification

Both players now see identical board states during evaluation:
1. `eval_vs_gnubg.jl` creates game with `GI.init(gspec)` → uses SHORT_GAME=true
2. `GnubgPlayer.think()` passes `game.game` (BackgammonGame) to gnubg
3. State cloning via `clone()` preserves all fields including `obs_type`

## Recommendations

1. **Always use MINIMAL features** for training when targeting strong play
2. **Evaluate vs GnuBG periodically** during training (not just random)
3. **Investigate color asymmetry** - may indicate training issue
4. **Consider full backgammon** - SHORT_GAME may not transfer perfectly
5. **Use 1-ply as primary benchmark** - more stable than 0-ply

## Sessions

| Session | Config | vs Random | vs GnuBG 0-ply | vs GnuBG 1-ply |
|---------|--------|-----------|----------------|----------------|
| `cluster_20260201_211302` | 100 iter, MINIMAL | +2.01 | +0.35 (62%) | +0.20 (55%) |
| `cluster_20260202_233010` | 100 iter, 128w×3b | +1.29 | +0.425 (63.5%) | **+0.215 (56.2%)** |
| `cluster_20260202_233013` | 100 iter, 128w×6b | +1.04 | +0.415 (65.5%) | +0.095 (50.0%) |

## Next Steps

- [ ] Verify short_game training transfers to standard positions
- [ ] Investigate color asymmetry root cause
- [ ] Test longer training (200+ iterations)
- [ ] Compare short_game=false training
