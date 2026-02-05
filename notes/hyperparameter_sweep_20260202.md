# Hyperparameter Sweep Results (2026-02-02)

## Objective
Find optimal network architecture and MCTS settings before long training run.

## Experiments

All experiments: 30 iterations, seed=42, 6 workers, 50 games/iteration, 200-game final eval.

### Experiment A: Baseline
- **Config**: 128w × 3b (281,641 params), 100 MCTS
- **Session**: `sessions/cluster_20260202_103525`
- **Time**: 49.6 minutes
- **Throughput**: ~62 games/min
- **Periodic evals**: iter10=1.52, iter20=1.80, iter30=1.82
- **Final eval**: White=1.76, Black=1.70, **Combined=1.73**

### Experiment B: Wider Network
- **Config**: 256w × 3b (857,513 params), 100 MCTS
- **Session**: `sessions/cluster_20260202_112558`
- **Time**: 95.12 minutes (1.9x slower)
- **Throughput**: ~20 games/min (3x slower)
- **Periodic evals**: iter10=1.86, iter20=2.06, iter30=1.64
- **Final eval**: White=1.74, Black=1.62, **Combined=1.68**
- **Note**: 3x more params but WORSE final score. Periodic evals were higher but didn't hold.

### Experiment C: Deeper Network
- **Config**: 128w × 6b (382,249 params), 100 MCTS
- **Session**: `sessions/cluster_20260202_130346`
- **Time**: 65.81 minutes (1.3x slower)
- **Throughput**: ~47 games/min
- **Periodic evals**: iter10=1.58, iter20=1.78, iter30=1.78
- **Final eval**: White=1.88, Black=1.85, **Combined=1.865**
- **Note**: Best color balance (only +0.03 delta). Modest param increase, good improvement.

### Experiment D: More MCTS Iterations
- **Config**: 128w × 3b (281,641 params), 200 MCTS
- **Session**: `sessions/cluster_20260202_141047`
- **Time**: 71.42 minutes (1.4x slower)
- **Throughput**: ~32 games/min (2x slower per game)
- **Periodic evals**: iter10=1.66, iter20=2.22, iter30=1.82
- **Final eval**: White=2.06, Black=1.84, **Combined=1.95**
- **Note**: Best overall score! iter20 hit 2.22 which was highest seen.

## Summary Table

| Config | Params | MCTS | Final | Time | Notes |
|--------|--------|------|-------|------|-------|
| A: Baseline | 281K | 100 | 1.73 | 50m | Reference |
| B: Wider | 857K | 100 | 1.68 | 95m | Worse despite 3x params |
| C: Deeper | 382K | 100 | 1.865 | 66m | Good balance |
| **D: More MCTS** | 281K | 200 | **1.95** | 71m | **Best score** |

## Key Findings

1. **MCTS quality > network size**: More search iterations produced better games for training.
   - 200 MCTS (1.95) beat 256-width network (1.68) with same iteration count

2. **Wider networks don't help (and may hurt)**:
   - 3x parameters, 2x slower, worse final score
   - May overfit faster or not converge in 30 iterations

3. **Deeper networks help modestly**:
   - +35% params, +30% time, +8% score improvement
   - Most balanced white/black performance

4. **Periodic eval variance is high**:
   - Experiment D showed 2.22 at iter20 but 1.95 final (200 games)
   - 50-game periodic evals have high variance

## Color Asymmetry

| Config | White | Black | Delta (W-B) |
|--------|-------|-------|-------------|
| A | 1.76 | 1.70 | +0.06 |
| B | 1.74 | 1.62 | +0.12 |
| C | 1.88 | 1.85 | +0.03 |
| D | 2.06 | 1.84 | +0.22 |

All experiments showed white slightly stronger, contrary to 100-iter run where black was stronger.
This may be due to short training (30 iter) not fully converging.

## Recommendation for Long Training

**Option 1 (Recommended): More MCTS**
```bash
julia --project --threads=8 scripts/train_cluster.jl \
    --network-width=128 --network-blocks=3 \
    --mcts-iters=200 \
    --total-iterations=100
```
- Best score in sweep
- Same network, just better quality games
- ~32 games/min throughput

**Option 2: Deeper + More MCTS**
```bash
julia --project --threads=8 scripts/train_cluster.jl \
    --network-width=128 --network-blocks=6 \
    --mcts-iters=200 \
    --total-iterations=100
```
- Combines both improvements
- ~25 games/min throughput (estimate)
- May produce best overall results

## GnuBG Evaluation (2026-02-02)

**Critical finding**: Rankings reverse completely when evaluating against GnuBG instead of random!

### Results vs GnuBG

| Config | vs Random | vs GnuBG 0-ply | vs GnuBG 1-ply |
|--------|-----------|----------------|----------------|
| **A: Baseline** | 1.73 (3rd) | **+0.290 (61%)** | **+0.190 (55%)** |
| C: Deeper | 1.865 (2nd) | +0.170 (57%) | -0.090 (47%) |
| D: More MCTS | **1.95 (1st)** | +0.180 (57%) | -0.040 (50%) |

### Key Findings

1. **Baseline wins vs GnuBG** - The simplest config generalizes best
2. **More MCTS hurts generalization** - 200 MCTS scored highest vs random but near-zero vs GnuBG 1-ply
3. **Deeper network doesn't help** - 6 blocks worse than 3 blocks vs GnuBG
4. **Random baseline is misleading** - Don't use it to compare models

### Color Asymmetry (all models)

| Color | Win Rate Range |
|-------|----------------|
| As White | 26-38% |
| As Black | 65-91% |

All models struggle as white but dominate as black - suggests defensive/reactive play style.

### Implications for Training

1. **Use GnuBG eval during training** - Random doesn't predict real strength
2. **Don't over-optimize MCTS iterations** - May overfit to self-play
3. **Simpler configs may generalize better** - Avoid overfitting

## 100-Iteration Training Runs (2026-02-03)

Extended training to validate sweep findings with longer runs.

### Baseline (128w×3b, 100 MCTS) - 100 iterations
- **Session**: `sessions/cluster_20260202_233010`
- **vs Random**: +1.29
- **vs GnuBG 0-ply**: +0.425 (63.5% wins)
- **vs GnuBG 1-ply**: **+0.215 (56.2% wins)**
  - As white: -0.330 (37.5% wins)
  - As black: +0.760 (75.0% wins)

### Deeper (128w×6b, 100 MCTS) - 100 iterations
- **Session**: `sessions/cluster_20260202_233013`
- **vs Random**: +1.04
- **vs GnuBG 0-ply**: +0.415 (65.5% wins)
- **vs GnuBG 1-ply**: +0.095 (50.0% wins)
  - As white: -0.440 (30.5% wins)
  - As black: +0.630 (69.5% wins)

### Summary

| Model | Params | Iters | vs Random | vs GnuBG 1-ply |
|-------|--------|-------|-----------|----------------|
| **Baseline** | 281K | 100 | **+1.29** | **+0.215** |
| Deeper | 382K | 100 | +1.04 | +0.095 |

**Conclusion**: Baseline config is the clear winner for the big run.
- Better vs random (+1.29 vs +1.04)
- Better vs GnuBG 1-ply (+0.215 vs +0.095)
- 36% fewer parameters, faster training
- Color asymmetry persists but baseline is better in both roles

## Recommendation for Big Run

```bash
julia --project --threads=8 scripts/train_cluster.jl \
    --network-width=128 --network-blocks=3 \
    --mcts-iters=100 \
    --total-iterations=200 \
    --games-per-iteration=100 \
    --final-eval-games=1000
```

## Scripts

Sweep script: `scripts/run_hyperparameter_sweep.sh`
