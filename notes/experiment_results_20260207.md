# Experiment Results — 2026-02-07

## Overview

Tested 4 configurations, all using FCResNetMultiHead 128w×3b (283K params), AdamW lr=0.001, 400 MCTS sims, batch=50, 600K buffer, seed=42.

## Results Summary

| Experiment | Iters | Loss (final) | vs GnuBG 0-ply | vs GnuBG 1-ply | Time (min) |
|-----------|-------|-------------|----------------|----------------|------------|
| **Baseline** | 200 | 3.89 | +1.31 (89%) | **+1.05 (78%)** | 254.5 |
| **Baseline** | 50 | 3.97 | +0.94 (79%) | +0.55 (66%) | 75.0 |
| **PER** | 50 | 4.07 | +1.08 (83%) | +0.79 (72%) | 74.4 |
| **Reanalyze** | 50 | 3.98 | +0.97 (79%) | +0.69 (71%) | 80.0 |
| **Bear-off** | 50 | 3.98 | +1.00 (81%) | +0.83 (72%) | 67.1 |

## Key Findings

### 1. Scaling (Baseline 200 iter)
- Loss: 5.40 → 3.89 (plateaus at ~3.95 around iter 50, then slowly continues to ~3.89)
- GnuBG 1-ply: +0.55 (50 iter) → +1.05 (200 iter) — **massive improvement from just training longer**
- Eval vs random continued improving throughout (2.2→2.45)
- Model capacity (128w×3b) appears to be a soft limit — loss plateau but play keeps improving

### 2. PER (50 iter) — Best 50-iter Result
- **+0.79 vs GnuBG 1-ply** (vs +0.55 baseline) — **+44% improvement in equity**
- Higher loss (4.07 vs 3.97) but stronger play — PER prioritizes hard positions, raising average loss but improving decision quality
- IS weights introduce variance but focus learning on informative samples
- No throughput penalty (74.4 min ≈ baseline 75 min)

### 3. Reanalyze (50 iter) — Moderate Improvement
- **+0.69 vs GnuBG 1-ply** (vs +0.55 baseline) — **+25% improvement**
- Same loss as baseline (3.98) — value target refreshing helps convergence
- Eval vs random was weaker early (1.77 vs 2.13 at iter 30) but recovered to 2.24
- Slight overhead (80 min vs 75 min from GPU reanalysis passes)

### 4. Bear-off Rollouts (50 iter) — Strongest 50-iter GnuBG Improvement
- **+0.83 vs GnuBG 1-ply** (vs +0.55 baseline) — **+51% improvement in equity**
- Uses 50 random rollouts for bear-off race positions → better endgame value targets
- Same loss as baseline (3.98)
- Actually faster (67.1 min vs 75 min) — unclear why, possibly faster game completion

## Ranking (50-iter GnuBG 1-ply performance)

1. **Bear-off rollouts**: +0.83 (72.0% win) — best
2. **PER**: +0.79 (71.8% win) — close second
3. **Reanalyze**: +0.69 (70.9% win) — solid improvement
4. **Baseline**: +0.55 (65.6% win) — reference

## Implications

1. **Training longer is the biggest win**: 200 iter baseline (+1.05) beats all 50-iter experiments. Before adding complexity, scale compute first.
2. **Bear-off + PER could combine well**: Bear-off improves endgame targets, PER focuses on hard positions — these address different aspects.
3. **Reanalyze + PER is the natural pair**: Reanalyze refreshes value targets and feeds PER with updated TD-errors.
4. **Next experiment**: Run 200 iter with PER + bear-off to see if improvements compound.

## Session Directories
- Baseline 200 iter: `sessions/distributed_20260206_223524/`
- PER 50 iter: `sessions/distributed_20260207_030412_per/`
- Reanalyze 50 iter: `sessions/distributed_20260207_043500_reanalyze/`
- Bear-off 50 iter: `sessions/distributed_20260207_061713_bearoff/`

## Observations & Lessons Learned

### On Loss vs Play Strength
- Loss is a poor proxy for play strength. PER had the WORST loss (4.07) but the SECOND BEST play strength.
- The 200-iter baseline improved from +0.55 to +1.05 vs GnuBG 1-ply while loss only dropped from 3.97 to 3.89.
- Hypothesis: loss measures average prediction accuracy, but play strength depends on getting critical positions right. PER explicitly optimizes for critical positions.

### On Bear-off Rollouts
- Bear-off positions are ~15-20% of all training positions in short games. Replacing NN value estimates with rollout equity for these gives the model "free" perfect endgame knowledge.
- The 67.1 min runtime (vs 75 min baseline) is surprising — may indicate bear-off positions were causing more MCTS exploration (uncertain values → more search), and exact values reduce unnecessary search.
- 50 rollouts per position is likely overkill for pure race positions. Could test 10-20 rollouts, or even exact bear-off database lookup.

### On PER
- α=0.6 and β=0.4→1.0 worked out of the box (standard Atari hyperparams). No tuning needed.
- IS weights are critical — without them, PER would bias the network toward hard positions at the expense of common ones.
- The zero throughput overhead is because PER sampling (cumulative sum binary search) is O(log N) and negligible vs MCTS/inference.

### On Reanalyze
- 25% reanalysis fraction with EMA α=0.5 blending was conservative. Could try higher fraction or sharper blending.
- The early weakness in eval vs random (1.77 vs 2.13 at iter 30) suggests reanalyze may temporarily destabilize learning before the refreshed targets help.
- Would likely pair well with PER — reanalyze updates values, PER ensures updated samples get trained on proportionally.

### On Compute Scaling
- 200 iterations took 254.5 min (~4.2 hours) at 394 g/m. This is reasonable for a single machine.
- The diminishing returns curve (loss plateau) vs the linear play strength improvement suggests the model is extracting increasingly subtle strategic knowledge that doesn't show in average loss.
- Larger models may break through the loss plateau and unlock faster play strength gains.

## Recommended Next Experiments

1. **PER + Bear-off (200 iter)** — ~4 hours, highest expected improvement
2. **PER + Bear-off + Reanalyze (200 iter)** — ~5 hours, test if all three compound
3. **256w×10b model (200 iter, baseline)** — ~8-12 hours, test capacity hypothesis
4. **PER + Bear-off (200 iter, 256w×10b)** — ~12 hours, full configuration

## Raw GnuBG Evaluation Details

### Baseline 200 iter (2000 games)
- vs 0-ply: +1.313 (89.0%), white +1.320/black +1.306
- vs 1-ply: +1.053 (78.1%), white +1.068/black +1.038

### PER 50 iter (2000 games)
- vs 0-ply: +1.079 (83.4%), white +1.004/black +1.154
- vs 1-ply: +0.789 (71.8%), white +0.762/black +0.816

### Reanalyze 50 iter (2000 games)
- vs 0-ply: +0.971 (79.1%), white +0.982/black +0.960
- vs 1-ply: +0.685 (70.9%), white +0.700/black +0.670

### Bear-off 50 iter (2000 games)
- vs 0-ply: +0.998 (80.7%), white +1.036/black +0.960
- vs 1-ply: +0.832 (72.0%), white +0.830/black +0.834
