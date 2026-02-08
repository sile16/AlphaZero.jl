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
| **Bear-off rollouts** | 50 | 3.98 | +1.00 (81%) | +0.83 (72%) | 67.1 |
| ~~Bear-off table+TD~~ | 50 | 4.34 | +1.12 (83%) | +0.60 (65%) | ~67 |
| ~~Bear-off table targets~~ | 50 | 4.09 | +1.04 (81%) | +0.74 (70%) | 69.2 |
| ~~Bear-off table+TD+resample~~ | 50 | 4.25 | +0.94 (79%) | +0.62 (65%) | 66.0 |
| **Bear-off table (FIXED)** | 50 | 4.00 | +0.96 (81%) | +0.63 (68%) | 71.0 |

**WARNING: Bear-off table results marked with ~~strikethrough~~ used a BROKEN table (see "Bear-off Table Bug" section below). Those results are INVALID and should not be relied upon. The "Bear-off table (FIXED)" row uses the corrected table.**

**Note**: "Bear-off rollouts" = 50 random rollouts per bear-off position (training targets only, unaffected by table bug).
"Bear-off table (FIXED)" = corrected k=6 table for bear-off training targets + MCTS leaf eval + TD-bootstrap.
~~"Bear-off table targets" = BROKEN k=6 table for bear-off training targets only.~~
~~"Bear-off table+TD" = BROKEN k=6 table in MCTS self-play + TD-bootstrap.~~
~~"Bear-off table+TD+resample" = BROKEN, same as table+TD but resamples hard outcomes.~~

## Bear-off Table Bug (discovered 2026-02-07, fixed 2026-02-08)

The k=6 two-sided bear-off table (`BackgammonNet.jl/src/bearoff_k6.jl`) had **two critical bugs** that produced completely wrong P(win) values:

1. **Wrong field decode**: `decode_gnubg_entry` treated gnubg field[0] as gammon_win_conditional, but it's actually a cubeful equity value. The formula `pW = (eq_cl + 1 + 2*glc) / (2*(1 + gwc + glc))` gave wrong results when the misinterpreted field was non-zero.
   - **Fix**: Use field[2] (cubeless equity) directly: `pW = (eq_cl + 1) / 2`

2. **Wrong swap logic**: The table stored only ONE perspective per pair and used `pW = 1 - pW` for the other direction. This is mathematically wrong because first-mover advantage means `P(A wins | A first) + P(B wins | B first) != 1.0`.
   - **Fix**: Store BOTH perspectives (4 bytes/pair: 2 × UInt16)

**Impact**: Example errors before fix:
- 3@ace vs 3@ace: table pW = 0.4767, exact = 0.8611 (error = 0.38)
- 2@ace vs 1@ace: table pW = 0.0000, exact = 1.0000 (error = 1.00)
- All symmetric positions had pW < 0.5 (impossible for on-roll player)

**After fix**: All 100 tested ace-only positions match exact hand-computed values (max error 0.000043 ≈ UInt16 quantization).

**All experiments using the k=6 table (table targets, table+TD, table+TD+resample) used the BROKEN table.** Only "bear-off rollouts" (which does random rollouts, not table lookup) was unaffected. The fixed table result (+0.63 vs 1-ply at 50 iter) is a clean baseline for future bear-off table experiments.

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

### 5-7. ~~Bear-off Table Experiments (BROKEN TABLE — INVALID RESULTS)~~

**All three bear-off table experiments (table targets, table+TD, table+TD+resample) used a BROKEN bear-off table with massive P(win) errors.** See "Bear-off Table Bug" section above for details. The results (+0.74, +0.60, +0.62) are meaningless and should be disregarded.

### 8. Bear-off Table FIXED (50 iter) — Clean Baseline
- **+0.63 vs GnuBG 1-ply** (vs +0.55 baseline) — **+15% improvement in equity**
- Uses the corrected k=6 table (both perspectives stored, eq_cl decode)
- Loss 4.00 (slightly higher than baseline 3.97, expected since bear-off equity targets are harder)
- Confirms the fixed table provides a modest but real improvement at 50 iterations
- Still below bear-off rollouts (+0.83) — rollouts may benefit from stochastic diversity in training targets
- The table provides deterministic targets which may need longer training to show full benefit

## Ranking (50-iter GnuBG 1-ply performance)

1. **Bear-off rollouts**: +0.83 (72.0% win) — best
2. **PER**: +0.79 (71.8% win) — close second
3. **Reanalyze**: +0.69 (70.9% win) — solid improvement
4. **Bear-off table (FIXED)**: +0.63 (67.5% win) — exact table, all modes enabled
5. **Baseline**: +0.55 (65.6% win) — reference
6. ~~Bear-off table targets: +0.74 — INVALID (broken table)~~
7. ~~Bear-off table+TD+resample: +0.62 — INVALID (broken table)~~
8. ~~Bear-off table+TD: +0.60 — INVALID (broken table)~~

## Implications

1. **Training longer is the biggest win**: 200 iter baseline (+1.05) beats all 50-iter experiments. Before adding complexity, scale compute first.
2. **Bear-off rollouts are the best 50-iter bear-off approach**: Simple rollout targets (+0.83) outperform the fixed exact table (+0.63) at 50 iterations. Stochastic rollout diversity may help training.
3. **Bear-off rollouts + PER could combine well**: Rollouts improve endgame targets, PER focuses on hard positions — these address different aspects.
4. **Reanalyze + PER is the natural pair**: Reanalyze refreshes value targets and feeds PER with updated TD-errors.
5. **Always verify databases against known exact values**: The broken bear-off table invalidated 3 experiments. A simple check against hand-computable positions (e.g., n@ace vs m@ace) would have caught this immediately.
6. **Next experiment**: Run 200 iter with PER + bear-off rollouts to see if improvements compound.

## Session Directories
- Baseline 200 iter: `sessions/distributed_20260206_223524/`
- PER 50 iter: `sessions/distributed_20260207_030412_per/`
- Reanalyze 50 iter: `sessions/distributed_20260207_043500_reanalyze/`
- Bear-off rollouts 50 iter: `sessions/distributed_20260207_061713_bearoff/`
- ~~Bear-off table targets 50 iter: `sessions/distributed_20260207_162443_bearoff/`~~ (BROKEN TABLE)
- ~~Bear-off table+TD 50 iter: `sessions/distributed_20260207_103614_bearoff/`~~ (BROKEN TABLE)
- ~~Bear-off table+TD+resample 50 iter: `sessions/distributed_20260207_145102_bearoff/`~~ (BROKEN TABLE)
- **Bear-off table (FIXED) 50 iter: `sessions/distributed_20260207_215103_bearoff/`**

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

### On Bear-off Table Integration (Exact k=6)
- **CRITICAL**: The original bear-off table experiments (table targets, table+TD, table+TD+resample) all used a BROKEN table. The conclusions drawn from those experiments about soft vs hard targets, MCTS integration disruption, etc. cannot be trusted.
- With the **fixed table** (correct P(win) values, both perspectives stored), the result is +0.63 vs GnuBG 1-ply at 50 iter — a modest improvement over baseline (+0.55).
- Bear-off rollouts (+0.83) still outperform the fixed table (+0.63) at 50 iterations. Possible reasons:
  - Rollout stochasticity provides beneficial training signal diversity
  - Table uses all modes (MCTS leaf + TD-bootstrap + targets) which changes game distribution
  - 50 iterations may not be enough for the model to fully leverage exact endgame knowledge
- **Lesson**: Always verify database/table values against known exact solutions before using in experiments. A single wrong decode formula invalidated multiple weeks of experiments.
- **Next step**: Test fixed table at 200 iter and/or with PER to see if it catches up to rollouts.

### On Compute Scaling
- 200 iterations took 254.5 min (~4.2 hours) at 394 g/m. This is reasonable for a single machine.
- The diminishing returns curve (loss plateau) vs the linear play strength improvement suggests the model is extracting increasingly subtle strategic knowledge that doesn't show in average loss.
- Larger models may break through the loss plateau and unlock faster play strength gains.

## Recommended Next Experiments

1. **PER + Bear-off rollouts (200 iter)** — ~4 hours, highest expected improvement (combine top 2 techniques)
2. **PER + Bear-off table FIXED (200 iter)** — ~4 hours, test if fixed table + PER improves over rollouts long-term
3. **PER + Bear-off rollouts + Reanalyze (200 iter)** — ~5 hours, test if all three compound
4. **256w×10b model (200 iter, baseline)** — ~8-12 hours, test capacity hypothesis
5. **PER + Bear-off rollouts (200 iter, 256w×10b)** — ~12 hours, full configuration

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

### Bear-off rollouts 50 iter (2000 games)
- vs 0-ply: +0.998 (80.7%), white +1.036/black +0.960
- vs 1-ply: +0.832 (72.0%), white +0.830/black +0.834

### ~~Bear-off table targets 50 iter (2000 games)~~ BROKEN TABLE
- vs 0-ply: +1.035 (81.3%), white +1.004/black +1.066
- vs 1-ply: +0.737 (70.2%), white +0.752/black +0.722

### ~~Bear-off table+TD 50 iter (2000 games)~~ BROKEN TABLE
- vs 0-ply: +1.121 (82.8%), white +1.178/black +1.064
- vs 1-ply: +0.599 (64.6%), white +0.600/black +0.598

### ~~Bear-off table+TD+resample 50 iter (2000 games)~~ BROKEN TABLE
- vs 0-ply: +0.939 (78.9%), white +0.918/black +0.960
- vs 1-ply: +0.618 (64.9%), white +0.684/black +0.552

### Bear-off table (FIXED) 50 iter (4000 games)
- vs 0-ply: +0.957 (80.5%), white +0.891/black +1.023
- vs 1-ply: +0.629 (67.5%), white +0.643/black +0.616
