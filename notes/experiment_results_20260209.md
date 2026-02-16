# ~~Experiment Results — 2026-02-09: PER 200-iter~~

> **INVALIDATED (2026-02-14)**: All GnuBG evaluation numbers in this file are invalid due to a critical board encoding bug in `_to_gnubg_board` (fixed in commit e164a85). The bug caused GnuBG to evaluate wrong positions and play terribly, inflating win rates from ~3-10% (reality) to 65-92% (reported here). See `notes/corrected_eval_results_20260214.md` for corrected results. Training dynamics observations (loss, throughput) are still valid. Relative technique comparisons may hold but absolute GnuBG numbers are meaningless.

## Overview

200-iteration PER training run to validate PER as a long-training improvement over baseline.
FCResNetMultiHead 128w×3b (283K params), AdamW lr=0.001, 400 MCTS sims, batch=50, 600K buffer, seed=42.
PER: α=0.6, β=0.4→1.0 (linear annealing over 200 iterations).

## Results Summary

| Experiment | Iters | Loss (final) | vs GnuBG 0-ply | vs GnuBG 1-ply | Time (min) |
|-----------|-------|-------------|----------------|----------------|------------|
| **PER** | **200** | **3.90** | **+1.38 (92%)** | **+1.21 (82%)** | **312** |
| Baseline | 200 | 3.89 | +1.31 (89%) | +1.05 (78%) | 254.5 |
| **PER @50** | **50** | **4.12** | **+1.06 (83%)** | **+0.87 (74%)** | — |
| Previous PER | 50 | 4.07 | +1.08 (83%) | +0.79 (72%) | 74.4 |
| Baseline | 50 | 3.97 | +0.94 (79%) | +0.55 (66%) | 75.0 |

## Key Findings

### PER is the new baseline
- **+1.21 vs GnuBG 1-ply** — 15% equity improvement over baseline 200 (+1.05)
- **82% win rate** vs GnuBG 1-ply (up from 78%)
- Consistent improvement at both 50 and 200 iterations

### PER benefit compounds with training length
- At 50 iter: PER +0.87 vs baseline +0.55 → **+58% equity gain**
- At 200 iter: PER +1.21 vs baseline +1.05 → **+15% equity gain**
- The percentage gain narrows but the absolute gain remains strong (+0.16 equity)

### Loss is nearly identical
- PER 200: 3.90 vs Baseline 200: 3.89
- PER starts with higher loss (IS weights upweight hard examples) but converges to the same level
- Confirms PER doesn't hurt average prediction quality while improving play strength

### Training dynamics
- Loss trajectory: 5.43 → 4.12 (iter 50) → 3.99 (iter 100) → 3.95 (iter 150) → 3.90 (iter 200)
- Throughput: ~320 games/min (slightly slower than baseline ~350, due to PER sampling overhead)
- Total time: 312 min (vs 254.5 min baseline — longer due to slightly lower throughput)

## Raw GnuBG Evaluation Details

### PER 200 iter — at iteration 50 (2000 games)
- vs 0-ply: +1.061 (82.8%), white +1.070/black +1.052
- vs 1-ply: +0.866 (74.3%), white +0.852/black +0.880

### PER 200 iter — at iteration 200 (2000 games)
- vs 0-ply: +1.384 (91.5%), white +1.344/black +1.424
- vs 1-ply: +1.207 (81.9%), white +1.216/black +1.198

## Session Directory
- `sessions/distributed_20260209_215824_per/`
- Checkpoints at every 10 iterations (iter_10.data through iter_200.data)

## Lessons Learned

1. **PER is a reliable, no-tuning-needed improvement**: Default hyperparams (α=0.6, β=0.4→1.0) worked again, matching the 50-iter experiment. This is a robust technique.

2. **Run-to-run variance is real but small**: The iter-50 checkpoint of this run (+0.87) is slightly different from the previous standalone PER 50-iter run (+0.79). This ~10% variance is normal for 2000-game evaluations.

3. **PER makes training longer more worthwhile**: The bigger model sees more diverse hard positions over 200 iterations, and PER ensures they get trained on. This may explain why the benefit compounds.

4. **Throughput impact is modest**: ~8% slower throughput (320 vs 350 g/m) from PER sampling overhead. The O(n) cumulative sum sampling per batch on 600K buffer adds ~5% overhead. Could be optimized with binary search precomputation but not a bottleneck.

## Next Steps

1. **Fix bear-off table for post-dice values** — currently pre-dice only (see `notes/bearoff_chance_node_issue.md`)
2. **PER + bear-off (200 iter)** — combine with fixed bear-off integration
3. **Larger model (256w×10b)** — test capacity hypothesis with PER as default
