# Corrected GnuBG Evaluation Results (2026-02-14)

## Background

A critical board encoding bug in `_to_gnubg_board` was found and fixed on 2026-02-14 (commit e164a85). Three bugs caused gnubg to evaluate completely wrong positions, making it play terribly:

1. **Off-by-one**: Points at index N instead of N-1 (bar at idx 0 instead of idx 24)
2. **Bar position**: Bar at Python index 0, gnubg expects index 24
3. **Opponent perspective**: Both arrays used on-roll's perspective; gnubg expects each player's OWN perspective

ALL previous GnuBG evaluation numbers were invalid and showed inflated results (82-92% win rates when the true rates are 2-10%).

## Verification

The fix was verified with:
- Position ID round-trip: initial position matches gnubg's known `4HPwATDgc/ABMA`
- 2,476 position checks across 25 games (random + gnubg-play), all passing:
  - Checker count validation (<=15 per side)
  - gnubg.probabilities() sanity checks
  - GnubgPlayerFast vs GnubgInterface encoding agreement
  - Bar checker positions included (907 total with bar)

## Re-evaluation Results

**Settings**: vs GnuBG 0-ply, 500 games per side (1000 total), 100 MCTS iterations, 8 workers

### All Models Ranked

| Rank | Experiment | Architecture | Iters | Equity | Win% | Old Rank |
|------|-----------|-------------|-------|--------|------|----------|
| 1 | PER+Reanalyze | 256w×5b | 200 | **-1.361** | **9.6%** | N/A (new) |
| 2 | PER | 128w×3b | 200 | -1.558 | 7.8% | #1 |
| 3 | PER+Reanalyze | 256w×5b | 50 | -1.573 | 7.0% | N/A (new) |
| 4 | Baseline | 128w×3b | 200 | -1.746 | 4.6% | #2 |
| 5 | PER | 128w×3b | 50 | -1.759 | 4.8% | #3 |
| 6 | Baseline | 128w×3b | 50 | -1.841 | 3.5% | #6 |
| 7 | Bearoff rollouts | 128w×3b | 50 | -1.993 | 2.1% | #4 (was "best 50-iter") |
| 8 | Baseline v2 | 128w×3b | 50 | -2.036 | 1.6% | N/A |
| 9 | Reanalyze | 128w×3b | 50 | -2.054 | 2.3% | #5 |

### Session Paths

| Experiment | Session Dir | Checkpoint |
|-----------|------------|------------|
| PER+Reanalyze 256w 200i | distributed_20260213_031243_per_reanalyze | latest.data |
| PER 128w 200i | 20260209_per_200iter | latest.data |
| PER+Reanalyze 256w 50i | distributed_20260213_010615_per_reanalyze | latest.data |
| Baseline 128w 200i | 20260206_baseline_200iter | latest.data |
| PER 128w 50i | 20260207_per_50iter | latest.data |
| Baseline 128w 50i | 20260206_baseline_50iter | latest.data |
| Bearoff rollouts 128w 50i | 20260207_bearoff_rollouts_50iter | latest.data |
| Baseline v2 128w 50i | 20260206_baseline_v2_50iter | latest.data |
| Reanalyze 128w 50i | 20260207_reanalyze_50iter | latest.data |

## Key Changes from Previous Rankings

### Rankings that changed dramatically
- **Bearoff rollouts**: Was #4 "best 50-iter technique" → Now #7 (worse than baseline). The apparent gain was entirely due to the encoding bug.
- **Reanalyze**: Was #5 "moderate gain" → Now #9 (worst of all). Reanalyze actively hurts performance.
- **Baseline**: Was #6 → Now #6 (held position, but now we know it beats bearoff and reanalyze)

### Rankings that held
- **PER 200-iter**: Was #1 → Still #2 (best 128w). PER is the only reliable improvement.
- **PER 50-iter**: Was #3 → Now #5 (still best 50-iter 128w technique)

### New findings
- **256w×5b is the best architecture** — larger model at 200 iter beats best 128w by +0.197 equity
- **All models are genuinely weak** — best is 9.6% wins vs gnubg 0-ply
- **Models get heavily gammoned** — equity -1.4 to -2.0 (average loss includes many gammons)

## Technique Analysis

### PER (Prioritized Experience Replay) — WORKS
- At 50 iter: -1.759 vs baseline -1.841 = **+0.082 equity** (+37% relative improvement)
- At 200 iter: -1.558 vs baseline -1.746 = **+0.188 equity** (+70% relative)
- PER benefit compounds with training length

### Bearoff Rollouts — DOES NOT WORK
- At 50 iter: -1.993 vs baseline -1.841 = **-0.152 equity** (REGRESSION)
- Previously appeared to help (+0.28 in buggy eval), but that was entirely artifact

### Reanalyze — DOES NOT WORK
- At 50 iter: -2.054 vs baseline -1.841 = **-0.213 equity** (WORST REGRESSION)
- Previously appeared to help (+0.14 in buggy eval), but that was entirely artifact

### Larger Model (256w×5b) — WORKS
- At 50 iter: -1.573 (256w PER+Rean) vs -1.759 (128w PER) = **+0.186 equity**
- At 200 iter: -1.361 (256w PER+Rean) vs -1.558 (128w PER) = **+0.197 equity**
- Note: 256w sessions used PER+Reanalyze together; the improvement might be purely from model size

### Longer Training — WORKS
- Baseline: -1.841 (50i) → -1.746 (200i) = **+0.095 equity**
- PER: -1.759 (50i) → -1.558 (200i) = **+0.201 equity**
- Benefits compound with PER

## Performance Notes

- **128w×3b eval**: ~5 games/sec with 8 workers (bottleneck: MCTS inference)
- **256w×5b eval**: ~0.7 games/sec with 8 workers (7x slower, larger NN)
- **Full 128w batch (7 checkpoints)**: ~22 minutes
- **Full 256w batch (2 checkpoints)**: ~48 minutes
- **GnuBG 0-ply**: ~48k evals/sec, negligible time vs MCTS

## Implications for Training Strategy

1. **Drop bearoff rollouts and reanalyze** — they don't help, only PER works
2. **Scale up**: 256w×5b + PER + 200+ iterations is the best recipe
3. **The model fundamentally can't play well yet** — 9.6% wins means it's making very bad moves
4. **Need to investigate WHY** — possible issues: value head accuracy, policy head quality, MCTS depth, training signal quality
5. **Consider training without MCTS** (TD-learning with self-play) as a diagnostic
