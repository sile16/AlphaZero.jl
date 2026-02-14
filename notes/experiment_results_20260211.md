# Experiment Results — Stochastic Wrapper (2026-02-11)

## Baseline
- **Locked-in config**: PER + Reanalyze, stochastic wrapper, AdamW lr=0.001
- All: 128w × 3b FCResNetMultiHead, MINIMAL obs, 400 MCTS sims, 14 workers

## Results (50 iter)

| Experiment | Loss | vs GnuBG 0-ply | vs GnuBG 1-ply | Time |
|-----------|------|----------------|----------------|------|
| **PER + Reanalyze (v0.6.0)** | **4.44** | **+1.403 (92%)** | **+1.146 (82%)** | **72 min** |
| PER + Reanalyze (v0.4.1) | 4.35 | +1.341 (90%) | +0.974 (76%) | 89 min |
| PER baseline | ~4.12 | +1.019 (82%) | +0.670 (67%) | ~80 min |
| PER + Bearoff Soft (trunc) | 4.43 | +0.982 (81%) | +0.587 (64%) | 67 min |
| PER + Bearoff Full Play | 4.30 | +0.942 (81%) | +0.587 (67%) | 73 min |

### Old step! interface (for reference)

| Experiment | Iters | Loss | vs GnuBG 0-ply | vs GnuBG 1-ply |
|-----------|-------|------|----------------|----------------|
| **PER** | **200** | **3.90** | **+1.38 (92%)** | **+1.21 (82%)** |
| Baseline | 200 | 3.89 | +1.31 (89%) | +1.05 (78%) |
| PER | 50 | 4.12 | +1.06 (83%) | +0.87 (74%) |
| Bear-off rollouts | 50 | 3.98 | +1.00 (81%) | +0.83 (72%) |
| Baseline | 50 | 3.97 | +0.94 (79%) | +0.55 (66%) |

## Key Findings

1. **PER + Reanalyze is best 50-iter config**: +0.974 vs GnuBG 1-ply, +45% equity over PER baseline (+0.670). Stochastic wrapper makes reanalyze synergistic (old interface: reanalyze gave +0.69, worse than PER alone).

2. **Bear-off table targets hurt training**: Both truncation (+0.587) and full-play (+0.587) give identical 23% regression from baseline. Table values override MCTS signal with mismatched endgame-calibrated values.

3. **MCTS bear-off evaluator is the right way to use the table**: Already active in all experiments. Provides perfect endgame play during search without corrupting training targets.

## Completed: PER + Reanalyze 200-iter
- Session: distributed_20260211_212006_per_reanalyze
- Result: **+1.466 (93%)** vs GnuBG 0-ply, **+1.338 (84%)** vs GnuBG 1-ply
- **New overall best.** Beats old PER 200-iter (+1.21) by +10.6% equity.

## Completed: BackgammonNet v0.6.0 Reproduction (50-iter PER + Reanalyze)
- Session: distributed_20260212_220302_per_reanalyze
- BackgammonNet upgraded v0.4.1 → v0.6.0 (obs 330→344 features, +14 cube/match/context)
- Result: **+1.403 (92%)** vs GnuBG 0-ply, **+1.146 (82%)** vs GnuBG 1-ply
- **New best 50-iter result!** +17.7% equity improvement over v0.4.1 baseline (+0.974)
- 72 min (faster than v0.4.1's 89 min due to throughput improvements)
- 14 new constant features (cube disabled in money play) appear to help as implicit regularization

---

## TODO: Pre-Big-Run Experiments (50 iter each)

All experiments start from locked-in baseline: PER + Reanalyze, stochastic wrapper.

### 1. Temperature Schedule — DONE (all worse)
- **Result**: All temperature schemes hurt performance. τ=1 everywhere is optimal.
  - Step30 (τ→0 at move 30): +0.747 vs 1-ply (**-23%**)
  - Soft20 (τ→0.3 at move 20): +0.852 vs 1-ply (**-13%**)
  - IterDecay (τ 1.0→0.3 linear): +0.830 vs 1-ply (**-15%**)
- **Why**: Backgammon's high branching (~680 actions) + dice randomness provide natural exploration. Reducing temperature loses training signal diversity without benefit.

### 2. Larger Model
- **Current**: 128w × 3b (283K params, loss plateaus at ~3.95)
- **Try**: 256w × 5b or 256w × 10b
- **Rationale**: Capacity bottleneck. If larger model shows lower loss AND better GnuBG score, use for big run.
- **Code**: Need CLI args for width/blocks in train_distributed.jl

### 3. Dirichlet Noise
- **Current**: α=0.3, ε=0.25
- **Try**: α=0.15 or α=0.10
- **Rationale**: Backgammon has ~680 actions (large space). High α spreads noise too uniformly. Go uses α=0.03.
- **Code**: Pure config change

### 4. CPUCT
- **Current**: 2.0
- **Try**: 1.0, 1.5, 3.0
- **Rationale**: Controls exploration/exploitation in MCTS tree. Never tuned.
- **Code**: Pure config change

### 5. Reanalyze Parameters
- **Current**: 25% of buffer per iter, EMA α=0.5
- **Try**: 50% reanalyze rate, or EMA α=0.3 (trust new values more)
- **Rationale**: Reanalyze is our biggest win. Tuning it could compound gains.
- **Code**: Config change (may need CLI args)

### 6. MCTS Simulations
- **Current**: 400 sims, batch_size=50
- **Try**: 800 sims
- **Rationale**: Deeper search = better policy targets. Trade throughput for quality.
- **Code**: Pure config change, but ~2x slower per game

### 7. Concurrent Reanalyze
- **Current**: Reanalyze runs sequentially on GPU after self-play completes (25% of buffer, random subset)
- **Try**: Run reanalyze concurrently with self-play using CPU network clone, target oldest experiences first
- **Rationale**: GPU is idle during self-play (CPU BLAS workers). Concurrent reanalyze = free compute. Oldest experiences have stalest values and benefit most from refresh. Could increase reanalyze fraction to 50%+ at zero throughput cost.
- **Design**: Spawn reanalyze thread alongside self-play. Use `cpu_network` (previous iter weights, read-only). Iterate buffer front-to-back (oldest first). Lock-free: reanalyze writes to existing buffer entries (value blend), self-play only appends.
- **Code**: Moderate — move `reanalyze_buffer!` to use CPU network, spawn as `Threads.@spawn`, add buffer locking if needed

### 8. BackgammonNet v0.6.0 Integration — DONE (+17.7% improvement!)
- **Upgrade**: v0.4.1 → v0.6.0 (obs 330→344 features, +14 cube/match/context channels)
- **Changes**: `clone_into!` uses `copy_state!` API, `vectorize_state_into!` uses public `observe_minimal_flat!`, `current_state` updated for new struct (no `history`, add `tavla`)
- **Result**: +1.146 vs GnuBG 1-ply (82% wins). New best 50-iter, exceeding v0.4.1's +0.974 by +17.7%.
- **Why**: 14 additional constant features (cube/match channels all zero in money play) may act as implicit regularization or provide a richer gradient landscape for the first layer.

### 9. Concurrent Reanalyze — DONE (worse)
- **Result**: Both approaches regressed from baseline:
  - Oldest-first targeting: +0.464 vs 1-ply (**-52%**). Always reanalyzes same entries about to be evicted.
  - Random targeting (2048 batch): +0.692 vs 1-ply (**-29%**). GPU reanalyze time exceeds self-play CPU window.
- **Conclusion**: Sequential reanalyze after buffer update with latest weights remains optimal.

### 10. Network Architecture + Observation Type Sweep — DONE

Tested 256w×5b (1.13M params, 4x larger) and min_plus_flat (350 features, +6 pre-computed) vs baseline.

| Model | Obs | Params | vs GnuBG 0-ply | vs GnuBG 1-ply | Time |
|-------|-----|--------|----------------|----------------|------|
| **128w×3b** | **minimal_flat (344)** | **283K** | **+1.403 (92%)** | **+1.146 (82%)** | **72 min** |
| 256w×5b | min_plus_flat (350) | 1.13M | +1.325 (90%) | +1.158 (82%) | 163 min |
| 256w×5b | minimal_flat (344) | 1.13M | +1.328 (90%) | +0.980 (75%) | 126 min |
| 128w×3b | min_plus_flat (350) | 284K | +1.054 (83%) | +0.667 (68%) | 114 min |

**Key findings**:
- **128w×3b minimal_flat remains the best at 50 iterations**. The smaller model learns more efficiently with less data.
- **256w×5b underfits at 50 iterations**: 4x more parameters need proportionally more training. With minimal_flat, 256w×5b (+0.980) is -14% worse than 128w×3b (+1.146).
- **min_plus_flat helps large models but hurts small ones**: For 256w×5b, min_plus raises performance from +0.980 to +1.158 (+18%). But for 128w×3b, min_plus *drops* performance from +1.146 to +0.667 (-42%).
- **Hypothesis**: The 6 extra pre-computed features (pip counts, contact flags, bearoff indicators) provide useful shortcuts for the larger model (saving capacity for other patterns), but the small model lacks capacity to integrate them and they dilute the core 344 features.
- **256w×5b min_plus_flat matches 128w×3b minimal_flat** (+1.158 vs +1.146) at 50 iter, but takes 2.3x longer. The larger model needs a 200-iter run to show its potential.

Sessions:
- `distributed_20260212_220302_per_reanalyze` — 128w×3b minimal_flat (best 50-iter)
- `distributed_20260213_010615_per_reanalyze` — 256w×5b minimal_flat
- `distributed_20260213_031243_per_reanalyze` — 256w×5b min_plus_flat
- `distributed_20260213_055608_per_reanalyze` — 128w×3b min_plus_flat

### 11. 200-Iteration Runs — DONE

Both 50-iter sessions resumed to 200 iterations with 2-ply evaluation added.

| Model | Obs | vs GnuBG 1-ply | vs GnuBG 2-ply | Train Time |
|-------|-----|---------------|---------------|------------|
| **256w×5b** | **min_plus_flat (350)** | **+1.484 (88.4%)** | **+1.526 (94.0%)** | **359 min** |
| 128w×3b | minimal_flat (344) | +1.353 (84.8%) | +1.409 (92.4%) | 211 min |
| *Old best (PER+RA v0.4.1)* | *minimal (330)* | *+1.338 (84%)* | *N/A* | *—* |

**Key findings**:
- **256w×5b min_plus_flat is the new overall best**: +1.484 vs 1-ply (88.4%), +9.7% equity over 128w×3b. Larger model excels with more training.
- **128w×3b 200-iter also improves**: +1.353 vs 1-ply (84.8%), +1.1% over old best (+1.338). v0.6.0 obs helps marginally.
- **2-ply anomaly**: Both models score HIGHER vs 2-ply than 1-ply. Verified ply is working (different moves/timing). Likely SHORT_GAME artifact — 2-ply's strategy in near-bearoff positions is more exploitable by MCTS-100 tree search.
- **Board/reward verification**: 20 full games verified with strict @assert — checker counts, board encoding, gnubg evaluation sanity, and reward calculation all match exactly.

Sessions:
- `distributed_20260212_220302_per_reanalyze` — 128w×3b minimal_flat (resumed 50→200)
- `distributed_20260213_031243_per_reanalyze` — 256w×5b min_plus_flat (resumed 50→200)

### Priority Order
1. ~~Temperature schedule~~ — **DONE, all worse. τ=1 is optimal.**
2. ~~Concurrent reanalyze~~ — **DONE, all worse. Sequential is optimal.**
3. ~~BackgammonNet v0.6.0~~ — **DONE, +17.7% improvement! New 50-iter best.**
4. ~~Larger model + obs type~~ — **DONE, 128w×3b minimal_flat still best at 50 iter. 256w×5b needs 200+ iter.**
5. ~~200-iter runs~~ — **DONE, 256w×5b min_plus_flat is new overall best (+1.484 vs 1-ply).**
6. Dirichlet noise (quick param change)
7. CPUCT (quick param change)
8. Reanalyze params (tune best feature)
9. MCTS sims (slowest experiment)
