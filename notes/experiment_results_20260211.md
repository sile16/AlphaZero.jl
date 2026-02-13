# Experiment Results — Stochastic Wrapper (2026-02-11)

## Baseline
- **Locked-in config**: PER + Reanalyze, stochastic wrapper, AdamW lr=0.001
- All: 128w × 3b FCResNetMultiHead, MINIMAL obs, 400 MCTS sims, 14 workers

## Results (50 iter)

| Experiment | Loss | vs GnuBG 0-ply | vs GnuBG 1-ply | Time |
|-----------|------|----------------|----------------|------|
| **PER + Reanalyze** | **4.35** | **+1.341 (90%)** | **+0.974 (76%)** | 89 min |
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

## Running: BackgammonNet v0.6.0 Reproduction (50-iter PER + Reanalyze)
- Session: distributed_20260212_*_per_reanalyze (started 2026-02-12)
- BackgammonNet upgraded v0.4.1 → v0.6.0 (obs 330→344 features)
- Target: reproduce +0.974 ± ~0.15 vs GnuBG 1-ply

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

### 8. BackgammonNet v0.6.0 Integration — IN PROGRESS
- **Upgrade**: v0.4.1 → v0.6.0 (obs 330→344 features, +14 cube/match/context channels)
- **Changes**: `clone_into!` uses `copy_state!` API, `vectorize_state_into!` uses public `observe_minimal_flat!`, `current_state` updated for new struct (no `history`, add `tavla`)
- **Impact**: New 344-dim input requires retraining from scratch. Network auto-adjusts (283K → 283.4K params).
- **Status**: Smoke tests pass. 2-iter sanity test pass. 50-iter reproduction running.

### 9. Concurrent Reanalyze — DONE (worse)
- **Result**: Both approaches regressed from baseline:
  - Oldest-first targeting: +0.464 vs 1-ply (**-52%**). Always reanalyzes same entries about to be evicted.
  - Random targeting (2048 batch): +0.692 vs 1-ply (**-29%**). GPU reanalyze time exceeds self-play CPU window.
- **Conclusion**: Sequential reanalyze after buffer update with latest weights remains optimal.

### Priority Order
1. ~~Temperature schedule~~ — **DONE, all worse. τ=1 is optimal.**
2. ~~Concurrent reanalyze~~ — **DONE, all worse. Sequential is optimal.**
3. **BackgammonNet v0.6.0 reproduction** — IN PROGRESS
4. Larger model (addresses known capacity bottleneck)
5. Dirichlet noise (quick param change)
6. CPUCT (quick param change)
7. Reanalyze params (tune best feature)
8. MCTS sims (slowest experiment)
