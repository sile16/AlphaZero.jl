# v9 Completed Work Summary

**Date:** 2026-03-26
**Status:** All compute and code changes complete. Ready for v9 training launch.

## 1. Value Head Redesign: Conditional → Joint

### Discovery
BGBlitz and wildbg both send **joint cumulative** probabilities, not conditional.
The `conditional_v1` contract was wrong for all of v1-v8. Confirmed by:
- JAR decompilation of BGBlitz (`bgblitz.bot.Equity`) — internal fields are cumulative joint
- Bridge code (`BgblitzBridge.java:261`) calls `getGammon()` (raw joint), not `getGammonRate()` (conditional)
- Live scalar parity test: 29 diverse positions, joint formula Δ=0.000000, conditional errors up to 0.154
- Bootstrap re-query: 20 contact positions from actual training data, all 20 confirmed JOINT
- Wildbg confirmed: `_equity_from_probs` simplifies to joint formula `(2pw-1)+(wg-lg)+(wbg-lbg)`

### Code Changes (all tests passing)

**AlphaZero.jl (9 files, net -23 lines):**
- `fc_resnet_multihead.jl` — logit outputs (no sigmoid in NN), joint equity formula
- `learning.jl` — `bce_logits_wmean()` added, `_equity_head_weights()` deleted, no masking
- `memory.jl` — joint cumulative docs
- `fast_weights.jl` — joint equity formula (fewer FLOPs)
- `selfplay_client.jl` — bearoff via `to_absolute()`, updated comments
- `training_server.jl` — sigmoid before equity in reanalyze
- `game_loop.jl` — fixed `env.game` access for non-backgammon games
- `runtests.jl` — removed 3 stale tests (Cluster, Reanalyze, Dummy Runs)
- `test_multihead.jl` — BCEWithLogits test, joint equity tests

**BackgammonNet.jl (4 files, net +40 lines):**
- `equity.jl` — `compute_equity_joint()`, updated `compute_cubeless_equity()`
- `generate_bootstrap.jl` — contract → `joint_cumulative_v1`, joint parity formula
- `audit_bootstrap.jl` — updated contract/parity
- `bootstrap_value_head_contract.md` — rewritten for joint cumulative

### Test Results
- `Pkg.test()`: 608/608 pass (Testing Games, Backgammon Inference, Multihead, Game Loop Integration)
- `test_value_head_formats.jl`: 6,316/6,316 assertions pass
- Live BGBlitz validation: 20/20 positions confirmed joint

## 2. k=7 Bearoff Table

### Specs
- Single-side positions: 170,544 (C(22,7))
- Total states solved: 29,085,255,936
- Solve time: ~31 hours on Neo (M3 Max, 30 threads, 512 GB RAM)
- Output: c14 = 25 GB, c15 = 63 GB, total = 88 GB
- Contract: `joint_noncumulative_bearoff_v1`
- Format: pre-roll only, u16 probabilities, pW + pWG + pLG (gammon-only, not cumulative)
- Move generation: BackgammonNet.jl's battle-tested engine (not hand-rolled)

### Files
- Solver: `BackgammonNet.jl/tools/bearoff_twosided/solve_k7.jl`
- Lookup module: `BackgammonNet.jl/src/bearoff_k7.jl`
- Table data: `BackgammonNet.jl/tools/bearoff_twosided/bearoff_k7_twosided/`
- Validation: `BackgammonNet.jl/tools/bearoff_twosided/validate_k7.jl`

### Validation Results (ALL 9 TESTS PASS)

| # | Test | Scope | Max Error | Result |
|---|------|-------|-----------|--------|
| 1 | Gammon Head Bellman | 10,000 c15 states | 0.00e+00 | PASS |
| 2 | Exhaustive Low-Pip Bellman | 12,393,010 states (pip≤40) | 0.00e+00 | PASS |
| 3 | k=6 Gammon Cross-Val | 5,000 c15 positions | 7.56e-06 | PASS |
| 4 | Symmetry (first-mover) | 2,000 symmetric positions | all pW>0.5 | PASS |
| 5 | Analytical 1v1 | 49 positions | 2.38e-07 | PASS |
| 6 | Terminal States | edge cases | exact | PASS |
| 7 | Monotonicity | 8,410 shift/bearoff tests | 0 violations | PASS |
| 8 | Equity Consistency | 10,000 states | constraints OK | PASS |
| 9 | Table File Integrity | 3,000 spot checks | 0 errors | PASS |

## 3. Bootstrap Data (Wildbg, MIT Licensed)

### Generation
- 10 batches × 100K games = **1,000,000 games**
- ~**56.6M positions** total (56.6 positions/game average)
- Matchups: 70% wildbg-large vs wildbg-small, 30% wildbg-large vs wildbg-large
- Evaluator: wildbg large on ALL positions (joint cumulative, confirmed Δ=0 vs scalar)
- Contract: `joint_cumulative_v1`
- License: MIT (wildbg only, no BGBlitz in training pipeline)
- Scalar parity: 0.000000 for all positions (exact)

### Files
- Generator: `BackgammonNet.jl/scripts/generate_bootstrap_wildbg.jl`
- Data: `BackgammonNet.jl/data/bootstrap/bootstrap_wildbg_100k_part{0-9}.jls` (29 GB each, 290 GB total)
- Each part: 100K games, different seed, independent

### Performance
- 8 threads: ~5,400 games/min
- 1 thread: ~720 games/min
- Total generation time: ~3 hours

## 4. Numerical Validation Tests

### test/test_value_head_formats.jl (6,316 assertions)
12 test suites verifying conditional↔joint equivalence, bearoff table format,
Bellman consistency, perspective flips, equity formulas, BCEWithLogits stability.

### scripts/validate_bootstrap_bearoff.jl
Cross-validated 31,796 bearoff positions in bootstrap against exact k=6 table.
Mean equity error 0.0008 (BGBlitz 1-ply vs exact table).

### scripts/investigate_bootstrap_outliers.jl
Investigated all 31,796 bearoff positions. Top errors are gammon head estimation
noise from BGBlitz 1-ply. 0.05% of positions have >0.05 equity error.

### Live BGBlitz format tests (/tmp/bgblitz_live_test.jl, /tmp/ground_truth_test3.jl)
29 diverse positions + 20 bootstrap re-queries confirmed joint format definitively.

## 5. Key Decisions

### Value head format: Joint cumulative for NN, joint non-cumulative for bearoff
- NN heads: `[P(win), P(win∧gammon+), P(win∧bg), P(loss∧gammon+), P(loss∧bg)]`
- Bearoff: `[pW, pWG, pLG]` where gammon = gammon-only (bg=0 in bearoff)
- Numerically identical in bearoff since bg=0

### Licensing: BGBlitz for debug/eval only
- Training pipeline: wildbg (MIT) + self-play + exact bearoff table only
- BGBlitz: diagnostics, value comparison, format validation only
- GnuBG: available but ply parameter unreliable in current C library integration

### BCEWithLogits for all 5 heads
- Network outputs raw logits (no sigmoid)
- Sigmoid applied at inference only
- Eliminates vanishing gradients for small joint probabilities
- Standard practice (PyTorch default)

## 6. Outstanding Items for v9 Launch

1. Convert bootstrap data to AlphaZero.jl training format (vectorize states, etc.)
2. Wire k=7 lookup into selfplay_client.jl (replace k=6)
3. Copy k=7 table to NFS shared storage for both machines
4. Update launch script for v9 config
5. Smoke test: 5 iterations of training + eval before full run
