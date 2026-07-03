# MCTS-iters Convergence Sweep vs Exact Table (2026-07-03)

Setup: `eval_table_vs_wildbg.jl --policy=mcts`, 1000 mutual-bearoff positions × 2
sides vs wildbg small, duplicate-dice paired, seed 42. MCTS = BatchedMCTS with the
exact k=7 turn-aware bearoff evaluator at every node (uniform dummy oracle; NN never
consulted in pure bearoff). Pure-table baseline: +0.003 ± 0.0059.

| iters | paired edge vs wildbg | mean Δ from optimal (raw pts) | max Δ |
|-------|----------------------|-------------------------------|-------|
| 30    | −0.075 ± 0.0159      | 0.0885                        | 1.000 |
| 100   | −0.006 ± 0.0078      | 0.0237                        | 0.285 |
| 400   | +0.003 ± 0.0059      | 0.0033                        | 0.111 |
| 800   | +0.002 ± 0.0062      | 0.0008                        | 0.024 |
| 1600  | +0.003 ± 0.0059      | 0.0004                        | 0.013 |

Findings:
- Monotone convergence to the exact table; **400 iters is the knee** — paired edge
  already indistinguishable from perfect play; 1600 is functionally perfect
  (mean move-quality loss 0.0004 pts).
- 30 iters plays measurably WORSE than the raw table (−0.075): after /3
  normalization, exact move-value gaps (~0.01–0.03 equity) are below what 30 PUCT
  sims can resolve — few-sim MCTS ADDS noise on top of an exact evaluator.
- Implications: (a) the 600-iter eval standard is safely past the knee;
  (b) progressive-MCTS-budget schedules should not drop below ~200–400 sims in
  phases where exact/strong values drive the tree; (c) a sign/perspective bug
  would show non-convergence — this doubles as a standing MCTS wiring check.

Raw log: session scratchpad `mcts_sweep_results.log` (regenerate with the harness;
same seed reproduces positions).
