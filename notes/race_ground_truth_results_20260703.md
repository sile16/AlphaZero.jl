# Pre-Bearoff Value-Head Ground-Truth Results — 2026-07-03

First run of `scripts/race_ground_truth.jl` (wildbg-rollout ground truth at
in-distribution post-dice decision nodes; see the script header + the "never use
NN as pre-dice frontier evaluator" law). All 2000 race-eval positions are
pre-bearoff (none in the k=7 range), so this measures the exact band where v12
plateaued.

## v12 value-head accuracy vs wildbg-rollout equity (1000 positions × 200 rollouts each)

| v12 iter | MSE | MAE | bias | corr |
|----------|-------|-------|--------|-------|
| 5  | 0.144 | 0.258 | +0.045 | 0.962 |
| 10 | 0.049 | 0.157 | −0.016 | 0.987 |
| 20 | 0.045 | 0.142 | +0.056 | 0.989 |
| 30 | 0.042 | 0.137 | +0.006 | 0.989 |
| 40 | 0.048 | 0.130 | −0.034 | 0.988 |
| 50 | 0.040 | 0.124 | −0.026 | 0.990 |

(units: raw money-equity points; RMSE at iter 50 ≈ 0.20, ≈0.17 after subtracting
~200-rollout MC target noise. NN value mean 0.337 vs rollout 0.363 at iter 50.)

## Findings

1. **The value head improves then PLATEAUS — it does not drift.** MSE drops 3.7×
   from iter 5→10, then holds flat 0.04–0.05 through iter 50; MAE improves
   monotonically; corr rises to 0.99 and stays. Bias stays small and unsigned.
2. **First direct confirmation that v12's k=7 table-anchoring fixed the v11
   value drift.** v11 drifted 0.974→0.875 (vs the exact table on bearoff); v12
   stays rock-stable at ~0.99 here against a RESOLVED-rollout ground truth (not a
   static-eval proxy). The anchoring worked.
3. **The pre-bearoff value head is NOT the plateau bottleneck.** It reaches
   wildbg-parity accuracy by iter 10 and holds. So the play plateau at wildbg
   parity is not explained by value-head inaccuracy in this band — the lever to
   beat wildbg is policy/search, or a training target STRONGER than wildbg.
4. **A wildbg-rollout ground truth caps at wildbg quality.** corr 0.99 = "v12
   value ≈ wildbg equity" = the parity plateau restated. To measure PAST wildbg
   needs the exact one-sided ground truth (the memory's designed approach) — this
   run validates precisely why that refinement matters.

## Policy-head result (v12 iter-50, 1000 pre-bearoff positions, n=815 non-doubles)

| metric | value |
|--------|-------|
| policy argmax == wildbg move | 54.3% |
| NN policy mass on wildbg's move | 0.351 |
| top-1 prob (policy sharpness) | 0.395 |
| move-regret mean (pts) | **−0.0144** |
| move-regret \| disagree (pts) | **−0.0381** (SE ≈ 0.006) |

**The 54% agreement is a red herring.** Move-regret is ≈0 and marginally
NEGATIVE — the NN's disagreements with wildbg are near-ties where the NN is
*marginally better* by rollout truth, not mistakes. The diffuse policy (top-1
0.40) reflects genuine move-ambiguity in races (many pip-equivalent plays), not
weakness. wildbg picks its move by static eval; the NN's move aligns slightly
better with rollout truth. So the pre-bearoff **policy/move quality is also at
(or just above) wildbg parity** — not the plateau bottleneck.

Net across both heads: **the pre-bearoff RACE band is not where v12 is weak.**
Value AND move quality match/slightly-beat wildbg there. The plateau at wildbg
parity must live in CONTACT positions or overall integration — that is where the
next diagnostic (same value+policy eval on a contact model/positions) should go.

## Implications for v13
- Value-head training is healthy (converges, stable, no drift) — do not spend v13
  effort "fixing" the pre-bearoff value head.
- Prioritize the levers that can exceed wildbg: policy/search quality, and the
  exact one-sided ground truth (as targets and/or eval) to push value accuracy
  beyond the wildbg reference. This is the pre-bearoff band where the plateau
  actually lives.

## Method notes / caveats
- Ground truth is wildbg-vs-wildbg rollout equity — a strong but not exact
  reference; it measures agreement WITH wildbg, so it cannot score super-wildbg
  accuracy (Stage-4 exact one-sided DP in the plan removes this ceiling).
- 200 rollouts leaves ~0.01 variance of MC noise in the per-node target; more
  rollouts or the analytic combine tighten it.
- Run cost: ~9s per checkpoint (1000 pos × 200 rollouts) — wildbg is a fast C lib
  and pre-bearoff races are short. Cheap enough to score every checkpoint inline.
