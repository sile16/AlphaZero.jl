# MCTS Deterministic Identity Staircase (2026-07-03)

Permanent regression tests proving the MCTS engines introduce **zero correctness
error** on positions whose ground truth is known exactly. New file:
`test/test_mcts_identity_staircase.jl` (wired into `test/runtests.jl` after the
bear-off doubles regression testset). Runtime ~17s with the k=7 table present;
skips gracefully (`@warn` + `@test_skip`) if the table is absent.

## Why these are *identity* tests, not play-tests

We deliberately separate two things the existing eval scripts conflate:

- **CORRECTNESS errors** — sign, mover-vs-white perspective, reward-scale (÷3),
  backprop arithmetic, virtual-loss leakage, chance-node probability weighting.
  These *must be identically zero*. They are checkable on the tree's **internal**
  statistics (`Q = W/N` per action) at machine-epsilon tolerance, deterministically.
- **finite-budget SUBOPTIMALITY** — imperfect visit allocation at low sim counts.
  This is *expected*, is a smooth function of the budget, and is measured
  elsewhere as convergence curves (`notes/mcts_convergence_sweep_20260703.md`).

Behavioral tests catch bugs only *statistically* and cannot distinguish "the
arithmetic is perfect" from "two sign errors cancelled over 500 games". Identity
tests on tree internals *can*: one mis-signed backup fails an exact equality by
O(1), far outside any rounding tolerance. (This is the same class of bug as the
2026-07-03 doubles mid-turn mis-scoring, which a statistical run barely caught at
−0.022 ± 0.010; an identity test would have failed it by ~1.0.)

Ground truth = the exact **k=7 two-sided bear-off table** (Rungs 1–2) and a
direct pure-Julia **expectimax recursion** over the same game (Rung 3,
cross-checked against the table where positions are in range).

## What each rung proves

### Rung 1 — evaluator / wrapper identities (209 assertions)
On ~200 random mutual-bear-off states plus hand-checked anchors:

- **1.0 Hand-checked mirror.** A player-role swap with point-number reflection
  (`_mirror`) is verified on an asymmetric position (P0 far ahead ↔ P1 far ahead):
  the racer's equity is seat-invariant (`E == Em`, atol 1e-6) and clearly > 0.5.
- **1a. Internal-consistency sign/perspective negation (EXACT).** For a
  turn-complete (chance/terminal) state, `bearoff_turn_value(g,0) == -bearoff_turn_value(g,1)`
  to the bit (`max|v0+v1| == 0.0`). *Only* asserted on turn-complete states — a
  mid-turn decision node's value is **not** a pure sign flip because each mover
  maximises their own outcome (documented in the test).
- **1b. Physical-mirror perspective symmetry.** White-relative value of a
  position == −(white-relative value of its player-swapped mirror), `< 1e-6`.
- **1c. Normalization.** The MCTS-facing evaluator output equals
  `bearoff_turn_value/3` **exactly** (`== 0.0`) at both chance nodes and decision
  nodes, and `|v| ≤ 1`.
- **1d. Chance-node = probability-weighted average of per-dice best moves.** The
  "sum of stochastic children, averaged" identity: the pre-dice table value equals
  `Σ P(dice)·(exact best post-dice move value)` over all 21 outcomes with weights
  1/18 (non-doubles) and 1/36 (doubles).

### Rung 2 — depth-1 search identity (854 assertions)
BatchedMCTS with uniform oracle (P=1/n, V=0) + exact bear-off evaluator,
cpuct=2.0, no noise, sims = 4·num_actions. For a **turn-completing** root action
(post-move state terminal or a chance node), every visit re-terminates at the
*same* exactly-evaluated leaf, so:

```
Q(a) = W/N  ==  bearoff_turn_value(post_move, root_mover)/3     EXACTLY (< 1e-9)
```

- Asserted over **300** states (batch_size 1). Measured `max|Q−exact| ≈ 5.6e-17`.
- **argmax-Q == exact table argmax** on all depth-1 states (exact-tie tolerance 1e-12).
- **Doubles mid-turn** actions (post-move is a same-player node that IS expanded
  and re-descended) can only *average down* toward exactly-evaluated descendants,
  so we assert the hard bound `Q(a) ≤ exact + eps` (measured violation ≈ 1e-16).
- **Virtual-loss unwinding.** The *same* exact identity is re-asserted at
  batch_size 8 and 16, where many simulations traverse concurrently applying and
  removing virtual loss. It still holds to ~1e-16 → VL is fully unwound (a stuck
  −VL in W would corrupt Q). Visit-count bookkeeping `nsims-1 ≤ ΣN ≤ nsims` is
  also checked (the −1 vs 0 depends on whether the root is single-action).

### Rung 3 — multi-level backprop identity (8 assertions, 3 tiny trees)
Classic MCTS (`src/mcts.jl`) with `chance_mode=:full` (exact all-outcome chance
averaging), **no** bear-off evaluator, uniform V=0 interior priors, γ=1, no noise,
20 000 sims. Each root action's `Q·3` (raw points) is compared to the exact
expectimax value from the pure-Julia recursion. Exercises the mechanisms Rung 2
cannot reach: reward recording (÷3), **pswitch sign flips**, chance-node
probability weighting, multi-level accumulation, and the **±2 gammon reward path**.

Positions (each side few checkers, ends in ≤3 plies):
- `A_race_3v3` — genuine 3-vs-3 race, mixed action values → real argmax check.
- `C_race_2v3_black` — 2-vs-3 race, **black** to move (losing) → mover=1 sign paths.
- `B_gammon_multilevel` — **guaranteed +2 white gammon**: white 1 checker on
  point 6, dice (2,1) can't bear off → post-move is a *black chance node*; black
  has 15 checkers on point 7 (outside home, 0 off) and can never bear off before
  white finishes. The ±2 reward backs up through a chance node + pswitch.

## Tolerances chosen (and why)

| Rung | Check | Tolerance | Measured | Justification |
|------|-------|-----------|----------|---------------|
| 1a | sign negation | `== 0.0` | 0.0 | exact fp negation for turn-complete states |
| 1b | mirror symmetry | 1e-6 | ~0 | same physical race → identical table index |
| 1c | normalization | `== 0.0` | 0.0 | evaluator is literally `helper/3` |
| 1d | chance = wavg | **1e-3** | ~2.5e-5 | table probs are UInt16/65535 (~1.5e-5/prob); equity sums a few + per-dice max → ~30× margin |
| 2 | depth-1 Q identity | **1e-9** | ~5.6e-17 | pure fp; same leaf every visit → machine epsilon |
| 2 | argmax | exact, ties 1e-12 | 0 fails | exact table argmax |
| 3 | expectimax value | **0.1 raw pts** | ≤0.074 | `:full` is deterministic; uniform V=0 priors truncate un-expanded leaves at 0 → small O(0.05) downward bias at finite sims. A sign/scale bug gives O(0.5–2.0), so 0.1 cleanly separates correctness from MC bias |

The Rung 3 residual is a **fixed deterministic** function of the sim budget
(`:full` uses all-outcome expansion + deficit selection, no sampling), not noise —
so 0.1 is a stable ceiling, not a flaky threshold.

## The staircase (rungs 0–7)

| Rung | Claim under test | Status |
|------|------------------|--------|
| 0 | Exact k=7 table is itself correct (vs external / wildbg) | **curve-covered** — `eval_table_vs_wildbg.jl` (paired edge > 0), table construction in BackgammonNet.jl |
| 1 | Evaluator/wrapper sign · perspective · normalization · chance-average | **IDENTITY-covered** — this file, Rung 1 |
| 2 | Depth-1 search: `Q(a)` == exact leaf value; argmax; virtual-loss unwind | **IDENTITY-covered** — this file, Rung 2 |
| 3 | Multi-level backprop == exact expectimax (reward ÷3, pswitch, chance wts, ±2) | **IDENTITY-covered** — this file, Rung 3 |
| 4 | Bear-off-evaluator-guided MCTS agrees with the perfect-table policy in play | **curve-covered** — `eval_table_vs_wildbg.jl --policy=mcts` move-agreement (`Δ>1e-9` non-tie disagreements = 0) |
| 5 | Trained-NN MCTS value accuracy vs table/wildbg (contact + race) | **pending** — `scripts/eval_value_accuracy.jl` (TODO in CLAUDE.md) |
| 6 | Self-play policy improvement (MCTS visits > raw prior) | **curve-covered** — training TB curves / convergence sweep |
| 7 | End-to-end strength vs wildbg / GnuBG | **curve-covered** — `eval_race.jl`, `eval_vs_wildbg.jl` |

Rungs 1–3 are the deterministic *foundation*: they guarantee that any strength
gap observed at rungs 4–7 is a **modelling / budget** limitation, never a hidden
arithmetic bug in the search core.

## Counts / summary
- Total: **1071 assertions**, all green, ~17s (k=7 table mmap-backed → cheap load).
- No identity violations found. The MCTS backprop/perspective/scale arithmetic is
  exact to machine epsilon (Rungs 1–2) and converges to exact expectimax within a
  documented deterministic bias (Rung 3).
- On failure, every rung prints an actionable dump (board hex, dice, player,
  per-action table value vs Q) via `@error` before the `@test` records the failure.
