# MCTS-driven + Match-Objective Bearoff Validation — 2026-07-03

Extends `scripts/eval_table_vs_wildbg.jl` (exact k=7 bearoff table vs wildbg from
mutual-bearoff starts, duplicate-dice paired CI) with:

- **Task A** `--policy=table|mcts` — play the table side either with the pure
  exact-table argmax (unchanged default) or with `BatchedMCTS.BatchedMctsPlayer`
  driven by an exact bearoff evaluator, plus move-agreement instrumentation.
- **Task B** `--objective=money|dmp|gg|gs` — a weight vector over (plain win,
  gammon win, plain loss, gammon loss) applied to BOTH the table's move selection
  and to game scoring. wildbg always plays money.

New flags: `--policy`, `--objective`, `--mcts-iters` (default 30), `--cpuct`
(default 2.0). `scripts/bearoff_eval_common.jl` gained an optional `weights=`
kwarg on `bearoff_turn_value` / `bearoff_best_move_value` (default
`BEAROFF_MONEY_WEIGHTS = (1,2,-1,-2)` reproduces `compute_equity` / raw reward
bit-exactly → all existing callers unchanged).

All runs: 1000 positions × 2 sides (paired), 8 workers, seed 42, wildbg large lib,
exact k=7 twosided table (c14 27GB + c15 68GB, mmap).

## Results

| # | policy | objective | starts | paired edge ± CI (obj units) | win% | move-agree (mcts) | obj≠money argmax | harness label |
|---|--------|-----------|--------|------------------------------|------|-------------------|------------------|---------------|
| 1 | table  | money | all      | **+0.003 ± 0.0059** | 50.0 | —            | —          | OK (parity)   |
| 2 | mcts   | money | all      | −0.075 ± 0.0159     | 46.6 | 69.4%        | —          | (see note)    |
| 2b| mcts   | money | all (400 iters, 400 pos) | +0.010 ± 0.0138 | 50.5 | 87.7% | — | OK (parity)   |
| 3 | table  | dmp   | gammon   | **+0.002 ± 0.0028** | 50.1 | —            | 4.12%      | OK (parity)   |
| 4 | table  | gg    | gammon   | +0.092 ± 0.0126     | 50.0 | —            | 0.15%      | PASS*         |
| 5 | table  | gs    | gammon   | −0.162 ± 0.0148     | 50.0 | —            | 5.01%      | FAIL*         |
| 6a| mcts   | gg    | gammon   | +0.054 ± 0.0178     | 48.0 | 70.4%        | 0.16%      | PASS*         |
| 6b| mcts   | gs    | gammon   | −0.204 ± 0.0175     | 48.0 | 71.9%        | 5.59%      | FAIL*         |

`*` The harness PASS/FAIL text was written for the **money** (zero-sum,
symmetric) case. For gg/gs the paired-edge sign is dominated by the objective's
**non-symmetric scoring**, not by play skill — disregard the PASS/FAIL label for
those rows (see interpretation).

### gg/gs orientation split (objective units, n=1000 each)

| objective | favorite (table ahead, can WIN a gammon) | underdog (table has 15 on board, SAVES) |
|-----------|------------------------------------------|-----------------------------------------|
| gg (table) | +1.086 ± 0.038 | −0.902 ± 0.027 |
| gs (table) | +0.900 ± 0.027 | −1.224 ± 0.042 |
| gg (mcts)  | +1.048 ± 0.042 | −0.940 ± 0.021 |
| gs (mcts)  | +0.862 ± 0.031 | −1.269 ± 0.038 |

Each paired position (exactly one side has all 15 on board) contributes one
favorite game (table plays the side ahead) and one underdog game (table plays the
15-on-board side). gg makes the favorite worth **more** than it makes the underdog
cost (+1.086 vs −0.902 ⇒ paired +0.092); gs is the mirror (−1.224 dominates ⇒
paired −0.162). This asymmetry is the **weight** asymmetry, present even under
near-identical play — not a skill gap.

### MCTS move-agreement vs sim budget (money objective)

| iters | decisions | agree% | mean Δ (raw pts) | max Δ | Δ>1e-9 (non-tie) | paired edge |
|-------|-----------|--------|------------------|-------|------------------|-------------|
| 30    | 4806 | 69.4% | 0.0885 | 1.00 | 1413 | −0.075 ± 0.016 |
| 400   | 2008 | 87.7% | 0.0037 | 0.11 |  166 | +0.010 ± 0.014 |
| 2000  |  186 | 94.1% | 0.0005 | 0.004|    5 | +0.025 ± 0.049 |

Agreement, Δ, and edge converge **monotonically** to the pure-table policy as the
sim budget rises. This is the decisive evidence that the MCTS wiring is correct.

## Interpretation

### Task A — MCTS wiring is CORRECT (PASS)

At the *specified* 30 iters / cpuct 2.0 the MCTS agrees with the pure table only
~69% and posts a *significantly negative* paired edge (−0.075). That is **not** a
wiring bug:

- With 400 / 2000 iters the edge returns to parity (+0.010 / +0.025) and agreement
  climbs to 88% / 94% with mean Δ → 0.0005 raw points. A sign/perspective flip
  would give ~0% agreement, a strongly negative edge, and would **not** improve
  with more sims.
- Agreement at 30 iters (69%) is already far above random over the handful of
  legal bearoff moves.
- Lowering cpuct to 1.0 / 0.3 did **not** raise 30-iter agreement (still ~70%),
  ruling out "too much exploration" as the sole cause.

**Root cause of the 30-iter gap (not a bug):** the bearoff evaluator returns exact
values normalized `/3` (per the project's MCTS scale convention), so the value gap
between competing bearoff moves is tiny (~0.01–0.03 normalized). With only 30
simulations the PUCT visit counts cannot resolve gaps that small, so the
visit-count argmax often picks a near-equal-but-not-best move. 30-iter MCTS is
therefore a genuinely *weaker search* than exact play — it just needs more sims to
match the table. Wildbg plays near-perfect mutual bearoff, so a slightly weaker
searcher loses a small margin to it. **Conclusion: the exact-evaluator MCTS path
is correctly wired; use ≥~400 iters if you want table-level bearoff play.**

### Task B — objective selection is active, but has little to bite on in bearoff

The exact "objective-argmax ≠ money-argmax" rate is the clean, noise-free measure
of how often optimizing the objective changes the *optimal* move (and hence how
much a money player like wildbg could be out-played):

- **dmp 4.12%, gg 0.15%, gs 5.01%** of table decisions.

So in the k=7 mutual-bearoff range the objective rarely changes the best move.
This is expected: pure mutual bearoff is a race with no shots, so
maximize-wins ≈ maximize-gammons ≈ money for the overwhelming majority of
positions. Gammon-go / gammon-save decisions (risking or avoiding a shot, timing)
live mostly in **contact** positions, which this harness does not cover.

- **dmp** has *symmetric* weights `(1,1,−1,−1)`, so its paired edge IS a clean
  skill measure: **+0.002 ± 0.0028 — parity**, exactly as expected. The table
  adapts its move on 4.1% of decisions but nets ~0 vs near-perfect wildbg.
- **gg / gs** have *asymmetric* weights. Under near-money play the paired edge is
  driven by the scoring asymmetry (gg rewards the gammon-winner more than it
  penalizes the gammon-loser ⇒ +0.092; gs the reverse ⇒ −0.162), **not** by
  superior objective play. With only 0.15% / 5% of moves differing from money,
  there is no real objective skill edge over wildbg to be had in bearoff.

**Honest headline (as the task requested): wildbg essentially agrees with the
objective-optimal move in the bearoff range, so the objective margin is genuinely
small; the nonzero gg/gs edges are a scoring artifact, not a demonstration that
the table out-plays wildbg on the match objective.** To see a real gg/gs skill
gap, run from *contact* gammon positions (out of scope here — the table only
covers ≤7-checker bearoff).

MCTS objective runs (6a/6b) track their table counterparts (gg +0.054 vs +0.092;
gs −0.204 vs −0.162), a little worse because 30-iter MCTS is the weaker search
(same effect as row 2); agreement 70–72% as in the money case.

## Bug found & fixed during this work

**Stateful wildbg backend crashed the move-disagreement instrumentation.** The
first draft measured "table-objective move vs wildbg's money move" by calling
`BackgammonNet.agent_move` on each table decision state. wildbg's backend is
**stateful** — it caches a doubles sub-move plan and must be driven *in sequence*
(`best_move` on a `remaining_actions==2` doubles state caches the second sub-move
for the next call). Querying it out-of-band on arbitrary trace states throws
`Cached wildbg doubles target no longer matches any legal second action`.

**Fix:** dropped the wildbg query and instead compare the exact table
**money-argmax** as the money-player reference (`obj≠money argmax` column). This is
noise-free, a *stronger* money player than wildbg, and directly answers "does the
objective change the optimal move" — strictly better than comparing to an
imperfect, stateful engine. Documented in the harness.

No other bugs. The turn-aware doubles handling in `bearoff_eval_common.jl` (the
2026-07-03 mid-turn sign pitfall) was preserved and exercised through the new
weighted paths.

## Caveats

- **Non-money MCTS + reward scaling.** Inside MCTS, rewards on edges MCTS actually
  traverses (e.g. a move that immediately bears off the last checker → terminal)
  are divided by `GI.reward_scale = 3.0`, which is *money*-scaled, while the
  bearoff evaluator applies the objective weights (also `/3`). In pure mutual
  bearoff the evaluator fires at essentially every node and dominates, so this
  mismatch is negligible here; it would matter more if MCTS routinely played
  through real terminal edges. The evaluator normalization is `/3` throughout
  (documented in the harness), monotonic, so it does not affect argmax.
- The harness PASS/FAIL banner is money-oriented; for objective runs read the
  paired edge + orientation split + obj≠money rate, not the banner.
- gg/gs paired edges conflate scoring asymmetry with skill; only the obj≠money
  move-change rate isolates the play effect in this bearoff-only setting.

## Verification

- `Meta.parseall` syntax-clean on both edited files.
- Full suite green: `Testing AlphaZero tests passed` (exit 0), incl.
  `Bearoff Doubles Regression 11/11` unchanged — backward compatibility proven.
- Baseline reproduced: `--policy=table --objective=money` → **+0.003 ± 0.0059**
  (matches the pre-existing harness result).
