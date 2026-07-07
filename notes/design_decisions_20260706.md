# Design Decisions — 2026-07-06 (post-refactor reset)

Clean slate. Old experiment logs (EXP2–EXP5, the "~29% ceiling"/"method-bound plateau"
conclusions) are **discarded** — they may reflect backend/training bugs rather than real signal.
Methodology going forward: build models and add techniques **one at a time**, verifying each
works with our implementation + tests and produces **solid, trustworthy** results before the next.

Target architecture reference: `docs/recent_ml_research_papers.md` (Explicit-Chance Gumbel
AlphaZero). Bring its pieces in incrementally; we already have Gumbel MCTS + sequential halving,
explicit chance nodes, a multi-head distributional value head, and cubeful values.

---

## Action space — DECIDED: keep 680, deterministic dice-movement

`NUM_ACTIONS = BackgammonNet.MAX_ACTIONS = 680`
- 1:676 = checker actions (deterministic dice-movement encoding; doubles split into two actions)
- 677:680 = cube actions

Rationale for keeping it (not switching representations):
- Not excessively large — chess has a larger action space; this is a good balance.
- Beats the tiny single-dice-move / micro-action space (~100 = 4 dice-positions × 24 points)
  used by the Stochastic-MuZero backgammon paper.
- ~80%+ of moves are non-doubles; splitting doubles into two actions is very doable.
- We keep a **real, learnable policy head**. Afterstate scoring (doc §7.2) is rejected: with
  afterstates you can't predict a policy — it's just raw NN value heads.

### Noted optimization (rejected for now): collapse duplicate dice-order actions
Different dice orders can be two distinct action indices that reach the **same target board**
(duplicates). We could mask to collapse them into one action. **Decision: don't** — the masking
rules would likely be *harder* for the NN to learn than the current deterministic
dice-movement action space. Kept here as a possible future lever only.

---

## Observation space — TO DECIDE (sets NN input size)

Configurable via `BACKGAMMON_OBS_TYPE` env var. Flat options and their input dims:

| obs type        | dim |
|-----------------|-----|
| minimal_flat    | 344 | (default)
| min_plus_flat   | 350 |
| full_flat       | 376 |
| biased_flat     | 436 |

Plus hybrid board+globals variants. Choose deliberately — this + action space drive NN size.

**DECIDED: `min_plus_flat` (350).** Adds the highest-value, lowest-cost engineered features
over minimal (pip counts, pip diff, contact/race indicator, can-bear-off) — matches the
"strongest reduced-hardware" stance in `docs/recent_ml_research_papers.md` §6.3/§8.7. Committed
as the baked default in `games/backgammon-deterministic/game.jl` (`OBS_TYPE_STR`). Larger tiers
(full 376, biased 436) can be A/B'd later as clean one-variable tests.

Gotcha fixed: `training_server.jl` and `selfplay_client.jl` hardcoded
`ENV["BACKGAMMON_OBS_TYPE"]="minimal_flat"` before including game.jl — changed to respect an
explicit env override, else default min_plus_flat. `OBS_TYPE_STR` is a const read from ENV at
game.jl include time; the server sets it right before the include so it takes effect.

---

## Milestone 1 — VERIFIED: exact-table supervised race value net (2026-07-06)

First trustworthy result of the reset. Pipeline: obs(min_plus_flat 350) → 128×3 multihead net
→ 5-head value + policy → BCEWithLogits/CE loss → Trainer.

- **Data:** `BackgammonNet/scripts/generate_race_table_supervised.jl` — 300k disengaged-race
  decision nodes, targets from the EXACT one-sided n18 table. Value target = POST-DICE optimal
  (max over legal moves; NOT pre-dice on-roll equity). 5-head mover-relative (1-pW, pLG, 0, pWG, 0);
  soft best-move policy (temp 0.05). Gen: 300k in 14s (16 threads). Held-out 20k test (disjoint seed).
- **Train:** `training_server.jl --training-mode race --bootstrap-only` on the 300k file, 40 iters,
  race_loss 1.79 → 1.245.
- **Verify:** `scripts/verify_race_supervised.jl` on held-out 20k, iter 40:
  **value corr 0.99869, MSE 0.0051, MAE 0.032 pts, bias +0.003**, policy top-1 65.6%.
- **Conclusion:** obs encoding + net + value head + loss + trainer are TRUSTWORTHY. The near-exact
  value net doubles as a reusable frozen RACE evaluator for later contact self-play.

## Milestone 2a — VERIFIED: race POLICY head via move-regret (2026-07-06)

Top-1 agreement (65.6%) is misleading for races (many near-tied best moves). The proper metric
is exact move-regret = equity lost vs the table's best move. `verify_race_supervised.jl` now loads
the n18 table and scores whatever move the net picks. iter-40 race net on held-out 20k:
- **mean regret 0.0022 pts, median 0.000, p95 0.012, optimal-move rate (regret<0.01) 94.2%**, max 1.0.
- Raw-net policy is near-optimal on races; the ~6% tail (regret up to 1 pt) is where MCTS should help.
- Confirms the earlier intuition (low agreement, ~0 regret). Policy head VERIFIED good.

## Milestone 2b — VERIFIED: MCTS improves race move quality (2026-07-06)

`verify_race_mcts.jl` compares raw-net argmax vs the production MCTS move, both scored against
the exact n18 table (MCTS uses the NN evaluator ONLY — no exact-table leaf — to isolate search).
3000 held-out positions, 100 iters:
- full set: raw mean regret 0.00255 → MCTS 0.00248 (2.5% lower, optimal% 93.9 both).
- **hard subset** (6.1% where raw net is suboptimal, regret ≥0.01): raw 0.0365 → **MCTS 0.0332
  (9.0% lower)**. Search helps where it matters, never hurts.
- Modest overall because the raw net is already near-optimal on races AND its value head is
  near-exact (so NN-leaf lookahead adds little). Search verified working.

**FINDING (actionable):** the production MCTS (`BatchedMctsPlayer`/`batched_mcts.jl`) is **pUCT**,
NOT Gumbel. `src/gumbel_mcts.jl` (sequential-halving Gumbel-root, the doc's #1 recommendation)
EXISTS but is **not wired into the batched player** used by self-play/eval. So "add Gumbel root
search" is a real, distinct future rung: wire gumbel into BatchedMCTS (or route the player to the
gumbel Env), then re-run this exact regret comparison (pUCT vs Gumbel vs raw) to measure its value.

## Milestone 3 — KEY FINDING: self-play PLATEAUS far below supervised on races (2026-07-06)

Ran self-play on the race sub-domain (seed from 5000 disengaged race starts, race training-mode,
128×3 net, mcts-iters 100, ~3000 samples/iter, 100 train steps/iter). Measured value-corr +
move-regret vs the EXACT table on the held-out 20k covered-band set each few iters.

Two configs, same result:
- **truncation ON** (exact frontier returns): corr flat ~0.72. (metric partly mismatched — truncation
  doesn't train the covered band, so this alone is inconclusive.)
- **pure self-play (--no-bearoff, no table crutch)**: covered band IS in-distribution (games play
  through to terminal), so the test set is a valid yardstick. Result — **corr FLAT ~0.74 across
  iters 5/15/30/60/100** (0.747, 0.741, 0.741, 0.747, 0.745); move-regret ~0.022; optimal% ~82.

Compare **supervised on the IDENTICAL net/obs/domain/test set: corr 0.9987, regret 0.0022, 94%
optimal.** The net is provably capable; **self-play stagnates ~0.74**. Both heads plateau together
(policy stuck at 82% optimal → value targets stuck → policy stuck — classic self-play stagnation).

**This isolates the long-standing "~0.74 plateau" against exact ground truth, with confounds ruled
out: NOT net capacity, NOT observation, NOT target representability (supervised reaches 0.9987).
The limiter is in the SELF-PLAY → target → training path.** corr is scale-invariant, so it is NOT a
pure value-scale bug (that wouldn't decorrelate); it's something that DEGRADES the signal:
search/policy improvement, policy/value target construction, or buffer/training dynamics.

Leading suspects to investigate next (one at a time):
1. MCTS not improving the policy (weak/mis-scaled search) — note [[review-findings-2026-07]] listed
   an "MCTS value scale mismatch" High finding; re-verify it in this clean setting.
2. Value/policy target construction from self-play traces (outcome bootstrapping, buffer [-3,3] vs
   MCTS [-1,1] boundary, policy-from-visits degeneracy).
3. Training dynamics (LR, steps/iter, buffer churn/overfit, PER off).

Diagnostic advantage: we now have a TRUSTED supervised reference + exact metric, so each fix can be
A/B'd cleanly (does self-play corr move toward 0.99?).

## Milestone 3 diagnosis (2026-07-06) — two findings

**Ruled out distribution confound:** re-ran pure self-play seeded from the COVERED band itself
(`covered_band_starts_8000.jls`, train == test distribution). Still corr flat ~0.74 (iters 10/30/60),
while race_loss dropped to ~1.07 (lower than far-race). So the net FITS its targets (loss ↓) but the
targets only ~0.74-correlate with exact equity → **the self-play TARGETS are the cap**, not net/obs/dist.

**Finding A — self-play is at the MC-outcome ceiling.** `BackgammonNet/scripts/diagnose_mc_target_variance.jl`
rolls out optimal (table-greedy) games from covered-band positions and correlates outcomes with the
table equity: **corr(single-game outcome, exact) = 0.73–0.77 ≈ the self-play net's 0.74.** The net is
saturated at the single-MC-outcome information limit. Supervised (exact expectation targets) → 0.99.
Lever = value-TARGET quality (lower variance): MCTS root value, reanalyze, exact-frontier truncation.

### RESOLUTION (2026-07-06, later): the "plateau" was a DOUBLES target-generation bug, NOT a real plateau

`validate_rollout_vs_k7.jl` on exact-k7 deep bearoff, split by dice type:
- **non-doubles: corr(rollout-mean, k7-exact) = 0.9996, MAE 0.013** — rollout + tables are EXACT.
- **doubles: corr 0.147, MAE 0.79** — garbage.

Root cause: my hand-rolled `move_eq` (in `generate_race_table_supervised.jl` AND the diagnostics)
does ONE `apply_action!` per move, but a DOUBLES roll is a multi-part turn — after one action the
SAME player is still on turn (mid-turn), so the unconditional sign flip `-compute_equity(lookup(child))`
is wrong. Production `BackgammonNet.bearoff_best_move_value` recurses through mid-turn doubles
correctly; my generator did not. So ~28% of my supervised TARGET values (doubles) are wrong.

Re-measuring corr split by dice type (test set = my generator's values):
- SUPERVISED net: non-dbl 0.9988 / dbl 0.9985 (it faithfully reproduced its own buggy targets).
- SELF-PLAY net (covered iter 60): **non-dbl 0.992** / dbl 0.117. Trained on real game outcomes
  (encoding-agnostic) → learns CORRECT values → matches exact on non-doubles (0.992 ≈ supervised),
  DISAGREES with the buggy doubles targets (0.117), which drags the average to the fake "0.74".

**CONCLUSIONS (Milestone 3 corrected):**
1. Tables (k7 AND n18) are EXACT. Finding B DISSOLVED.
2. The self-play RL loop WORKS — corr 0.992 on the trustworthy (non-doubles) subset, vs supervised
   0.999. There is NO plateau. The small remaining gap (0.992 vs 0.999) is the expected MC-target
   noise (Finding A's mechanism, but tiny — not a ceiling).
3. My `generate_race_table_supervised.jl` (and verify/diagnostic move_eq) have a DOUBLES bug →
   supervised targets wrong for ~28% of positions. FIX: use production `bearoff_best_move_value` /
   post-dice recursion for doubles, regenerate train+test, re-measure full-set (expect self-play to
   jump to ~0.99 overall once the reference is correct).

Historical note: this doubles bug is a strong candidate for the ORIGINAL "~0.74 contact plateau" too
(same move_eq pattern anywhere doubles are evaluated one-action-at-a-time).

### CONFIRMED FIX (2026-07-06): doubles-correct generator → self-play scores 0.99 both

Fixed `generate_race_table_supervised.jl` to use combined k7+n18 tables +
`BackgammonNet.bearoff_turn_value_equity` (recurses through mid-turn doubles). Regenerated the 20k
test set (v2) and re-measured the UNCHANGED nets against CORRECT targets:
- **Self-play net (covered iter 60): non-doubles 0.992, doubles 0.991, overall 0.9917.** (was 0.745
  vs the buggy test set). The self-play RL loop was correct all along — NO plateau, ever.
- Old supervised net (trained ON buggy targets): non-doubles 0.999, doubles **0.122**, overall 0.75.
  It faithfully learned the garbage doubles targets → its "0.9987 gold standard" was an illusion
  (self-consistent with its own bad targets). The self-play net is BETTER on doubles (0.99 vs 0.12).

Regenerated the 300k train set (v2, seed 42) and retraining supervised on it (sessions/race-
supervised-v2) to confirm supervised now reaches ~0.99 on both dice types with correct targets.

### LOOP CLOSED + report-driven hardening (2026-07-06)
- Supervised RETRAINED on corrected 300k v2 (sessions/race-supervised-v2, iter 20): **non-doubles
  corr 0.9986 / doubles 0.9981 / overall 0.9984** (doubles was 0.122 with buggy targets). Move-regret
  (now doubles-correct) mean 0.0023, 94.5% optimal. Full pipeline confirmed: tables→generator→
  supervised 0.998 both, self-play 0.99 both.
- Engineer's report reviewed (same bug class). Fixes applied:
  - P0 `verify_race_supervised.jl` + `verify_race_mcts.jl`: move-regret now uses
    `bearoff_turn_value_equity` (combined table) — doubles-correct.
  - P0 `validate_rollout_vs_k7.jl`: `move_eq` clearly marked as the INTENTIONAL naive reproducer.
  - P1 generator doc header corrected (no longer claims the "child = opponent, flip" assumption).
  - P1 `generate_contact_bootstrap.jl` + P2 `src/eval_backends.jl`: these use the SAFER pattern
    (flip only when current_player actually changed) but rely on the backend valuing MID-TURN states;
    documented the invariant + TODO parity test. (Also flags the second failure mode: forced-pass /
    blocked opponent returning control to the mover.)
  - Added `test/test_turn_aware_bearoff.jl` (CI-safe, mock table): guards the game-mechanics invariant
    (doubles ⇒ mid-turn, current_player unchanged) and that turn-aware eval differs from the naive
    one-ply flip on mid-turn states. 256 assertions, green.

**Net takeaways:**
- Exact bearoff tables (k7, n18) are TRUSTWORTHY.
- The self-play RL loop WORKS end-to-end on races (corr 0.99). The whole multi-session "self-play
  caps at ~0.74 / method-bound plateau" narrative was a doubles TARGET-COMPUTATION bug, not an
  algorithmic ceiling. Re-examine the contact path for the same one-action-at-a-time doubles pattern.
- Any script doing move-enumeration + table lookup must evaluate at TURN BOUNDARIES (recurse through
  doubles), never one action + sign flip. `verify_race_supervised.jl`/`verify_race_mcts.jl`/
  `diagnose_mc_target_variance.jl` still carry the naive move_eq for their move-regret metric — only
  valid on non-doubles; fix or restrict before trusting their doubles regret numbers.

---
_Superseded investigation note (kept for context):_
**Finding B (open, foundational) — optimal-rollout mean ≠ one-sided-table equity by ~0.29 MAE.**
Averaging rollouts barely helps: mean-of-25 corr 0.75, mean-of-200 corr 0.80, **MAE stuck ~0.29**
(variance-only would give MAE ≤ ~0.11 at K=200). So there is a SYSTEMATIC gap between actual
optimal-play outcomes and the n18 one-sided-table equity. Either (a) the diagnostic's greedy rollout
policy is suboptimal / has a bug, or (b) the n18 one-sided table is NOT exact on the full race band
(it was only validated MAE 0.0007 vs k7 on the DEEP-bearoff OVERLAP; the broader ≤18-frame band is
untested against true 2-sided outcomes). This matters: the table is the ground-truth reference behind
the supervised targets, truncation, and the corr metric itself. MUST resolve before trusting the 0.74
gap magnitude — validate the rollout on k7-exact deep bearoff (does rollout-mean → k7 equity?), and
validate n18 vs k7 across the overlap + n18 vs rollouts.
