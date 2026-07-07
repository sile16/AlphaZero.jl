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

Next rung: self-play on the solved race sub-domain (does the full RL loop converge to table play?)
OR wire+verify Gumbel root search.
