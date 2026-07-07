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

Follow-ups: policy top-1 undercounts (race near-ties) — add a move-regret metric later. Then next
rung: verify the policy head + MCTS improvement, then self-play on the solved race sub-domain.
