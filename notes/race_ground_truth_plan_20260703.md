# Pre-Bearoff Ground-Truth Eval — Build Plan (2026-07-03)

Turnkey staged plan for `scripts/race_ground_truth.jl`, grounded in an API
survey of BackgammonNet.jl + the k=7 table. Design rationale lives in memory
`race-ground-truth-rollouts`; this is the concrete implementation path.

**Why it matters:** v12 plateaued invisibly at wildbg parity because nothing
measures ground truth in the pre-bearoff race band. This gives that band a
fixed-position ground-truth eval (value MSE/corr/move-regret per checkpoint),
mirroring `eval_bearoff_accuracy.jl`.

**Non-negotiable:** a wrong ground truth is worse than none. Every stage is
validated before the next depends on it.

## Building blocks available
- `BackgammonNet`: `legal_actions`, `apply_action!`, `clone`, `copy_state!`,
  `sample_chance!`, `is_chance_node`, `is_race_position`, `game_terminated`,
  `winner`, board encoding (UInt128, 4 bits/point). No one-sided move API.
- `BearoffK7`: `BearoffTable`, `lookup`, `is_bearoff_position`, `compute_equity`
  — two-sided EQUITY only (pW, pWG, pLG), NOT a rolls-to-off distribution.
- Template: `scripts/eval_bearoff_accuracy.jl` (position cache, per-checkpoint
  value MSE/corr/move-regret scoring, checkpoint iteration).
- Data: `race_starts_tuples*.jls` on NFS; `race_eval_2000.jls` fixed set.

## Stage 1 — One-sided rollout engine (Tier 1, no truncation)
Simulate ONE side racing home, reusing BackgammonNet move-gen (do NOT
reimplement movement — bearing-off/overshoot/doubles rules are subtle).
- Construct a game with the racing side to move and a fixed non-terminal
  placeholder opponent that is never moved. After each of the side's turns the
  engine lands on the opponent's chance node; hijack the loop — reset
  `current_player` back to the racing side + fresh dice — so only that side ever
  moves. (Requires direct struct-field handling; unit-test against hand-checked
  positions.)
- Per rollout record the objective-independent primitives: `T_all` (rolls to all
  15 off), `T_first` (rolls to first checker off), `n_off(k)` (checkers off after
  roll k). Cache per (position, personality).
- **Validate:** Monte-Carlo `E[T_all]` for a few hand-computable positions
  (e.g. 2 checkers on the 1-point → geometric-ish), and that `winner`/off-count
  bookkeeping matches BackgammonNet on completed rollouts.

## Stage 2 — Personality movers (deterministic corners of objective space)
Fixed move-selection heuristics over `legal_actions` (no NN — see memory: never
use the NN as rollout policy):
1. **Efficiency** — minimize rolls-to-off (max pip progress / standard race play).
2. **Gammon-go** — maximize early bear-off pace (play FOR the gammon).
3. **Gammon-save** — minimize rolls to FIRST checker off (avoid being gammoned).
- **Validate:** on gammon-live positions, gammon-go should raise P(15-0) and
  gammon-save should lower P(0 off) vs efficiency; sanity-check directions.

## Stage 3 — Analytic combine
- Convolve the two sides' `T_all` distributions with the turn offset (who is on
  roll) → P(win). Gammon from `T_first`/`n_off` (loser has 0 off when winner
  finishes). Weight terminals by objective (money weights now; MET later reuses
  the SAME cached primitives — never re-simulate per score).
- Personality selection per side by favorite/underdog status at combine time.
- **Validate (gate):** compare multi-personality one-sided targets against a
  small sample of full TWO-SIDED rollouts (money equity). Bound the coupling
  error (target: equity MAE well under a checker-play margin) BEFORE trusting as
  ground truth. This is the go/no-go.

## Stage 4 — Optional Tier-2 exact DP + k=7 truncation (variance kill)
- Exact one-sided rolls-to-off DP for states small enough (home board + few
  points), k=7 as base case. Truncate rollouts into it once checkers enter range.
- Validates Tier 1 and removes deep-tail variance. This is the "missing
  primitive" — needed for the sharpest targets but not for a first usable eval.

## Stage 5 — Fixed eval set + scoring harness
- Build/reuse a fixed 2000 pre-bearoff race positions (from `race_starts`).
- Score every checkpoint: value MSE / corr / move-regret vs the ground-truth
  targets, parallel to the bearoff eval. Reuse `eval_bearoff_accuracy.jl`'s
  per-checkpoint scaffold.

## Effort / risk
Stages 1–3 + 5 give a usable, validated ground-truth eval (Tier 1). Stage 4 is
a follow-on refinement. The correctness-critical pieces are the Stage-1 turn/
dice state hijack and the Stage-3 combine — both must pass their validation
gates before any number is reported as "ground truth."
