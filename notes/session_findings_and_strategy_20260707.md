# Session Findings & Strategy — 2026-07-06/07

Durable consolidation so nothing is lost across sessions. Companion to the detailed milestone log
in `notes/design_decisions_20260706.md` and the auto-memory `project-status.md`.

---

## 1. The headline: the "plateau" was a DOUBLES bug, and the foundation is now trustworthy

The single defect behind every historical "plateau": evaluating a **multi-part doubles turn** with one
`apply_action!` + a sign flip. That's correct for non-doubles (one action = full turn) but wrong for
doubles (after one action the SAME player is mid-turn) and forced-pass (control returns to the mover).
The fix: recurse through the rest of the turn carrying the original mover, flipping only at true turn
boundaries — `BackgammonNet.bearoff_turn_value_equity` / `bearoff_turn_value`.

- It corrupted ~28% of targets (the doubles positions) → manufactured a fake ~0.74 value-corr ceiling.
- Fixed across the race generator, both verifiers, the contact bootstrap, and eval; added a CI-safe
  regression test (`BackgammonNet/test/test_turn_aware_bearoff.jl`). Three external reviews confirmed
  the production paths (best_move bridges, MCTS, board conversions) were already correct.
- RULE: any move-enumeration + table/engine lookup must evaluate at TURN BOUNDARIES, never one action
  + sign flip. Backends must value mid-turn states or the caller must recurse.

**Verified trustworthy now:** exact bearoff tables (k7 + n18); race self-play → value-corr 0.998;
full pipeline runs on both machines; wildbg reduced to a single OPTIONAL bootstrap.

---

## 2. The value-target principle (confirmed twice)

Low-variance value targets are the lever for the VALUE head:
- RACE: exact-table targets → value-corr **0.998**.
- CONTACT: wildbg PER-POSITION equity (`raw_evaluate_with_probs`, low-variance) instead of noisy game
  OUTCOME → imitation contact value-corr **0.39 → 0.9945**, bias +0.82 → −0.01.
- Outcome-as-value is the MC-variance ceiling; wildbg's direct eval is the low-variance analogue of
  the exact table. (Same idea as §13.5.3: relative/differential value matters most.)

And an accurate value makes SEARCH productive (see §3).

---

## 3. Contact self-play: what works, what the ceiling is

Progression of experiments (Win% vs wildbg; @mcts-200 unless noted):
- imitation warm-start (256×5, wildbg-equity value): 31% @200, 36% @400; value-corr 0.9945.
- **Value alone doesn't fix PLAY** — play is policy/search-limited, not value-limited.
- Unanchored self-play (mcts-200): value drifts to over-optimistic "vs-self", Win% 26→14→4% (crash).
- Anchored self-play (wbeq bootstrap pinned in buffer, mcts-200): value held ~0.88 but Win% FLAT
  28–30% — value calibration necessary but NOT sufficient.
- **mcts-600 self-play: Win% CLIMBED to 38.8%@200 / 40.8%@400** — higher SELF-PLAY search beats the
  imitation policy prior and yields improving policy targets. Search is a REAL lever.
- **mcts-800 (200 iters): CONFIRMS a ~40% ceiling** — bounced 35–39%, no gain over 600. MCTS budget
  exhausted as a lever. Value stays calibrated (0.87–0.94) → ceiling is POLICY/CAPACITY, not value.

**Two structural laws learned:**
- (a) Self-play value = "equity vs a copy of yourself" → drifts to over-optimism vs a strong external
  opponent unless anchored. RACE escapes this via the exact-table ABSOLUTE anchor; CONTACT has none.
- (b) With value calibrated, self-play improves PLAY only if MCTS search is deep enough to beat the
  policy prior (mcts-200 too weak; 600 works; 800 no better).

Best contact nets so far: **mcts-800 iter-110 (39.2%@200)**, **mcts-600 iter-120 (40.8%@400)**.

---

## 4. Optimization backlog / experiment ladder (next levers to break ~40%)

Ranked, to step through one at a time and verify each:

1. **Capacity** — bigger contact net (retest scaling; old "scale doesn't help" was on BUGGY data).
   IN PROGRESS: 512×8 (5.5M params vs 256×5's 1.1M) imitation → does it beat 31%@200 / 36%@400?
2. **Chance-node expansion: split TRAINING vs EVAL** (user strategy, 2026-07-07 — see §5). Current code
   uses PASSTHROUGH (single-sample) everywhere. Eval could use exact-expectation for more accurate play
   → may recover part of the "ceiling" with NO retraining. High value, eval-only change.
3. **Gumbel root search** — `src/gumbel_mcts.jl` exists but is NOT wired into `BatchedMctsPlayer`
   (production is pUCT). Gumbel is strongest at LOW sim budgets; unclear if the edge persists at high
   budgets (user's open question). Wire it + A/B at low and high sims.
4. **Temperature annealing / MCTS-budget schedules** — flags exist (`--temp-*`, `--mcts-budget-mode
   progressive`, `--progressive-sim-*`). Tune for self-play exploration vs exploitation by phase.
5. **Better policy targets / exploration** — Dirichlet at root, phase-aware temperature (doc §4.2).
6. **Late-training switch to slower/more-accurate** (user) — near the end, switch to exact chance +
   higher sims + lower temperature to extract the final SOTA edge.

---

## 5. Training-vs-Eval is a DIFFERENT paradigm (user strategy, 2026-07-07)

Key insight: learning-from-all-games (training) and playing-the-single-best-game (eval) can and should
use different search/expansion strategies. Concretely:

- **Chance nodes in TRAINING — passthrough (single randomly-sampled outcome):** more efficient. The
  full width of chance outcomes is covered across MANY full games rather than by expanding one node;
  the afterstate is still trained on the game's value outcome. (Current default — keep.)
- **Chance nodes in EVAL — exact-expectation + progressive expansion:** on FIRST hit, the NN value at
  the chance node ≈ the network's estimate of the average over all random outcomes; on SECOND visit,
  expand ALL child outcomes at once (explore every dice result), then sample later visits by the known
  (1/36, 2/36) probabilities. This is the doc's §3.5 Mode A ("exact 21-roll expectation near root")
  and gives strictly more accurate move choice at eval for the same net. NOT yet implemented in
  `batched_mcts.jl` (which is passthrough-only) — a real code change, high value.
- **Gumbel**: clear advantage at LOW sim budgets (Gumbel-Top-k + sequential halving). Open question:
  does it keep its edge at HIGH sim budgets, or does pUCT catch up? Test both regimes.
- **Temperature + sim budget**: separate schedules for training (exploration, diversity) vs eval
  (greedy, deep). Late-training may switch to the slow/accurate eval-style search to finalize.

Implication for measuring "beat wildbg": our current ~40% may be partly an EVAL-accuracy artifact
(passthrough chance + mcts-200). Evaluating the SAME best net with exact-expectation chance +
higher/gumbel search could raise the measured strength without any retraining. Worth an early test.

---

## 6. Infra lessons (durable)

- **Neo (M3/ARM) is ~8× SLOWER than Jarvis for wildbg-FFI** (bootstrap gen) but ~EQUAL per-core for
  pure-Julia MCTS self-play. Do wildbg generation on Jarvis; use Neo for self-play CLIENTS.
- **Self-play throughput bottleneck = the SINGLE inference batcher per client process** (one thread,
  BLAS=1), NOT cores. Run MULTIPLE client processes per machine (5 on Neo → ~4× a Jarvis client).
- **The curriculum is GPU-TRAINING-bound** — more self-play throughput beyond "enough fresh data"
  doesn't speed iterations, only reduces staleness.
- Sharded wildbg gen = many single-process shards + `merge_contact_shards.jl` (forces
  obs_type=min_plus_flat; synthesizes value-head-contract metadata).
- Anchored self-play = pre-fill buffer with the wbeq bootstrap + `--bootstrap-train-iters 0` + a
  buffer big enough not to evict it (proper fix = constant-fraction mixed-batch, not yet built).
- Eval robustness: wildbg `best_move` can fail to match its move on rare doubles → ExternalAgent now
  falls back to a legal move.
- Metrics: adopt ERROR RATE / PR (equity loss per unforced decision) as PRIMARY over win% — it's the
  race move-regret we already compute; doc §13.5. Luck-adjusted match scoring shrinks CIs further.

---

## 7. Key artifacts

- Bootstrap (wildbg-equity, doubles-correct): `BackgammonNet/data/bootstrap/contact_bootstrap_wbeq_300k.jls`
- Corrected race net (0.998): `sessions/race-supervised-v2/checkpoints/race_train_latest.data`
- Best contact nets: `sessions/contact-selfplay-mcts800/checkpoints/contact_iter_110.data`,
  `sessions/contact-selfplay-mcts600/checkpoints/contact_iter_120.data`
- Generators: `BackgammonNet/scripts/generate_contact_bootstrap.jl` (turn-aware policy + wildbg-equity
  value), `generate_race_table_supervised.jl` (exact-table, doubles-correct).
- Verifiers: `AlphaZero/scripts/verify_race_supervised.jl`, `verify_race_mcts.jl`,
  `validate_contact_doubles_policy.jl`, `validate_rollout_vs_k7.jl`, `eval_vs_wildbg.jl`.

---

## 8. CORRECTED FINAL VERDICT (late 2026-07-07) — the honest strength picture

The §3 win%-based claims ("beat wildbg", "~gnubg-0ply parity") turned out to be INFLATED. Rigorous
per-decision metrics on a COMMON benchmark corrected them. This is the trustworthy conclusion:

**Our contact net is genuinely WEAK.** True-scale PR (gnubg-ply1 native reference, common 1500-position
benchmark, floor ~1 PR): **gnubg-0ply ~1.9 PR (near-perfect) | wildbg-large ~13 PR (intermediate) |
i140 (ours) ~30 PR (weak)**. Raw net (mcts-1, no search) wins only **6.7%** vs wildbg — a blowout loss.

**"Beating wildbg" was two artifacts, not net strength:** (1) a HARNESS BUG — `best_move` threw on rare
doubles-match failures, and the eval's ExternalAgent then played an arbitrary bad move FOR the opponent,
inflating win% ~2-4 pts (fixed: `best_move` now falls back to rank-by-own-eval argmax; committed
`fad1b95`, tests green, validated 0 throws / ~18500 decisions). (2) ASYMMETRIC SEARCH — i140 ran
mcts-800 lookahead vs wildbg's shallow best-move, letting a weak net reach ~even. Corrected true
head-to-heads: **i140 vs wildbg 52%, vs gnubg-0ply 33.3%.**

**Two structural lessons (hold across the cubeful redesign):**
- **Win% vs a weak/handicapped opponent is a mirage; per-decision PR is the truth.** Only PR caught this.
- **Deep search lifts win% but NOT per-decision quality** (~48@800 ≈ ~54@1600 self-play PR) — the net's
  policy/value is the cap. The path to world-class is a fundamentally STRONGER NETWORK, not more search.

**Trustworthy metric/eval suite (use these for the cubeful net):**
- `scripts/benchmark_pr.jl` — cross-comparable checker PR on a FIXED common position set (the RIGHT way
  to track net versions; self-play PR is NOT cross-comparable).
- `scripts/analyze_pr_native.jl` — true-scale checker PR (~0 floor via gnubg native move-list + match-
  by-board, sidesteps the gnubg player-0 action-id bug).
- `BackgammonNet/scripts/analyze_cube_pr.jl` — cube-ER (doubling + take) via gnubg arDouble, floor 0.
- `scripts/calibrate_pr_ladder.jl` — external engines' own PR (the calibration ladder above).
- `scripts/eval_vs_gnubg.jl` / `eval_vs_wildbg.jl` — head-to-heads (now trustworthy post `fad1b95`;
  gnubg ply>=2 deadlocks — use ply-0/1 or single-worker small batches).

**Cubeful redesign (user-driven, validated game-side):** 4 MCTS node types (checker 676=26x26 micro-
action, chance 21, double 2, take 2) + per-non-chance value heads; game rules VALIDATED (unit 231328,
full-game integration, cube-decision reference — see `test_cubeful_integration.jl`,
`validate_cube_reference.jl`). AZ-side MCTS/heads/training is the next build. PR-track the new net on
`benchmark_pr.jl` (common set) — NOT win%.
