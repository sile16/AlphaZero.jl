# Full-Codebase Correctness Review — 2026-07-03 (pre-v13)

A systematic fan-out review across the whole training/eval stack (beyond the
day's per-commit reviews), to de-risk before committing GPU-days to a v13 run.
Four independent agents covered: training loop + learning, distributed
coordination, self-play + target generation, and MCTS + inference. Findings
verified in-code before fixing.

## Verdict

The core is sound. The self-play target path and the MCTS/inference core came
back **clean** (all dimensions traced correct). The training-math dimensions the
review flagged were correct except one MED priority-scale bug. The distributed
layer had two HIGH network-robustness bugs. All HIGH/MED items are now fixed
with regression tests; a validated ladder (full suite green) confirms no
regression.

## Fixed (this batch)

- **HIGH — buffer copy trusts network array dims** (`buffer.jl` `per_add_batch!`).
  The `@inbounds` column copy used `n = length(batch_values)` and the buffer's
  own row counts with no validation. A worker on a stale git commit / different
  `BACKGAMMON_OBS_TYPE` (realistic during rolling `git pull` restarts) could send
  a wrong `state_dim` or length-skewed columns → OOB read → silent buffer
  corruption or a segfault taking down the whole training process. Added a
  dim/length guard before the copy (rejects → `handle_samples` 400). Test: 6
  assertions (wrong state_dim / num_actions / value skew / flag skew, buffer
  untouched on reject).
- **HIGH — unguarded `finalize_eval` can kill training** (`training_server.jl`,
  `server.jl`). `finalize_eval` does `value_nn .- value_opp` / `cor(...)`; a
  client submitting length-skewed value arrays throws `DimensionMismatch`, and
  the finalize call ran unguarded inside the training task → training silently
  dies while HTTP stays up. Fixed both ends: validate `length(value_nn) ==
  length(value_opp)` in `handle_eval_submit` (400 on mismatch), and wrap the
  training-loop finalize in try/catch with a `finally` that always clears
  `EVAL_JOB[]`.
- **MED — reanalyze PER priority on the wrong scale** (`buffer.jl`
  `reanalyze_update!`). Training writes priorities from the normalized TD error
  `|V̂/3 − V/3|` (∈[0,2]); reanalyze wrote `|new − old|` on the raw [-3,3] scale
  (∈[0,6]) into the same `buf.priorities`. Reanalyzed slots were oversampled
  ~3^0.6 ≈ 1.9× — a silent, compounding PER-distribution skew (exactly the kind
  of latent corruption v13 must avoid). Normalized by /3 to match. Test: 2
  assertions.
- **LOW — empty root Dirichlet vector could crash** (`mcts.jl` `best_uct_action`).
  `η = Float64[]` when `noise_α == 0`; a forced-move root reached with
  `noise_ϵ > 0` would index `η[i]` OOB. Not reachable today (self-play filters
  single-action roots; eval sets ε=0), but a v13 config raising eval ε would
  crash. Guarded: `iszero(ϵ) || isempty(η)`.
- **LOW — bearoff diagnostic sampled the wrong subset** (`training_server.jl`).
  `bearoff_mask[randperm(n_sample)]` only ever drew the first `n_sample` (oldest)
  bearoff slots; fixed to `bearoff_mask[randperm(n_bo)[1:n_sample]]`. TB
  diagnostic only, no gradient effect.

## Verified correct (no action — cited in agent reports)

IS-weight computation + application, chance-node policy masking, value/equity
target scale in `losses()`, reanalyze generation guard, per-partition sampling,
promotion gate (regressed model can't publish; fail-closed; resume state split),
eval work-queue state machine (no double-count / lost-chunk under the lease/
expiry race), weight pinning/versioning, buffer read/write locking + lock
ordering, MCTS value-backup scale across all three leaf sources, bearoff
override perspective, chance passthrough signs, PUCT, dual-network routing,
`sim_budget_fn` consumption, self-play perspective/flags/RNG/policy-alignment,
terminal-gammon target (both target + evaluator paths).

## Deferred (noted, not blocking v13 — weigh before/after launch)

- **MED — action mask reconstructed from `policy > 0`** (`training_server.jl`
  `prepare_batch_columnar`). The buffer stores no legal mask, so the invalid-
  action penalty is computed over the *visited* subset; legal-but-unvisited
  actions get mildly penalized as invalid. Real impact small at 400–600 sims
  (KL already drives this). Proper fix needs a legal-mask buffer column + wire
  field — a schema change; defer.
- **MED — sample upload has no idempotency** (`client.jl` `flush_samples!`).
  `empty!` runs before the status check → samples dropped on non-200; a thrown
  upload after the server processed could duplicate on retry. Impact: minor
  data loss / mild PER bias. Fix needs a dedup key or retain-on-failure with
  explicit semantics; defer.
- **LOW** — `init_game` `env.rng` assignment is a silent no-op (Xoshiro vs the
  env's MersenneTwister field type); harmless today (explicit sub-RNG threaded
  everywhere) but a determinism trap if future code reads `env.rng`. Reanalyze
  runs the untrained contact net in race mode (wasted GPU). Eval metrics during
  a gate-block reflect the last *published* weights (attribution note). MCTS
  bearoff decision-node override is stored in the tree (wastes sims vs the
  chance-node path). `V_equity` aliases `ln_mean` (safe now, fragile).

## A/B fix-impact check (fixed weights, old vs new code)

v12 iter-50 race weights, pre-fix `26a6a45` vs post-fix HEAD, 1000 paired race
games @600 MCTS: **play strength unchanged** (equity −0.006 vs −0.025, win 48.2%
vs 47.7% — within noise), confirming the fixes are eval-neutral for play (their
value is in training targets). Value MSE nearly halved (0.772 → 0.474) with
identical weights — the MCTS value-scale fix correcting the nn-value-vs-wildbg
comparison; 0.47 is the correct number.
