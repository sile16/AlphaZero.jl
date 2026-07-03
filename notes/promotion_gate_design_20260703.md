# Weight Promotion Gate — Design (2026-07-03)

Closes the last open finding of the 2026-07-03 external review
(`notes/review_fixes_20260703.md` §5). The training server used to PUBLISH fresh
weights every iteration unconditionally; a regressed race model immediately fed
its own regression back into the replay buffer via self-play. This adds a gate on
PUBLICATION only — training is never gated.

## Files

- `src/distributed/promotion_gate.jl` — NEW. Pure decision logic + JSON sidecar.
  Included top-level (no module) by `training_server.jl`, same as `buffer.jl`.
- `scripts/training_server.jl` — integration (CLI flags, config/banner, resume
  seeding, gate decision in the bearoff-eval block, gated publication, gated
  `*_latest.data` writes, `race_best.data` + sidecar saves).
- `test/test_promotion_gate.jl` — NEW, wired into `test/runtests.jl` (53 assertions).

Nothing under `src/` was modified except the new file; `server.jl` is untouched.

## Gate rule

Metric `m` = the fixed-set bearoff eval's **value MAE** (race value head vs exact
k=7 table, normalized equity units; lower is better). Best-so-far `best` is the
lowest MAE seen. For each eval:

```
threshold = best * (1 + tol_frac) + tol_abs
publish   = m <= threshold
```

- `tol_frac` = `--gate-tolerance`, default **0.10** (10%).
- `tol_abs` = `GATE_TOL_ABS` = **0.003** (normalized ≈ 0.009 points) — an absolute
  floor so the band never collapses to zero width when `best` is tiny.
- First finite eval always publishes and seeds `best` (no baseline yet).
- `best` only decreases; updated on any publishing improvement (`improved`).
- A non-finite metric (broken eval) blocks defensively and leaves `best` intact.

The decision **persists** between eval iterations. The gate signal is only
computed every `--bearoff-eval-interval` iters; intermediate iterations reuse the
last decision (`GateState.last_published`). A later passing eval resumes
publication with the then-current weights.

### Why value MAE (not move accuracy)

- MAE is smooth and continuous — it tracks value-head quality monotonically, so a
  percentage tolerance is meaningful. Move-accuracy metrics (`policy_top1`,
  `nn_top1`) are coarse: they are argmax step-functions that can stay flat across
  a genuine value regression and jump noisily on a single position, making a
  tolerance band hard to calibrate.
- MAE measures exactly the feared failure mode: the value head drifting from
  ground truth during self-play (v11 saw corr 0.974→0.875 while loss fell). That
  drift is what poisons the buffer.
- It is already computed synchronously in-server at negligible cost.

A combined MAE + move-accuracy rule was considered but rejected for a single
tunable knob and smoothness. Move-accuracy stays logged to TB for human
oversight; promoting it to a secondary guard later is a one-line change in
`gate_evaluate`'s caller.

### Tolerance rationale

Self-play evals are noisy, so a strict "never worse than best" rule would block on
sampling noise and stall progress. 10% fractional tolerance absorbs eval noise
while still catching real regressions (a value head that has drifted enough to
poison the buffer moves MAE well beyond 10%). The 0.003 absolute floor keeps the
band usable once the net is near-perfect on bearoff (best MAE → small). Both are
CLI/const-tunable if v13 data suggests otherwise.

## Publication-point analysis (no torn weights/version pair)

Two publication surfaces, both now gated in `training_server.jl`:

1. **Served weights** — `update_weight_cache!(...)` (in `server.jl`, unchanged).
   It bumps `contact_version`/`race_version` and rewrites
   `contact_weight_bytes`/`race_weight_bytes`, all under `state.weight_lock`.
   Clients pull via `handle_weights` (`server.jl:225`), which reads *both* bytes
   and version **inside the same `state.weight_lock`**. So a client always gets a
   consistent `(bytes, version)` pair. On a gate hold we simply do NOT call
   `update_weight_cache!`, so both bytes and version remain at the last-good
   value — still a consistent pair, just an older one. No torn read is possible.

   (`handle_weights_version` reads `race_version` as a lock-free atomic; it is
   monotonic and only changes together with the bytes under the lock, so at worst
   a polling client re-fetches. Pre-existing, unaffected by the gate. Under a hold
   nothing changes at all.)

2. **`*_latest.data` files** — written in the checkpoint block. `*_latest.data`
   are the published/resume weights, so they are written only when
   `publish_this_iter`. `*_iter_N.data` are history and are ALWAYS written, as is
   `iter.txt`.

### Iteration ordering

The publication was moved to AFTER the fixed bearoff eval within the iteration so
the gate can act on the current iteration's eval before serving its weights.
Concretely: train → (deferred) → fixed bearoff eval computes `gate_evaluate` and
updates `GATE_STATE` → `publish_this_iter = !GATE_ENABLED || GATE_STATE.last_published`
→ conditional `update_weight_cache!` → checkpoint (gated `*_latest`, ungated
`*_iter_N`). Nothing between the old and new publication points depends on the
freshly published bytes (the eval uses the in-memory `race_network` directly), so
the move is behavior-preserving when the gate is disabled.

## `race_best.data` (rollback)

On every publishing improvement, `race_best.data` is saved alongside the regular
checkpoints, plus a `gate_state.json` sidecar. This is the best-by-gated-metric
race net, available for manual rollback if a run degrades.

## Resume

`gate_state.json` (best_metric, last_published, n_evals, n_blocked, metric_name,
tolerances) is written into the checkpoint dir on every improvement and every
checkpoint. On `--resume`, `GATE_STATE` is seeded from
`<resume_dir>/gate_state.json` via `load_gate_state`, which returns `nothing` and
starts the gate fresh if the file is absent or unparseable (graceful — a
pre-gate checkpoint just re-seeds `best` on its first post-resume eval).

## Escape hatches

- `--no-promotion-gate` — disable; publish every iteration unconditionally.
- The gate is auto-disabled (with a startup println) when the fixed bearoff eval
  is off, since that eval is the gate signal.
- `--gate-tolerance FRAC` — fractional tolerance (default 0.10).
- Startup banner prints enabled/disabled + metric + tolerance.
- TB scalars: `gate/published` (0/1), `gate/metric`, `gate/best_metric`,
  `gate/threshold`, `gate/n_blocked`. A block also emits a loud `@warn`.

## Limitations / flags for v13

- **Contact model is ungated.** `update_weight_cache!` bumps both versions
  atomically and lives in `server.jl` (which the task scoped out of edits), so the
  gate holds/releases contact and race *together*. In the current race-only
  pipeline (`--training-mode race`) the contact net is not meaningfully training,
  so this is harmless; on a dual run, contact publication would be coupled to the
  race gate. Independent contact gating needs a contact-side signal and a split of
  `update_weight_cache!` — deferred.
- **Gate is blind between evals.** With `--bearoff-eval-interval N`, up to N−1
  iterations publish (or hold) on a stale decision. If N is large relative to how
  fast the model can regress, a bad model could publish for several iterations
  before the next eval catches it. Keep N modest (default 10) or lower it if
  regressions are observed mid-window.
- **Bearoff-eval dependence.** The gate exists only when the bearoff eval + k=7
  table + race start positions are all available. No table → no gate.
- **Value MAE is a bearoff-region signal.** It does not measure pre-bearoff race
  strength (the known unmeasured plateau, review §"Also this session"). It gates
  against value-head *drift* well, but a regression that only hurts contact/early
  race play and leaves bearoff MAE intact would pass. Adding the periodic wildbg
  eval as a slower secondary gate is the natural next step.

## Verification

- `test/test_promotion_gate.jl`: 53/53 pass (first-eval-passes, improvement,
  within-tolerance pass, at-threshold inclusive, big-regression block, absolute
  floor, recovery-after-block, non-finite defensive, between-eval persistence,
  JSON round-trip, graceful fallback).
- Standalone smoke (`scratchpad/gate_smoke.jl`): realistic 10-eval sequence →
  8 publishes, 2 holds at the regression spike, resume after recovery, sidecar
  round-trip. Matches expectation.
- Full suite green in isolation (`test_backgammon_inference_regressions` 390/390,
  Promotion Gate 53/53, etc.). One threaded-consistency test flaked once under the
  concurrent eval sweep load and passed on isolated re-run — unrelated to the gate.
- `Meta.parseall` OK on `training_server.jl` and `promotion_gate.jl`.
