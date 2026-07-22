# Backgammon Integration Status

Last reset: 2026-07-16

This file is the single active status, conclusion, and TODO record for the
AlphaZero/BackgammonNet integration. Dated files under `notes/` and the archived
report addendum are historical evidence only.

## 2026-07-22 — API consumption migration

AlphaZero was migrated to consume BackgammonNet's finalized API surface. Code is
treated as final; BackgammonNet is still completing artifact verification, so the
frozen verified-release pin and preflight remain open TODOs below.

Consumed changes (all landed, full test suite green except the deferred item):

- **Bearoff tiers.** The removed n18 one-sided table and
  `default_bearoff_onesided_dir()` were replaced by the k7 → n15 `CombinedBearoff`
  tier (`BearoffOneSidedCompact.CompactBundle`, `default_bearoff_n15_dir()`,
  `load_combined_bearoff`). Updated in `selfplay_client.jl`,
  `verify_race_supervised.jl`, `verify_race_mcts.jl`, and AGENTS.md.
- **Cubeless money `:auto`.** `search_value(g, heads; mode=:auto)` is now cubeless
  for money play even with the cube enabled. The 676-head wrapper keeps the cube
  disabled, so the oracle value is the cubeless money equity; the inference
  regression test was updated to assert this contract.
- **Training artifact `v2` → `v4`.** BackgammonNet now emits/requires
  `backgammon_training_v4`, which stores RAW checker action equities
  (`checker_action_ids` / `checker_action_equities`) and forbids generation-time
  policy-shaping metadata. AlphaZero consumes bootstrap artifacts only through
  `load_training_artifact` + `fill_training_batch!`, so production needed only a
  field rename in `validate_training_data.jl` (`policy_actions` →
  `checker_action_ids`) and doc-string updates. The hand-built artifact
  integration test (`test/test_backgammon_training_artifact.jl`) was **skipped**
  rather than reproducing the v4 contract — see the TODO below.
- **Equity-formula de-duplication.** The joint money-equity formula is
  consolidated onto one vectorized `FluxLib.compute_equity` kernel (the scalar
  `EquityOutput` method now delegates to it), locked to
  `BackgammonNet.compute_equity_joint` by the existing equivalence test in
  `test_multihead.jl`. `compute_equity_joint` is scalar-only and cannot flow
  through Flux batched autodiff, so one local vectorized mirror is retained by
  design. `memory.jl`'s `equity_targets_from_outcome` was reviewed and kept: it
  adapts AlphaZero's own `GI.GameOutcome` (for which BackgammonNet exposes no
  equivalent) to the value-head order it already sources from BackgammonNet.

## Evidence policy

Every claim must be assigned one of these states:

- **Established:** covered by current code tests or a versioned public
  BackgammonNet contract and reproducible without historical generated data.
- **Provisional:** plausible and useful for planning, but requires a fresh run
  from validated inputs before it can guide model selection.
- **Quarantined:** depends on questioned GNUBG labels/equities, an old bridge,
  an unversioned dataset, or a mutable evaluation set. It cannot be used for
  training, promotion, or strength claims.

An old result does not become established merely because a new run produces a
similar number. The new run must satisfy the acceptance criteria below.

## Established implementation facts

- AlphaZero consumes BackgammonNet as a package dependency and requires version
  0.7.x on Julia 1.12.6+.
- The wrapper uses BackgammonNet's public value-head probability contract and
  side-to-move equity convention.
- The ML wrapper exposes 21 dice outcomes at every dice chance node. Initial
  player selection is game setup, not an MCTS chance node.
- The current checker policy head has 676 actions. Cube actions remain outside
  this network wrapper, and cube play must remain disabled.
- The distributed protocol fingerprints the observation, value, action,
  chance, and rule contract; mismatched clients are rejected.
- Sample upload is idempotent, replay samples retain source iteration, and
  transactional checkpoints checksum model, optimizer, RNG, publication, and
  optional replay-buffer state.
- Preflight, immutable evaluation manifests, numerical-safety rejection,
  health/drain behavior, and curated TensorBoard metrics have direct tests.
- AlphaZero delegates native engine bridges, capability checks, artifact
  validation, and bearoff lookup to public BackgammonNet APIs.

These facts establish plumbing and invariants. They do **not** establish label
quality, model quality, PR, equity accuracy, or playing strength.

## Quarantined conclusions and artifacts

Until revalidated, the following are not current conclusions:

- All GNUBG-derived label-quality, equity, move-regret, PR, floor, agreement,
  and teacher-strength numbers recorded before this reset.
- Claims that a historical model reached GNUBG parity, beat WildBG, or had a
  particular win rate, PR, correlation, error rate, or backgammon frequency.
- Conclusions about MCTS benefit, policy/value plateaus, DAgger improvements,
  teacher calibration, or architecture quality when their reference labels or
  scoring path used the questioned GNUBG data.
- Historical checkpoints trained or fine-tuned with those labels. They may be
  retained for forensic comparison but are ineligible as bootstrap weights,
  promotion candidates, or named baselines.
- Historical training/evaluation corpora without a current BackgammonNet
  artifact manifest, contract fingerprint, provenance, and validation report.

WildBG-, BGBlitz-, outcome-, and bearoff-based conclusions are also provisional
when they shared an old evaluation harness or equity conversion. They must be
rerun even if their raw source was not GNUBG.

## BackgammonNet fixes that define the reset boundary

The implementation fixes below are established in BackgammonNet history. Their
existence does not rehabilitate artifacts or measurements produced before them.
Every affected artifact must be regenerated and independently audited.

| Area | Fix | BackgammonNet commits | Disposition |
|---|---|---|---|
| Canonical value heads | Full-game values had been read from GNUBG cubeful move-record heads instead of the cubeless turn-boundary label path; GNUBG evaluation context also forced cubeful lookahead for callers that requested cubeless heads. | `be738f3`, `5fad31f` | Quarantine old GNUBG full-game value labels and every model/metric derived from them. |
| Context-independent values | Real match/cube context leaked into canonical heads, with observed deviations up to 0.133 in DMP and 0.059 in match positions, allowing match-aware continuation values to be MET-converted a second time during search. Canonical labels are now evaluated in a normalized money context; real-context labels are separate auxiliaries. | `5d4de2d` | Quarantine old match, Crawford, post-Crawford, and DMP canonical value labels. Require `money_normalized_v1` or `exact_table` context metadata. |
| Checker policy density | Top-5 serialization discarded as much as 68% of teacher probability mass on measured hard positions. Producers now store the full legal-action softmax at the declared temperature. | `4c37f49` | Quarantine truncated checker-policy artifacts, even when the retained probabilities were renormalized. Require `full_legal_softmax_v1`. |
| Cube response | Perspective handling produced wrong-signed cube-response labels. | `bc5837c`, `61f0b63` | Quarantine old take/pass and cube-response labels and cube metrics. The current 676-head AlphaZero wrapper does not consume them, but full artifacts may still contain them. |
| Bearoff perspective | Bearoff recursion negated a scalar instead of flipping the probability distribution into the mover frame. Money weights were unaffected; non-money personalities could return materially wrong values. | `ade9af8` | Revalidate non-money bearoff conclusions. Money-race artifacts are not invalidated by this fix alone, but still need the current independent race audit. |
| Artifact loading | The ML loader previously accepted stale encodings, inconsistent lengths, invalid action types, and unnormalized or truncated policies. It now fails closed. | `59ac9be`, `4354899` | AlphaZero accepts only the current `BackgammonNet.TRAINING_ARTIFACT_SCHEMA` (now `backgammon_training_v4`, raw checker action equities) through `BackgammonNet.load_training_artifact`; legacy converted samples and older schemas are unsupported. |
| Audit independence | Value and race audits reused the producer calculations and could reproduce the same bug. They now distinguish replay reproducibility from independent label-contract validation and use an independent race oracle. | `774f0e1`, `3c42bb8` | Every old “audit passed” conclusion is quarantined. Require the current strict/live BackgammonNet audit and v2 release manifest. |
| Validation reports | BGBlitz match-cube results used the wrong MWC/EMG scale; corpus reports could route cube rows into checker statistics. | `bf65e8b`, `69c5cad` | Labels are not invalidated solely by these fixes, but old BGBlitz match-cube and corpus-summary conclusions are invalid. |

The hash written in a historical artifact is evidence of which code produced
it, not proof that the code was correct. AlphaZero does not maintain a second
copy of these validation rules; BackgammonNet's strict audit and release
manifest remain authoritative.

## Current conclusions

1. The AlphaZero integration and distributed safety mechanisms are ready for
   compatibility validation against a frozen BackgammonNet 0.7.x revision.
2. No current training-data family is approved solely because it was used by a
   previous successful run.
3. No engine-relative strength claim is currently established.
4. Exact bearoff data remains the preferred independent race reference, but
   each generated artifact must pass the current BackgammonNet validator and
   doubles/turn-boundary checks before use.
5. New training must wait until the data and evaluation gates below are green.

## Current validation snapshot

### 2026-07-22 — Jarvis + Neo compatibility (post-migration, BackgammonNet HEAD, NOT frozen)

After the API consumption migration, **both machines are green and contract-matched**:

- **Full test suite green on both** (0 failures; the v4 artifact integration test
  is intentionally skipped) — Jarvis (x86/CUDA) and Neo (M3/ARM, CPU,
  `--gcthreads=1`).
- **No-training server preflight passed all 11 checks on both machines.**
- Identical ML contract fingerprint on both:
  `8c3cc18431da9f58718025fc65f7f0c8c3a2988e9e66624eb6bca3486339a84c`
  (config fingerprint differs by machine, as expected).
- Julia `1.12.6`; AlphaZero clean; BackgammonNet `0.7.0` at commit
  `456beced8b9457e68307f0b6d3894e887dea7685` ("Validate relabel contract in
  source lineage"), clean on both. The contract fingerprint is stable across the
  `18e1718 → 456beced` validator commit (verified by re-running preflight), i.e.
  that commit is contract-neutral.
- state dimension **366** (was 352 — observation encoding changed in the
  BackgammonNet update), checker actions 676, chance outcomes 21.

This is a compatibility check against **current BackgammonNet HEAD, which is NOT
the frozen verified release** (BackgammonNet is still in final artifact
verification). It establishes plumbing/invariants and cross-machine client
compatibility only — no artifact quality or model strength. Re-pin the frozen
release commit + fingerprint and re-run preflight on both machines once the
release lands (P1 TODO). The prior snapshot below is historical.

> **SUPERSEDED (2026-07-16 snapshot).** Pins BackgammonNet `0.7.0` at commit
> `65eb189…` with state dim 352 and fingerprint `a6e6cf10…` — both changed in the
> 2026-07-22 migration (see above). Kept for history only.

On Jarvis, the AlphaZero working tree passed 3,377 assertions on 2026-07-16.
The no-training server preflight passed all 11 checks with:

- Julia `1.12.6`;
- BackgammonNet `0.7.0` at clean commit
  `65eb189d57824baabf64eed83a508619c462c0e3`;
- ML contract fingerprint
  `a6e6cf10c0397757fbc6f463b3be86f098d8762ceb9da5591e0351741a1a25d6`;
- state dimension 352, checker actions 676, chance outcomes 21, and distributed
  protocol version 4.

This was a compatibility check of an uncommitted AlphaZero working tree. It ran
without bootstrap, start-position, or evaluation artifacts and without a
bearoff table. It establishes neither artifact quality nor model strength. Neo
preflight remains pending.

## Supported operational surface

The supported distributed stack is `training_server.jl`,
`selfplay_client.jl`, and `src/distributed/`. Supported data/evaluation helpers
are limited to immutable-manifest construction, fail-closed artifact/corpus
validation, exact-race verification, and implementation/performance tests.

Old GNUBG label builders, PR ladders, afterstate/DAgger fine-tuning tools,
teacher-calibration scripts, and standalone WildBG/GNUBG match harnesses were
removed. They used private bridge internals, permissive historical formats, or
the quarantined methodology. Git history is their code archive; they must not
be restored piecemeal. Any replacement must use BackgammonNet's public backend
and artifact APIs plus the acceptance criteria in this document.

## Revalidation TODO

### P0 — establish trusted inputs

- [ ] Record the frozen BackgammonNet version, commit, dirty state, native
  backend configuration, and value-head contract in every validation report.
- [ ] Inventory every candidate train/eval artifact by checksum, generator,
  teacher/backend, backend settings, perspective, value-head order, rules,
  observation encoding, and generation commit.
- [ ] Mark pre-reset GNUBG-derived artifacts `quarantined`; do not silently
  relabel or overwrite them.
- [ ] Require each candidate block to pass BackgammonNet's current strict audit,
  the applicable live or independent-label lane, and inclusion in a verified
  `backgammon_artifact_manifest_v2` release.
- [ ] Regenerate a small GNUBG calibration corpus through the current public
  BackgammonNet artifact API.
- [ ] Validate probability bounds/order, equity reconstruction, mover and
  side-to-move perspective, player 0/player 1 symmetry, legal-action mapping,
  doubles continuation, forced pass, terminal rewards, and cube-disabled rules.
- [ ] Compare a stratified sample against at least one independent reference
  path where supported (exact bearoff, WildBG, or BGBlitz). Disagreement is a
  diagnostic, not an automatic majority vote.

### P0 — rebuild evaluation authority

- [ ] Build immutable, checksummed evaluation artifacts with disjoint seeds and
  explicit contact/race/dice/rule strata.
- [ ] Validate the evaluation artifact and manifest before any benchmark.
- [ ] Re-run engine self-consistency/floor tests with both players, doubles,
  forced moves, and board-result matching represented explicitly.
- [ ] Define primary metrics before looking at results. Report sample count,
  confidence intervals, invalid/unmatched counts, and phase splits alongside
  every aggregate.

### P1 — compatibility and operations

- [ ] Run the AlphaZero server preflight against the frozen BackgammonNet
  revision on Jarvis and Neo without starting training. Record the frozen
  release commit + contract fingerprint and restate the "Current validation
  snapshot" above (which is superseded as of the 2026-07-22 migration).
- [ ] Re-enable `test/test_backgammon_training_artifact.jl` against a REAL
  BackgammonNet-produced `backgammon_training_v4` artifact (or a public
  BackgammonNet v4 fixture builder). It is currently `@test_skip`ped rather than
  hand-reproducing the v4 teacher/action-equity contract; trust the
  BackgammonNet release manifest until a real artifact is available.
- [ ] Run protocol, reconnect/idempotency, checkpoint/resume, drain, and
  evaluation-chunk fault tests on both architectures.
- [ ] Capture CPU/GPU inference throughput, queue pressure, cache hit rates, and
  evaluation throughput from the curated metric set.

### P1 — only after both P0 gates pass

- [ ] Generate fresh bootstrap corpora with new artifact identities.
- [ ] Train small smoke models first; do not reuse quarantined weights.
- [ ] Re-establish race value/policy accuracy against validated exact tables.
- [ ] Re-establish contact label calibration and MCTS benefit on the immutable
  evaluation set.
- [ ] Promote a result to `Established` only after a second reproducible run or
  an independent reference agrees within the declared tolerance.

## Acceptance criteria for a trustworthy result

A reported number must include:

- AlphaZero and BackgammonNet commits and clean/dirty state;
- immutable input/output artifact checksums and contract fingerprint;
- backend name, role, quality, settings, and machine-local configuration;
- rules, perspective, value-head order, action/chance contract, seeds, and
  contact/race/dice strata;
- sample counts, exclusions, failures/unmatched cases, uncertainty, and the
  exact metric implementation;
- a machine-readable report and enough configuration to reproduce it.

If any required field is missing, the result is provisional. If it uses a
quarantined artifact, it remains quarantined regardless of apparent quality.
