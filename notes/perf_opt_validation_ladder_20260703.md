# Perf-Optimization Validation Ladder — 2026-07-03

Re-ran the full validation ladder after the day's core performance
optimizations to confirm they introduced **no correctness regressions**.
Verdict: **ladder green, perf opts are regression-clean.** One regression was
found and fixed — but it was in the *ladder harness itself*, not in a shipped
training/eval path.

## Commits under test (today, perf/correctness)

| commit | change | risk class |
|--------|--------|-----------|
| `f18c1d2` | GEMM: FMA via muladd + 4-step k-unroll | numerical |
| `293cd32` | scratch backgammon action lookup | aliasing/threads |
| `3cf2df4` | avoid env rebuilds for state actions | staleness/aliasing |
| `498f66e` | batched-MCTS oracle packing | indexing/perspective |
| `c8322cc` | Xoshiro self-play RNG | reproducibility/races |
| `19abe68` | explicit Xoshiro streams for MCTS | reproducibility/races |
| `11c46b2` | eval oracle setup + buffer reset | eval-weight crossing / partial reset |
| `639ed90` | strict distributed sample schema | field drop / silent default |
| `6f9e12e` | FastInference regression tests | (tests) |

## Correctness review — all four dimensions CLEAN

Four independent focused reviews (one per risk cluster):

1. **GEMM (`f18c1d2`)** — clean. Kernel verified against a Float64 reference
   across 2,160 shapes (max rel err 9.3e-7). The 4-step unroll's cleanup loop
   covers non-multiple-of-4 `k` exactly (traced k∈{1,3,5,7,63,65,67,…}); FMA
   only reassociates within each `muladd`, no terms dropped/duplicated; no new
   shared state. The 28-min hang seen once was diagnosed as OpenBLAS
   threadpool oversubscription in the test's *reference* path (`Float64.*` /
   Flux Dense), not the kernel — it did not reproduce under `Pkg.test`.
2. **Action caching (`293cd32`,`3cf2df4`)** — clean. The per-thread scratch
   alias never escapes: both callers (`available_actions`, `actions_mask`)
   consume it synchronously into freshly-owned containers with **no yield point**
   between grab and consume, so task migration cannot tear it. `legal_actions!`
   doesn't mutate game state (no staleness); cloned envs stay independent.
   LOW: scratch pool sized by `maxthreadid()` at include time → a later-adopted
   thread would `BoundsError` (loud crash, never miscompute); can't happen in
   the pure-`@spawn` path today.
3. **Oracle packing + RNG (`498f66e`,`c8322cc`,`19abe68`)** — clean. Packing
   preserves index/perspective/value semantics exactly (input `idx` ↔ output
   slot, column `j` ↔ net output + action list); action-aware path pairs
   `pv[k]=P[actions[k]]` with `leaf_actions[k]` by construction. Each worker
   gets an independent, deterministic Xoshiro stream (`MAIN_SEED + w*104729` →
   per-game sub-stream); chance/Dirichlet/temperature all draw from `env.rng`;
   no hot-loop reseeding. LOW: unchecked OOB `P[action,j]` read (invariant
   prevents it); eval `@threads` per-thread-agent pattern safe only while the
   CPU oracle doesn't yield; legacy `play.jl` not plumbed (inactive path).
4. **Buffer reset + strict schema (`11c46b2`,`639ed90`)** — clean, and these
   *fixed* real bugs: flux-eval oracle crash (`primary_net===nothing`), an
   orphaned heartbeat task, a gammon-**loss** equity-target mislabel in
   `bearoff_turn_value_equity`, and `is_chance` being silently hard-coded
   `false` in the buffer (chance samples would have received policy loss).
   `PERBuffer.reset!` resets priorities/flags/generations under lock — strictly
   more correct than the old inline reset. Schema fails **loudly** on a missing
   field. MED operational note: strict schema makes resuming a *pre-schema*
   training session a hard error (intentional; matters for v13 resume).

## Ladder rungs

- **Rung 0 — full unit suite (`Pkg.test`): GREEN.** 10/10 testsets, incl.
  MCTS Identity Staircase 1071/1071 (search exact to machine epsilon — the
  direct guard on action-lookup/oracle-packing/GEMM regressions), Fast
  Inference 311/311, Scale & Buffer 47/47, Bearoff Doubles 13/13.
- **Rung 1 — exact-table policy vs wildbg (game code): GREEN, no regression.**
  `eval_table_vs_wildbg.jl --policy=table`, 1000 paired positions.
  Mutual-bearoff money: **+0.003 ± 0.006, 50.1% win**. Gammon-starts:
  **+0.002 ± 0.0028**, gammon-rate 16.1%. Parity is the documented post-
  doubles-fix reality (wildbg is near-optimal in pure bearoff); we are nowhere
  near the **−0.022 ± 0.010 loss** that the pre-fix doubles bug produced — the
  signature this harness is sensitive to.
- **Rung 2 — MCTS+exact-evaluator vs wildbg (oracle packing + MCTS): GREEN,
  matches pre-perf-opt baseline.** `--policy=mcts --mcts-iters=400`, 200 paired
  positions. Paired edge **+0.0125 ± 0.044**; move-agreement vs pure table
  **86.95%**, mean Δ 0.0027 raw pts — statistically identical to the pre-perf-
  opt baseline (87.7%, mean Δ 0.0037, +0.010 ± 0.014 from
  `mcts_objective_validation_20260703.md`). The 87 non-tie disagreements are
  finite-budget visit-argmax noise on near-ties, not a wiring bug (large-Δ +
  negative-edge is the bug signature; we see tiny-Δ + parity).

## Regression found and fixed

`19abe68` routed `rng=` into every agent's `create_player`, but the local
`TableAgent.create_player` in `eval_table_vs_wildbg.jl` still had the old
zero-kwarg signature → the ladder harness crashed on game 1
("unsupported keyword argument rng"). Fixed in `2597cb4` (accept & ignore
`rng`; `TableAgent` has no player object). No shipped training/eval path was
affected — only this harness.

## Not re-run (redundant)

The full MCTS convergence sweep (1600-iter tail) was not re-run: the tail is
already covered by the identity-staircase unit test (exact-to-epsilon), and the
400-iter integration point matches baseline. The error-response curves are a
measurement, not a pass/fail regression gate.

## Blocked

Longer-race validation rung (18-point race starts vs wildbg) is blocked on
Neo's `n=18` one-sided table build — 21% done at 16:35, ETA ~16h.

## Deferred improvements (post-green, each its own validated commit)

Much of the proposed idea-1/2/3 *correctness* content was already delivered by
today's commits (see review #4). Remaining, in ROI order:
1. **Wire progressive sim budgets** (`ProgressiveSimParams`/
   `TurnProgressiveSimParams` exist but the client still uses constant
   `MCTS_ITERS`) — highest ROI; training-behavior change, must be ladder-
   validated after wiring.
2. Eval-path dispatch isolation / dedup (cleanup, not a correctness gap).
3. `PERBuffer` two-phase-mark polish + documented invariants.
4. Bearoff Float32/64 unification + expanded regression tests.
5. LOW hardening: `maxthreadid()` scratch-pool guard; OOB action-index nit.
