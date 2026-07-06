# External Review Fixes — 2026-07-03

> Superseded 2026-07-06: AlphaZero bearoff helpers now live in BackgammonNet
> v0.6.2+; legacy `games/backgammon/` and `scripts/bearoff_eval_common.jl`
> references are historical.

Five High findings from an external engineer review. All verified against the code
before acting; four fixed this session, one open.

## 1. Terminal bearoff moves scored as simple wins (FIXED)

**Bug**: `scripts/selfplay_client.jl` hard-coded `mover_val = 1.0` / `eq = [1,0,0,0,0]`
when a move terminated the game (bore off the last checker), in BOTH:
- `bearoff_post_dice_equity` (training-target path)
- `make_bearoff_evaluator` (MCTS evaluator path)

A terminal bearoff win is a **gammon (2.0)** whenever the opponent has 0 checkers
off — exactly the c15 states the k=7 table has gammon conditionals for. Non-terminal
moves in the same comparison loop DID get gammon values from the table, so the
comparison was inconsistent: MCTS/targets systematically preferred non-terminal
moves' correct values over terminal moves' understated ones, and training targets
for terminal moves lost the gammon signal.

**Fix**: derive value and 5-head target from `bg_copy.reward`, which BackgammonNet
sets at termination with the win multiplier (verified: `_compute_win_multiplier` →
`compute_game_reward` at `apply_single_move!`). Empirically confirmed: constructed
c15 position → terminal reward 2.0; opponent-1-off control → 1.0.

**Test**: `test/test_terminal_bearoff_rewards.jl` (14 assertions).

## 2. MCTS value scale inconsistency (FIXED)

**Bug**: three value sources mixed into the same MCTS Q totals on different scales:
- NN forward: equity/3 ∈ [-1,1] (`fc_resnet_multihead.jl`, `fast_weights.jl`)
- Game rewards: raw ±1/±2/±3 (`GI.white_reward`)
- Bearoff evaluator: raw points ∈ [-2,2]

Raw sources dominate 3× over NN values, distorting PUCT exploration and making
Q-value comparisons near terminal/bearoff nodes wrong.

**Fix — one convention**:
- **Buffer/training targets: raw points [-3,3]** (unchanged; `compute_td_errors`
  already normalizes for priority computation)
- **MCTS-internal: normalized [-1,1]**:
  - New `GI.reward_scale(::AbstractGameSpec) = 1.0` interface; backgammon
    overrides to 3.0 (`games/backgammon-deterministic/game.jl`)
  - `batched_mcts.jl`: `BatchedEnv.inv_reward_scale`, rewards multiplied at record time
  - `mcts.jl`: `Env.inv_reward_scale`, same treatment; `RolloutOracle` also normalized
  - `make_bearoff_evaluator` (selfplay_client.jl): all values normalized /3
    (matching the already-correct newer eval code in eval_bearoff_accuracy.jl
    and training_server.jl which use `normalized_points* = v/3`)

Other games are unaffected (default scale 1.0). Note: cpuct=2.0 was tuned under
the mixed-scale regime; near-bearoff search behavior changes with this fix.

## 3. Race mode trained on parts.all (FIXED)

`training_server.jl` race mode trained on `parts.all`. Harmless in v11/v12
(race-start bootstrap → buffer contact count 0, confirmed in TB) but any full-game
bootstrap (e.g. v9's bootstrap_wildbg_1M) contaminates the race net with contact
states. Now trains on `parts.race`, warns when contact samples exist, and logs
partition counts at bootstrap load.

## 4. Replay buffer torn reads (FIXED)

`extract_batch` and `partition_indices` were lock-free on the claim "HTTP thread
only appends" — but `per_add_batch!` OVERWRITES entries in place once the circular
buffer fills (3M buffers do fill). A concurrent read could see a torn sample
(state from old entry, policy from new). Both now hold the `ReentrantLock`; the
copy is O(batch) and cheap vs a training step.

Residual benign race: a sample can be fully swapped between partition selection
and extraction — yields a valid sample, possibly in the wrong partition batch.
Acceptable; a full fix would re-check flags under the extract lock.

## 5. No promotion gate (OPEN — design)

Server publishes `race_latest.data` every iteration unconditionally; clients sync
it for self-play. A regressed model immediately generates its own training data.

**Planned design** (next session):
- Gate publication on the fixed bearoff eval (fast, every iteration when enabled)
  + periodic race eval: publish only if metrics don't regress beyond confidence bounds
- Keep `race_best.data` (best-so-far by gated metric) for rollback
- Never stall silently: on gate failure, keep serving last-good weights and log loudly

## Also this session

- **v11/v12 TB analysis**: v11 value head drifted from exact-table truth during
  self-play (corr 0.974→0.875 while loss fell); v12's table anchoring fixed it
  (0.999). v12 play strength flat at wildbg parity → plateau is in pre-bearoff
  race positions, unmeasured by anything.
- **Ground-truth plan for pre-bearoff race positions**: one-sided rollouts with 3
  personalities (efficiency / gammon-go / gammon-save), rich signatures
  (T_all, T_first, n_off(k)), k=7 table truncation, analytic combine; validate vs
  small two-sided rollout sample. Match-play (MET) reuses the same cached
  primitives with different combine weights. See memory: race-ground-truth-rollouts.

## 6. Doubles mid-turn bear-off scoring (FIXED — found by the new validation harness)

**Bug**: with 2-checker action encoding, a doubles turn is TWO sequential actions
(`remaining_actions = 2`). After the first action the state is still the SAME
player's decision node — but every bear-off evaluator scored each post-move state
as "opponent pre-dice table value, negated". The negation flips the sign for the
first half of every doubles turn (1/6 of rolls), so argmax actively picked the
WORST first move. Affected: selfplay_client.jl (both training-target and MCTS
paths), eval_bearoff_accuracy.jl, training_server.jl fixed eval.

**How it was caught**: `scripts/eval_table_vs_wildbg.jl` plays a pure exact-table
policy vs wildbg from mutual-bearoff starts with duplicate-dice pairing (both side
assignments of a position share the dice seed, so the paired per-position sum
cancels the on-roll advantage). An exact policy can never significantly lose;
it measured **-0.0225 ± 0.0103** → bug. Post-fix: **+0.003 ± 0.0059** (parity;
+0.0025 ± 0.0035 on gammon-live starts). Wildbg plays near-perfect mutual bearoff,
so parity is the expected healthy state — significance of a NEGATIVE edge is the
regression signal.

**Fix**: turn-aware helpers in `scripts/bearoff_eval_common.jl`
(`bearoff_turn_value`, `bearoff_turn_value_equity`, `bearoff_best_move_value`):
terminal from `reward`, completed-turn chance nodes via table lookup with correct
perspective, mid-turn states via recursion over the remaining actions. All four
call sites now use them. Regression test: `test/test_bearoff_doubles_regression.jl`
(mid-turn recursion exercised without the 88GB table).

## Round-2 review findings (2026-07-03, all addressed)

1. **Value eval metrics scale-mismatched** — NN V (equity/3) was compared raw
   against wildbg points at eval_race.jl, training_server.jl, selfplay_client.jl
   value-comparison fns. Fixed: NN ×3 at all three producers (eval_manager.jl is
   just an aggregator). Historical value_mse/value_corr numbers in CLAUDE.md
   tables were computed with the mismatch — treat as relative-only.
2. **reanalyze_update! unlocked writes** — now holds the buffer lock (stale-NN
   blend into swapped samples remains possible in principle since NN outputs are
   computed pre-lock; full fix would need per-index epochs. Reanalyze is off in
   the current pipeline).
3. **Stale partition membership** — an index can flip race→contact between
   partitioning and extraction with a full buffer. Fixed: `_train_model_on_samples!`
   takes `expect_contact` and zeroes the loss weight of any extracted sample whose
   current flag mismatches.
4. **games/backgammon/game.jl reward_scale** — legacy wrapper now also overrides
   3.0. (gumbel_mcts/async_mcts are out of compilation; play.jl only serves
   scale-1.0 games in tests.)
5. **eval_table_vs_wildbg autodetect** — was already fixed before the review
   landed (auto-detect + explicit error; smoke-verified).

## Verification

- Full test suite: 608/608 pass post-fix (plus 39 new regression tests:
  terminal rewards 14, scale/buffer 14, doubles 11)
- Empirical gammon-multiplier check on constructed c15 positions: pass
- Table-vs-wildbg paired harness: -0.0225 (bug) → +0.003 (fixed), CI ±0.006

## Round 3 findings (2026-07-03)

Seven review findings, all verified against current code before acting, all fixed
this session. Full suite green afterward: 1799 assertions (Promotion Gate 71,
Scale/Buffer 24, Identity Staircase 1071, etc.).

### 1. HIGH — gate resume inconsistency (training_server.jl) — FIXED

**Verify**: confirmed. Resume loaded `race_latest.data` (gated/published weights)
with `START_ITER` from `iter.txt`. On a gate BLOCK `*_latest.data` stays at
last-good while `iter.txt` advances → resume silently loads stale weights under a
newer iter count.

**Fix**: TRAINING vs PUBLISHED state split on disk. Checkpoint block now
unconditionally writes `contact_train_latest.data` / `race_train_latest.data`
(lock-step with `iter.txt`) alongside the gated `*_latest.data`. `--resume` prefers
`*_train_latest.data`, falls back to `*_latest.data` for pre-gate sessions, prints
which was used. Gate-state seeding and `race_best.data` semantics unchanged.
Documented in a new "Resume semantics" section of the gate design note.

**Test**: exercised by the existing suite (no new unit — it is a file-IO/resume
path); logic reviewed. Notes updated.

### 2. MEDIUM — gate fails open on eval error (training_server.jl) — FIXED

**Verify**: confirmed. Catch block only `@warn`'d; `GATE_STATE` untouched, so the
gate reused `last_published` (init `true`) → a persistently broken eval publishes
forever.

**Fix**: new pure `gate_on_eval_error(state)` in `promotion_gate.jl` — "calibrated
fail-closed": finite `best_metric` (ever calibrated) → BLOCK; `Inf` best (cold
start) → publish (fail-open); both `@warn`. Added `n_eval_failures` consecutive
counter to `GateState` (reset by any finite eval; incremented on exception AND
non-finite metric), logged to TB `gate/n_eval_failures`. Catch block wired via a
`gate_updated_this_eval` flag so a post-decision throw can't double-apply.

**Test**: `test/test_promotion_gate.jl` +3 testsets (cold-start fail-open,
post-calibration fail-closed + streak reset, non-finite counts as failure) +
sidecar round-trip of the new field. Pure functions, no server dependency.

### 3. HIGH-if-enabled — reanalyze generation check (buffer.jl + server) — FIXED

**Verify**: confirmed dormant (reanalyze off in current launches) but real:
`reanalyze_update!` writes blends by index after slow unlocked NN inference; a
slot overwritten in between gets a stale prediction.

**Fix**: added `generation::Vector{UInt32}` to `PERBuffer` (zeros init),
incremented per slot in `per_add_batch!`. `extract_batch` now returns
`generations` (NamedTuple — backward compatible; verified all callers destructure
by name). `reanalyze_update!` takes `expected_generations`, skips any index whose
current generation differs under the lock, returns the skip count. Server passes
`col_data.generations[sub_local_idx]` through and `@info`-logs total skipped.

**Test**: `test/test_scale_and_buffer_regressions.jl` new testset — write, extract
(snapshot gens), wrap the circular buffer to overwrite slots 1-2, `reanalyze_update!`
with stale gens → asserts skip count == 2, overwritten slots untouched, matching
slots blended.

### 4. MEDIUM — eval_vs_wildbg.jl value-metric scale — FIXED

**Verify**: confirmed. `value_oracle_fn` stored NN V (equity/3 ∈ [-1,1]) raw
against wildbg points ∈ [-3,3] at eval_game (~line 220); `compute_value_stats`
(~185) then computed MSE/MAE on the mismatch.

**Fix**: scaled NN `× 3.0` at the single producer site (line 223) with the same
comment as eval_race.jl. Verified this is the ONLY NN-value producer — all
downstream (`nn_val` samples, `compute_value_stats`) flow from it.

**Test**: covered by existing eval integration; scale change is a one-liner mirror
of the already-tested eval_race.jl fix.

### 5. MEDIUM/LOW — zero-weight batch NaN path (training_server.jl) — FIXED

**Verify**: confirmed. Stale-partition guard zeroes mismatched IS weights; if ALL
masked, `Wmean = mean(W) = 0` and `losses()` divides by it → NaN.

**Fix**: after the mask, if `!any(!iszero, is_weights)` → `continue` (skip batch)
with an `n_masked_skipped` counter and occasional `@warn` (1st + every 100th).
Loss denominators use `max(1, num_batches - n_masked_skipped)`. Normal path (no
partition mismatch) never enters the branch, so it stays alloc-free.

**Test**: logic reviewed; guarded branch only reachable with a full wrapped buffer
under partition churn (hard to force deterministically in-unit). Existing partition
tests still green.

### 6. LOW — per_sample_partition priorities race (buffer.jl) — FIXED

**Verify**: confirmed. Priority cumsum read `buf.priorities` lock-free while
`per_update_priorities!`/`per_add_batch!` mutate them.

**Fix**: wrapped ONLY the priority-snapshot cumsum loop in `lock(buf.lock)`; the
sampling math (binary search, IS weights) stays outside on the local snapshot.
`buf.lock` is a `ReentrantLock` and callers (`_train_model_on_samples!`) do not
hold it, so no deadlock; no long compute moved under the lock.

**Test**: covered by existing buffer regression suite (green).

### 7. LOW — eval_table_vs_wildbg.jl close! typo — FIXED

**Verify**: confirmed. `BackgammonNet.close!` doesn't exist; the wildbg backend
provides `Base.close(::WildbgBackend)` (frees handle + dlclose, wildbg.jl:246).
The old `try ... catch end` swallowed the MethodError → each backend's native
handle leaked.

**Fix**: call `close(wb)` with a targeted `try/catch e … @warn` (no swallow-all).
Single surgical edit.

**Test**: script-level (not in unit suite); verified the target function exists in
`~/github/BackgammonNet.jl/src/wildbg.jl`.
