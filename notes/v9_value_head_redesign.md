# v9 Value Head Redesign: Conditional → Joint (Logit-Space)

**Date:** 2026-03-24
**Status:** Validated, needs implementation plan review before coding
**CRITICAL UPDATE:** BGBlitz and Wildbg both send JOINT values. See "BGBlitz Format Discovery" section.

## Executive Summary

Switch the 5-head equity network from **conditional probabilities with masking** to **joint probabilities with BCEWithLogits**. This aligns with GnuBG/TD-Gammon/XtremeGammon, doubles effective training data for gammon heads, simplifies the loss function, and is numerically validated as lossless.

## Background: The 5 Value Heads

Backgammon has three win types: single (1pt), gammon (2pt), backgammon (3pt). The network predicts 5 probabilities that together determine expected equity.

### Current System (Conditional, v1–v8)

| Head | Meaning | Training |
|------|---------|----------|
| p_win | P(win) | All samples |
| p_gammon_win | P(gammon \| win) | **Only won samples** (masked on losses) |
| p_bg_win | P(backgammon \| win) | **Only won samples** |
| p_gammon_loss | P(gammon \| loss) | **Only lost samples** (masked on wins) |
| p_bg_loss | P(backgammon \| loss) | **Only lost samples** |

**Equity formula:** `equity = p_win * (1 + gw + bgw) - (1 - p_win) * (1 + gl + bgl)`

**Problem:** When a game is lost, P(gammon|win) is *undefined* (not zero). Storing 0.0 and masking it out wastes 50% of training signal for each gammon head. Self-play data is expensive to generate — this is a significant efficiency loss.

### Proposed System (Joint, v9)

| Head | Meaning | Training |
|------|---------|----------|
| p_win | P(win) | All samples |
| p_wg | P(win ∧ gammon+) — cumulative, includes bg | **All samples** |
| p_wbg | P(win ∧ backgammon) | **All samples** |
| p_lg | P(lose ∧ gammon+) — cumulative, includes bg | **All samples** |
| p_lbg | P(lose ∧ backgammon) | **All samples** |

**Equity formula (GnuBG-style):** `equity = (2*p_win - 1) + (p_wg - p_lg) + (p_wbg - p_lbg)`

**Key insight:** With joint probabilities, every sample has a valid target for every head. A loss means P(win∧gammon) = 0, which is a *true statement* the network should learn — not an undefined value to mask out.

## Who Uses What

| Engine | Format | Evidence |
|--------|--------|----------|
| **GnuBG** | Joint cumulative | Source code: `OUTPUT_WINGAMMON` includes bg |
| **TD-Gammon** | Joint cumulative | Seminal paper |
| **XtremeGammon** | Joint cumulative (believed) | Commercial SOTA |
| **BGBlitz** | **Joint cumulative** | JAR decompilation + live test (2026-03-24) |
| **Wildbg** | **Joint cumulative** | C struct `CProbabilities`; `_equity_from_probs` uses joint formula |
| **Our bearoff table (c15)** | Conditional | Confirmed via Bellman validation |
| **Us (v1-v8 training)** | Conditional masking + mismatched bootstrap | Latent bug |

**Every external engine uses joint cumulative.** Our bearoff table is the sole conditional
source. The `conditional_v1` contract label applied to the wrong data.

## BGBlitz and Wildbg Both Use Joint

BGBlitz's internal `Equity` class (decompiled via jadx) stores cumulative joint fields:
`myWins`, `myGammon` (gammon+backgammon), `myBackGammon`. Its bridge exposes these raw
values via `getGammon()`, not the derived conditional `getGammonRate()`. The UI may display
conditional rates, but the programmatic API returns joint.

Wildbg's C FFI struct `CProbabilities` has `win_g` (cumulative G+BG) and `win_bg`. Its
equity function `_equity_from_probs` simplifies to the joint formula
`(2pw-1) + (wg-lg) + (wbg-lbg)`.

Our earlier assumption that both used conditional was based on misleading variable names
in the bridge code (`pGamWin` suggesting "P(gammon|win)") and the `conditional_v1` contract
label. The live scalar parity test definitively proved joint.

## The Logit-Space Addition

Joint gammon heads have smaller absolute values than conditional (e.g., P(win∧gammon) ≈ 0.08 vs P(gammon|win) ≈ 0.15). To prevent vanishing gradients near zero:

**Architecture change:** Remove sigmoid from the 4 gammon heads. Output raw logits. Use BCEWithLogits loss (which uses the log-sum-exp trick for numerical stability).

```
Current:  trunk → Dense(width, 1) → sigmoid → BCE(sigmoid_output, target)
New:      trunk → Dense(width, 1) → raw logit → BCEWithLogits(logit, target)
```

At inference only: apply sigmoid to get probabilities, then compute equity.

This is standard practice in modern deep learning (PyTorch's default) and eliminates the small-value tradeoff entirely.

## Data Pipeline Changes

### Self-Play Targets (Simpler — no masking)

```
Won by gammon:    [1.0, 1.0, 0.0, 0.0, 0.0]  →  equity = +2
Won single:       [1.0, 0.0, 0.0, 0.0, 0.0]  →  equity = +1
Lost by bg:       [0.0, 0.0, 0.0, 1.0, 1.0]  →  equity = -3
Lost single:      [0.0, 0.0, 0.0, 0.0, 0.0]  →  equity = -1
```

**The vectors are numerically identical to current conditional targets.** The difference is that zeros are now valid targets (trained on), not undefined placeholders (masked out).

### Bootstrap

**The specific production artifact** (`bootstrap_race_samples.jls`, 2.37M samples built
from 2-ply BGBlitz batches) contains joint values — no conversion needed for v9.

**However**, the checked-in generator (`BackgammonNet.jl/scripts/generate_bootstrap.jl`)
and audit tool (`audit_bootstrap.jl`) still reference the `conditional_v1` contract and
use the conditional scalar parity formula. These must be updated as part of this migration:

1. Update `generate_bootstrap.jl`: change contract to `joint_cumulative_v1`, use joint
   parity formula `scalar ≈ (2pw-1) + (gw-gl) + (bgw-bgl)`
2. Update `audit_bootstrap.jl`: same contract and parity formula change
3. Update `bootstrap_value_head_contract.md`: document the correct joint semantics

**Two distinct contracts exist in the system:**
- **NN + bootstrap + external engines:** `joint_cumulative_v1` — "gammon" means gammon+
- **k=7 bearoff table:** `joint_noncumulative_bearoff_v1` — "gammon" means gammon-only
  (numerically identical in bearoff since bg=0, but semantically distinct)

**For any FUTURE bootstrap artifacts**, the generator already produces correct values
(bridge sends joint). Only the metadata labels and parity check formula are wrong.

```julia
# Production artifact values are ALREADY:
# [P(win), P(gammon+ wins), P(bg wins), P(gammon+ losses), P(bg losses)]
# Use as-is with joint equity formula
```

### Bear-off Table

**Current k=6 table** stores conditional values. For v9, use `to_absolute()` at lookup
sites to get joint values. This is a temporary bridge.

**New k=7 table** (in progress) will store joint non-cumulative natively:
- `pW` = P(win)
- `pWG` = P(win ∧ gammon) — gammon-only, not cumulative (bg=0 in bearoff)
- `pLG` = P(lose ∧ gammon) — gammon-only, not cumulative

No conversion needed at lookup sites. Drops into v9 as a native joint source. For the
equity formula, `pWG` works directly since `pWG_cumulative = pWG + 0 = pWG` in bearoff.

**Why non-cumulative:** Values are identical to cumulative in bearoff (bg=0). But labeling
as "cumulative gammon-or-better" would add a semantic claim not verifiable from the
artifact — the same class of error as the `conditional_v1` mislabeling.

k=7 table specs: 170,544 single-side positions, ~95.2 GB disk, u16 probabilities,
pre-roll only (post-roll computed on-the-fly via move enumeration).

### FastWeights Inference (Fewer FLOPs)

```julia
# Current conditional: 4 multiplications + 4 additions
equity = p_win * (1 + p_gw + p_bgw) - p_loss * (1 + p_gl + p_bgl)

# New joint: 3 additions + 3 subtractions
equity = (2*p_win - 1) + (p_wg - p_lg) + (p_wbg - p_lbg)
```

### Loss Function (Delete masking code)

```julia
# Current: complex masking per head
W_equity, W_win, W_loss = _equity_head_weights(W, EqWin, HasEquity)
Lv_gw = bce_wmean(V̂_gw, EqGW, W_win)     # only won samples
Lv_gl = bce_wmean(V̂_gl, EqGL, W_loss)     # only lost samples

# New: simple BCE on all heads, all samples
W_eq = W .* HasEquity
Lv_wg  = bce_logits_wmean(logit_wg,  target_wg,  W_eq)   # all samples
Lv_lg  = bce_logits_wmean(logit_lg,  target_lg,  W_eq)   # all samples
```

`_equity_head_weights()` is deleted entirely.

## Numerical Validation

### Tests Written and Passing (test/test_value_head_formats.jl)

12 test suites, 6,315 individual assertions, all passing:

| # | Test | Result | Key Finding |
|---|------|--------|-------------|
| 1 | Conditional↔Joint roundtrip | ✓ | Algebraically equivalent, exact roundtrip |
| 2 | Edge cases (pW=0, pW=1) | ✓ | Both formulas agree at boundaries |
| 3 | Self-play binary targets | ✓ | **Vectors identical** under both interpretations |
| 4 | Perspective flip negates equity | ✓ | Double-flip = identity |
| 5 | Bearoff to_conditional↔to_absolute | ✓ | 200 positions, clean roundtrip |
| 6 | Bearoff c15 format determination | ✓ | **Confirmed conditional** via equity-maximizing Bellman |
| 7 | Probability invariants | ✓ | 500 positions, all constraints satisfied |
| 8 | Gammon head Bellman (equity-optimal) | ✓ | 50 positions, max err 0.00001 |
| 9 | Conditional↔Joint equity algebraic equivalence | ✓ | Both formulas give identical equity (machine epsilon) |
| 10 | Masking analysis | ✓ | **Confirmed 50% data waste** for gammon heads |
| 11 | Pre-dice = E[post-dice] | ✓ | 50 positions, max err 0.00002 |
| 12 | BearoffK6↔FluxLib equity | ✓ | Perfect agreement across 100 positions |

### Cross-Validation: Bootstrap vs Bearoff Table (scripts/validate_bootstrap_bearoff.jl)

Ran on Jarvis against 5,000 bearoff positions from the raw 1-ply bootstrap (BackgammonGame objects):

**POST-ROLL BGBlitz vs exact bearoff table:**

Note: BGBlitz returns joint values, bearoff table stores conditional. For bearoff positions
(p_win near 0 or 1), both formulas converge numerically, so the equity comparison is valid
even though the raw head values have different semantics. Head-by-head comparison is only
meaningful for p_win (same in both formats) and equity (formula-independent at extremes).

| Metric | Mean |Δ| | Max |Δ| |
|--------|---------|---------|
| p_win | 0.000282 | 0.0087 |
| **equity** | **0.000780** | **0.0564** |

BGBlitz 1-ply is extremely accurate for bearoff positions — mean equity error 0.0008 vs exact table.

**Bootstrap data facts:**
- 284,183 total positions (5,000 games between wildbg and BGBlitz)
- 31,796 (11.2%) are bearoff positions
- **All are decision nodes** (post-dice) — zero chance nodes in bootstrap
- Raw format has BOTH `equity_preroll` and `equity_postroll` fields
- 437/5000 sampled bearoff positions are c15 (gammons possible), 57 have non-trivial gammon values

**Conditional↔Joint format consistency:** 0/1000 mismatches, max diff 2.22e-16 (machine epsilon)

### Discovery: Raw Bootstrap Has Pre-Roll AND Post-Roll

The original 1-ply bootstrap (`bootstrap_5000g_bgblitz1ply.jls`, 284K samples) stores
separate `equity_preroll` and `equity_postroll` fields.

The 2-ply batches (`bootstrap_5000g_bgblitz2ply_batch*.jls.zst`, 20 batches × ~564K
samples each) store a single `equity` field. Investigation of batch 1 shows:
- 278,532 chance nodes (pre-dice) → equity contains BGBlitz preroll value
- 285,493 decision nodes (post-dice) → equity contains BGBlitz postroll value
- Each sample gets the **appropriate** value for its state type

### Production Bootstrap Format (VERIFIED)

The production file (`bootstrap_race_samples.jls`, 2.37M samples) was built from the
2-ply batches. Verified on Jarvis:
- **Zero chance nodes** — all 2,370,908 are decision nodes
- **Equity = postroll** — the accurate post-dice BGBlitz evaluation
- 1,132,896 bearoff samples with equity, 246,404 without
- This is the correct choice: decision nodes should use post-dice evaluations

**No pre-roll contamination bug exists.** The production bootstrap uses the right values.

### Outlier Investigation (scripts/investigate_bootstrap_outliers.jl)

Investigated all 31,796 bearoff positions in the raw 1-ply bootstrap to understand error patterns.

**Post-roll error distribution (BGBlitz 1-ply vs exact bearoff table):**

| Error Range | Count | % |
|------------|-------|---|
| > 0.001 | 5,785 | 18.2% |
| > 0.01 | 396 | 1.25% |
| > 0.05 | 15 | 0.05% |
| > 0.10 | 5 | 0.02% |

**All top-20 post-roll outliers are gammon head disagreements, NOT p_win errors.**

Two patterns:
1. **Mover losing (p_win≈0), gammon loss rate disagreement:** BGBlitz underestimates
   P(gammon|loss). Example: table says 0.222, BGBlitz says 0.041 (error: 0.18). Worst
   position: 15 checkers on pts 5-6 vs opponent with 2 on pt1.
2. **Mover winning (p_win≈1), gammon win rate disagreement:** BGBlitz disagrees on
   P(gammon|win). Example: table says 0.352, BGBlitz says 0.210 (error: 0.14).

**Root cause:** BGBlitz at 1-ply is a neural network estimator. It's very accurate for
p_win (max error 0.009) but less precise for gammon conditionals, which require looking
many moves ahead. The bearoff table is computed by exact dynamic programming.

**Pre-roll outliers fully explained — not a bug.** All top-20 have the same pattern:
near-terminal positions (1-2 checkers each), ALL doubles, where knowing the specific dice
changes evaluation from ~0.11 (pre-dice average) to ~1.0 (certain win with these doubles).
This is comparing two different quantities, not an error.

**Impact on training:** Only 0.05% of bearoff positions have > 0.05 equity error. Mean
error 0.0008. This is estimation noise from BGBlitz 1-ply, not systematic bias. The
production bootstrap uses 2-ply BGBlitz, which should be even more accurate. Acceptable for
bootstrap data under either conditional or joint format.

## BGBlitz Format Discovery (2026-03-24, CRITICAL)

**Decompilation of BGBlitz JAR + live testing revealed: the bridge sends JOINT
(cumulative) probabilities, not conditional.**

### Evidence

1. **JAR decompilation** (`bgblitz.bot.Equity`): Internal fields are `myWins`,
   `myGammon` (cumulative G+BG), `myBackGammon`. `calculateEquity()` =
   `myWins + myGammon + myBG - oppWins - oppGammon - oppBG`.

2. **Bridge code** (`BgblitzBridge.java:261`): Calls `eq.getGammon(true)` (returns
   raw `myGammon`), NOT `eq.getGammonRate()` (which would return the conditional
   `(myGammon - myBG) / myWins`).

3. **Live test** (29 diverse positions, BGBlitz 0-ply on Jarvis): Joint formula
   matches scalar equity with **zero error** (Δ = 0.00000) for every position.
   Conditional formula has errors up to **0.154**.

```
pos | pw     | gw     | scalar  | eq_cond | eq_joint | Δcond   | Δjoint
 13 | 0.3793 | 0.0338 | -0.6687 | -0.5148 | -0.6687  | 0.15384 | 0.00000
  7 | 0.8636 | 0.6998 | +1.4245 | +1.3448 | +1.4245  | 0.07972 | 0.00000
  8 | 0.6843 | 0.4092 | +0.7201 | +0.6343 | +0.7201  | 0.08578 | 0.00000
```

### Impact

| What | Status |
|------|--------|
| Bootstrap data format | Was always JOINT — `conditional_v1` label was wrong |
| Training with conditional formula | Had mean ~0.05 equity error on contact positions |
| Bearoff table (c15) | Stores TRUE conditional — confirmed by Bellman |
| v1-v8 training | Latent bug: joint bootstrap fed into conditional loss formula |
| v9 joint redesign | Now even MORE important — fixes this latent bug |

### What the 5-tuple actually contains

The bridge sends (confirmed by decompilation + live test):

| Index | Bridge Call | Actual Value | Our Old Label | Correct Label |
|-------|-----------|-------------|--------------|---------------|
| 1 | `getWins(true)` | P(win) | p_win | p_win |
| 2 | `getGammon(true)` | P(gammon+ wins) | P(gammon\|win) | P(win ∧ gammon+) **JOINT** |
| 3 | `getBackGammon(true)` | P(bg wins) | P(bg\|win) | P(win ∧ bg) **JOINT** |
| 4 | `getGammon(false)` | P(gammon+ losses) | P(gammon\|loss) | P(loss ∧ gammon+) **JOINT** |
| 5 | `getBackGammon(false)` | P(bg losses) | P(bg\|loss) | P(loss ∧ bg) **JOINT** |

**"gammon" means gammon-or-better (cumulative).** It includes backgammon.

### Why existing tests didn't catch this

1. **Bearoff comparison was inconclusive**: For bearoff positions, p_win is near 0 or 1,
   causing conditional and joint formulas to converge numerically. The mean equity error
   of 0.0008 is identical regardless of which formula is used. Reconfirmed: corrected
   comparison gives identical results to the original (both mean 0.000827).

2. **Parity check wasn't run on production data**: The `audit_bootstrap.jl` parity check
   compares equity_from_probs (conditional formula) against the BGBlitz scalar (joint
   formula). These differ for midgame positions. But the production artifact was built
   by a parallel generation script that didn't include this check — the 1-ply metadata
   has no `scalar_parity_checked` field.

3. **Self-play targets are format-agnostic**: Binary 0/1 targets are numerically identical
   under both conditional and joint interpretations. The format only matters for continuous
   targets (bootstrap, bearoff).

4. **The definitive test required midgame positions**: Only the live BGBlitz scalar
   comparison on contact positions (p_win in 0.3-0.7 range, significant gammons)
   can distinguish the formats — joint formula matched with Δ≤0.000002, conditional
   had errors up to 0.088. Tested on 20 positions from the actual bootstrap file
   (re-queried live), all 20 confirmed JOINT. Stored bootstrap values matched live
   query with Δ=0.000000 — no transformation was applied during bootstrap creation.

### Design implication for v9

**The bootstrap data needs NO conversion for v9 — it's already joint!**

The only conversion needed is for the bearoff table (conditional → joint via
`to_absolute()`), which was already planned. This actually simplifies the migration.

## Pros and Cons

### Pros

1. **2× training signal for gammon heads** — every sample trains all 5 heads
2. **Eliminates masking complexity** — `_equity_head_weights()` deleted
3. **BCEWithLogits is numerically superior** — no vanishing gradients near 0
4. **Simpler equity formula** — addition only, fewer FLOPs in inference hot path
5. **Bear-off table alignment** — k=6 uses `to_absolute()` temporarily; k=7 table (in progress) stores joint natively
6. **Proven approach** — GnuBG, TD-Gammon, XG all use joint
7. **No division-by-zero edge cases** — conditional has P(gammon|win) undefined when P(win)→0

### Cons

1. **Bootstrap tooling update needed** — production artifact is already joint, but generator/auditor scripts still reference `conditional_v1` contract and use wrong parity formula
2. **Weights not transferable** — v8 conditional weights can't seed v9 (fresh bootstrap start anyway)
3. **Probability constraints harder to enforce** — P(wbg) ≤ P(wg) ≤ P(win) not guaranteed by NN (same issue exists with conditional; GnuBG doesn't enforce it either — clamps at inference)
4. **BackgammonNet.jl changes needed** — `BearoffResult`, `compute_equity()`, contract docs

### Risk Assessment

**Medium risk — cross-repo value-contract migration.** The direction is correct (aligning
with every successful backgammon NN, and fixing a latent bug where joint bootstrap data
was fed into a conditional loss formula). However, the blast radius is significant:

- **Two repos affected:** AlphaZero.jl (NN, training, inference, buffer) and BackgammonNet.jl
  (bearoff table, equity computation, bootstrap tooling, contract docs)
- **Every component touching the 5-head vector must be audited:** NN forward pass, loss
  function, FastWeights inference, MCTS backprop, sample conversion, perspective flips,
  bearoff evaluator, bootstrap loader, eval scripts
- **Artifact metadata must be updated:** `conditional_v1` → `joint_cumulative_v1` in
  bootstrap generator, auditor, and contract docs. Bearoff table uses separate
  `joint_noncumulative_bearoff_v1` contract (numerically identical in bearoff, distinct label)
- **Self-play targets are numerically unchanged** (binary 0/1 targets are identical under
  both interpretations) — this limits the blast radius for the most common code path

Mitigations:
1. Comprehensive test suite already exists (6,315+ assertions)
2. Live BGBlitz validation confirms the target format
3. Self-play targets don't change (reducing risk of subtle bugs)
4. Can validate via smoke test: 5 iterations of training + eval before full run

## Files That Change

| File | Change |
|------|--------|
| **AlphaZero.jl — NN and training** | |
| `src/networks/architectures/fc_resnet_multihead.jl` | Remove sigmoid from gammon heads, joint equity formula, update docs |
| `src/learning.jl` | BCEWithLogits loss, delete `_equity_head_weights()` masking |
| `src/memory.jl` | Update EquityTargets docs (joint semantics), update `flip_equity_perspective` for joint |
| `src/inference/fast_weights.jl` | Joint equity formula (fewer FLOPs) |
| **AlphaZero.jl — scripts** | |
| `scripts/training_server.jl` | Remove any conditional conversion; bootstrap loads as-is (already joint). Bearoff equity targets use `to_absolute()` |
| `scripts/selfplay_client.jl` | Update `convert_trace_to_samples`: self-play targets are joint (same vectors, remove masking comments). Bearoff targets via `to_absolute()` |
| `scripts/eval_race.jl` | Update equity computation for joint formula |
| **BackgammonNet.jl — bearoff and contract** | |
| `src/bearoff_k6.jl` | Temporary: use `to_absolute()` at call sites for joint values |
| `src/bearoff_k7.jl` | New: k=7 solver + lookup, natively joint non-cumulative |
| `src/equity.jl` | Add `compute_equity_joint()` alongside existing conditional version |
| `notes/bootstrap_value_head_contract.md` | Change contract from `conditional_v1` to `joint_cumulative_v1` |
| `scripts/generate_bootstrap.jl` | Update contract label and scalar parity formula |
| `scripts/audit_bootstrap.jl` | Update contract check and parity formula |

## Verification Summary

### Completed

| # | Verification | Script/Test | Result |
|---|-------------|------------|--------|
| 1 | Conditional↔Joint algebraic equivalence | `test/test_value_head_formats.jl` | ✓ 6,315 assertions pass |
| 2 | Bearoff table format (conditional confirmed) | `test/test_value_head_formats.jl` #6 | ✓ Bellman match |
| 3 | Gammon head Bellman (equity-optimal) | `test/test_value_head_formats.jl` #8 | ✓ max err 0.00001 |
| 4 | Bootstrap vs bearoff table (5K positions) | `scripts/validate_bootstrap_bearoff.jl` | ✓ mean |Δ eq| = 0.0008 |
| 5 | Outlier investigation (31K positions) | `scripts/investigate_bootstrap_outliers.jl` | ✓ All explained |
| 6 | Production bootstrap uses postroll equity | Jarvis REPL inspection | ✓ Confirmed |
| 7 | No pre-roll contamination in production data | Jarvis REPL inspection | ✓ Zero chance nodes |
| 8 | 2-ply batch format (correct equity per state type) | Jarvis REPL inspection | ✓ 278K pre + 285K post |
| 9 | BGBlitz JAR decompilation | jadx on Equity.java | ✓ Internal fields are joint cumulative |
| 10 | Live BGBlitz scalar vs formula (29 positions) | `/tmp/bgblitz_live_test.jl` on Jarvis | ✓ **Joint matches scalar Δ=0. Conditional errors up to 0.154** |
| 11 | Bootstrap re-query (20 contact positions) | `/tmp/ground_truth_test3.jl` on Jarvis | ✓ **20/20 JOINT. Stored=live Δ=0. Conditional off by 0.003-0.088** |

### Before Deploying v9

1. Verify joint loss function produces same gradients on binary self-play targets (where conditional and joint are numerically identical)
2. Verify equity formula change in FastWeights gives identical MCTS values
3. Smoke test: train 5 iterations, verify loss decreases and eval doesn't crash
