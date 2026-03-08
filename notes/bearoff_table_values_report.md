# Bear-off Table Values: What We Need and How They're Used

## 1. The 5-Head Network Architecture

The network outputs 5 sigmoid heads, all in [0, 1]:

| Head | Symbol | Meaning |
|------|--------|---------|
| 1 | `p_win` | P(current player wins) |
| 2 | `p_gw` | P(gammon \| win) — conditional |
| 3 | `p_bgw` | P(backgammon \| win) — conditional |
| 4 | `p_gl` | P(gammon \| loss) — conditional |
| 5 | `p_bgl` | P(backgammon \| loss) — conditional |

These are combined into a scalar equity for MCTS:

```
equity = p_win × (1 + p_gw + p_bgw) − (1 − p_win) × (1 + p_gl + p_bgl)
```

Range: [-3, +3]. Normalized to [-1, +1] by dividing by 3.

## 2. How Training Targets Are Set

### Without bear-off table (baseline)

Each game plays to completion. Every position in the game trace receives the
**actual game outcome** as its target:

```
eq[1] = won ? 1.0 : 0.0           # p_win: hard 0/1
eq[2] = won && gammon ? 1.0 : 0.0  # p_gw: 1 if won with gammon
eq[3] = won && bg ? 1.0 : 0.0      # p_bgw: 1 if won with backgammon
eq[4] = lost && gammon ? 1.0 : 0.0  # p_gl: 1 if lost with gammon
eq[5] = lost && bg ? 1.0 : 0.0     # p_bgl: 1 if lost with backgammon
```

These are **hard targets** (0 or 1) — the same outcome is assigned to every
position in the game. The network learns soft probabilities through averaging
over many games.

### With bear-off table (TD-bootstrap)

When `--use-bearoff` is enabled, the code in `convert_trace_to_samples()`
(line 990-1001) **replaces** the game outcome with table values for every
position that is a bear-off position:

```julia
if USE_BEAROFF && is_bearoff_position(state.p0, state.p1)
    bo = bearoff_table_equity(state)   # lookup from table
    z = wp ? bo.value : -bo.value      # scalar equity
    eq = [p_win, p_gw, p_bgw, p_gl, p_bgl]  # 5-tuple from table
end
```

These are **soft targets** (continuous probabilities) from the exact table,
replacing the hard 0/1 from the actual game outcome.

**Key point**: Only bear-off positions get table targets. Non-bearoff positions
in the same game still use the actual game outcome (hard 0/1). This creates a
**discontinuity** in target style within a single game trace.

## 3. The Loss Function

All 5 heads use **binary cross-entropy** (BCE), weighted by sample importance:

```
L_head = -mean(y × log(ŷ) + (1-y) × log(1-ŷ))
L_value = L_win + L_gw + L_bgw + L_gl + L_bgl
L_total = L_value + L_policy + L_invalid + L_reg
```

BCE treats each head independently. A target of 0.034 for `p_gw` produces
a very different gradient than a target of 0.0 or 1.0.

## 4. What the Bear-off Table Currently Stores

### c14 table (both sides ≤14 checkers, ≥1 borne off)

4 bytes per pair: `[pW_ci:u16, pW_cj:u16]`

- No gammons possible (both sides have borne off at least 1 checker)
- p_gw = p_gl = p_bgw = p_bgl = 0
- **Only need pW. This is correct and uncontroversial.**

### c15 table (at least one side has all 15 checkers)

8 bytes per pair: `[pW_ci:u16, pW_cj:u16, gwc_ci:u16, gwc_cj:u16]`

Where:
- `pW_ci` = P(ci wins | **ci moves first**) — from gnubg optimal play
- `pW_cj` = P(cj wins | **cj moves first**) — from gnubg optimal play
- `gwc_ci` = P(ci gammons cj | ci wins, **ci moves first**) — from pip-min distributions
- `gwc_cj` = P(cj gammons ci | cj wins, **cj moves first**) — from pip-min distributions

## 5. How the Table Is Looked Up

From `bearoff_k6.jl` line 386-395, when the mover is ci:

```julia
BearoffResult(pW_ci, gwc_ci, 0.0, gwc_cj, 0.0)
#              ^       ^              ^
#              |       |              |
#              |       |              p_gammon_loss = gwc_cj
#              |       p_gammon_win = gwc_ci
#              p_win = pW_ci
```

Then in training (line 765):
```julia
eq = [r.p_win, r.p_gammon_win, r.p_bg_win, r.p_gammon_loss, r.p_bg_loss]
   = [pW_ci,   gwc_ci,         0,           gwc_cj,          0          ]
```

## 6. The Four Values We Actually Need

For a given game position where the **mover** faces an opponent, we need:

| Value | Definition | Current Source | Status |
|-------|-----------|---------------|--------|
| **pW** | P(mover wins) | gnubg optimal play | ✅ Verified correct |
| **gwc_win** | P(gammon \| mover wins, mover first) | pip-min distributions | ⚠️ See issues |
| **gwc_loss** | P(gammon \| mover loses, mover first) | pip-min distributions | ⚠️ See issues |
| **bgc** | P(backgammon \| win or loss) | hardcoded 0 | ✅ Correct for bearoff |

## 7. Known Issues

### Issue 1: Policy mismatch between pW and gwc

**pW** comes from gnubg (optimal play). **gwc** comes from pip-minimization
heuristic. These are different play policies.

The conditional gwc = P_gammon / P_win was computed as:
```
gwc = P_gammon_pipmin / P_win_pipmin
```

But training uses it with gnubg's pW. The equity computation becomes:
```
P_gammon_for_equity = gwc_pipmin × pW_gnubg
                    = (P_gammon_pipmin / P_win_pipmin) × pW_gnubg
```

If P_win_pipmin ≠ pW_gnubg (differences up to 0.425 observed), this
product is inconsistent. The absolute P_gammon is neither from gnubg's
perspective nor from pip-minimization's perspective.

**However**: gwc values are very small (0.001-0.08), so the absolute error
from this mismatch is bounded by ~0.08 × 0.425 ≈ 0.034. Probably negligible
for training.

### Issue 2: First-mover asymmetry in p_gammon_loss

The table stores gwc_cj = P(cj gammons ci | **cj first**). But when the
mover is ci, it's used as p_gammon_loss = P(cj gammons ci | **ci first**).

The difference: when ci moves first and cj finishes in k rolls, ci has had
**k** rolls (not k-1). The correct formula would use P_no_off[ci, k+1]
instead of P_no_off[ci, k].

Since gwc_cj assumes cj first, it slightly **overestimates** the gammon loss
probability (opponent had one fewer roll to bear off checkers). But gwc values
are tiny, so this is negligible.

### Issue 3: Soft vs hard targets create training asymmetry

Non-bearoff positions get hard targets (0 or 1) from actual game outcomes.
Bear-off positions get soft targets (e.g., 0.034) from the table.

For the gammon heads specifically:
- Non-bearoff: targets are 0 or 1 (strong gradient signal)
- Bear-off: targets are ~0.001-0.08 (weak, near-zero signal)

This means the gammon heads receive **strong signal from middlegame** (where
gammons actually happen) but **near-zero signal from endgame** (where gammons
are rare). This asymmetry may cause the gammon heads to under-train for
endgame positions, though this is arguably correct behavior.

The old broken gwc (~0.88) accidentally created strong gammon signal from
bear-off positions, effectively giving the gammon heads extra supervised
targets correlated with pW. This may explain why broken gwc outperformed.

## 8. What Correct Values Would Look Like

The ideal table would store values from a single, consistent play policy:

**Option A: All from gnubg**
- pW from gnubg ✅ (already have)
- gwc from gnubg ❌ (v4 database values are wrong)
- Would need gnubg rollouts or fixed gnubg database

**Option B: All from distributions (pip-minimization)**
- pW from distributions (differs from gnubg by up to 0.425)
- gwc from distributions ✅ (already have, MC-validated)
- Self-consistent, but less accurate pW

**Option C: pW from gnubg, gwc = 0**
- pW from gnubg ✅
- gwc = 0 for all positions (gammons are <8% conditional, <0.08 absolute)
- Simple, avoids all policy mismatch issues
- Loss: ~0.04 equity error on rare gammon-relevant positions

**Option D: Don't use table for gammon targets**
- Use table only for pW and scalar equity in MCTS leaf evaluation
- Let gammon heads learn from actual game outcomes (hard 0/1 targets)
- Bear-off positions get: eq = [pW_table, ?, ?, ?, ?] where ? comes from
  actual game outcome, not table
- This preserves the strong gradient signal for gammon heads

## 9. Recommendation

**Option D** (table pW only, game outcome for gammons) is likely the best
approach. It gives:
1. Accurate pW targets from gnubg
2. Natural gammon learning from actual game results
3. No policy mismatch
4. No soft/hard target asymmetry for gammon heads

To implement: modify `convert_trace_to_samples()` to only overwrite eq[1]
(p_win) from the table, keeping eq[2:5] from the actual game outcome.

The **bear-off rollouts** approach (+0.83 vs GnuBG 1-ply) already implicitly
does something similar — it uses noisy but unbiased rollout estimates for all
5 values, avoiding the policy mismatch entirely.
