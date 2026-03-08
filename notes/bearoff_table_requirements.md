# Bear-off Table Requirements: Exact Two-Sided k=6 Database

## Overview

Two separate bear-off databases for backgammon, covering all positions where
both players have checkers only on their 6-point home board (no contact, no
bar). All values must be computed under **optimal play** (equity-maximizing
moves), not heuristics like pip-minimization.

Two use cases with different requirements:
1. **Training** — 5 value head targets for AlphaZero self-play
2. **Game play** — cubeful equity with Match Equity Table (MET) support

---

## Definitions

### Position Space

A **one-sided k=6 position** is a distribution of 0-15 checkers across 6
points. Represented as a tuple (n₁, n₂, ..., n₆) where nᵢ = checkers on
point i and Σnᵢ ∈ {0, 1, ..., 15}.

Number of positions: C(15 + 6, 6) = C(21, 6) = **54,264**

A **two-sided position** is a pair (pos_A, pos_B) of one-sided positions plus
a flag for who moves next.

### Position Classes

| Class | Condition | Gammons possible? | Count of pairs |
|-------|-----------|-------------------|---------------|
| **c14** | Both sides have ≤14 checkers (both have borne off ≥1) | No | 751,188,180 |
| **c15** | At least one side has exactly 15 checkers | Yes | 721,129,800 |

c15 subdivides into:
- **c15-mixed** (Region A): One side ≤14, other side =15. Count: 600,935,040
- **c15-both** (Region B): Both sides =15. Count: 120,194,760

### Optimal Play

**Optimal play** = the move that maximizes cubeless money equity from the
mover's perspective:

```
E = P(win) × (1 + P(gammon|win)) − P(loss) × (1 + P(gammon|loss))
```

Range: [-2, +2] for bearoff (no backgammons possible).

In bearoff, both sides race independently (no blocking). Optimal play differs
from pip-minimization in positions where distributional shape matters (e.g.,
spreading checkers across low points to avoid wasting pips on future rolls).

### Game Flow

```
P1 pre-roll → P1 rolls dice → P1 moves → P2 pre-roll → P2 rolls dice → P2 moves → ...
```

There are two distinct position types:
- **Pre-roll**: A board state where the next action is a dice roll. The
  player to move has not yet rolled.
- **Post-roll**: A board state + specific dice, where the next action is
  choosing a move.

### First-Mover Convention

When A moves first (A is "on roll" at the pre-roll state):
- A finishes on A's k-th roll → B has completed k−1 rolls
- B finishes on B's k-th roll → A has completed k rolls

This asymmetry means **first-mover has an advantage**. All stored values must
specify whose turn it is.

---

## What the Table Stores: Pre-Roll Values

The table stores **pre-roll values only** — expected probabilities before the
mover's dice are known, averaged over all 21 dice outcomes, assuming optimal
play from both sides going forward.

**Post-roll values are not stored.** They are computed on-the-fly via 1-ply
search: given specific dice, enumerate all legal moves, look up each
resulting position in the pre-roll table (from opponent's perspective), and
pick the move maximizing equity. This gives exact post-roll evaluation with
zero additional storage.

---

## Use Case 1: Training (5 Value Heads)

### Purpose

Provide exact training targets for the AlphaZero multi-head equity network.
Used for TD-bootstrap: when a self-play game reaches a bear-off position,
replace the learned network estimate with the exact table value.

### Network Head Definitions

The network outputs 5 sigmoid heads. All probabilities are **from the
perspective of the player to move** (the "mover"), at a **pre-roll** state:

| Head | Symbol | Definition | Range |
|------|--------|-----------|-------|
| 1 | `pW` | P(mover wins) | [0, 1] |
| 2 | `pGW` | P(gammon \| mover wins) — conditional | [0, 1] |
| 3 | `pBGW` | P(backgammon \| mover wins) — conditional | always 0 in bearoff |
| 4 | `pGL` | P(gammon \| mover loses) — conditional | [0, 1] |
| 5 | `pBGL` | P(backgammon \| mover loses) — conditional | always 0 in bearoff |

All values are expectations over all possible dice outcomes for the mover
(pre-roll), assuming optimal play from both sides.

### Precise Mathematical Definitions

Given two-sided pre-roll position (A, B) where A is the mover (about to roll):

**pW** = P(A finishes before B | A rolls first, both play optimally)

This is the pre-roll win probability: averaged over A's first dice roll, then
A's optimal move, then B's dice roll, B's optimal move, etc.

**pGW** = P(B has all 15 checkers when A finishes | A wins, A rolls first, optimal play)

Gammon conditional: given that A wins, what fraction of the time does A also
gammon B? A gammon occurs when the winner bears off all checkers while the
loser still has all 15 on the board.

**pGL** = P(A has all 15 checkers when B finishes | B wins, A rolls first, optimal play)

Note the first-mover convention: A rolls first, but B wins. When B finishes
on B's k-th roll, A has had k rolls (A rolls first each turn). So:

```
pGL = [Σ_k P_finish_B(k) × P_no_off_A(k)] / P(B wins | A first)
```

This is **different** from the case where B rolls first (where A would only
have had k−1 rolls when B finishes in k).

**pBGW** = 0, **pBGL** = 0 (no backgammons in bearoff).

### What Must Be Stored

#### c14 Table

Both sides ≤14 checkers. No gammons possible (both have borne off ≥1).

Per pair: **pW only** from each perspective.

| Field | Type | Meaning |
|-------|------|---------|
| pW_ci | u16 | P(ci wins \| ci rolls first) |
| pW_cj | u16 | P(cj wins \| cj rolls first) |

Note: pW_ci + pW_cj ≠ 1 in general (first-mover advantage).

Format: **4 bytes per pair**
Total: 751,188,180 × 4 = **3.00 GB**

Training target when mover = ci: `[pW_ci, 0, 0, 0, 0]`

#### c15 Table

At least one side has 15 checkers. Gammons possible.

Per pair: **pW + gammon conditionals**, all 4 gammon values stored explicitly
to handle both first-mover conventions correctly.

| Field | Type | Meaning |
|-------|------|---------|
| pW_ci | u16 | P(ci wins \| ci rolls first) |
| pW_cj | u16 | P(cj wins \| cj rolls first) |
| gwc_ci_first | u16 | P(ci gammons cj \| ci wins, **ci first**) |
| glc_ci_first | u16 | P(cj gammons ci \| cj wins, **ci first**) |
| gwc_cj_first | u16 | P(cj gammons ci \| cj wins, **cj first**) |
| glc_cj_first | u16 | P(ci gammons cj \| ci wins, **cj first**) |

Reading convention — when mover = ci (ci rolls first):
```
pW  = pW_ci
pGW = gwc_ci_first       # mover wins and gammons
pGL = glc_ci_first       # mover loses and gets gammoned
```

When mover = cj (cj rolls first):
```
pW  = pW_cj
pGW = gwc_cj_first       # mover wins and gammons
pGL = glc_cj_first       # mover loses and gets gammoned
```

Format: **12 bytes per pair** [6 × u16]
Total: 721,129,800 × 12 = **8.65 GB**

#### c15-mixed Simplification

For c15-mixed pairs where ci ≤14 and cj = 15:
- ci has borne off ≥1 checker → can never be gammoned
- glc_ci_first = 0 (cj cannot gammon ci when ci rolls first)
- gwc_cj_first = 0 (cj cannot gammon ci when cj rolls first)
- Only gwc_ci_first and glc_cj_first can be > 0 (ci gammoning cj)
- And gwc_ci_first ≠ glc_cj_first (different first-mover = different
  number of rolls for cj to start bearing off)

Still store all 6 values for format uniformity. The zeros compress well if
needed.

### Scalar Equity for MCTS

Computed on-the-fly from stored values, not stored:

```
E = pW × (1 + pGW) − (1 − pW) × (1 + pGL)
```

Normalized to [-1, 1] by dividing by 3.

### Post-Roll Evaluation for MCTS

When MCTS encounters a bear-off position **after** dice are rolled (post-roll
decision node), compute the exact value via 1-ply search:

```
For each legal move with the given dice:
    new_pos = apply_move(pos, move)
    value = -lookup_preroll(new_pos, opponent_perspective)
Pick move with maximum value
```

This gives exact post-roll values using only the pre-roll table.

---

## Use Case 2: Game Play (Cubeful + MET)

### Purpose

During actual game play (human vs AI, tournament play), provide exact
endgame evaluation with proper cube handling and match play scoring.

### Requirements

For each two-sided bear-off position, compute on-the-fly:

1. **Cubeless equity** (money game)
   ```
   E_cl = pW − pL + 2×(pWG − pLG)
   ```
   (Backgammon terms are 0 in bearoff)

2. **Cubeful equity** (money game, with doubling cube)
   Using Janowski's formula or equivalent:
   - Depends on cube ownership (I own / opponent owns / centered)
   - Depends on cube efficiency parameter
   - Requires: pW, pWG, pLG, and cubeless equity
   - All computed from the stored pre-roll probability values

3. **Match equity** (match play, with cube + MET)
   - Convert point equity to match winning chances using MET
   - Cube decisions (double/take/pass) depend on match score
   - Requires: all cubeless probabilities + match score + cube state

### What Must Be Stored

**Same pre-roll probability values as the training table.** Cubeful equity,
MET lookups, and cube decisions are all computed on-the-fly.

Key difference: game play may benefit from higher precision. Consider
Float32 (4 bytes) per value instead of UInt16 (2 bytes).

### Table Size Estimates (Game Play)

If Float32 (4 bytes per value):
- c14: 751M × 2 values × 4 bytes = **6.0 GB**
- c15: 721M × 6 values × 4 bytes = **17.3 GB**
- Total: **23.3 GB**

If UInt16 is sufficient:
- Same as training table: **11.65 GB**

---

## Computation Method

### Step 1: One-Sided Optimal Policy

For each of 54,264 one-sided positions, determine the optimal move for
every dice outcome. "Optimal" = maximizes cubeless equity including gammon
values.

This is backward induction over positions ordered by decreasing pip count:
1. Terminal position (all checkers off): finished
2. For each position, for each of 21 dice outcomes, enumerate all legal
   moves, evaluate resulting position value, pick the best

The resulting **optimal policy π(pos, dice) → move** defines the transition
dynamics.

### Step 2: One-Sided Distributions Under Optimal Play

Given the optimal policy from Step 1, compute by forward propagation:

- **P_finish[pos, t]**: P(finish in exactly t rolls | start at pos, play π)
- **P_no_off[pos, t]**: P(still have all 15 checkers after t rolls | start at pos, play π)

Method: sparse transition matrix × distribution vector, iterated for
t = 0, 1, ..., T_MAX.

### Step 3: Two-Sided Convolution

For each pair (ci, cj), compute the 6 values via dot products of the
one-sided distributions:

**pW (ci first):**
```
pW_ci = Σ_{k=1}^{T} P_finish_ci(k) × [1 − F_finish_cj(k−1)]
```

**Gammon: ci gammons cj, ci first** (cj has had k−1 rolls when ci finishes in k):
```
pGam_ci = Σ_{k=1}^{T} P_finish_ci(k) × P_no_off_cj(k−1)
gwc_ci_first = pGam_ci / pW_ci
```

**Gammon: cj gammons ci, ci first** (ci has had k rolls when cj finishes in k):
```
pGam_cj = Σ_{k=1}^{T} P_finish_cj(k) × P_no_off_ci(k)
glc_ci_first = pGam_cj / (1 − pW_ci)
```

Swap ci ↔ cj for cj-first values (gwc_cj_first, glc_cj_first).

### Step 4: Verification

#### Exact checks
- c14 pairs: all gammon values = 0
- c15-mixed (ci ≤14): glc_ci_first = 0, gwc_cj_first = 0
- 15@ace vs 15@ace: gwc ≈ 0 (any die bears off from point 1)
- All values in [0, 1]
- pW_ci > 0.5 for most symmetric pairs (first-mover advantage)

#### Cross-validation with gnubg
- Compare pW against gnubg two-sided database for 10,000+ random pairs
- **Must match to quantization precision** (max error ≈ 0.00003 for UInt16)
- If pW doesn't match gnubg, the optimal policy differs → investigate

#### Monte Carlo validation
- For 1,000+ random c15 pairs, run 1M rollouts **using the computed optimal
  policy** (not pip-minimization)
- Compare simulated pW, gwc, glc against table values
- Expected agreement: within 3 standard errors (≈ 0.003 for 1M sims)

#### Self-consistency
- P_finish sums to 1.0 for all positions
- P_no_off monotonically non-increasing in t
- gwc × pW ≤ pW (absolute gammon prob ≤ win prob)
- glc × (1−pW) ≤ (1−pW) (absolute gammon loss prob ≤ loss prob)
- pW from distributions matches gnubg pW (confirms same optimal policy)

---

## Summary of Deliverables

### Training Table (Priority 1)

| File | Pairs | Format | Size |
|------|-------|--------|------|
| c14.bin | 751,188,180 | [pW_ci, pW_cj] × u16 | 3.00 GB |
| c15.bin | 721,129,800 | [pW_ci, pW_cj, gwc_ci_first, glc_ci_first, gwc_cj_first, glc_cj_first] × u16 | 8.65 GB |
| **Total** | | | **11.65 GB** |

- All values pre-roll, under optimal play
- Verified against gnubg pW and Monte Carlo simulation
- Post-roll values computed on-the-fly via 1-ply search using this table

### Game Play Table (Priority 2, future)

- Same stored values as training table
- Potentially Float32 precision (23.3 GB total)
- Cubeful equity via Janowski's formula (on-the-fly)
- Match equity via MET integration (on-the-fly)
- Cube decision logic: double/take/pass/beaver (on-the-fly)

### Verification Artifacts

- pW cross-validation vs gnubg (10K+ pairs, must match to quantization)
- Monte Carlo validation with optimal policy (1K+ pairs, 1M rollouts each)
- Distribution self-consistency report
