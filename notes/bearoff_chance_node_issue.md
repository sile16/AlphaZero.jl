# Bear-off Table: Chance Node Issue

## The Problem

Bear-off table values are **pre-dice expectations**: E[equity | board_position], averaged over all 21 possible dice outcomes. They are valid at **chance nodes** (before dice are rolled).

In the current "deterministic" game wrapper, dice are auto-rolled inside `step!()`. Every game state the NN sees has dice already rolled. This means:

- **NN input**: (board, dice) — knows which dice were rolled
- **Bear-off table target**: E[equity | board] — doesn't condition on dice
- **Correct target**: E[equity | board, dice] — should condition on the specific roll

This is a **signal mismatch**. A bear-off position with double-6s is worth much more than the same board with 1-2, but the table returns the same value for both.

## Current Usage (training targets only)

In `train_distributed.jl`, bear-off table values replace game outcomes as training targets for bear-off positions. During self-play MCTS, the NN evaluates all positions (no table lookup). So the mismatch only affects training signal quality, not search.

## Why It Partially Works Anyway

Over many training examples, the NN sees the same board position with different dice rolls, all mapped to the same average target. The NN learns a compromise that approximates the average. But it can't learn dice-specific value differences in bear-off, which limits endgame precision.

## Solutions

### Option A: Explicit Chance Nodes in MCTS (Best)

Switch bear-off evaluation to happen at **chance nodes** (pre-dice). The MCTS code in `src/mcts.jl` already has full chance node support with 4 modes (`:full`, `:sampling`, `:stratified`, `:progressive`). The infrastructure exists but is hidden by the deterministic game wrapper.

**Approach:**
1. Create a "stochastic" game variant or modify the deterministic wrapper to expose chance nodes specifically for bear-off positions
2. At bear-off chance nodes, use the table value directly (it IS the correct pre-dice value)
3. Non-bear-off positions continue using the deterministic wrapper

**Pros:** Mathematically correct. No approximation.
**Cons:** Requires game interface changes. MCTS tree grows (21 children per chance node).

### Option B: Compute Post-Dice Values (Practical)

For each (board, dice) pair, compute the exact post-dice value:
1. Enumerate all legal moves with these specific dice
2. For each resulting board, look up opponent's bear-off table value
3. Take the max (optimal move in pure race = pip minimization)
4. Value = 1 - opponent_table_value (flip perspective)

This gives the correct E[equity | board, dice] without changing the MCTS architecture.

**Pros:** Works with current deterministic wrapper. Exact values.
**Cons:** Requires legal move enumeration for bear-off. Non-trivial but feasible (BackgammonNet has move generation). Computational cost per lookup increases.

### Option C: Use Table at Chance Nodes Only (Hybrid)

Modify the game to expose chance nodes only when:
- Position is a bear-off position
- We want to evaluate using the table

At non-bear-off positions, keep the deterministic wrapper. This is a middle ground.

### Option D: Accept the Noise (Status Quo)

The pre-dice average is a noisy but unbiased estimator of position value. Over many training examples, the NN learns a reasonable approximation. This is what we're doing now.

**Note:** The previous experiment showed bear-off table targets gave only modest improvement (+0.63 vs +0.55 baseline at 50 iter). The signal mismatch may explain why the gain was small despite exact values being available.

## Recommendation

Option B is the most practical near-term fix. It gives exact post-dice values without MCTS architecture changes. Option A is the correct long-term solution for when we want table lookups during MCTS search (not just training targets).

## Key Insight

The bear-off table is a **state-value function V(s)** at chance nodes. To use it at decision nodes (post-dice), we need **Q(s, dice) = max_move V(result(s, dice, move))** — which requires move enumeration.
