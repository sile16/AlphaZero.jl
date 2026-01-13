# [Gumbel MCTS](@id gumbel_mcts)

```@meta
CurrentModule = AlphaZero.GumbelMCTS
```

Gumbel MCTS is an alternative to standard MCTS based on the paper
["Policy improvement by planning with Gumbel"](https://openreview.net/forum?id=bERaNdoegnO)
(Danihelka et al., ICLR 2022).

## Key Differences from Standard MCTS

| Aspect | Standard MCTS | Gumbel MCTS |
|--------|---------------|-------------|
| Root selection | UCT formula | Gumbel-max trick |
| Simulation allocation | All actions equally | Sequential halving |
| Subtree selection | UCT | UCT (same) |

## Algorithm Overview

1. **Sample Gumbel noise** for each action: `g(a) ~ Gumbel(0, 1)`
2. **Compute initial scores**: `σ(a) = log(π(a)) + g(a)`
3. **Sequential halving**: Progressively eliminate actions
   - Start with top-k actions by σ score
   - Simulate each, compute Q-values
   - Keep top half, repeat until budget exhausted
4. **Select action**: `argmax σ(a) + Q_completed(a)`

The Q-completion formula interpolates between value estimate (unvisited) and
empirical Q (visited):
```
Q_completed(a) = V + (c_scale * N(a) / (c_visit + N_total)) * (Q(a) - V)
```

## When to Use Gumbel MCTS

- **Low simulation budgets** (< 100 sims): Gumbel is more sample-efficient
- **Training**: Can improve policy targets with same compute
- **High branching factor**: Sequential halving focuses on promising actions

```@docs
GumbelMCTS
```

## Environment

```@docs
Env
explore!
policy
reset!
```

## Utilities

```@docs
approximate_memory_footprint
average_exploration_depth
```
