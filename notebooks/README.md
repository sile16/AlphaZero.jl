# Stochastic AlphaZero Comparison

This directory contains notebooks and scripts to compare Standard AlphaZero vs Stochastic AlphaZero on the Game of Pig.

## Overview

We compare three approaches:

1. **Standard AlphaZero** - Treats dice rolls as hidden environment dynamics. MCTS doesn't know about stochasticity.

2. **Stochastic AlphaZero** - Explicitly models chance nodes with expectimax in MCTS. Knows about all 6 dice outcomes.

3. **Hold20 Baseline** - Simple heuristic: hold when turn total ≥ 20. Known to be a strong Pig strategy.

## Files

- `stochastic_comparison.ipynb` - Jupyter notebook for Colab/JuliaHub
- `run_comparison.jl` - Standalone Julia script

## Running on Google Colab

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `stochastic_comparison.ipynb`
3. Run the first cell to install the package:
   ```julia
   using Pkg
   Pkg.add(url="https://github.com/sile16/AlphaZero.jl", rev="stochastic-mcts")
   ```
4. Run all cells

## Running on JuliaHub

1. Create a new Julia notebook
2. Add the package from the stochastic-mcts branch
3. Copy cells from `stochastic_comparison.ipynb`

## Running Locally

```bash
cd AlphaZero.jl
julia --project=. notebooks/run_comparison.jl
```

## Expected Results

**Hypothesis**: Stochastic AlphaZero should outperform Standard AlphaZero because:

- **Better value estimates**: Expectimax computes true expected values over dice outcomes
- **Correct exploration**: Explores all dice possibilities, not random samples
- **Cleaner training signal**: Network learns values that properly account for stochasticity

Standard AlphaZero sees the same state leading to different outcomes (depending on hidden dice roll), causing noisy value estimates and confused learning.

## Metrics

- **Win Rate vs Hold20**: Primary metric - percentage of games won
- **Sample Efficiency**: Win rate / total MCTS simulations
- **Iterations to Beat Hold20**: How many training iterations to reach 55%+ win rate

## Training Configuration

Default (light) parameters for faster iteration:
- MCTS iterations per turn: 50
- Self-play games per iteration: 50
- Evaluation games: 50
- Max iterations: 20
- Win threshold: 55%

Adjust in the notebook/script for more thorough training.
