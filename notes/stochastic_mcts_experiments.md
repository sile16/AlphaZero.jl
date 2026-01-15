# Stochastic MCTS Experiments

## Experiment 1: Sampling Mode vs Standard MCTS (2024-01-14)

### Configuration
- **Training time**: 25 minutes per approach (~50 min total)
- **Game**: Pig (TARGET_SCORE=50, MAX_TURNS=200)
- **Benchmark**: 1200 games (600 per direction) vs Hold20 baseline

### Hyperparameters (Both Approaches)
| Parameter | Value |
|-----------|-------|
| Network | SimpleNet (width=64, depth=4) |
| Self-play games/iter | 200 |
| MCTS iters/turn | 50 |
| CPUCT | 1.0 |
| Temperature | 1.0 (self-play), 0.2 (arena) |
| Dirichlet noise ε | 0.25 (self-play), 0.05 (arena) |
| Dirichlet noise α | 1.0 |
| GPU | true (inference + training) |
| Workers | 16 |
| Batch size | 16 |
| Memory buffer | 100,000 |
| L2 regularization | 1e-4 |
| Optimizer | CyclicNesterov (lr: 1e-3 to 1e-2) |

### Key Difference
- **Standard**: No explicit chance nodes (stochasticity hidden in `play!`)
- **Stochastic Sampling**: `chance_mode=:sampling` - samples 1 outcome per MCTS visit

### Results

#### Training Efficiency
| Metric | Standard | Stochastic |
|--------|----------|------------|
| Training time | 1567s | 1559s |
| Iterations | 17 | 16 |
| Sec/iteration | 92.2s | 97.4s |

#### Final Performance (vs Hold20)
| Metric | Standard | Stochastic |
|--------|----------|------------|
| Agent first | 56% | 53% |
| Agent second | 49% | 46% |
| Overall | 52.5% | 49.5% |

#### Statistical Significance
- **Difference**: 3.0% (Standard - Stochastic)
- **Z-score**: 1.47
- **P-value**: 0.1416
- **Significant at α=0.05?**: NO

### Conclusions
1. Sampling mode performs comparably to standard MCTS (within statistical error)
2. No significant advantage for either approach in this test
3. Standard was slightly faster per iteration with GPU batching
4. Simple 1-sample approach may not capture enough information per visit

### Next Steps
- Implement **progressive expansion** for chance nodes
- Each visit expands a different outcome, building up coverage over time
- Use principled integration of NN prior with progressive samples

---

## Experiment 2: Progressive Expansion (IMPLEMENTED)

### Implementation Details

Added `chance_mode=:progressive` to MCTS with the following behavior:

#### Progressive Widening Formula
- Expand new outcome when `N^α > num_expanded`
- Default `α = 0.5` means expansion at visits: 1, 4, 9, 16, 25, 36...
- For 6-sided die: all outcomes expanded by ~36 visits

#### Expansion Order
- Outcomes sorted by probability (descending)
- Highest probability outcomes expanded first
- Ensures most likely outcomes get attention early

#### Prior Integration with Virtual Visits
- Each outcome initialized with `prior_virtual_visits` (default 1.0)
- Initial W = V * virtual_visits, N = virtual_visits
- As real visits accumulate, prior gets "washed out"
- Smooth transition from prior-dominated to data-dominated estimates

#### Value Calculation
```
For each outcome i:
  if expanded: mean_value[i] = W[i] / N[i]  (includes virtual visits)
  else:        mean_value[i] = V_prior       (use NN value)

total_value = Σ prob[i] * mean_value[i]
```

#### Selection Among Expanded Outcomes
- Uses visit-deficit selection (like expectimax mode)
- Select outcome with largest: `(prob / total_prob) - (N / total_N)`
- Ensures visit distribution converges to probability distribution

### New Parameters in MctsParams
```julia
chance_mode :: Symbol = :progressive
progressive_widening_alpha :: Float64 = 0.5   # N^α > num_expanded
prior_virtual_visits :: Float64 = 1.0         # Weight for NN prior
```

### Quick Sanity Test Results
- 1 iteration test completed successfully
- Self-play: 725 samples/sec
- Win rate improved from 15% → 35% after 1 iteration

### References
- Coulom (2007) "Efficient Selectivity and Backup Operators in MCTS"
- Silver et al. (2017) "Mastering the Game of Go without Human Knowledge" (PUCT prior)

---

## Experiment 3: Progressive vs Sampling vs Standard (TODO)

Run full comparison with same wall-clock time:
- Standard MCTS (hidden stochasticity)
- Sampling mode (`chance_mode=:sampling`)
- Progressive mode (`chance_mode=:progressive`)

Hypothesis: Progressive should outperform sampling by spending more effort on likely outcomes while maintaining prior information for unexplored outcomes.
