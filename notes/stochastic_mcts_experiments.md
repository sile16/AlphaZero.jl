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

---

## Session Management Guidelines

### Preserving Training Data

**DO preserve session data when:**
- Training ran for a meaningful duration (more than a few minutes)
- The run completed without obvious code bugs
- Results are useful for comparison or analysis

**OK to delete session data when:**
- Short test runs (couple minutes just to verify code works)
- Obvious code bugs caused incorrect behavior
- Debugging/development runs with known issues

### Session Documentation

For each meaningful training session, write a short summary including:

1. **Goal**: What hypothesis or question is this testing?
2. **Configuration**: Key parameters that differ from defaults
3. **Duration**: Training time and iterations completed
4. **Results**: Key metrics (win rates, loss values, etc.)
5. **Conclusions**: What did we learn?

Example:
```
Session: backgammon-sampling-v1
Goal: Test if sampling mode with prior integration outperforms deterministic MCTS
Config: chance_mode=:sampling, prior_virtual_visits=1.0, 100 iters/turn
Duration: 4 hours, 150 iterations
Results: 65% vs GnuBG, 80% vs Random
Conclusion: [pending comparison with deterministic baseline]
```

### Session Naming Convention

Use descriptive names: `{game}-{variant}-{date}` or `{game}-{experiment}-v{N}`

Examples:
- `backgammon-sampling-20260118`
- `backgammon-progressive-v2`
- `pig-baseline-comparison`

---

## Key Insights

### Sampling vs Deterministic MCTS (2026-01-18)

Even with O(1) sampling at chance nodes, stochastic MCTS differs fundamentally from deterministic:

**Deterministic MCTS** (hidden stochasticity):
- Backs up value from ONE sampled path
- No explicit aggregation across dice outcomes
- Q(s,a) = average of whatever dice happened to come up

**Stochastic MCTS with sampling**:
- Samples ONE outcome but backs up EXPECTATION across ALL outcomes
- `V = Σ prob_i × (W_i / N_i)` at each chance node
- Prior fills in for unsampled/undersampled outcomes
- Explicitly computes E[V | all dice]

The key advantage is **explicit expectimax aggregation at backup time**, not just the prior.

### Prior Integration Formula (Fixed 2026-01-18)

Correct formula for mixing prior with samples:
```
W_initial = V_prior × virtual_N
N_initial = virtual_N

After k samples: mean = (V_prior × virtual_N + Σ samples) / (virtual_N + k)

Prior weight: virtual_N / (virtual_N + k)
Sample weight: k / (virtual_N + k)
```

Bug fixed: Was using `W = V × prob × virtual_N` (wrong), now `W = V × virtual_N` (correct).
