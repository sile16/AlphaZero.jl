# Progressive Simulation Budget Testing Results

This document summarizes experiments testing progressive simulation budgets for AlphaZero training on Connect Four.

## Background

The MiniZero paper (arXiv:2310.11305) suggests that early training iterations can use fewer MCTS simulations since the neural network is weak and provides unreliable evaluations. The hypothesis is that compute can be saved by starting with low simulation counts and ramping up as training progresses.

We tested two approaches:
1. **Iteration-Progressive**: Vary simulation count by training iteration (MiniZero approach)
2. **Turn-Progressive**: Vary simulation count by turn number within each game (novel approach)

## Baseline Configuration

- Game: Connect Four
- Network: ResNet (128 filters, 5 blocks)
- Training: 15 iterations, 5000 self-play games per iteration
- MCTS: 600 simulations per turn (constant)
- Benchmark: AlphaZero vs MCTS with 1000 rollouts

**Baseline Results:**
- Total training time: 650.5 minutes
- Final benchmark: +1.0 (perfect win rate vs MCTS-1000)
- Total simulations: ~1.12 billion

---

## Experiment 1: Iteration-Progressive Simulation

### Approach
Based on MiniZero paper methodology:
- `sim_min = 2` (iteration 1)
- `sim_max = 1198` (iteration 15)
- Linear interpolation: `sims = sim_min + (sim_max - sim_min) * iter / num_iters`
- Total budget designed to match baseline: `(2 + 1198) / 2 * 15 = 600 * 15`

### Configuration
```julia
progressive_sim = ProgressiveSimParams(
  sim_min=2,
  sim_max=1198)
```

### Results

| Metric | Baseline | Iter-Progressive |
|--------|----------|------------------|
| Total Time | 650.5 min | 646.3 min |
| Final Benchmark | +1.0 | -0.371 |
| Total Sims | 1.12B | 0.83B |
| Speedup | - | 1.01x |

### Benchmark Progression
```
Iter   Sims/Turn   Benchmark
1      2           -0.785
5      321         -0.664
10     720         -0.508
15     1118        -0.371
```

### Analysis
- **No wall clock speedup** (1.01x) - early iterations are fast but later iterations with 1000+ sims are very slow
- **Uses 26% fewer simulations** but at significant performance cost
- **Final benchmark much worse** (-0.371 vs +1.0)
- Network never recovers from poor early training with 2 sims/turn

---

## Experiment 2: Turn-Progressive Simulation

### Approach
Novel approach varying simulations by turn number within each game:
- Early moves: 2 simulations (pattern-based, less tactical)
- Late moves: 600 simulations (tactical depth needed)
- Ramp speed increases with training iteration:
  - Iter 1: Reach target by turn 30 (slow ramp)
  - Iter 15: Reach target by turn 3 (fast ramp)

### Intuition
1. Opening moves may be more pattern-based, needing fewer simulations
2. Mid/late game positions need tactical depth
3. As network improves, it learns openings faster, so ramp up quicker

### Configuration
```julia
turn_progressive_sim = TurnProgressiveSimParams(
  turn_sim_min=2,
  turn_sim_target=600,
  ramp_turns_initial=30,
  ramp_turns_final=3)
```

### Simulation Schedule Examples
```
Iter 1 (ramp over 30 turns):
  Turn 0: 2 sims, Turn 10: 201 sims, Turn 20: 401 sims, Turn 29: 580 sims

Iter 8 (ramp over 16.5 turns):
  Turn 0: 2 sims, Turn 10: 364 sims, Turn 16+: 600 sims

Iter 15 (ramp over 3 turns):
  Turn 0: 2 sims, Turn 1: 201 sims, Turn 2: 401 sims, Turn 3+: 600 sims
```

### Results

| Metric | Baseline | Turn-Progressive |
|--------|----------|------------------|
| Total Time | 650.5 min | 634.5 min |
| Final Benchmark | +1.0 | -0.461 |
| Total Sims | 1.12B | 0.73B |
| Speedup | - | 1.03x |
| Sim Savings | - | 35% |

### Benchmark Progression
```
Iter   Benchmark
1      -0.867
5      -0.777
10     -0.574
11     -0.496 (best mid-training)
15     -0.461
```

### Analysis
- **Modest wall clock speedup** (2.5% faster)
- **Saves 35% simulations** (0.73B vs 1.12B)
- **Still much worse final performance** (-0.461 vs +1.0)
- Benchmark improves steadily but never catches up to baseline

---

## Comparison Summary

| Approach | Wall Clock | Sim Savings | Final Benchmark | Verdict |
|----------|------------|-------------|-----------------|---------|
| Baseline (600 constant) | 650.5 min | - | **+1.0** | Best |
| Iteration-Progressive | 646.3 min | 26% | -0.371 | Poor |
| Turn-Progressive | 634.5 min | 35% | -0.461 | Poor |

---

## Conclusions

### Why Progressive Simulation Fails for Connect Four

1. **Early training signals matter**: When MCTS uses only 2 simulations, the policy targets are essentially noise. The network learns poor play patterns that persist.

2. **Opening quality is critical**: Connect Four openings are highly tactical. Playing poorly in the opening leads to losing positions that the network then learns as "normal."

3. **Cascading effects**: Poor early play leads to poor training data, which leads to a poor network, creating a negative feedback loop.

4. **No recovery path**: Unlike the MiniZero results on Go, Connect Four training doesn't recover from weak early iterations even when later iterations use high simulation counts.

### Recommendations

For Connect Four with this network architecture and training setup:
- **Use constant simulation budget** (600 sims/turn)
- Progressive simulation does not improve training efficiency
- The compute savings (~35%) come at too high a performance cost

### Future Work

If pursuing progressive simulation further, consider:
1. **Higher minimum sims** (e.g., 100-200 instead of 2)
2. **Shorter ramp period** (reach target by turn 5 instead of 30)
3. **Different games**: May work better for Go or games with longer horizon
4. **Larger networks**: May be more robust to noisy training signals

---

## Implementation Details

### Files Modified/Created
- `src/params.jl`: Added `TurnProgressiveSimParams` struct
- `src/play.jl`: Added `TurnProgressiveMctsPlayer`
- `src/training.jl`: Modified `self_play_step!` to support turn-progressive
- `games/connect-four/params_progressive.jl`: Iteration-progressive config
- `games/connect-four/params_turn_progressive.jl`: Turn-progressive config

### How to Run

```julia
using AlphaZero

# Baseline
Scripts.train("connect-four")

# Iteration-Progressive
Scripts.train("connect-four-progressive")

# Turn-Progressive
Scripts.train("connect-four-turn-progressive")
```

---

*Testing performed: January 2026*
*AlphaZero.jl version: Latest (master branch)*
