# Stochastic MCTS Experiments

> **GnuBG RESULTS INVALIDATED (2026-02-14)**: Any GnuBG evaluation numbers in this file (e.g., "63.6% vs GnuBG 0-PLY", "65% vs GnuBG") are invalid due to a critical board encoding bug in `_to_gnubg_board` (fixed in commit e164a85). The MCTS design discussion, implementation details, bug fixes, and "vs Random" results are still valid. See `notes/corrected_eval_results_20260214.md` for corrected GnuBG results.

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

---

## Future Optimizations to Try

### 1. Stratified Sampling for Chance Nodes

Instead of pure probability-proportional sampling, use stratified sampling with coverage guarantee:

```
if any outcome has 0 visits:
    sample uniformly from unvisited outcomes
else:
    sample proportionally to probability
```

**Benefits:**
- Guarantees all outcomes visited before any repeated
- Reduces variance in early estimates
- Maintains probability-proportional sampling after initial coverage

**Trade-off:** Initial phase slightly biased toward low-probability outcomes, but this washes out quickly.

### 2. Gumbel AlphaZero (Root Sampling)

From "Policy Improvement by Planning with Gumbel" (Danihelka et al., 2022):

**Key idea:** Instead of visit counts for action selection at the root, use Gumbel-Top-k sampling:
- Add Gumbel noise to Q + prior values
- Select top-k actions for MCTS expansion
- Provides better exploration with fewer simulations

**Benefits for stochastic games:**
- More efficient exploration at decision nodes
- Could combine with stratified sampling at chance nodes
- Proven to match or exceed AlphaZero with fewer simulations

**Reference:** https://arxiv.org/abs/2104.06303

### 3. MuZero-style Learned Dynamics

From "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (Schrittwieser et al., 2020):

**Key idea:** Learn a latent dynamics model instead of using game rules:
- Representation function: observation → hidden state
- Dynamics function: (hidden state, action) → next hidden state, reward
- Prediction function: hidden state → policy, value

**Benefits for stochastic games:**
- Dynamics model can learn to handle stochasticity implicitly
- No need for explicit chance nodes in MCTS
- Works even when game rules aren't available

**Challenges:**
- More complex architecture
- Requires learning dynamics (not just value/policy)
- May need modifications for explicit stochasticity

**Reference:** https://arxiv.org/abs/1911.08265

### 4. Stochastic MuZero

Extension of MuZero for stochastic environments:
- Learn a stochastic dynamics model
- Sample from learned transition distribution
- Could combine learned dynamics with explicit chance modeling

**Reference:** https://arxiv.org/abs/2104.06303 (Section on stochastic environments)

---

## Summary: Optimization Priority List

| Priority | Optimization | Effort | Expected Impact |
|----------|-------------|--------|-----------------|
| 1 | Stratified sampling | Low | Medium - better early estimates |
| 2 | Gumbel root sampling | Medium | High - fewer sims needed |
| 3 | Higher virtual_visits tuning | Low | Low-Medium - tune prior weight |
| 4 | MuZero dynamics | High | High - but major rewrite |
| 5 | Stochastic MuZero | Very High | Unknown - research frontier |

### Quick Wins (< 1 day):
- Stratified sampling implementation
- Tune `prior_virtual_visits` parameter (try 0.5, 2.0, 5.0)
- Tune `cpuct` for stochastic games

### Medium Projects (1-3 days):
- Gumbel AlphaZero implementation
- Progressive widening with k parameter
- Hybrid: progressive for training, sampling for eval

### Research Projects (weeks):
- MuZero adaptation for backgammon
- Stochastic MuZero with explicit chance modeling
- Learned outcome importance weighting

---

## Conclusions: Gumbel and Stochastic MCTS (2026-01-24)

### What We Tried

1. **Gumbel MCTS** (6-hour runs with 50 and 100 simulations)
   - Did not show improvement over standard MCTS
   - Added complexity without measurable benefit at this scale

2. **Stochastic MCTS with Sampling** (`chance_mode=:sampling`)
   - 80.0% vs Random (vs 82.5% for deterministic)
   - 63.6% vs GnuBG 0-PLY (vs 67.0% for deterministic)
   - Head-to-head: 48% vs 52% (not statistically significant)

3. **Stochastic MCTS with Progressive Widening** (`chance_mode=:progressive`)
   - Similar results to sampling mode
   - Added overhead without clear benefit

### Key Findings

**Neither Gumbel nor stochastic chance node handling improved performance at this scale.**

Possible explanations:
- 4-6 hour training may be insufficient to see benefits
- Network capacity (width=128, depth=6) may be too small
- Standard MCTS with hidden stochasticity works surprisingly well for backgammon
- The dice sampling inherent in MCTS rollouts may already provide sufficient exploration

### Recommendation

**Focus on deterministic MCTS baseline with larger networks before revisiting stochastic approaches.**

Next steps:
1. Establish strong deterministic baseline with larger NN
2. If GPU allows, test if larger network improves learning efficiency
3. Revisit stochastic approaches only after seeing diminishing returns from network scaling

---

## Experiment 4: Larger Network Baseline (2026-01-24)

### Goal
Test if larger neural network improves performance, given GPU is underutilized.

### Experimental Metadata (Publication Record)

| Field | Value |
|-------|-------|
| Git commit | fd39c82bfd14b5737ef3b3709db6806b9df7c9e3 |
| Hardware | NVIDIA GeForce RTX 4090, 24GB VRAM |
| Julia version | 1.12.4 |
| Start time (Large NN) | 2026-01-24 10:49:56 |
| End time | 2026-01-24 15:37:00 |
| Total runtime | 4hr 47min |
| Log file | results/largenn_4hr_20260124_104941.log |
| Session dir | /homeshare/projects/AlphaZero.jl/sessions/largenn-4hr-20260124-104946/ |

### Results: Large Network Run (FCResNet 256×8)

#### Network Statistics
- Parameters: 1,523,365 (1,506,048 regularized)
- Memory per MCTS node: 16,712 bytes
- Samples/sec: 24-37 (avg ~25)

#### Training Progress

| Iter | Arena Reward | AZ vs Random (1000g) | Random vs AZ (1000g) |
|------|--------------|---------------------|---------------------|
| 0 | - | -0.84 | -1.10 |
| 1 | +0.25 | -0.39 | -1.16 |
| 2 | +0.20 | -0.37 | -1.22 |
| 3 | +0.20 | -0.27 | -1.32 |
| 4 | +0.60 | -0.13 | -1.28 |
| 5 | +0.50 | -0.20 | -1.30 |

#### Performance Interpretation

**Benchmark rewards** (avg reward per game, range -2 to +2 for gammon/backgammon):
- "AZ vs Random": AlphaZero plays first (white)
- "Random vs AZ": Random plays first (white)

**Key observations:**
1. **First-player disadvantage**: AZ consistently performs worse when playing first
   - Best: -0.13 (iter 4) when playing first vs -1.28 when Random plays first
   - This suggests backgammon may favor the second player, or network hasn't learned first-player strategy

2. **Improvement trajectory**:
   - Untrained: AZ loses badly in both positions
   - After 5 iterations: AZ wins convincingly as second player, struggles as first player

3. **Arena vs Benchmark discrepancy**:
   - Arena (20 games): Shows +0.50 to +0.60
   - Benchmark (2000 games): Shows AZ still losing as first player
   - Arena may have high variance due to small sample size

#### Timing Analysis

| Phase | Time |
|-------|------|
| Iteration 0 (init benchmark) | ~25 min |
| Per iteration average | ~50 min |
| - Self-play (250 games) | ~12 min |
| - Learning | ~3 min |
| - Arena (20 games) | ~2 min |
| - Benchmark (2000 games) | ~33 min |

**Bottleneck**: The 2000-game per-iteration benchmark consumed 66% of training time.

#### Issues Identified

1. **Benchmark overhead**: Running 2000 games per iteration severely limited iterations
   - Got 5 iterations instead of target 24
   - Should move benchmark to final-only for future runs

2. **Low GPU utilization**: Only 2-8% GPU usage
   - Large network (1.5M params) still not saturating RTX 4090
   - Could potentially use even larger network

3. **Slow samples/sec**: 24-37 samples/sec vs expected higher throughput
   - May be CPU-bound during self-play
   - Consider batch size tuning or async improvements

### Design
- **Duration**: 4 hours
- **Iterations**: 24 (targeting ~10 min each)
- **Mode**: Deterministic MCTS (no explicit chance nodes)

### Two Runs

**Run A: Baseline (current network)**
- SimpleNet width=128, depth=6
- ~232K parameters

**Run B: Large Network**
- FCResNet width=256, num_blocks=8
- 1,523,365 parameters (1,506,048 regularized)

### Hyperparameter Adjustments for Larger Network

| Parameter | Baseline | Large NN | Rationale |
|-----------|----------|----------|-----------|
| Network | SimpleNet(128, 6) | FCResNet(256, 8) | More capacity |
| LR range | 1e-3 → 1e-2 | 5e-4 → 5e-3 | Lower LR for larger nets |
| Batch size | 64 | 128 | Better GPU utilization |
| L2 reg | 1e-4 | 5e-5 | Less reg per parameter |
| Games/iter | 250 | 250 | ~10 min iterations |
| MCTS sims | 100 | 100 | Keep constant |
| Arena | 20 games, threshold=0.0 | Same | Always accept, track only |
| Final eval | 1000 games vs Random | Same | Statistical power |

---

## Statistical Significance Guidelines

### Required Sample Sizes for Backgammon

Backgammon has high variance due to dice. Use these guidelines for eval games:

| Effect Size | Games Needed | 95% CI Width |
|-------------|--------------|--------------|
| 10% diff | ~400 | ±5% |
| 5% diff | ~1600 | ±2.5% |
| 3% diff | ~4400 | ±1.5% |

### Formula
```
n = (z² × p × (1-p)) / E²

Where:
- z = 1.96 for 95% confidence
- p = 0.5 (expected win rate)
- E = margin of error (half of CI width)

For 5% margin: n = (1.96² × 0.25) / 0.05² = 384 games
For 2.5% margin: n = (1.96² × 0.25) / 0.025² = 1537 games
```

### Reporting Template

When summarizing results, always include:
```
Win Rate: X% (n games)
95% CI: [X - E, X + E]
Z-score vs baseline: Z
P-value: p
Significant at α=0.05: YES/NO
```

### Final Eval Settings
- Use 1000+ games for meaningful comparisons
- Report exact counts and confidence intervals
- Don't claim significance without p < 0.05

---

## Bug Fix: Player Perspective Mismatch (2026-01-24)

### Symptom
Large asymmetry in benchmark results between AZ playing as P0 vs P1:
- AZ as P0 (white): -0.84 avg reward → ~8% win rate
- AZ as P1 (black): -1.10 avg reward (from P0's perspective)

Going first in backgammon provides only ~1-2% advantage, so this large discrepancy indicated a bug.

### Root Cause Analysis

The issue was in how `push_trace!` (memory.jl) determines the player perspective for value targets.

**Old code:**
```julia
wp = GI.white_playing(GI.init(mem.gspec, s))
z = wp ? wr : -wr
```

The problem: `GI.init(gspec, state)` creates a GameEnv and calls `set_state!`, which could have side effects. In the backgammon implementation, `set_state!` includes:

```julia
if BackgammonNet.is_chance_node(g.game) && !BackgammonNet.game_terminated(g.game)
    BackgammonNet.sample_chance!(g.game, g.rng)
end
```

If a state somehow had `dice == (0, 0)` (chance node), `sample_chance!` would be called, which could:
1. Roll dice
2. Auto-apply PASS|PASS if that's the only legal move
3. Change `current_player` to the next player

This would cause `white_playing` to return the wrong value, flipping the sign of value targets.

### Fix Applied

1. **Added `GI.white_playing(gspec, state)` to game.jl (line 139)**
   - A two-argument version that works directly on states without creating a GameEnv
   - Default implementation falls back to creating a temporary GameEnv

2. **Updated `push_trace!` in memory.jl (line 94)**
   ```julia
   wp = GI.white_playing(mem.gspec, s)  # Direct state access, no side effects
   ```

3. **Implemented direct accessor in both backgammon games**
   ```julia
   GI.white_playing(::GameSpec, state::BackgammonNet.BackgammonGame) = state.current_player == 0
   ```

### Files Modified
- `src/game.jl`: Added `white_playing(gspec, state)` function signature with default
- `src/memory.jl`: Updated `push_trace!` to use direct state access
- `games/backgammon-deterministic/game.jl`: Added direct accessor
- `games/backgammon/game.jl`: Added direct accessor

### Verification
After this fix, training should produce networks that perform symmetrically when playing as P0 or P1, with only the natural ~1-2% first-player advantage seen in backgammon.

### Lesson Learned
When implementing game interfaces with canonical (current-player-relative) observations:
1. Avoid side effects in `set_state!`
2. Provide direct state accessors for perspective-sensitive functions
3. Be especially careful with stochastic games where state restoration may trigger game logic

---

## Experiment 5: Observation Feature Engineering Comparison (2026-01-27)

### Goal
Compare different observation representations to determine which features help learning:
- **MINIMAL**: Basic board state only (780 features)
- **FULL**: Board + game rule features like pip count, home count (1612 features)
- **BIASED**: Full + heuristic/strategic features (3172 features)

### Experimental Metadata

| Field | Value |
|-------|-------|
| Git commit | cc921129 (dirty) |
| BackgammonNet version | v0.2.8 |
| Hardware | NVIDIA GeForce RTX 4090, 24GB VRAM |
| Julia version | 1.12.4 |
| Start time | 2026-01-26 23:16 |
| End time | 2026-01-27 02:54 |
| Total runtime | ~3.5 hours (3 runs) |
| WandB project | alphazero-jl |

### Configuration (All Runs)

| Parameter | Value |
|-----------|-------|
| Network | FCResNetMultiHead (width=128, blocks=3) |
| Total iterations | 70 |
| Games per iteration | 50 |
| MCTS iterations | 100 |
| Workers | 6 |
| Final eval games | 1000 |
| Buffer capacity | 100,000 |

### Results

| Observation | Features | Network Params | Combined Reward | Training Time | Games/min |
|-------------|----------|----------------|-----------------|---------------|-----------|
| MINIMAL | 780 | 339,241 | **1.23** | 57.6 min | 190.6 |
| FULL | 1612 | 429,097 | **1.318** (+7.2%) | 67.4 min | 147.2 |
| BIASED | 3172 | 588,457 | **1.339** (+8.9%) | 72.5 min | 95.6 |

### WandB Runs
- Minimal: https://wandb.ai/sile16-self/alphazero-jl/runs/sxflbfvn
- Full: https://wandb.ai/sile16-self/alphazero-jl/runs/u4xxvj0p
- Biased: https://wandb.ai/sile16-self/alphazero-jl/runs/6rcbhux2

### Analysis

**Performance Ranking**: BIASED > FULL > MINIMAL

1. **Feature engineering helps**: Adding game rule features (FULL) improved performance by 7.2%
2. **Heuristics help more**: Adding strategic bias features improved another 1.6% (8.9% total)
3. **Diminishing returns**: Jump from minimal→full was larger than full→biased
4. **Training speed tradeoff**: More features = slower training (190 → 95 games/min)

### Observation Feature Details (BackgammonNet v0.2.8)

**MINIMAL (30 channels × 26 width = 780)**:
- Basic one-hot board encoding
- Current player indicator
- Dice values

**FULL (62 channels × 26 width = 1612)**:
- Everything in MINIMAL plus:
- Pip counts (normalized)
- Home board counts
- Bar counts
- Bearing off progress
- Game phase indicators

**BIASED (122 channels × 26 width = 3172)**:
- Everything in FULL plus:
- Blot exposure risk
- Prime structure detection
- Racing vs contact indicators
- Anchor positions
- Blocking point values

### Key Insights

1. **Pre-computed features reduce what NN must learn**: The network doesn't need to learn pip counting or blot detection from raw positions - it's provided directly.

2. **Heuristic features encode domain knowledge**: Features like "prime strength" or "anchor quality" encode backgammon strategy that would otherwise require many training iterations to discover.

3. **Speed vs performance tradeoff**: For same wall-clock training time:
   - MINIMAL: ~10,000 games, Combined=1.23
   - BIASED: ~6,400 games, Combined=1.339

   Despite 36% fewer games, BIASED still won due to richer features.

4. **Recommendation**: Use BIASED features for production training. The slower throughput is offset by faster learning per game.

### Lessons Learned

1. **Feature engineering still matters in deep RL**: Even with neural networks, hand-crafted features can significantly accelerate learning.

2. **Domain knowledge via features**: Rather than learning everything from scratch, encoding known-good heuristics as features provides a strong inductive bias.

3. **Test feature sets systematically**: Running controlled experiments with different observation spaces identifies which features actually help.

### Next Steps

1. ~~Train longer with BIASED features to see final performance ceiling~~
2. ~~Test if benefits persist at higher iteration counts (200+)~~
3. ~~Consider hybrid: BIASED for training, MINIMAL for fast eval~~

**UPDATE**: See Experiment 6 below - GnuBG evaluation revealed surprising results!

---

## Experiment 6: GnuBG Evaluation of Observation Types (2026-01-27)

### Goal
Evaluate the three models from Experiment 5 against GnuBG (GNU Backgammon) at 0-ply and 1-ply to test performance against a strong, well-calibrated opponent instead of just a random baseline.

### Experimental Metadata

| Field | Value |
|-------|-------|
| Evaluation script | scripts/eval_vs_gnubg.jl |
| Games per matchup | 500 |
| Matchups per model | 4 (white/black × 0-ply/1-ply) |
| Total games per model | 2000 |
| GnuBG interface | PyCall via BackgammonNet/GnubgInterface.jl |

### Models Evaluated

| Model | Checkpoint | Features | Network Params |
|-------|------------|----------|----------------|
| MINIMAL | /homeshare/projects/AlphaZero.jl/sessions/cluster_20260126_231628/checkpoints/latest.data | 780 | 339,241 |
| FULL | /homeshare/projects/AlphaZero.jl/sessions/cluster_20260127_001636/checkpoints/latest.data | 1612 | 445,737 |
| BIASED | /homeshare/projects/AlphaZero.jl/sessions/cluster_20260127_012254/checkpoints/latest.data | 3172 | 645,417 |

### Results vs GnuBG 0-ply (Neural Network Only)

| Model | As White | As Black | **Combined** | **Win Rate** |
|-------|----------|----------|--------------|--------------|
| **MINIMAL** | -0.034 ± 0.106 | +1.140 ± 0.073 | **+0.553** | **70.2%** |
| FULL | -0.128 ± 0.101 | +1.060 ± 0.077 | +0.466 | 66.8% |
| BIASED | -0.250 ± 0.106 | +1.098 ± 0.081 | +0.424 | 64.4% |

### Results vs GnuBG 1-ply (1-ply Lookahead)

| Model | As White | As Black | **Combined** | **Win Rate** |
|-------|----------|----------|--------------|--------------|
| **MINIMAL** | -0.212 ± 0.111 | +0.994 ± 0.104 | **+0.391** | **62.1%** |
| FULL | -0.240 ± 0.106 | +0.850 ± 0.120 | +0.305 | 58.8% |
| BIASED | -0.298 ± 0.108 | +0.768 ± 0.126 | +0.235 | 55.6% |

### Summary Table

| Model | vs Random | vs GnuBG 0-ply | vs GnuBG 1-ply |
|-------|-----------|----------------|----------------|
| MINIMAL | +1.23 (3rd) | **+0.553 (1st)** | **+0.391 (1st)** |
| FULL | +1.318 (2nd) | +0.466 (2nd) | +0.305 (2nd) |
| BIASED | **+1.339 (1st)** | +0.424 (3rd) | +0.235 (3rd) |

### Key Finding: Rankings Reversed!

**The order completely flips when evaluating against GnuBG vs random:**
- vs Random: BIASED > FULL > MINIMAL
- vs GnuBG: MINIMAL > FULL > BIASED

### Analysis

#### 1. Why BIASED Fails Against GnuBG

The BIASED observation includes heuristic features like:
- Blot exposure risk
- Prime structure detection
- Anchor positions
- Blocking point values

These heuristics encode **one particular style of play**. The network may learn to rely on these pre-computed features rather than understanding the underlying position. When facing GnuBG (which uses different heuristics), this creates a **feature mismatch**.

#### 2. Why MINIMAL Succeeds

With only basic board state (780 features), the MINIMAL network must learn:
- Position evaluation from first principles
- Generalizable patterns that work against any opponent
- Robust features that transfer to different playing styles

This leads to **more generalizable learning** even though it's harder initially.

#### 3. The Random Baseline is Misleading

A random opponent doesn't punish strategic weaknesses:
- BIASED features help exploit random's mistakes
- But these same features may not generalize to rational opponents
- Random play is not representative of competitive backgammon

### Implications

#### For Feature Engineering
1. **Domain heuristics can hurt generalization**: Hand-crafted features may encode brittle strategies
2. **Simpler observations may force better learning**: Network must discover robust representations
3. **Evaluation opponent matters critically**: Always test against calibrated opponents, not just random

#### For Training Pipeline
1. **Use GnuBG for evaluation** during training, not just random baseline
2. **Consider minimal features** for training despite slower initial progress
3. **Heuristic features may be a form of overfitting** to the training distribution

### Color Asymmetry Observation

All models show strong asymmetry:
- As Black (second player): 72-93% win rate
- As White (first player): 38-48% win rate

This suggests:
1. Models may have learned a defensive/reactive style
2. Or the MCTS + NN combination works better in reactive positions
3. Worth investigating in future work

### Updated Recommendations

**Previous recommendation** (from Experiment 5): Use BIASED features for production training.

**New recommendation**:
1. **Use MINIMAL features** for training to develop generalizable play
2. **Evaluate against GnuBG** (not just random) to catch overfitting to weak opponents
3. **Consider FULL features** as a middle ground if minimal is too slow
4. **Avoid heavy heuristic features** unless validated against strong opponents

### Lessons Learned

1. **Benchmark choice is critical**: Results that look good against random may not generalize
2. **More features ≠ better generalization**: Feature engineering can introduce biases
3. **Simple can be stronger**: Forcing the network to learn from scratch may produce more robust play
4. **Test against calibrated opponents early**: Don't wait until the end to test against GnuBG

### Scripts Created

- `scripts/eval_vs_gnubg.jl` - Sequential evaluation against GnuBG
- `scripts/eval_vs_gnubg_parallel.jl` - Parallel version (PyCall threading issues)
- `scripts/eval_vs_gnubg_batch.jl` - Batch evaluation helper
- `scripts/run_gnubg_eval_all.sh` - Runner for all 3 models

### Next Steps

1. **Retrain with MINIMAL features** for longer (200+ iterations) and track vs GnuBG
2. **Add GnuBG evaluation to training loop** as a periodic benchmark
3. **Investigate color asymmetry** - why do all models struggle as white?
4. **Test intermediate feature sets** - find the sweet spot between minimal and biased
5. **Try curriculum learning** - start with minimal, gradually add features

---

## Experiment 7: BackgammonNet v0.3.2 Observation Formats (2026-01-28)

### Goal
Compare different observation formats in BackgammonNet v0.3.2, which introduced simplified and restructured observation types. Test whether the hybrid format (separating board from globals) improves learning over flat vectors.

### BackgammonNet v0.3.2 Changes

The new version significantly simplified observation sizes:

| Obs Type | v0.3.1 Size | v0.3.2 Size | Format |
|----------|-------------|-------------|--------|
| minimal_flat | 780 | **330** | Flat vector |
| full_flat | 1612 | **362** | Flat vector |
| full_hybrid | N/A | **362** | (board=12×26, globals=50) |

The hybrid format separates board representation (12 channels × 26 points = 312) from global features (dice, bar, off counts = 50).

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Game | backgammon-deterministic (short_game=true) |
| Network | FCResNetMultiHead (width=128, blocks=3) |
| Workers | 6 |
| Iterations | 70 |
| Games/iteration | 50 |
| MCTS iterations | 100 |
| Final eval games | 1000 |

### Models Trained

| Model | Obs Type | Features | Network Params | Session |
|-------|----------|----------|----------------|---------|
| full_hybrid | :full_hybrid | 362 | 285,737 | cluster_20260127_173723 |
| full_flat | :full_flat | 362 | 285,737 | cluster_20260127_181701 |
| minimal_flat | :minimal_flat | 330 | 285,737 | cluster_20260128_105330 |

### All Training Runs Summary (70 iterations each)

| Model | Version | Features | Wall Clock | Notes |
|-------|---------|----------|------------|-------|
| MINIMAL | v0.3.1 | 780 | **57.6 min** | Fastest, best generalization |
| FULL | v0.3.1 | 1612 | **67.4 min** | |
| BIASED | v0.3.1 | 3172 | **72.5 min** | Slowest |
| full_hybrid | v0.3.2 | 362 | **~75 min** | Includes ~15 min final eval |
| full_flat | v0.3.2 | 362 | **~81 min** | Includes ~15 min final eval |
| minimal_flat | v0.3.2 | 330 | **61 min** | 59 min train + 2 min eval |

**Notes:**
- All runs used identical config: 70 iterations, 50 games/iter, 100 MCTS iters, 6 workers
- v0.3.1 times are training only; v0.3.2 times include 1000-game final evaluation (~15 min)
- Larger feature counts = slower training (more compute per forward pass)
- Network params vary by input size: 339K (780 features) to 645K (3172 features)

### Training Observations

**Loss increased during training** for both models:
- Iterations 1-10: Loss dropped 5.4 → 3.9 (normal)
- Iterations 11-70: Loss rose 4.2 → 6.9 (unusual)

This occurred after the replay buffer filled at 100K samples. Possible causes:
1. Target instability from self-play
2. Buffer turnover replacing "easier" positions
3. Policy entropy collapse

Despite rising loss, playing strength remained stable.

### Results vs Random (1000 games)

| Model | Combined | As White | As Black |
|-------|----------|----------|----------|
| **full_hybrid** | **+1.201** | **+0.604** | +1.798 |
| minimal_flat | +1.172 | +0.636 | +1.708 |
| full_flat | +1.109 | +0.382 | +1.836 |

**full_hybrid wins vs random**, especially as white. minimal_flat is second.

### Results vs GnuBG 0-ply (500 games each direction)

| Model | As White | As Black | **Combined** | **Win Rate** |
|-------|----------|----------|--------------|--------------|
| full_flat | -0.204 ± 0.100 | +1.114 ± 0.076 | **+0.455** | **66.3%** |
| full_hybrid | -0.258 ± 0.104 | +1.096 ± 0.081 | +0.419 | 65.3% |
| minimal_flat | -0.262 ± 0.105 | +1.100 ± 0.083 | +0.419 | 64.7% |

### Results vs GnuBG 1-ply (500 games each direction)

| Model | As White | As Black | **Combined** | **Win Rate** |
|-------|----------|----------|--------------|--------------|
| full_flat | -0.332 ± 0.102 | +0.912 ± 0.076 | **+0.290** | **57.6%** |
| full_hybrid | -0.448 ± 0.106 | +0.720 ± 0.081 | +0.136 | 53.1% |
| minimal_flat | -0.292 ± 0.107 | +0.782 ± 0.113 | +0.245 | 57.7% |

### Complete Comparison Across All Experiments

| Model | Obs Size | vs Random | vs GnuBG 0-ply | vs GnuBG 1-ply |
|-------|----------|-----------|----------------|----------------|
| **Previous (v0.3.1):** |
| MINIMAL | 780 | +1.23 | **+0.553** | **+0.391** |
| FULL | 1612 | +1.318 | +0.466 | +0.305 |
| BIASED | 3172 | +1.339 | +0.424 | +0.235 |
| **New (v0.3.2):** |
| full_flat | 362 | +1.109 | +0.455 | **+0.290** |
| minimal_flat | 330 | +1.172 | +0.419 | +0.245 |
| full_hybrid | 362 | +1.201 | +0.419 | +0.136 |

### Key Findings

#### 1. Rankings Reverse Again
- **vs Random**: full_hybrid (+1.201) > minimal_flat (+1.172) > full_flat (+1.109)
- **vs GnuBG**: full_flat (+0.290) > minimal_flat (+0.245) > full_hybrid (+0.136)

Same pattern as Experiment 6: better vs random ≠ better generalization.

#### 2. Flat Format Generalizes Better
Despite identical feature count (362), flat vectors outperform hybrid format:
- 0-ply: +0.455 vs +0.419 (+8.6%)
- 1-ply: +0.290 vs +0.136 (+113%!)

The hybrid separation may be a form of inductive bias that hurts generalization.

#### 3. Smaller Observations Competitive
v0.3.2 flat formats nearly match old versions with far fewer features:
- FULL (1612): +0.305 vs GnuBG 1-ply
- full_flat (362): +0.290 → only 5% worse with **4.5× fewer features**
- minimal_flat (330): +0.245 → still competitive with **4.9× fewer features**
- BIASED (3172): +0.235 → minimal_flat beats it with **10× fewer features**!

#### 4. More Features in Same Format = Better
Comparing flat variants in v0.3.2:
| Model | Features | vs GnuBG 1-ply |
|-------|----------|----------------|
| full_flat | 362 | +0.290 |
| minimal_flat | 330 | +0.245 |
| full_hybrid | 362 | +0.136 |

Within flat format, more features (362 vs 330) improves performance by ~18%.

### Analysis

#### Why Flat Beats Hybrid?

1. **MLP architecture favors flat inputs**: FCResNet processes all features equally through dense layers. The hybrid structure doesn't provide spatial locality benefits without conv layers.

2. **Forced feature mixing**: Flat format mixes board and globals immediately, potentially enabling richer feature interactions in early layers.

3. **Hybrid may introduce harmful bias**: Separating board from globals might prevent useful cross-feature learning.

#### Implications for Feature Engineering

1. **Format matters as much as content**: Same information in different structures yields different results.

2. **Match format to architecture**: Use hybrid/structured observations only with architectures that exploit structure (e.g., conv networks for board, separate heads for globals).

3. **Potential for pruning**: Since larger flat observations help, we could:
   - Analyze feature importance via gradients
   - Identify unused features
   - Prune network for efficiency

### Feature Importance Analysis (Future Work)

To identify which features the network actually uses:

1. **Gradient-based importance**:
   ```julia
   # Compute gradient w.r.t. inputs
   grads = gradient(x -> sum(network(x)), observation)
   importance = mean(abs.(grads), dims=batch)
   ```

2. **First layer weight analysis**:
   ```julia
   W = network.layers[1].weight
   importance = sum(abs.(W), dims=1)
   ```

3. **Ablation study**: Zero out feature groups and measure performance drop

4. **Permutation importance**: Shuffle features and measure accuracy degradation

### Updated Rankings (All Models)

**vs GnuBG 1-ply (most meaningful benchmark):**
1. MINIMAL (780 features) = **+0.391**
2. FULL (1612 features) = +0.305
3. **full_flat (362 features) = +0.290** ← 4.5× smaller than FULL!
4. **minimal_flat (330 features) = +0.245** ← Smallest, still competitive!
5. BIASED (3172 features) = +0.235
6. full_hybrid (362 features) = +0.136

### Conclusions

1. **MINIMAL (v0.3.1) remains best** for generalization despite being oldest/simplest
2. **full_flat is highly competitive** with 4.5× fewer features than FULL
3. **minimal_flat nearly matches BIASED** with 10× fewer features (330 vs 3172)
4. **Hybrid format hurts this architecture** - don't use without conv layers
5. **Feature count helps within same format** - full_flat > minimal_flat
6. **Rising loss during training** didn't prevent learning useful features

### Next Steps

1. ~~**Train minimal_flat (330 features)** with v0.3.2 for fair comparison~~ ✅ Done!
2. **Implement feature importance analysis** to identify unused features
3. **Try conv architecture with hybrid format** to test spatial benefits
4. **Investigate training loss dynamics** - why does loss rise after buffer fills?
5. **Longer training** to see if full_flat catches up to MINIMAL

---

## Experiment 8: Stochastic Wrapper — Debugging & Verification (2026-02-10/11)

### Goal
Replace the deterministic `step!` wrapper (which hides dice inside `play!`) with a stochastic wrapper that exposes dice as explicit chance nodes. This is the prerequisite for using the bear-off table at its mathematically correct pre-dice location in MCTS.

### Implementation
- `games/backgammon-deterministic/game.jl`: `GI.play!` → `apply_action!` only (no auto-dice). `GI.apply_chance!` rolls dice. Chance nodes visible to MCTS.
- `src/batched_mcts.jl` + `src/mcts.jl`: Passthrough mode — sample dice immediately at chance nodes, no tree entries for chance nodes.
- Self-play + eval game loops handle forced passes + dice externally.

### Commits
| Commit | Description |
|--------|-------------|
| 9d167d5 | Add stochastic game wrapper + passthrough chance nodes in MCTS |
| 7de741a | Fix: stop recording chance nodes in trace (2x buffer dilution) |
| 8bfd746 | Optimize MCTS passthrough: eliminate allocations per chance node |
| 77f68d7 | Fix pswitch bug: remove _handle_forced_pass! from GI.play!/GI.apply_chance! |
| 9efef1b | Skip oracle evaluation for single-option states in batched MCTS |
| 3f7bb4f | Skip oracle evaluation for single-option states in standard MCTS |

### Bugs Found & Fixed

#### Bug 1: Trace Dilution (commit 7de741a)
Recording chance nodes in the training trace doubled the buffer sample count, halving data diversity. Fix: only record decision nodes in trace.

#### Bug 2: pswitch Bug (commit 77f68d7)
`_handle_forced_pass!` inside `GI.play!`/`GI.apply_chance!` auto-played opponent forced passes, causing invisible player switches. MCTS saw stale `pswitch=TRUE` when the turn bounced back to the same player, flipping the value sign incorrectly (catastrophic). Fix: removed `_handle_forced_pass!` from both functions. Forced passes are now visible to MCTS as single-option decision nodes.

#### Bug 3: Oracle Waste (commits 9efef1b + 3f7bb4f)
Forced-pass states consumed ~40% of MCTS oracle budget during training. Fix: single-option states skip oracle evaluation — create trivial tree entry (P=[1.0], Vest=0) and continue traversal. Value backpropagates from child.

### Progressive Fix Impact (same old PER 50-iter model, eval only)

| Eval Version | vs GnuBG 0-ply | vs GnuBG 1-ply |
|---|---|---|
| All bugs present | +0.919 (81%) | +0.513 (65%) |
| + Allocation fix | +0.903 (81%) | +0.545 (66%) |
| + Pswitch fix | +0.995 (81%) | +0.733 (69%) |
| + Skip-force eval | +1.105 (84%) | +0.813 (71%) |
| Original step! eval (reference) | +1.06 (83%) | +0.87 (74%) |

### Training Quality (50-iter PER, skip-force eval for all)

| Model | vs GnuBG 0-ply | vs GnuBG 1-ply |
|---|---|---|
| Old PER (step! training) | +1.105 (84%) | +0.813 (71%) |
| **New PER (stochastic training)** | **+1.048 (82%)** | **+0.764 (69%)** |
| Gap | -0.057 | -0.049 (within 1σ) |

### Key Metrics
- Throughput: ~350 games/min (stochastic) vs ~390 games/min (old step!). ~10% slower due to intermediate state traversal.
- Samples/game: ~163 (stochastic) vs ~198 (old step!). Difference is game length variance.
- Training time: 69.4 min for 50 iterations.

### Conclusions

1. **Stochastic wrapper matches old step! baseline** within statistical noise (+0.764 vs +0.813).
2. **Three bugs caused cumulative 60% regression**: trace dilution + pswitch + oracle waste.
3. **Skip-force optimization is critical**: without it, forced passes waste ~40% of oracle budget, causing major training regression.
4. **pswitch is the most dangerous bug**: invisible player switches cause catastrophic value sign flips. Never auto-play forced passes inside MCTS-visible functions.
5. **Stochastic wrapper is ready** for bear-off table integration at chance nodes.

### Sessions
| Session | Description | vs GnuBG 1-ply |
|---|---|---|
| distributed_20260210_181748_per | Stochastic, all bugs | +0.081 |
| distributed_20260210_201425_per | Trace fix only | +0.286 |
| distributed_20260211_000703_per | + Pswitch fix | +0.210* |
| distributed_20260211_013048_per | + Skip-force (final) | +0.764 |

*Pswitch fix model evaluated without skip-force in eval MCTS, so eval oracle also wasted on forced passes.
