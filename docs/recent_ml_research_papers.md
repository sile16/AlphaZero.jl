# SOTA AlphaZero-Style Backgammon Report, 2026

**Topic:** MCTS structure, stochastic/chance nodes, temperature scaling, dice handling, MuZero vs AlphaZero, observation design, neural-network design, training procedure, and reduced-hardware guidance for building a strong AlphaZero-style backgammon engine.

**Main conclusion:** the best practical architecture for backgammon in 2026 is not vanilla AlphaZero and not deterministic MuZero. It is an **Explicit-Chance Gumbel AlphaZero** system:

```text
known legal move generator
+ known dice simulator
+ AlphaZero policy/value learning
+ afterstate tree
+ decision nodes for checker/cube choices
+ chance nodes for dice
+ Gumbel/Sequential-Halving root search for low-simulation regimes
+ distributional value head for win/gammon/backgammon/cube equity
+ replay + reanalyse
+ paired-dice evaluation
```

Use **Stochastic MuZero** as the main conceptual reference for stochastic tree structure. Use **AlphaZero-style known-rule search** for production backgammon, because the dice probabilities and legal transitions are exact, cheap, and known.

---

## Table of contents

1. [Executive architecture recommendation](#1-executive-architecture-recommendation)
2. [MCTS structure findings](#2-mcts-structure-findings)
3. [Stochastic nodes, dice, and random-event handling](#3-stochastic-nodes-dice-and-random-event-handling)
4. [Temperature scaling and root exploration](#4-temperature-scaling-and-root-exploration)
5. [MuZero vs AlphaZero for backgammon](#5-muzero-vs-alphazero-for-backgammon)
6. [Observation design](#6-observation-design)
7. [Action and policy design](#7-action-and-policy-design)
8. [Neural-network design](#8-neural-network-design)
9. [Training targets and loss design](#9-training-targets-and-loss-design)
10. [Training loop and curriculum](#10-training-loop-and-curriculum)
11. [Training-resource guidance](#11-training-resource-guidance)
12. [Reduced-hardware plans](#12-reduced-hardware-plans)
13. [Evaluation methodology](#13-evaluation-methodology)
14. [Implementation blueprint](#14-implementation-blueprint)
15. [High-impact engineering tricks](#15-high-impact-engineering-tricks)
16. [Reference list](#16-reference-list)

---

# 1. Executive architecture recommendation

## 1.1 Recommended system

Build:

```text
Explicit-Chance Gumbel AlphaZero for Backgammon

- exact rule simulator
- exact legal checker-play generation
- legal final-afterstate action representation
- decision/chance/afterstate MCTS
- exact 21-roll chance expansion near the root
- stratified passthrough chance sampling deeper in the tree
- Gumbel root search when simulation count is low
- candidate-action policy scorer
- distributional value head over normal/gammon/backgammon outcomes
- native cube and match-equity heads
- replay + reanalyse
- paired-dice evaluation against GNUbg/XG or other strong baselines
```

## 1.2 Why this is the right structure

Backgammon is not a deterministic perfect-information game in the AlphaZero sense. It alternates between:

```text
player choice  -> deterministic checker/cube action
random event   -> dice roll
player choice  -> deterministic checker/cube action
random event   -> dice roll
...
```

A vanilla AlphaZero tree that assumes action -> next state is structurally incomplete. The correct tree uses:

```text
Decision node: player chooses a legal checker or cube action
Afterstate: board after the chosen action, before dice
Chance node: dice outcome sampled or enumerated
Decision node: next player chooses a legal action
```

The Stochastic MuZero paper formalizes this idea with afterstates and chance nodes, and its backgammon experiments are the closest direct research reference for this project. In backgammon, Stochastic MuZero with 1600 simulations per move reached GNUbg-level play while deterministic MuZero was not the right fit for stochastic transitions. See Reference [1].

---

# 2. MCTS structure findings

## 2.1 Use three node types

Use explicit node types in the tree:

```text
DECISION node
    - owned by a player
    - contains legal checker plays or cube decisions
    - uses pUCT/Gumbel/Sequential-Halving selection

AFTERSTATE node
    - board after a player action
    - before the next dice event
    - may be folded into the parent action edge, but should be explicit in code/data

CHANCE node
    - dice roll event
    - uses true dice probabilities
    - backed up by expectation, not maximization
```

The tree structure should look like:

```text
(board, dice, player)
    decision: choose legal play
        -> afterstate board
            chance: roll dice
                -> (new board, new dice, opponent)
                    decision: opponent legal play
```

## 2.2 Decision nodes are optimized; chance nodes are averaged

At a decision node, use search to improve the policy:

```text
select action ~= argmax_a Q(s,a) + exploration_bonus(s,a)
```

At a chance node, use the true dice expectation:

```text
V(afterstate) = sum_roll P(roll) * V(next_state_after_roll)
```

Never treat dice as an adversary. Never take a max over dice outcomes.

## 2.3 Use afterstates as first-class training objects

The cleanest backgammon value interpretation is:

```text
Q(board, dice, play) = V(afterstate)
```

Then:

```text
V(afterstate) = E_roll[V(next_board, roll)]
```

This gives smoother comparisons between candidate checker plays, because many legal move sequences collapse into the same final afterstate.

## 2.4 Collapse duplicate move sequences

Backgammon move-generation creates duplicates because different move orders can reach the same final board.

Example:

```text
roll 3-1:
    move A by 3, move B by 1
    move B by 1, move A by 3
```

These may produce the same final board. Treat the final board as the action identity:

```text
action_id = hash(canonical_afterstate)
```

This improves:

- policy-target stability
- MCTS branching factor
- replay efficiency
- supervised/distillation quality

## 2.5 Root search and internal search should differ

Inside the tree, pUCT is fine.

At the root, the objective is different: identify the best final move with limited simulations. Gumbel AlphaZero/MuZero is directly useful here because it targets improved action selection at low simulation budgets using Gumbel-Top-k and Sequential Halving.

Recommended split:

```text
root:
    Gumbel sample top-m candidate plays
    allocate simulations by Sequential Halving
    choose final play from searched candidates

internal nodes:
    pUCT with legal-action masking
```

Good starting values:

```text
m = min(num_legal_plays, 16 or 32)
S = 32, 64, 128, 256, 800, or 1600 depending on hardware
```

---

# 3. Stochastic nodes, dice, and random-event handling

## 3.1 Backgammon dice distribution

Use 21 unordered dice outcomes:

```text
6 doubles:       probability 1/36 each
15 non-doubles:  probability 2/36 each
21 roll classes total
```

Do not learn this distribution unless the goal is specifically to study learned stochastic models. For production backgammon, the dice model is exact and should be hard-coded.

## 3.2 Full chance expansion is accurate but expensive

A simple branching calculation:

```text
average legal checker plays: ~20
unordered dice outcomes:      21
opponent legal replies:       ~20

one decision + chance + reply ~= 20 * 21 * 20 = 8400 branches
```

Full chance expansion everywhere is usually too expensive. Use exact chance expansion selectively.

Recommended budget policy:

```text
root action layer:
    enumerate all legal plays

first chance layer:
    enumerate all 21 dice outcomes if budget allows

deep chance layers:
    stratified sampling or progressive widening
```

## 3.3 Passthrough chance nodes

A chance node can be implemented as a passthrough event:

```text
decision node -> afterstate -> sampled dice -> next decision node
```

Instead of materializing all 21 dice children immediately, sample or stratify one dice child per visit. This keeps the tree narrow while preserving unbiased or low-bias estimates if the dice sampling is correct.

## 3.4 Controlled random-event exploration

Use deterministic stratified dice scheduling rather than naive randomness.

For known dice probabilities:

```text
choose roll r = argmax_r P_true(r) / (N_r + 1)
```

Where:

```text
P_true(r) = 1/36 for doubles
P_true(r) = 2/36 for non-doubles
N_r       = number of times this chance child has been sampled
```

This produces coverage proportional to true dice probability and reduces variance.

## 3.5 Three chance-node modes

Use three modes:

```text
Mode A: exact expectation
    expand all 21 dice outcomes
    value = sum_r P(r) * V(child_r)

Mode B: stratified passthrough
    pick r = argmax_r P(r)/(N_r+1)
    evaluate one child per visit

Mode C: progressive widening
    initially expose only a few dice outcomes
    add more outcomes as visit count grows
```

Recommended use:

```text
root / high-visit afterstates: exact expectation
middle tree:                  stratified passthrough
very deep tree:               sampled passthrough
```

## 3.6 Oversampling rare dice

Do not temperature-scale dice. Do not bias the actual environment distribution.

Acceptable:

```text
- stratified dice sampling during MCTS
- paired/common-random dice during evaluation
- separate rare-event diagnostic datasets
- importance-weighted oversampling for auxiliary training
```

Dangerous:

```text
- changing dice probabilities in value backup without correction
- training the main value target on biased dice without importance weighting
- treating rare dice as adversarial events
```

---

# 4. Temperature scaling and root exploration

## 4.1 Policy target temperature

AlphaZero-style training uses visit counts as the improved policy target:

```text
pi(a | s) proportional to N(s,a)^(1/tau)
```

Use this only at decision nodes.

Do not apply a policy loss to chance nodes. Dice outcomes are environment events, not actions chosen by the agent.

## 4.2 Self-play temperature schedule

A direct Go-style “first 30 moves tau=1” rule is not ideal for backgammon. Backgammon phases are more important than raw move count.

Recommended phase-aware schedule:

```text
opening / early contact:
    tau = 1.0

complex middle game:
    tau = 0.5 or 0.25

race, bearoff, forced play, cube-critical positions:
    tau -> 0

evaluation:
    tau -> 0
```

A stronger adaptive version:

```text
if top-2 searched candidate values are close:
    sample using tau = 0.5 or 1.0
else:
    play greedily
```

This keeps diversity where the position is genuinely ambiguous and avoids intentionally bad obvious plays.

## 4.3 Dice temperature

Dice should not be temperature-scaled.

Correct:

```text
P(1-1) = 1/36
P(1-2) = 2/36
...
P(6-6) = 1/36
```

Wrong for normal training/evaluation:

```text
P_train(roll) = softmax(dice_logits / tau)
```

## 4.4 Dirichlet noise scaling

AlphaZero adds root Dirichlet noise to encourage exploration:

```text
P_root = (1 - epsilon) * P_network + epsilon * Dirichlet(alpha)
```

AlphaZero used different alpha values for chess, shogi, and Go based on typical action count. A practical rule is to keep total concentration roughly constant:

```text
alpha ~= 10 / number_of_legal_actions
```

For backgammon with roughly 20 legal final-afterstate actions:

```text
alpha ~= 0.5
epsilon ~= 0.10 to 0.25
```

If using Gumbel root search, root Dirichlet noise is usually unnecessary.

---

# 5. MuZero vs AlphaZero for backgammon

## 5.1 AlphaZero is the better production base

AlphaZero assumes known rules and uses the simulator inside MCTS. That is ideal for backgammon because:

```text
- legal moves are exact
- dice probabilities are exact
- state transitions are cheap
- terminal scoring is exact
```

For a production-strength backgammon engine, use known-rule AlphaZero plus explicit chance nodes.

## 5.2 Deterministic MuZero is not enough

Standard MuZero learns a deterministic latent dynamics model. This is powerful for deterministic or nearly deterministic domains, but stochastic games require separate modeling of controllable actions and random events.

Backgammon needs:

```text
state --player action--> afterstate --dice outcome--> next state
```

Not:

```text
state --player action--> next state
```

## 5.3 Stochastic MuZero is the main conceptual reference

Stochastic MuZero factors dynamics into:

```text
representation h:
    observation -> latent state

afterstate dynamics phi:
    latent state + action -> afterstate

chance predictor sigma:
    afterstate -> distribution over chance codes

chance dynamics g:
    afterstate + chance code -> next state + reward

prediction f:
    latent state -> policy + value
```

For backgammon, replace learned chance prediction and chance dynamics with the true dice simulator:

```text
sigma(afterstate) = true dice distribution
g(afterstate, roll) = exact backgammon transition to next decision state
```

That keeps the good structure and removes model error.

## 5.4 When MuZero still matters

Use Stochastic MuZero if the research goal is:

```text
- learning stochastic rules from observation
- transferring across stochastic games
- partially observable or hidden dynamics
- no access to the simulator
- studying learned latent chance variables
```

Use explicit-chance AlphaZero if the goal is strongest backgammon performance per unit compute.

---

# 6. Observation design

## 6.1 Encode from current-player perspective

Always normalize the board so the side to move is “own side.”

```text
own checkers move from high points toward home
opponent checkers move in the opposite direction
own home board always occupies the same encoded region
```

Benefits:

```text
- halves the effective state space
- simplifies value sign handling
- improves data efficiency
- improves candidate-afterstate comparison
```

## 6.2 Minimal Markov state

A cubeless checker-play state needs:

```text
- 24 board points
- own and opponent checker counts
- own and opponent bar counts
- own and opponent borne-off counts
- side to move
- dice roll if already rolled
```

A full competitive cubeful/match state also needs:

```text
- cube value
- cube owner: centered / owned by player / owned by opponent
- match score
- match length
- Crawford state
- Jacoby / beaver / variant flags if supported
```

## 6.3 Board-point features

Recommended point features:

```text
for each of 24 points:
    own_count_raw_normalized
    opp_count_raw_normalized
    own_ge_1
    own_ge_2
    own_ge_3
    own_ge_4
    own_ge_5
    opp_ge_1
    opp_ge_2
    opp_ge_3
    opp_ge_4
    opp_ge_5
```

Global scalar or one-hot features:

```text
own_bar
opp_bar
own_borne_off
opp_borne_off
pip_count_own
pip_count_opp
pip_diff
dice_roll_one_hot_21
is_double
cube_value_log2
cube_owner
match_score
Crawford flag
```

If the project goal is strict tabula rasa, omit pip count and other engineered features. If the goal is strongest reduced-hardware performance, include them.

## 6.4 History planes are usually unnecessary

Go-style history planes are not essential because backgammon’s current board, dice, cube, and match state are enough for Markov play.

Possible exceptions:

```text
- opponent modeling
- clock/time management
- detecting repeated cube-offer behavior in human play
- nonstandard rules or external metadata
```

## 6.5 Observation variants worth testing

### Variant A: engineered vector

```text
flat feature vector
+ MLP / residual MLP
```

Best for speed and reduced hardware.

### Variant B: pointwise 1D board tensor

```text
24 point positions
x point features
+ 1D residual CNN
+ global feature fusion
```

Best balance of speed and structure.

### Variant C: point-token transformer

```text
24 point tokens
+ bar token
+ bearoff token
+ dice token
+ cube/match token
+ relative position embeddings
```

Useful for high-end experiments, but slower during search.

---

# 7. Action and policy design

## 7.1 Avoid a giant flat action vocabulary

Backgammon actions are hard to encode as a static action space:

```text
- legal actions depend strongly on dice
- doubles allow up to four moves
- move order can differ while final board is identical
- legal action count changes sharply by position
```

A giant policy head over all possible move sequences will waste capacity and create masking complexity.

## 7.2 Use candidate-afterstate scoring

Generate all legal final afterstates for the current board and dice, then score them.

```text
state_embedding = trunk(board, dice, cube, match)
afterstate_embedding = encoder(afterstate)
delta_embedding = encode(afterstate - board)

logit(play) = MLP([state_embedding, afterstate_embedding, delta_embedding])
```

Then:

```text
policy = softmax(logits over legal candidate afterstates only)
```

Benefits:

```text
- handles variable action counts naturally
- collapses duplicate move orders
- works with sampled-action MCTS
- aligns naturally with afterstate value learning
- easier to extend to cube decisions
```

## 7.3 Cube actions

For cubeful play, decision states are not only checker-play states.

Pre-roll cube decision:

```text
actions:
    no double
    double
```

If double is offered:

```text
opponent actions:
    take
    pass
```

Then:

```text
if no double or take:
    roll dice -> checker-play state
if pass:
    terminal cube result
```

Use separate heads for:

```text
checker-play policy
cube-offer policy
take/pass policy
value / match-equity
```

---

# 8. Neural-network design

## 8.1 Shared trunk with multiple heads

Use one shared representation trunk and several heads:

```text
shared board trunk
    -> candidate checker-play policy head
    -> value / outcome distribution head
    -> cube decision head
    -> optional auxiliary heads
```

AlphaGo Zero and AlphaZero showed the practical strength of a shared residual trunk with policy and value heads. For backgammon, add distributional/cube-specific heads.

## 8.2 Small reduced-hardware model

Recommended first strong model:

```text
Residual MLP

input:
    engineered board vector
    dice features
    cube/match features

trunk:
    4-8 residual MLP blocks
    width 256-1024

heads:
    candidate-action scorer
    distributional outcome value
    cube policy
    auxiliary features
```

This is fast, easy to batch, and good for single-GPU training.

## 8.3 1D residual CNN model

Recommended balanced model:

```text
point tensor: [24 points x features]
1D residual convolution blocks
global pooling
fusion with dice/cube/match features
candidate-afterstate scoring head
value distribution head
```

This exploits the one-dimensional structure of the board and is faster than a transformer.

## 8.4 Transformer model

High-end option:

```text
point tokens
bar/bearoff tokens
dice token
cube/match token
relative position embedding
candidate-afterstate cross-attention
```

Use only when inference throughput is still high enough to support many MCTS simulations. Larger networks can reduce playing strength at a fixed time limit if they reduce the number of simulations too much.

## 8.5 Distributional value head

Backgammon terminal outcomes are multi-outcome, not binary.

Cubeless money-game outcome classes:

```text
-3: lose backgammon
-2: lose gammon
-1: lose normal
+1: win normal
+2: win gammon
+3: win backgammon
```

Recommended head:

```text
p = softmax(outcome_logits)
equity = dot(p, [-3, -2, -1, +1, +2, +3])
```

For cubeful/match play, add:

```text
- cubeless outcome distribution
- cubeful equity
- match-winning probability
- double/no-double logits
- take/pass logits
```

## 8.6 Relative value accuracy matters

TD-Gammon’s absolute equity estimates could be off by meaningful amounts while its move choices remained strong because errors across similar positions were correlated. The practical lesson is:

```text
ranking similar candidate afterstates correctly matters more than perfect absolute equity early in training
```

That strongly supports a candidate-afterstate scorer and afterstate value design.

## 8.7 Useful auxiliary heads

For reduced hardware, auxiliary targets can improve representation quality:

```text
pip count
race/contact/crashed phase
bearoff database value
hit probability next roll
bar-enter probability
gammon probability
backgammon probability
next-roll equity distribution
cube take point / cash point estimate
```

For a pure AlphaZero experiment, keep these out. For a strongest-practical-engine project, use them.

---

# 9. Training targets and loss design

## 9.1 Per-position training record

Store:

```text
board
side to move
dice
cube state
match state
legal candidate afterstates
selected action
MCTS visit counts over legal candidates
MCTS root value
final game outcome
final match outcome
```

For cubeless distributional training, store terminal outcome:

```text
z_class in {-3, -2, -1, +1, +2, +3}
```

## 9.2 Policy target

At each decision node:

```text
pi_target(a) = N(s,a)^(1/tau) / sum_b N(s,b)^(1/tau)
```

At evaluation time:

```text
choose argmax_a N(s,a)
```

Do not train policy targets for chance nodes.

## 9.3 Value target

Options:

```text
scalar equity target:
    z = final points won/lost, normalized if desired

distributional target:
    one-hot or smoothed distribution over {-3,-2,-1,+1,+2,+3}

match target:
    final match win/loss or match-equity value
```

Recommended:

```text
train both distributional outcome and scalar equity
```

## 9.4 Loss

```text
L =
    CE(policy_target, policy_logits)
  + lambda_outcome * CE(outcome_target, outcome_logits)
  + lambda_value   * MSE(equity_target, predicted_equity)
  + lambda_cube    * CE(cube_target, cube_logits)
  + lambda_aux     * auxiliary_losses
  + lambda_l2      * ||theta||^2
```

For pure AlphaZero:

```text
L = value_error - policy_target dot log(policy) + c ||theta||^2
```

For practical backgammon, use the richer loss.

## 9.5 Reanalyse targets

Reanalyse old replay states with the newest network and search:

```text
old state -> fresh search -> refreshed policy target and value estimate
```

Prioritize:

```text
- close checker decisions
- cube decisions
- high-volatility positions
- positions where latest network disagrees with old search
- positions near terminal/gammon boundary
```

---

# 10. Training loop and curriculum

## 10.1 Baseline self-play loop

```text
initialize network theta
initialize replay buffer B

for iteration in training:
    generate self-play games using MCTS(theta)
    store decision states and search targets in B
    train theta on minibatches from B
    periodically reanalyse selected B states
    evaluate new theta against previous best using paired dice
    promote if statistically stronger
```

## 10.2 Training stages

Recommended curriculum:

```text
Stage 1: cubeless checker play
    reward in {-3,-2,-1,+1,+2,+3}
    no cube decisions

Stage 2: cubeless with better distributional value
    emphasize gammon/backgammon probabilities
    add bearoff/contact/race auxiliary heads

Stage 3: cubeful money play
    add cube value and cube ownership
    train double/no-double and take/pass

Stage 4: match play
    add match score, match length, Crawford
    train match-equity value

Stage 5: full-strength reanalyse/distillation
    reanalyse hard positions
    optionally distill from GNUbg/XG rollouts
```

## 10.3 Search-budget curriculum

Do not spend huge search on a weak early network.

Recommended progressive simulations:

```text
first 20% of training:
    8-16 simulations

20%-60%:
    32-64 simulations

60%-90%:
    128-256 simulations

late training / reanalyse:
    400-800 simulations

final evaluation:
    800-1600+ simulations if time allows
```

## 10.4 Dice randomness helps exploration

Backgammon’s stochastic dice already create substantial state-space exploration. This is one reason TD-Gammon was able to learn effectively from self-play.

Use root exploration, but do not over-randomize moves once the network becomes competent.

## 10.5 Distillation options

Three valid training philosophies:

### Pure AlphaZero

```text
no human data
no GNUbg/XG data
only rules + self-play
```

Best for research purity.

### Practical SOTA engine

```text
self-play
+ optional supervised warm-start from GNUbg/XG rollouts
+ reanalyse
+ hard-position mining
```

Best for performance per compute.

### Hybrid curriculum

```text
supervised warm-start -> self-play -> reanalyse -> expert-position fine-tuning
```

Best reduced-hardware path.

---

# 11. Training-resource guidance

## 11.1 Published resource signals

| System | Reported resource signal | Lesson for backgammon |
|---|---:|---|
| TD-Gammon 1.0 | about 300,000 self-play games; 1-ply search | Tiny networks can learn useful backgammon from self-play. |
| TD-Gammon 2.0 | 800,000 games; 2-ply search; 40 hidden units | Search depth and features matter. |
| TD-Gammon 2.1 | 1,500,000 games; 2-ply search; 80 hidden units | Strong master / near-world-class human play was possible with tiny 1990s networks. |
| Tesauro/Galperin Monte Carlo search | IBM SP1/SP2 parallel-RISC supercomputers; rollout-based online policy improvement | Raw Monte Carlo rollouts are expensive; neural value + selective search is essential. |
| AlphaGo Zero | 4.9M self-play games; 1600 MCTS simulations; 700k minibatches of 2048; 20 residual blocks; about 3 days | Deep AlphaZero-style training can be huge, but Go is much larger than backgammon. |
| AlphaZero chess/shogi/go preprint | 700k steps, minibatch size 4096; 5000 first-generation TPUs for self-play; 64 second-generation TPUs for training | DeepMind-scale AlphaZero is an upper extreme, not a requirement for backgammon. |
| AlphaZero Science version | 5000 first-generation TPUs for self-play and 16 second-generation TPUs for training; about 9h chess, 12h shogi, 13 days Go | Even within AlphaZero papers, reported hardware varies by version and setup. |
| Stochastic MuZero backgammon | 1600 simulations per move; matched GNUbg-level strength in reported experiment | 1600 simulations is a proven high-end search budget for stochastic backgammon. |
| Gumbel MuZero | strong low-simulation behavior in several domains | Use Gumbel root search when hardware is limited. |

## 11.2 Backgammon compute formula

Use this planning estimate:

```text
decision_positions ~= games * 55
NN_evals ~= games * 55 * simulations * chance_factor
```

Where:

```text
55 = rough average decision positions per game
chance_factor = 1.0     for passthrough sampled dice
chance_factor = 2 to 5  for exact root dice + deeper sampled dice
chance_factor = 21      for full chance expansion everywhere
```

Examples:

```text
300k games * 55 decisions * 32 sims ~= 528M NN evals
1M games   * 55 decisions * 16 sims ~= 880M NN evals
1M games   * 55 decisions * 64 sims ~= 3.52B NN evals
3M games   * 55 decisions * 64 sims ~= 10.56B NN evals
```

These are raw search-evaluation counts. Batching, caching, exact simulator speed, and network size determine actual wall-clock time.

## 11.3 What reduced hardware can realistically do

Backgammon is more forgiving than Go/chess for small hardware because:

```text
- the board is small
- average legal move count is moderate
- dice inject exploration
- terminal outcomes arrive relatively quickly
- afterstate action representation reduces branching
- TD-Gammon already reached strong play with tiny networks
```

The main bottleneck is not model size. It is:

```text
self-play throughput * MCTS simulations * neural inference speed
```

---

# 12. Reduced-hardware plans

## 12.1 CPU/laptop prototype

Goal: validate correctness.

```text
model:
    small MLP value net or candidate scorer

search:
    0-8 simulations
    or 1-ply afterstate evaluation

training:
    10k-100k self-play games

expected result:
    useful development baseline, not SOTA
```

Focus:

```text
- correct legal move generator
- duplicate afterstate collapse
- board encoding
- value target correctness
- paired-dice evaluation harness
```

## 12.2 Single consumer GPU

Goal: strong checker-play engine.

```text
model:
    residual MLP or small 1D ResNet

search:
    Gumbel root
    16-64 simulations
    stratified chance passthrough

training:
    300k-3M games
    replay + light reanalyse

expected result:
    strong checker play; likely much stronger with supervised/distillation help
```

## 12.3 Single high-end GPU / two GPUs

Goal: serious GNUbg lower-ply challenger.

```text
model:
    wider residual MLP or 1D ResNet

search:
    64-256 simulations during training
    exact 21-roll chance expansion near root
    deeper stratified chance nodes

training:
    1M-10M games depending throughput
    prioritized replay
    reanalyse
    cube curriculum
```

## 12.4 Four to eight GPUs

Goal: practical route toward Grandmaster-level benchmarks.

```text
model:
    strong 1D ResNet or compact transformer

search:
    256-800 simulations training/reanalyse
    800-1600 simulations evaluation

training:
    full cubeful/match training
    heavy reanalyse
    hard-position mining
    paired-dice leagues
```

## 12.5 Cluster-scale

Goal: research-grade comparison to Stochastic MuZero/AlphaZero-style papers.

```text
search:
    800-1600+ simulations

training:
    many millions of games
    large replay and reanalyse
    large evaluation leagues
```

---

# 13. Evaluation methodology

## 13.1 Use paired dice

Backgammon has high outcome variance. Use common random numbers.

For each dice sequence D:

```text
Game 1:
    Agent A plays white, Agent B black, dice sequence D

Game 2:
    Agent B plays white, Agent A black, same dice sequence with colors swapped
```

This sharply reduces evaluation noise.

## 13.2 Track multiple metrics

Do not rely only on win rate.

Track:

```text
points per game
cubeless equity error
checker-play error rate
cube decision error rate
match equity error
paired-dice Elo
normal/gammon/backgammon calibration
policy entropy by phase
root value calibration
search improvement over raw network
```

## 13.3 Evaluate by phase

Break positions into:

```text
opening
contact
blitz
holding game
backgame
race
bearoff
cube decision
take/pass decision
Crawford / post-Crawford match play
```

Backgammon engines often fail in narrow phase-specific ways. Aggregate Elo can hide serious cube or gammon errors.

## 13.4 Baselines

Use:

```text
- random/legal baseline
- one-ply neural evaluator
- no-search policy net
- previous-best self-play agent
- GNUbg/XG at several settings
- human expert-position test suites, if available
```

## 13.5 Luck-adjusted evaluation and error-rate metrics (PRIMARY progress metric)

Paired dice (§13.1) is correct but incomplete. Win rate over N games is a HIGH-VARIANCE, low-
information estimator of strength. The backgammon community measures strength differently, and this
project should too.

### 13.5.1 Error rate / PR as the primary metric — not win rate

Analyze the agent's own games with a reference analyzer (GNUbg via its scriptable Python interface,
or XG) and track **average equity loss per unforced decision** — the "error rate", and its
normalized cousin **PR (Performance Rating)**.

```text
error_rate = (1000 / N_unforced) * sum over unforced decisions of
             ( equity(best move) - equity(move actually played) )     # milli-equity per decision
PR         = error rate scaled to the community's normalized-luck convention (lower = stronger)
```

Reference points: wildbg reports error rate ~5.9 vs GNUbg 2-ply; Strehl (a top neural bot) reports
PR ~1.06. This metric:

```text
- has FAR lower variance than win rate (every decision is a datum, not every game)
- is the community's lingua franca for strength (directly comparable to gnubg/XG/wildbg/Strehl)
- gives FREE phase + category breakdowns (checker-play error vs cube error; §13.3) at no extra cost
- needs no opponent to "beat" — it measures distance to optimal play directly
```

Adopt error rate / PR as the PRIMARY progress metric. Keep win% only as a secondary, human-legible
sanity check. Note: this is exactly the "move-regret" quantity — equity lost vs the best move —
already computed against the EXACT bearoff table for the race band; extend the same idea to contact
using a reference engine (wildbg/GNUbg per-move equities) as the "best move" oracle.

### 13.5.2 Luck adjustment / variance reduction on evaluation matches

GNUbg's rollout variance reduction uses the **ply-to-ply equity difference of each dice roll as a
control variate** ("luck"): a roll that swings equity by +Δ was lucky by +Δ, independent of skill.

```text
luck(roll)          = V(position after roll, best play) - V(position before roll, on-roll)   # a control variate
adjusted_score      = raw_score - sum(luck over the game)      # subtract the dice luck each side got
```

Apply the SAME luck-adjusted scoring to head-to-head evaluation matches ON TOP OF paired dice — it
further shrinks the confidence interval on the equity/points estimate for the same number of games.
(Paired dice removes symmetric luck; the control variate removes residual per-roll luck.)

### 13.5.3 Relative vs absolute equity — justifies truncated-rollout reanalyse targets

GNUbg's practical observation: **truncated rollouts estimate RELATIVE equities (move A vs move B)
better than ABSOLUTE equities** — the correlated evaluation errors at the truncation horizon cancel
in the difference. This is the same phenomenon as §8.6 (ranking candidates matters more than absolute
value) and directly justifies using **truncated-rollout targets for reanalyse**: even a biased
truncation evaluator yields good MOVE-SELECTION targets because the bias is common-mode across the
candidate set.

---

# 14. Implementation blueprint

## 14.1 Core modules

```text
BackgammonEnv
    generate_legal_plays(board, dice) -> list[MoveSequence]
    apply_play(board, dice, play) -> afterstate
    canonicalize_afterstate(afterstate) -> action_id
    roll_distribution() -> list[(roll, probability)]
    apply_roll(afterstate, roll) -> decision_state
    terminal_score(board, cube, match) -> outcome

NeuralNet
    encode_state(board, dice, cube, match) -> state_embedding
    score_candidates(state, afterstates) -> logits
    value_distribution(state) -> outcome probabilities
    cube_policy(state) -> cube logits

Search
    decision_node_select(node)
    chance_node_select(node)
    expand(node)
    evaluate(node)
    backup(path, value)

Trainer
    self_play_workers
    replay_buffer
    reanalyse_workers
    optimizer
    evaluator
```

## 14.2 MCTS pseudocode

```text
for simulation in 1..S:
    node = root
    path = []

    while node is expanded:
        if node.type == DECISION:
            action = select_decision_action(node)
            path.append((node, action))
            node = node.child_afterstate(action)

        else if node.type == CHANCE:
            roll = select_chance_outcome(node)
            path.append((node, roll))
            node = node.child_decision(roll)

    value = evaluate_or_terminal(node)
    backup(path, value)
```

## 14.3 Chance selection pseudocode

```text
function select_chance_outcome(chance_node):
    if chance_node.visit_count >= exact_threshold:
        ensure_all_21_rolls_expanded(chance_node)
        return exact_expectation_marker

    return argmax_roll P_true(roll) / (N_roll + 1)
```

If exact expectation marker is returned:

```text
value = sum_roll P_true(roll) * V(child_roll)
```

Otherwise:

```text
value = V(sampled_child)
```

## 14.4 Backup sign handling

Use current-player perspective consistently.

If value is always from the player-to-act perspective at the node:

```text
when backing up across a player switch:
    value = -value

when backing up across chance:
    do not change sign unless player-to-act changes after dice
```

Be explicit. Most bugs in stochastic two-player MCTS come from incorrect sign flips at chance nodes.

## 14.5 Candidate policy target

At root:

```text
for each legal afterstate a:
    pi_target[a] = visit_count[a]^(1/tau) / sum_b visit_count[b]^(1/tau)
```

Store the candidate list with the replay item, because the legal action set is variable.

---

# 15. High-impact engineering tricks

## 15.1 Batch neural inference

Tree search will generate many small inference requests. Batch them aggressively:

```text
- across simulations
- across self-play games
- across workers
- across candidate afterstate scorings
```

Inference throughput matters more than raw model expressiveness early.

## 15.2 Move generator should be native and heavily tested

Implement legal move generation in a fast systems language or optimized Python extension.

Good choices:

```text
Rust
C++
Go
Cython / pybind11
Numba for prototype
```

Test:

```text
- doubles
- entering from bar
- forced higher die rule
- no legal move
- bearoff exact/off rules
- duplicate afterstate collapse
- cube terminal transitions
```

## 15.3 Cache afterstates and evaluations

Use transposition keys:

```text
hash(board, side_to_move, dice, cube, match)
hash(afterstate, cube, match)
```

Backgammon has many recurring local structures. Caching helps both search and reanalyse.

## 15.4 Keep the model small until the search is good

A large network can hurt at fixed move time by reducing simulations. Start with a fast model and prove that search improves it.

Track:

```text
raw policy/value performance
MCTS-improved performance
performance per millisecond
performance per simulation
```

## 15.5 Hard-position mining

Mine positions where:

```text
- top two moves are close
- cube action changes after search
- value head disagrees with rollout/reanalyse
- rare gammon/backgammon outcomes occur
- agent blunders against GNUbg/XG
```

Then oversample these positions in training, with correct value weighting.

## 15.6 Separate checker and cube debugging

Many engines hide cube weakness behind good checker play. Evaluate cube decisions separately.

Use datasets of:

```text
- no-double/take
- double/take
- double/pass
- too-good-to-double
- redouble decisions
- Crawford/post-Crawford match scores
```

## 15.7 Use rollout/distillation tactically

If compute is limited, the fastest route to strength is often:

```text
1. supervised warm-start from GNUbg/XG evaluations or rollouts
2. self-play AlphaZero training
3. reanalyse with current search
4. hard-position fine-tuning
```

Pure self-play remains possible, but distillation can save large amounts of compute.

---

# 16. Reference list

## Core stochastic/MCTS/MuZero references

1. **Ioannis Antonoglou, Julian Schrittwieser, Sherjil Ozair, Thomas K. Hubert, David Silver. _Planning in Stochastic Environments with a Learned Model._ ICLR 2022.**  
   Primary Stochastic MuZero paper. Important for afterstates, stochastic/chance nodes, learned stochastic models, and backgammon results.  
   <https://openreview.net/forum?id=X6D9bAHhBQ1>  
   PDF: <https://openreview.net/pdf?id=X6D9bAHhBQ1>

2. **Julian Schrittwieser et al. _Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model._ Nature 2020.**  
   Original MuZero paper. Important baseline for latent planning without known rules.  
   <https://arxiv.org/abs/1911.08265>

3. **Ivo Danihelka, Arthur Guez, Julian Schrittwieser, David Silver. _Policy Improvement by Planning with Gumbel._ ICLR 2022.**  
   Gumbel AlphaZero/MuZero. Important for low-simulation root search and policy improvement under small search budgets.  
   <https://openreview.net/forum?id=bERaNdoegnO>  
   PDF: <https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf>

4. **Thomas Hubert et al. _Learning and Planning in Complex Action Spaces._ ICML 2021.**  
   Sampled MuZero. Useful for variable or large action spaces and candidate-action search.  
   <https://proceedings.mlr.press/v139/hubert21a.html>  
   PDF: <https://proceedings.mlr.press/v139/hubert21a/hubert21a.pdf>

5. **Yazhe Niu et al. _LightZero: A Unified Benchmark for Monte Carlo Tree Search in General Sequential Decision Scenarios._ NeurIPS 2023.**  
   Practical ecosystem for AlphaZero/MuZero/EfficientZero/Sampled MuZero/Stochastic MuZero-style algorithms.  
   <https://arxiv.org/abs/2310.08348>  
   Code: <https://github.com/opendilab/LightZero>

6. **Chunyu Xuan et al. _ReZero: Boosting MCTS-based Algorithms by Backward-view and Entire-buffer Reanalyze._ 2024/2025.**  
   Relevant for reducing the wall-clock cost of reanalyse in MCTS-based training.  
   <https://arxiv.org/abs/2404.16364>

7. **Yuan Pu et al. _UniZero: Generalized and Efficient Planning with Scalable Latent World Models._ TMLR / arXiv 2024.**  
   Transformer-based scalable latent world model for MuZero-style planning. More relevant for general multi-task planners than a pure backgammon engine.  
   <https://arxiv.org/abs/2406.10667>  
   <https://openreview.net/forum?id=Gl6dF9soQo>

8. **Shengjie Wang et al. _EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data._ ICML 2024.**  
   Useful for sample-efficient MuZero-family ideas and search-based value estimation.  
   <https://arxiv.org/abs/2403.00564>  
   <https://github.com/shengjiewang-jason/efficientzerov2>

## AlphaZero / AlphaGo Zero references

9. **David Silver et al. _Mastering the Game of Go without Human Knowledge._ Nature 2017.**  
   AlphaGo Zero. Important for pUCT, root temperature, root noise, residual policy/value network, self-play training resources.  
   <https://www.dcsc.tudelft.nl/~sc4081/2018/assign/pap/alphago_paper1.pdf>

10. **David Silver et al. _Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm._ arXiv 2017.**  
    AlphaZero preprint. Important for generic game formulation, resource scale, and search/training recipe.  
    <https://arxiv.org/abs/1712.01815>  
    PDF: <https://arxiv.org/pdf/1712.01815>

11. **David Silver et al. _A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go through Self-Play._ Science 2018.**  
    Peer-reviewed AlphaZero paper; important for resource-scale comparisons.  
    <https://www.science.org/doi/10.1126/science.aar6404>

## Backgammon-specific references

12. **Gerald Tesauro. _TD-Gammon, a Self-Teaching Backgammon Program, Achieves Master-Level Play._ 1993.**  
    Core historical backgammon self-play result; reports TD-Gammon versions, game counts, hidden units, and human-level comparisons.  
    <https://bkgm.com/articles/tesauro/TDGammonAchievesMasterLevelPlay.pdf>

13. **Gerald Tesauro. _Temporal Difference Learning and TD-Gammon._ Communications of the ACM, 1995.**  
    Explains TD-Gammon, stochastic dice exploration, relative vs absolute evaluation accuracy, and practical learning behavior.  
    <https://www.bkgm.com/articles/tesauro/tdl.html>  
    PDF mirror: <https://www.csd.uwo.ca/~xling/cs346a/extra/tdgammon.pdf>

14. **Gerald Tesauro and Gregory R. Galperin. _On-line Policy Improvement using Monte-Carlo Search._ NeurIPS 1996.**  
    Backgammon Monte Carlo rollout/search paper; useful for understanding rollout cost and online policy improvement.  
    <https://papers.neurips.cc/paper/1302-on-line-policy-improvement-using-monte-carlo-search.pdf>

## Stochastic MCTS / uncertainty references

15. **Daniel Dam et al. _Power Mean Estimation in Stochastic Monte-Carlo Tree Search._ AISTATS 2024.**  
    Useful for stochastic backup design and alternatives to ordinary expectation/mean backup.  
    <https://proceedings.mlr.press/v244/dam24a.html>

16. **Daniel Dam et al. _Monte-Carlo Tree Search with Uncertainty Propagation via Optimal Transport._ AISTATS 2025.**  
    Relevant for distributional/uncertainty-aware tree backups.  
    <https://proceedings.mlr.press/v267/dam25c.html>

---

# Final recommended build target

```text
Explicit-Chance Gumbel AlphaZero for Backgammon

Search:
    decision nodes for checker/cube actions
    afterstate nodes after chosen plays
    chance nodes for dice
    exact 21-roll expectation near root
    stratified passthrough dice deeper
    Gumbel root search for low simulation counts

Representation:
    current-player normalized board
    dice one-hot over 21 unordered rolls
    bar/bearoff/cube/match features
    candidate-afterstate actions

Network:
    fast residual MLP or 1D ResNet first
    candidate-action policy scorer
    distributional value head over normal/gammon/backgammon outcomes
    cube and match-equity heads

Training:
    cubeless checker-play curriculum first
    add cube and match play later
    replay + reanalyse
    progressive simulation schedule
    hard-position mining
    paired-dice evaluation

Reduced-hardware strategy:
    prioritize fast inference and many self-play positions
    use Gumbel root search with 16-64 simulations early
    use exact root dice only when affordable
    keep the network small until search is clearly improving raw policy/value
```
