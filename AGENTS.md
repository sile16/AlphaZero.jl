# AlphaZero / Backgammon Integration Notes

This repo consumes `BackgammonNet.jl` as a package dependency. Do not copy
BackgammonNet source files into AlphaZero or include them by path.

## Current Contract

- Required BackgammonNet version: `0.6.3+`.
- Backgammon value-head contract: use `BackgammonNet.VALUE_HEAD_CONTRACT`,
  `BackgammonNet.VALUE_HEAD_ORDER`, and `BackgammonNet.check_probability_contract`.
- Equity perspective is side-to-move unless a local API explicitly says
  white-relative.
- Bearoff table generation and runtime lookup code belongs in BackgammonNet.
- Canonical local bearoff artifact root is:
  - `../BackgammonNet.jl/data/bearoff/`

Expected table subdirectories:

- `bearoff_k7_twosided`
- `bearoff_n18`

Use the BackgammonNet helpers instead of hardcoded paths:

- `BackgammonNet.default_bearoff_root()`
- `BackgammonNet.default_bearoff_k7_dir()`
- `BackgammonNet.default_bearoff_onesided_dir()`

Environment overrides are reserved for unusual deployments:

- `BACKGAMMONNET_BEAROFF_DIR`
- `BACKGAMMONNET_BEAROFF_K7_DIR`
- `BACKGAMMONNET_BEAROFF_ONESIDED_DIR`

## Active Distributed Code

The active distributed training stack is:

- `scripts/training_server.jl`
- `scripts/selfplay_client.jl`
- `src/distributed/`

Legacy archive trees were removed. Use git history for old launchers and
experiments.

## Cube Scope

BackgammonNet supports cube, match equity tables, Janowski conversion, and
rule-aware search values. The current `games/backgammon-deterministic/`
AlphaZero wrapper intentionally exposes only the 676 checker actions. Cube
actions are out of the current training curriculum until the policy/action
contract is expanded.

# Goal 1
Train a NN that outputs value and policy for board positions for Race positions that uses MCTS for better tree search of best moves.

# Goal 2
Train a NN that outputs value and policy for board positions for Contact positions that uses MCTS for better tree search of best moves.

# Goal 3
Beat wildbg in play , by using alphazero style self play to make the models better.  We can use bear off table to train the model, and bootstrap games to make training faster.


Use both Jarvis (x86) and Neo (M3 Mac Studio) , Balance incremental testing with faster wall clock time to acheive our goals.
