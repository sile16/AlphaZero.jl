# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaZero.jl is a Julia implementation of DeepMind's AlphaZero algorithm. The core algorithm is ~2,000 lines of Julia code that combines Monte Carlo Tree Search (MCTS) with neural network learning through self-play.

## Common Commands

```bash
# Install dependencies
julia --project -e 'import Pkg; Pkg.instantiate()'

# Run all tests
julia --project -e 'using Pkg; Pkg.test()'

# Train an agent (connect-four, tictactoe, mancala, grid-world)
julia --project -e 'using AlphaZero; Scripts.train("connect-four")'

# Interactive explorer for trained agent
julia --project -e 'using AlphaZero; Scripts.explore("connect-four")'

# Play against trained agent
julia --project -e 'using AlphaZero; Scripts.play("connect-four")'

# Quick validation (2 iterations, fast)
julia --project -e 'using AlphaZero; Scripts.dummy_run("tictactoe")'

# Build documentation
julia --project=docs docs/make.jl
```

## Architecture

### Core Modules (src/)

- **`AlphaZero.jl`** - Main module, exports and includes
- **`mcts.jl`** - Standard MCTS with UCT selection
- **`gumbel_mcts.jl`** - Gumbel MCTS variant (sequential halving + Gumbel-max trick)
- **`game.jl`** - `GameInterface` (GI) abstract interface for games
- **`play.jl`** - Player types: `MctsPlayer`, `GumbelMctsPlayer`, `NetworkPlayer`, `RandomPlayer`
- **`training.jl`** - Main training loop: self-play → memory → learning → arena evaluation
- **`params.jl`** - All parameter structs: `MctsParams`, `GumbelMctsParams`, `SimParams`, `Params`
- **`memory.jl`** - Experience buffer storing `TrainingSample` from self-play
- **`learning.jl`** - Neural network training with policy/value losses
- **`simulations.jl`** - Distributed game simulation with batched inference
- **`benchmark.jl`** - Player comparison framework

### Neural Networks (src/networks/)

- **`network.jl`** - `AbstractNetwork` interface
- **`flux.jl`** - Flux.jl implementations: `SimpleNet`, `ResNet`

### Games (games/)

Each game implements `GameInterface`:
- `tictactoe/` - 3x3 tic-tac-toe
- `connect-four/` - 7x6 Connect Four
- `mancala/` - Mancala board game
- `grid-world/` - Single-player navigation

### Key Patterns

**Game Interface**: Games implement `GI.init`, `GI.play!`, `GI.available_actions`, `GI.game_terminated`, `GI.white_reward`, `GI.vectorize_state`, etc.

**Oracle Pattern**: MCTS uses an oracle (network or rollout) that returns `(policy, value)` for a state.

**Training Loop** (`training.jl:train!`):
1. `self_play_step!` - Generate games with best network + MCTS
2. `memory_report` - Analyze collected samples
3. `learning_step!` - Train network, evaluate in arena, update best network

**Player Types**: All inherit from `AbstractPlayer` and implement `think(player, game) -> (actions, π)`.

## Testing

```bash
# Full test suite (slow - runs training iterations)
julia --project -e 'using Pkg; Pkg.test()'

# Quick game validation only
julia --project -e 'using AlphaZero; Scripts.test_game("tictactoe")'
```

The test suite includes "Dummy Runs" that execute full training iterations, which can take 10+ minutes.
