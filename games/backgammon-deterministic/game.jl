#####
##### Backgammon (Deterministic) - AlphaZero.jl wrapper for BackgammonNet.jl
#####
#
# This is the DETERMINISTIC version where dice rolls are handled automatically.
# Stochasticity is hidden inside step! - MCTS never sees chance nodes.
# This matches the "standard" AlphaZero approach for stochastic games.
#
# Configuration:
# - short_game=true: Faster games with pieces closer to bearing off
# - doubles_only=true: Only doubles dice rolls (simpler game)

import AlphaZero.GI
using StaticArrays
using Random

# Import from BackgammonNet.jl
using BackgammonNet

# Configuration constants
const SHORT_GAME = true
const DOUBLES_ONLY = true

# Action space: 676 actions (26*26 locations)
const NUM_ACTIONS = 676

# Observation size (using observe_full: 86 features)
const OBS_SIZE = 86

const Player = Bool
const WHITE = true   # Player 0
const BLACK = false  # Player 1

#####
##### Game Specification
#####

struct GameSpec <: GI.AbstractGameSpec end

GI.two_players(::GameSpec) = true

function GI.actions(::GameSpec)
  return collect(1:NUM_ACTIONS)
end

# NO chance outcomes - stochasticity is hidden
GI.num_chance_outcomes(::GameSpec) = 0

function GI.vectorize_state(::GameSpec, game::BackgammonNet.BackgammonGame)
  return BackgammonNet.observe_full(game)
end

#####
##### Game Environment
#####

mutable struct GameEnv <: GI.AbstractGameEnv
  game::BackgammonNet.BackgammonGame
  rng::MersenneTwister
end

GI.spec(::GameEnv) = GameSpec()

function GI.current_state(g::GameEnv)
  # Return a COPY of the game state, not a reference
  # This is important because MCTS captures state before playing actions
  game = g.game
  return BackgammonNet.BackgammonGame(
    game.p0, game.p1, game.dice, game.remaining_actions,
    game.current_player, game.terminated, game.reward,
    copy(game.history), game.doubles_only,
    Int[], Int[], Int[]  # Fresh buffers for the copy
  )
end

function GI.set_state!(g::GameEnv, state::BackgammonNet.BackgammonGame)
  # Copy state from another game
  g.game.p0 = state.p0
  g.game.p1 = state.p1
  g.game.dice = state.dice
  g.game.remaining_actions = state.remaining_actions
  g.game.current_player = state.current_player
  g.game.terminated = state.terminated
  g.game.reward = state.reward
  g.game.doubles_only = state.doubles_only
  empty!(g.game.history)
  append!(g.game.history, state.history)
  # If at chance node, auto-roll to get to player turn
  if BackgammonNet.is_chance_node(g.game) && !BackgammonNet.game_terminated(g.game)
    BackgammonNet.sample_chance!(g.game, g.rng)
  end
end

GI.white_playing(g::GameEnv) = g.game.current_player == 0

function GI.init(::GameSpec)
  game = BackgammonNet.initial_state(; short_game=SHORT_GAME, doubles_only=DOUBLES_ONLY)
  rng = MersenneTwister()
  genv = GameEnv(game, rng)
  # Auto-roll initial dice to get to first decision point
  BackgammonNet.sample_chance!(game, rng)
  return genv
end

function GI.game_terminated(g::GameEnv)
  return BackgammonNet.game_terminated(g.game)
end

function GI.white_reward(g::GameEnv)
  return Float64(g.game.reward)
end

#####
##### NO Chance Node Interface (deterministic version)
#####

function GI.is_chance_node(g::GameEnv)
  # Never expose chance nodes - always return false
  return false
end

function GI.chance_outcomes(g::GameEnv)
  # Should never be called since is_chance_node always returns false
  return Tuple{Int, Float64}[]
end

function GI.apply_chance!(g::GameEnv, outcome)
  # Should never be called
  error("apply_chance! should not be called in deterministic mode")
end

#####
##### Actions
#####

function GI.actions_mask(g::GameEnv)
  mask = falses(NUM_ACTIONS)

  # Handle internal chance nodes by auto-rolling (shouldn't happen after init)
  if BackgammonNet.is_chance_node(g.game) && !BackgammonNet.game_terminated(g.game)
    BackgammonNet.sample_chance!(g.game, g.rng)
  end

  if BackgammonNet.game_terminated(g.game)
    return mask
  end

  # Get legal actions from BackgammonNet
  legal = BackgammonNet.legal_actions(g.game)

  for action in legal
    if 1 <= action <= NUM_ACTIONS
      mask[action] = true
    end
  end

  return mask
end

function GI.play!(g::GameEnv, action)
  # Use step! which applies action AND auto-rolls dice for next turn
  BackgammonNet.step!(g.game, action, g.rng)
end

#####
##### Interactive Interface
#####

function GI.render(g::GameEnv)
  game = g.game
  println("=" ^ 50)
  println("Backgammon DETERMINISTIC (short_game=$SHORT_GAME, doubles=$DOUBLES_ONLY)")
  println("-" ^ 50)

  # Display board state using canonical indexing (27=my off, 28=opp off)
  my_off = game[27]
  opp_off = abs(game[28])
  if game.current_player == 0
    println("Player 0 (White) off: $my_off")
    println("Player 1 (Black) off: $opp_off")
  else
    println("Player 0 (White) off: $opp_off")
    println("Player 1 (Black) off: $my_off")
  end

  if BackgammonNet.game_terminated(game)
    winner = game.reward > 0 ? "Player 0 (White)" : "Player 1 (Black)"
    mult = abs(game.reward)
    wtype = mult == 1 ? "single" : (mult == 2 ? "gammon" : "backgammon")
    println("Game Over! $winner wins ($wtype)")
  else
    player = game.current_player == 0 ? "Player 0" : "Player 1"
    d1, d2 = game.dice
    remaining = game.remaining_actions
    println("$player's turn - Dice: ($d1, $d2), Actions remaining: $remaining")
  end
  println("=" ^ 50)
end

function GI.action_string(::GameSpec, action)
  return BackgammonNet.action_string(action)
end

function GI.parse_action(::GameSpec, str)
  parts = split(str, "|")
  if length(parts) != 2
    return nothing
  end

  function parse_loc(s)
    s = strip(lowercase(s))
    if s == "bar"
      return 0
    elseif s == "pass"
      return 25
    else
      loc = tryparse(Int, s)
      return loc !== nothing && 1 <= loc <= 24 ? loc : nothing
    end
  end

  loc1 = parse_loc(parts[1])
  loc2 = parse_loc(parts[2])

  if loc1 === nothing || loc2 === nothing
    return nothing
  end

  return BackgammonNet.encode_action(loc1, loc2)
end

#####
##### Heuristic Value
#####

function GI.heuristic_value(g::GameEnv)
  obs = BackgammonNet.observe_full(g.game)
  pip_diff = obs[38]
  if g.game.current_player == 1
    pip_diff = -pip_diff
  end
  return Float64(pip_diff)
end

#####
##### Random Baseline Player
#####

import AlphaZero: AbstractPlayer, think, reset!

struct RandomPlayer <: AbstractPlayer end

function think(p::RandomPlayer, game)
  genv = game
  mask = GI.actions_mask(genv)
  valid_actions = findall(mask)

  if isempty(valid_actions)
    return collect(1:NUM_ACTIONS), zeros(NUM_ACTIONS)
  end

  π = zeros(NUM_ACTIONS)
  prob = 1.0 / length(valid_actions)
  for a in valid_actions
    π[a] = prob
  end

  return collect(1:NUM_ACTIONS), π
end

reset!(::RandomPlayer) = nothing
