#####
##### Backgammon - AlphaZero.jl wrapper for BackgammonNet.jl
#####
#
# This wrapper integrates BackgammonNet.jl with AlphaZero.jl's GI interface.
# Supports both stochastic MCTS (explicit chance nodes) and deterministic modes.
#
# Configuration:
# - short_game=true: Faster games with pieces closer to bearing off
# - doubles_only=true: Only doubles dice rolls (6 outcomes vs 21)

import AlphaZero.GI
using StaticArrays

# Import from BackgammonNet.jl (assumes it's available in the project)
using BackgammonNet

# Configuration constants
const SHORT_GAME = true  # Shorter games for faster learning (intentionally different starting position)
const DOUBLES_ONLY = false  # Full 21 dice outcomes for proper stochastic testing

# Action space: 676 actions (26*26 locations)
const NUM_ACTIONS = 676

# Chance outcomes for doubles_only mode
const DOUBLES_INDICES = [1, 7, 12, 16, 19, 21]  # 1-1, 2-2, 3-3, 4-4, 5-5, 6-6
const NUM_CHANCE_OUTCOMES = DOUBLES_ONLY ? 6 : 21
const CHANCE_PROB = 1.0 / NUM_CHANCE_OUTCOMES

const Player = Bool
const WHITE = true   # Player 0
const BLACK = false  # Player 1

#####
##### Game Specification
#####

struct GameSpec <: GI.AbstractGameSpec end

GI.two_players(::GameSpec) = true

# Return all possible action indices (1-676)
# The action mask will indicate which are valid
function GI.actions(::GameSpec)
  return collect(1:NUM_ACTIONS)
end

GI.num_chance_outcomes(::GameSpec) = NUM_CHANCE_OUTCOMES

function GI.vectorize_state(::GameSpec, game::BackgammonNet.BackgammonGame)
  return BackgammonNet.observe_full(game)
end

#####
##### Game Environment
#####

mutable struct GameEnv <: GI.AbstractGameEnv
  game::BackgammonNet.BackgammonGame
end

GI.spec(::GameEnv) = GameSpec()

function GI.current_state(g::GameEnv)
  return BackgammonNet.clone(g.game)
end

function GI.set_state!(g::GameEnv, state::BackgammonNet.BackgammonGame)
  game = g.game
  game.p0 = state.p0
  game.p1 = state.p1
  game.dice = state.dice
  game.remaining_actions = state.remaining_actions
  game.current_player = state.current_player
  game.terminated = state.terminated
  game.reward = state.reward
  game.doubles_only = state.doubles_only
  game.obs_type = state.obs_type
  game.cube_value = state.cube_value
  game.cube_owner = state.cube_owner
  game.phase = state.phase
  game.cube_enabled = state.cube_enabled
  game.my_away = state.my_away
  game.opp_away = state.opp_away
  game.is_crawford = state.is_crawford
  game.is_post_crawford = state.is_post_crawford
  game.jacoby_enabled = state.jacoby_enabled
  game.tavla = state.tavla
  game._legal_actions_valid = false
end

GI.white_playing(g::GameEnv) = g.game.current_player == 0

# Direct state access for white_playing - avoids creating GameEnv which may have side effects
# This is used by push_trace! in memory.jl to determine value perspective
GI.white_playing(::GameSpec, state::BackgammonNet.BackgammonGame) = state.current_player == 0

function GI.init(::GameSpec)
  game = BackgammonNet.initial_state(; short_game=SHORT_GAME, doubles_only=DOUBLES_ONLY)
  return GameEnv(game)
end

function GI.game_terminated(g::GameEnv)
  return BackgammonNet.game_terminated(g.game)
end

function GI.white_reward(g::GameEnv)
  # BackgammonNet: g.reward > 0 means P0 (white) won, < 0 means P1 (black) won
  return Float64(g.game.reward)
end

#####
##### Chance Node Interface
#####

function GI.is_chance_node(g::GameEnv)
  return BackgammonNet.is_chance_node(g.game)
end

function GI.chance_outcomes(g::GameEnv)
  if DOUBLES_ONLY
    # Return 6 outcomes with equal probability 1/6
    return [(i, CHANCE_PROB) for i in 1:6]
  else
    # Return all 21 dice outcomes with their probabilities
    outcomes = BackgammonNet.chance_outcomes(g.game)
    return [(idx, Float64(prob)) for (idx, prob) in outcomes if prob > 0]
  end
end

function GI.apply_chance!(g::GameEnv, outcome)
  if DOUBLES_ONLY
    # Map outcome 1-6 to the actual dice indices
    dice_idx = DOUBLES_INDICES[outcome]
    BackgammonNet.apply_chance!(g.game, dice_idx)
  else
    BackgammonNet.apply_chance!(g.game, outcome)
  end
end

#####
##### Actions
#####

function GI.actions_mask(g::GameEnv)
  mask = falses(NUM_ACTIONS)

  if BackgammonNet.is_chance_node(g.game)
    # At chance node: no player actions available
    return mask
  end

  # Get legal actions from BackgammonNet
  legal = BackgammonNet.legal_actions(g.game)

  # Set mask for legal actions
  for action in legal
    if 1 <= action <= NUM_ACTIONS
      mask[action] = true
    end
  end

  return mask
end

function GI.play!(g::GameEnv, action)
  BackgammonNet.apply_action!(g.game, action)
end

#####
##### Interactive Interface
#####

function GI.render(g::GameEnv)
  game = g.game
  println("=" ^ 50)
  println("Backgammon (short_game=$SHORT_GAME, doubles_only=$DOUBLES_ONLY)")
  println("-" ^ 50)

  # Display board state using canonical indexing (27=my off, 28=opp off)
  # Note: g[27] returns current player's borne-off, g[28] returns opponent's
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
  elseif BackgammonNet.is_chance_node(game)
    player = game.current_player == 0 ? "Player 0" : "Player 1"
    println("$player rolling dice...")
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
  # Parse "loc1 | loc2" format
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
##### Heuristic Value (optional)
#####

function GI.heuristic_value(g::GameEnv)
  obs = BackgammonNet.observe_full(g.game)
  # Use pip count difference as heuristic (index 38 in observe_full)
  pip_diff = obs[38]  # Already normalized
  # Flip sign if current player is player 1 (black)
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
    # No valid actions (shouldn't happen in normal play)
    return collect(1:NUM_ACTIONS), zeros(NUM_ACTIONS)
  end

  # Uniform distribution over valid actions
  π = zeros(NUM_ACTIONS)
  prob = 1.0 / length(valid_actions)
  for a in valid_actions
    π[a] = prob
  end

  return collect(1:NUM_ACTIONS), π
end

reset!(::RandomPlayer) = nothing
