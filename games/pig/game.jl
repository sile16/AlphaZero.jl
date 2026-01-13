#####
##### Game of Pig - A simple dice game for testing stochastic MCTS
#####
#
# Rules:
# - Two players take turns
# - On your turn, roll a die:
#   - Roll 2-6: add to turn total, choose to roll again or hold
#   - Roll 1: lose turn total, turn ends (score unchanged)
# - Hold: add turn total to your score, turn ends
# - First to reach TARGET_SCORE wins
#
# This game demonstrates chance nodes (dice rolls) with the stochastic MCTS.

import AlphaZero.GI
using StaticArrays

const TARGET_SCORE = 100
const NUM_DIE_FACES = 6

# Actions
const ROLL = 1
const HOLD = 2
const NUM_ACTIONS = 2

const Player = Bool
const WHITE = true   # Player 1
const BLACK = false  # Player 2

# State: (p1_score, p2_score, turn_total, current_player, awaiting_dice)
# awaiting_dice: true if we're at a chance node waiting for dice roll
const State = @NamedTuple{
  p1_score::Int,
  p2_score::Int,
  turn_total::Int,
  curplayer::Player,
  awaiting_dice::Bool
}

const INITIAL_STATE = State((0, 0, 0, WHITE, false))

#####
##### Game Specification
#####

struct GameSpec <: GI.AbstractGameSpec end

GI.two_players(::GameSpec) = true
GI.actions(::GameSpec) = [ROLL, HOLD]

# Chance outcomes: die faces 1-6, each with probability 1/6
GI.num_chance_outcomes(::GameSpec) = NUM_DIE_FACES

function GI.vectorize_state(::GameSpec, state)
  # Normalize scores to [0, 1] range
  p1_norm = Float32(state.p1_score / TARGET_SCORE)
  p2_norm = Float32(state.p2_score / TARGET_SCORE)
  turn_norm = Float32(state.turn_total / TARGET_SCORE)
  curplayer = state.curplayer == WHITE ? 1f0 : 0f0
  awaiting = state.awaiting_dice ? 1f0 : 0f0
  return Float32[p1_norm, p2_norm, turn_norm, curplayer, awaiting]
end

#####
##### Game Environment
#####

mutable struct GameEnv <: GI.AbstractGameEnv
  state::State
end

GI.spec(::GameEnv) = GameSpec()
GI.current_state(g::GameEnv) = g.state
GI.set_state!(g::GameEnv, s) = (g.state = s)
GI.white_playing(g::GameEnv) = g.state.curplayer == WHITE

function GI.init(::GameSpec)
  return GameEnv(INITIAL_STATE)
end

function GI.game_terminated(g::GameEnv)
  return g.state.p1_score >= TARGET_SCORE || g.state.p2_score >= TARGET_SCORE
end

function GI.white_reward(g::GameEnv)
  if g.state.p1_score >= TARGET_SCORE
    return 1.0  # White (player 1) wins
  elseif g.state.p2_score >= TARGET_SCORE
    return -1.0  # Black (player 2) wins
  else
    return 0.0
  end
end

#####
##### Chance Node Interface
#####

function GI.is_chance_node(g::GameEnv)
  return g.state.awaiting_dice
end

function GI.chance_outcomes(g::GameEnv)
  # Die faces 1-6, each with probability 1/6
  prob = 1.0 / NUM_DIE_FACES
  return [(face, prob) for face in 1:NUM_DIE_FACES]
end

function GI.apply_chance!(g::GameEnv, outcome)
  die_face = outcome
  s = g.state

  if die_face == 1
    # Rolled a 1: lose turn total, switch players
    new_player = !s.curplayer
    g.state = State((s.p1_score, s.p2_score, 0, new_player, false))
  else
    # Rolled 2-6: add to turn total, player decides next
    new_turn_total = s.turn_total + die_face
    g.state = State((s.p1_score, s.p2_score, new_turn_total, s.curplayer, false))
  end
end

#####
##### Actions
#####

function GI.actions_mask(g::GameEnv)
  if g.state.awaiting_dice
    # At chance node: no player actions available
    return [false, false]
  else
    # Both ROLL and HOLD are always available at decision nodes
    return [true, true]
  end
end

function GI.play!(g::GameEnv, action)
  s = g.state

  if action == ROLL
    # Player chose to roll: transition to chance node
    g.state = State((s.p1_score, s.p2_score, s.turn_total, s.curplayer, true))
  elseif action == HOLD
    # Player chose to hold: add turn total to score, switch players
    if s.curplayer == WHITE
      new_p1_score = s.p1_score + s.turn_total
      g.state = State((new_p1_score, s.p2_score, 0, BLACK, false))
    else
      new_p2_score = s.p2_score + s.turn_total
      g.state = State((s.p1_score, new_p2_score, 0, WHITE, false))
    end
  end
end

#####
##### Interactive Interface
#####

function GI.render(g::GameEnv)
  s = g.state
  println("=" ^ 40)
  println("Player 1 (White): $(s.p1_score) / $TARGET_SCORE")
  println("Player 2 (Black): $(s.p2_score) / $TARGET_SCORE")
  println("-" ^ 40)
  if GI.game_terminated(g)
    winner = s.p1_score >= TARGET_SCORE ? "Player 1 (White)" : "Player 2 (Black)"
    println("Game Over! $winner wins!")
  elseif s.awaiting_dice
    player = s.curplayer == WHITE ? "Player 1" : "Player 2"
    println("$player rolling... (turn total: $(s.turn_total))")
  else
    player = s.curplayer == WHITE ? "Player 1" : "Player 2"
    println("$player's turn (turn total: $(s.turn_total))")
    println("Actions: 'r' = roll, 'h' = hold")
  end
  println("=" ^ 40)
end

function GI.action_string(::GameSpec, action)
  return action == ROLL ? "roll" : "hold"
end

function GI.parse_action(::GameSpec, str)
  s = lowercase(strip(str))
  if s == "r" || s == "roll"
    return ROLL
  elseif s == "h" || s == "hold"
    return HOLD
  else
    return nothing
  end
end

#####
##### Heuristic Value (optional, for baselines)
#####

function GI.heuristic_value(g::GameEnv)
  s = g.state
  # Simple heuristic: score difference normalized
  if s.curplayer == WHITE
    return (s.p1_score - s.p2_score + s.turn_total) / TARGET_SCORE
  else
    return (s.p2_score - s.p1_score + s.turn_total) / TARGET_SCORE
  end
end

#####
##### Hold20 Baseline Player
#####
# Simple strategy: hold when turn_total >= 20, otherwise roll

import AlphaZero: AbstractPlayer, think, reset!

struct Hold20Player <: AbstractPlayer
  threshold::Int
end

Hold20Player() = Hold20Player(20)

function think(p::Hold20Player, game)
  s = GI.current_state(game)
  actions = [ROLL, HOLD]

  # If turn_total >= threshold, hold; otherwise roll
  if s.turn_total >= p.threshold
    π = [0.0, 1.0]  # 100% hold
  else
    π = [1.0, 0.0]  # 100% roll
  end

  return actions, π
end

reset!(::Hold20Player) = nothing
