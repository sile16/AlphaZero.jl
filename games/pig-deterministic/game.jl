#####
##### Deterministic Pig - Hidden Stochasticity Version for Standard AlphaZero
#####
#
# This is the same game as Pig, but the dice roll happens INSIDE play!
# so that standard AlphaZero (without chance nodes) can be compared against
# the stochastic version.
#

import AlphaZero.GI
using Random

const TARGET_SCORE = 50
const MAX_TURNS = 200
const NUM_DIE_FACES = 6

# Actions
const ROLL = 1
const HOLD = 2
const NUM_ACTIONS = 2

const Player = Bool
const WHITE = true   # Player 1
const BLACK = false  # Player 2

# State: no awaiting_dice - dice is rolled immediately inside play!
const State = @NamedTuple{
  p1_score::Int,
  p2_score::Int,
  turn_total::Int,
  curplayer::Player,
  turn_count::Int
}

const INITIAL_STATE = State((0, 0, 0, WHITE, 0))

#####
##### Game Specification
#####

struct GameSpec <: GI.AbstractGameSpec end

GI.two_players(::GameSpec) = true
GI.actions(::GameSpec) = [ROLL, HOLD]

# NO chance outcomes - this is "deterministic" from MCTS perspective
GI.num_chance_outcomes(::GameSpec) = 0

function GI.vectorize_state(::GameSpec, state)
  # Normalize scores to [0, 1] range
  p1_norm = Float32(state.p1_score / TARGET_SCORE)
  p2_norm = Float32(state.p2_score / TARGET_SCORE)
  turn_norm = Float32(state.turn_total / TARGET_SCORE)
  curplayer = state.curplayer == WHITE ? 1f0 : 0f0
  turn_count_norm = Float32(state.turn_count / MAX_TURNS)
  return Float32[p1_norm, p2_norm, turn_norm, curplayer, turn_count_norm]
end

#####
##### Game Environment
#####

mutable struct GameEnv <: GI.AbstractGameEnv
  state::State
  rng::AbstractRNG
end

GI.spec(::GameEnv) = GameSpec()
GI.current_state(g::GameEnv) = g.state
GI.set_state!(g::GameEnv, s) = (g.state = s)
GI.white_playing(g::GameEnv) = g.state.curplayer == WHITE

function GI.init(::GameSpec)
  return GameEnv(INITIAL_STATE, Random.default_rng())
end

function GI.init(::GameSpec, state)
  return GameEnv(state, Random.default_rng())
end

function GI.game_terminated(g::GameEnv)
  return g.state.p1_score >= TARGET_SCORE ||
         g.state.p2_score >= TARGET_SCORE ||
         g.state.turn_count >= MAX_TURNS
end

function GI.white_reward(g::GameEnv)
  if g.state.p1_score >= TARGET_SCORE
    return 1.0  # White (player 1) wins
  elseif g.state.p2_score >= TARGET_SCORE
    return -1.0  # Black (player 2) wins
  else
    return 0.0  # Draw (max turns reached)
  end
end

#####
##### Chance Node Interface - NOT a chance game
#####

function GI.is_chance_node(g::GameEnv)
  return false  # Never a chance node - dice is hidden
end

#####
##### Actions
#####

function GI.actions_mask(g::GameEnv)
  # Both ROLL and HOLD are always available
  return [true, true]
end

function GI.play!(g::GameEnv, action)
  s = g.state

  if action == ROLL
    # Dice roll happens INSIDE play! - hidden from MCTS
    die_face = rand(g.rng, 1:NUM_DIE_FACES)

    if die_face == 1
      # Rolled a 1: lose turn total, switch players, increment turn count
      new_player = !s.curplayer
      g.state = State((s.p1_score, s.p2_score, 0, new_player, s.turn_count + 1))
    else
      # Rolled 2-6: add to turn total
      new_turn_total = s.turn_total + die_face
      g.state = State((s.p1_score, s.p2_score, new_turn_total, s.curplayer, s.turn_count))
    end
  elseif action == HOLD
    # Add turn total to score, switch players, increment turn count
    if s.curplayer == WHITE
      new_p1_score = s.p1_score + s.turn_total
      g.state = State((new_p1_score, s.p2_score, 0, BLACK, s.turn_count + 1))
    else
      new_p2_score = s.p2_score + s.turn_total
      g.state = State((s.p1_score, new_p2_score, 0, WHITE, s.turn_count + 1))
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
  println("Turn: $(s.turn_count) / $MAX_TURNS")
  println("-" ^ 40)
  if GI.game_terminated(g)
    if s.turn_count >= MAX_TURNS
      println("Game Over! Max turns reached - Draw!")
    elseif s.p1_score >= TARGET_SCORE
      println("Game Over! Player 1 (White) wins!")
    else
      println("Game Over! Player 2 (Black) wins!")
    end
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
##### Heuristic Value
#####

function GI.heuristic_value(g::GameEnv)
  s = g.state
  if s.curplayer == WHITE
    return (s.p1_score - s.p2_score + s.turn_total) / TARGET_SCORE
  else
    return (s.p2_score - s.p1_score + s.turn_total) / TARGET_SCORE
  end
end

#####
##### Hold20 Baseline Player
#####

import AlphaZero: AbstractPlayer, think, reset_player!

struct Hold20Player <: AbstractPlayer
  threshold::Int
end

Hold20Player() = Hold20Player(20)

function think(p::Hold20Player, game)
  s = GI.current_state(game)
  actions = [ROLL, HOLD]

  if s.turn_total >= p.threshold
    π = [0.0, 1.0]  # 100% hold
  else
    π = [1.0, 0.0]  # 100% roll
  end

  return actions, π
end

reset_player!(::Hold20Player) = nothing
