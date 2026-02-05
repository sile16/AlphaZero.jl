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
# - OBSERVATION_TYPE: :minimal_flat (330), :full_flat (362), :minimal (780), :full (1612)
#
# BackgammonNet v0.3.2+ required

import AlphaZero.GI
using StaticArrays
using Random

# Import from BackgammonNet.jl
using BackgammonNet

# Configuration constants
const SHORT_GAME = true  # Shorter games for faster learning (intentionally different starting position)
const DOUBLES_ONLY = false  # Full 21 dice outcomes for proper stochastic testing

# Observation type: Symbol-based configuration (v0.3.2+)
# Can be overridden via environment variable BACKGAMMON_OBS_TYPE
# Available types:
#   :minimal_flat (330)   - Flat vector for MLP (RECOMMENDED)
#   :full_flat (362)      - Flat vector with extra features
#   :biased_flat (?)      - Flat vector with heuristic features
#   :minimal (30×1×26)    - Tensor for conv networks
#   :full (62×1×26)       - Tensor with extra features
#   :minimal_hybrid       - Named tuple (board=12×26, globals) for hybrid nets
#   :full_hybrid          - Named tuple (board=12×26, globals=50) for hybrid nets
#   :biased_hybrid        - Named tuple with heuristic features
const OBS_TYPE_MAP = Dict(
    "minimal" => :minimal_flat,      # Default to flat for MLP networks
    "minimal_flat" => :minimal_flat,
    "full" => :full_flat,
    "full_flat" => :full_flat,
    "biased" => :biased_flat,
    "biased_flat" => :biased_flat,
    "minimal_conv" => :minimal,      # For conv networks
    "full_conv" => :full,
    "minimal_hybrid" => :minimal_hybrid,  # For hybrid networks
    "full_hybrid" => :full_hybrid,
    "biased_hybrid" => :biased_hybrid,
)
const OBS_TYPE_STR = get(ENV, "BACKGAMMON_OBS_TYPE", "minimal")
const OBSERVATION_TYPE = get(OBS_TYPE_MAP, OBS_TYPE_STR, :minimal_flat)

# Action space: 676 actions (26*26 locations)
const NUM_ACTIONS = 676

# Get observation size from BackgammonNet
# Handle hybrid observations (named tuples) by computing flattened size
function _compute_obs_size(obs_type)
  dims = BackgammonNet.obs_dims(obs_type)
  if dims isa NamedTuple
    # Hybrid: (board = (12, 26), globals = 50) -> 12*26 + 50 = 362
    return prod(dims.board) + (dims.globals isa Tuple ? prod(dims.globals) : dims.globals)
  elseif dims isa Tuple
    return prod(dims)
  else
    return dims
  end
end
const OBS_SIZE = _compute_obs_size(OBSERVATION_TYPE)

const Player = Bool
const WHITE = true   # Player 0
const BLACK = false  # Player 1

#####
##### Game Specification
#####

struct GameSpec <: GI.AbstractGameSpec end

GI.two_players(::GameSpec) = true
GI.supports_equity_targets(::GameSpec) = true

function GI.actions(::GameSpec)
  return collect(1:NUM_ACTIONS)
end

# NO chance outcomes - stochasticity is hidden
GI.num_chance_outcomes(::GameSpec) = 0

function GI.vectorize_state(::GameSpec, game::BackgammonNet.BackgammonGame)
  # Use observe() which dispatches based on game.obs_type
  obs = BackgammonNet.observe(game)
  # Handle hybrid observations (named tuples with board and globals)
  if obs isa NamedTuple
    return vcat(vec(obs.board), vec(obs.globals))
  else
    return vec(obs)
  end
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
  # Use clone() for safe deep copy with all fields including obs_type
  return BackgammonNet.clone(g.game)
end

function GI.set_state!(g::GameEnv, state::BackgammonNet.BackgammonGame)
  # Clone the state into our game
  cloned = BackgammonNet.clone(state)
  g.game = cloned

  # If at chance node, auto-roll to get to player turn
  if BackgammonNet.is_chance_node(g.game) && !BackgammonNet.game_terminated(g.game)
    BackgammonNet.sample_chance!(g.game, g.rng)
  end
end

GI.white_playing(g::GameEnv) = g.game.current_player == 0

# Direct state access for white_playing - avoids creating GameEnv which may have side effects
# This is used by push_trace! in memory.jl to determine value perspective
GI.white_playing(::GameSpec, state::BackgammonNet.BackgammonGame) = state.current_player == 0

function GI.init(::GameSpec)
  game = BackgammonNet.initial_state(;
    short_game=SHORT_GAME,
    doubles_only=DOUBLES_ONLY,
    obs_type=OBSERVATION_TYPE
  )
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
  # BackgammonNet: g.reward > 0 means P0 (white) won, < 0 means P1 (black) won
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
  println("Observation type: $(game.obs_type) ($(OBS_SIZE) features)")
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
  # Get pip difference as a simple heuristic
  # Note: This uses internal game state, not observation
  game = g.game

  # Calculate pip counts manually from board state
  p0_pip = 0
  p1_pip = 0
  for i in 1:24
    v = game[i]
    if v > 0
      p0_pip += v * (25 - i)  # White moving toward point 0
    elseif v < 0
      p1_pip += abs(v) * i    # Black moving toward point 25
    end
  end
  # Add bar pieces (25 pips to enter)
  p0_pip += game[0] * 25   # White on bar
  p1_pip += abs(game[25]) * 25  # Black on bar

  pip_diff = p1_pip - p0_pip  # Positive = white ahead

  if game.current_player == 1
    pip_diff = -pip_diff
  end

  # Normalize to roughly [-1, 1] range
  return Float64(pip_diff) / 100.0
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
