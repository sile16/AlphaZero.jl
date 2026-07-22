#####
##### Backgammon (Stochastic) - AlphaZero.jl wrapper for BackgammonNet.jl
#####
#
# Dice rolls are exposed as explicit chance nodes. MCTS sees chance nodes
# and can use the bear-off table at its mathematically correct location
# (pre-dice, where table values = E[equity | board]).
#
# Configuration:
# - short_game=true: Faster games with pieces closer to bearing off
# - every ML dice node has the standard 21 outcomes, including the first roll
# - BACKGAMMON_TAVLA_ENABLED=true: Tavla scoring (backgammons pay as gammons)
# - OBSERVATION_TYPE: baseline :min_plus_flat (352). Also :minimal_flat (346), :full_flat (378),
#   :biased_flat (438), :minimal (46×1×26), :full (78×1×26)
# NOTE: OBS_TYPE_STR is a const read from ENV at PRECOMPILE time (baked). Changing
#   BACKGAMMON_OBS_TYPE at runtime has no effect without a recompile — change the default below.
#
# BackgammonNet v0.7.0+ required

import AlphaZero.GI
using StaticArrays
using Random

# Import from BackgammonNet.jl
using BackgammonNet

# Configuration constants
const SHORT_GAME = true  # Shorter games for faster learning (intentionally different starting position)

_env_bool(name::String, default::Bool) = begin
  v = lowercase(strip(get(ENV, name, default ? "true" : "false")))
  v in ("1", "true", "yes", "on") && return true
  v in ("0", "false", "no", "off") && return false
  error("Invalid boolean value for $name: $(get(ENV, name, ""))")
end

const CUBE_ENABLED = _env_bool("BACKGAMMON_CUBE_ENABLED", false)
const JACOBY_ENABLED = _env_bool("BACKGAMMON_JACOBY_ENABLED", CUBE_ENABLED)
const TAVLA_ENABLED = _env_bool("BACKGAMMON_TAVLA_ENABLED", false)
const STRICT_STATE_CHECKS = _env_bool("BACKGAMMON_STRICT_STATE_CHECKS", false)
CUBE_ENABLED && error(
  "BACKGAMMON_CUBE_ENABLED=true is unsupported by the current 676-action " *
  "checker-policy network; add split cube-policy heads before enabling it")

# AlphaZero uses one configuration name and translates it to BackgammonNet's
# separate observation tier and format arguments.
# Can be overridden via environment variable BACKGAMMON_OBS_TYPE
# Available types (flat = MLP input, conv = tensor, hybrid = named tuple):
#   :minimal_flat (346)       - Flat vector for MLP (RECOMMENDED)
#   :min_plus_flat            - Flat vector with additional features
#   :full_flat                - Flat vector with all features
#   :biased_flat              - Flat vector with heuristic features
#   :minimal (46×1×26)        - Tensor for conv networks
#   :min_plus                 - Tensor with additional features
#   :full (78×1×26)           - Tensor with all features
#   :minimal_hybrid           - Named tuple (board=12×26, globals) for hybrid nets
#   :min_plus_hybrid          - Named tuple with additional globals
#   :full_hybrid              - Named tuple with all globals
#   :biased_hybrid            - Named tuple with heuristic features
const OBS_TYPE_MAP = Dict(
    "minimal_flat" => :minimal_flat,
    "min_plus_flat" => :min_plus_flat,
    "full_flat" => :full_flat,
    "biased_flat" => :biased_flat,
    "minimal" => :minimal,            # Conv tensor format
    "min_plus" => :min_plus,
    "full" => :full,
    "minimal_hybrid" => :minimal_hybrid,
    "min_plus_hybrid" => :min_plus_hybrid,
    "full_hybrid" => :full_hybrid,
    "biased_hybrid" => :biased_hybrid,
)
const OBS_TYPE_STR = get(ENV, "BACKGAMMON_OBS_TYPE", "min_plus_flat")
const OBSERVATION_TYPE = get(OBS_TYPE_MAP, OBS_TYPE_STR) do
    error("Unsupported BACKGAMMON_OBS_TYPE=$OBS_TYPE_STR; expected one of " *
          join(sort!(collect(keys(OBS_TYPE_MAP))), ", "))
end

function _observation_parts(obs_type::Symbol)
  name = String(obs_type)
  if endswith(name, "_flat")
    return Symbol(chop(name; tail=5)), :flat
  elseif endswith(name, "_hybrid")
    return Symbol(chop(name; tail=7)), :hybrid
  elseif obs_type in (:minimal, :min_plus, :full, :biased)
    return obs_type, :spatial
  end
  throw(ArgumentError("unsupported backgammon observation type: $obs_type"))
end

const OBSERVATION_TIER, OBSERVATION_FORMAT = _observation_parts(OBSERVATION_TYPE)

# Checker policy head width (engine IDs 1:676). Cube actions 677:680 are not part of
# this training head — use split cube heads / cube_enabled=false for ML.
# Engine still accepts MAX_ACTIONS via BackgammonNet.apply_action! for cubeful eval.
const NUM_ACTIONS = BackgammonNet.CHECKER_ACTIONS

# Get observation size from BackgammonNet
# Handle hybrid observations (named tuples) by computing flattened size
function _compute_obs_size(tier, format)
  dims = BackgammonNet.obs_dims(tier, format)
  if dims isa NamedTuple
    # Hybrid: flatten the board planes followed by the selected tier's globals.
    return prod(dims.board) + (dims.globals isa Tuple ? prod(dims.globals) : dims.globals)
  elseif dims isa Tuple
    return prod(dims)
  else
    return dims
  end
end
const OBS_SIZE = _compute_obs_size(OBSERVATION_TIER, OBSERVATION_FORMAT)

const Player = Bool
const WHITE = true   # Player 0
const BLACK = false  # Player 1
const LEGAL_ACTIONS_SCRATCHES = [BackgammonNet.LegalActionsScratch() for _ in 1:Threads.maxthreadid()]

legal_actions_scratch() = LEGAL_ACTIONS_SCRATCHES[Threads.threadid()]

"""
    backgammon_game(p0, p1, dice, remaining_actions, current_player,
                    terminated, reward; observation_type=:minimal_flat)

Build a BackgammonNet 0.7 game for AlphaZero fixtures and imported starting
positions. This isolates the dependency's intentionally non-stable concrete
state layout behind public construction plus field assignment.
"""
function backgammon_game(p0::UInt128, p1::UInt128, dice,
                         remaining_actions::Integer, current_player::Integer,
                         terminated::Bool, reward::Real;
                         observation_type::Symbol=:minimal_flat)
  tier, format = _observation_parts(observation_type)
  game = BackgammonNet.initial_state(;
    first_player=current_player,
    doubles_only=false,
    obs_tier=tier,
    obs_format=format,
    cube_enabled=false,
    jacoby_enabled=false,
  )
  game.p0 = p0
  game.p1 = p1
  game.dice = typeof(game.dice)(dice)
  game.remaining_actions = Int8(remaining_actions)
  game.current_player = Int8(current_player)
  game.terminated = terminated
  game.reward = Float32(reward)
  game.phase = iszero(game.dice[1]) && iszero(game.dice[2]) ?
      BackgammonNet.PHASE_CHANCE : BackgammonNet.PHASE_CHECKER_PLAY
  game.result_multiplier = terminated ? _terminal_result_multiplier(game) : Int8(0)
  return game
end

function _terminal_result_multiplier(game)
  p0_off = BackgammonNet.get_count(game.p0, 25)
  p1_off = BackgammonNet.get_count(game.p1, 0)
  if p0_off == 15
    p1_off > 0 && return Int8(1)
    on_bar = BackgammonNet.get_count(game.p1, 27) > 0
    in_home = any(i -> BackgammonNet.get_count(game.p1, i) > 0, 19:24)
    return Int8(on_bar || in_home ? 3 : 2)
  elseif p1_off == 15
    p0_off > 0 && return Int8(1)
    on_bar = BackgammonNet.get_count(game.p0, 26) > 0
    in_home = any(i -> BackgammonNet.get_count(game.p0, i) > 0, 1:6)
    return Int8(on_bar || in_home ? 3 : 2)
  end
  return Int8(1) # declined double
end

#####
##### Game Specification
#####

struct GameSpec <: GI.AbstractGameSpec end

GI.two_players(::GameSpec) = true
GI.supports_equity_targets(::GameSpec) = true

function GI.actions(::GameSpec)
  return collect(1:NUM_ACTIONS)
end

# AlphaZero's ML environment chooses the initial mover during setup. Every dice
# node visible to search, including the first playable roll, therefore uses the
# same standard 21-outcome distribution.
GI.num_chance_outcomes(::GameSpec) = 21

function GI.vectorize_state(::GameSpec, game::BackgammonNet.BackgammonGame)
  obs = BackgammonNet.observe(game, OBSERVATION_TIER, OBSERVATION_FORMAT)
  # Handle hybrid observations (named tuples with board and globals)
  if obs isa NamedTuple
    return vcat(vec(obs.board), vec(obs.globals))
  else
    return vec(obs)
  end
end

# In-place version: writes observation directly into buffer (zero intermediate allocation).
# Uses BackgammonNet v0.7 public tier/format observation API.
function vectorize_state_into!(buf::AbstractVector{Float32}, ::GameSpec, game::BackgammonNet.BackgammonGame)
  OBSERVATION_FORMAT === :flat ||
    throw(ArgumentError("vectorize_state_into! requires a flat observation format"))
  BackgammonNet.observe!(buf, game, Val(OBSERVATION_TIER), Val(:flat))
  return buf
end

#####
##### Game Environment
#####

mutable struct GameEnv <: GI.AbstractGameEnv
  game::BackgammonNet.BackgammonGame
  rng::Union{MersenneTwister,Random.Xoshiro}

  function GameEnv(game::BackgammonNet.BackgammonGame,
                   rng::Union{MersenneTwister,Random.Xoshiro})
    game.phase == BackgammonNet.PHASE_OPENING_ROLL && throw(ArgumentError(
      "AlphaZero ML states must resolve first-player selection during setup"))
    game.doubles_only && throw(ArgumentError(
      "AlphaZero ML states require the standard 21-outcome dice distribution"))
    return new(game, rng)
  end
end

"""Fail closed on any state that is not reachable by legal play.

The strict check is intentionally opt-in because MCTS mutates many cloned
states. It uses BackgammonNet's public exact-checker-count validator, which
catches the corruption left by decrementing an empty bitboard nibble even
though `sanity_check_bitboard` alone cannot identify that underflow.
"""
function strict_validate_state!(game::BackgammonNet.BackgammonGame,
                                context::AbstractString;
                                force::Bool=STRICT_STATE_CHECKS)
  force || return nothing
  try
    BackgammonNet.validate_reachable_state(game)
  catch err
    error("Strict Backgammon state validation failed after $context: " *
          sprint(showerror, err))
  end
  return nothing
end

GI.spec(::GameEnv) = GameSpec()

# Fast clone: 1 full clone instead of default's 2 (init + set_state! each clone)
# Shares RNG to avoid MersenneTwister allocation (2496 bytes each).
# Safe: MCTS clones are sequential within a batch, and RNG is only used for
# chance node sampling during tree traversal (stochastic nature is acceptable).
function GI.clone(g::GameEnv)
  return GameEnv(BackgammonNet.clone(g.game), g.rng)
end

# Zero-allocation clone: copies game state fields into pre-allocated dst,
# reusing dst's internal buffers. Used by batched MCTS game pool.
# Uses BackgammonNet v0.7 copy_state! API (handles all fields, invalidates cache).
function GI.clone_into!(dst::GameEnv, src::GameEnv)
  BackgammonNet.copy_state!(dst.game, src.game)
  dst.rng = src.rng
  return dst
end

function GI.current_state(g::GameEnv)
  # Must return an owning clone.
  #
  # BatchedMCTS stores states in the search tree while reusing pooled GameEnv
  # instances across simulations. If current_state aliases the pool's internal
  # action/source buffers, later clone_into! calls can mutate states that are
  # already in the tree, which breaks the oracle/action contract and can crash
  # MCTS with policy/action-length mismatches.
  return BackgammonNet.clone(g.game)
end

function GI.set_state!(g::GameEnv, state::BackgammonNet.BackgammonGame)
  state.phase == BackgammonNet.PHASE_OPENING_ROLL && throw(ArgumentError(
    "AlphaZero ML states must resolve first-player selection during setup"))
  state.doubles_only && throw(ArgumentError(
    "AlphaZero ML states require the standard 21-outcome dice distribution"))
  # Clone the state into our game
  cloned = BackgammonNet.clone(state)
  g.game = cloned
end

GI.white_playing(g::GameEnv) = g.game.current_player == 0

# Direct state access for white_playing - avoids creating GameEnv which may have side effects
# This is used by push_trace! in memory.jl to determine value perspective
GI.white_playing(::GameSpec, state::BackgammonNet.BackgammonGame) = state.current_player == 0
GI.state_key(::GameSpec, state::BackgammonNet.BackgammonGame) =
  BackgammonNet.game_state_fingerprint(state)

function init_with_rng(::GameSpec, rng::Union{MersenneTwister,Random.Xoshiro})
  first_player = rand(rng, 0:1)
  game = BackgammonNet.initial_state(;
    first_player=first_player,
    short_game=SHORT_GAME,
    doubles_only=false,
    cube_enabled=CUBE_ENABLED,
    jacoby_enabled=JACOBY_ENABLED,
    tavla=TAVLA_ENABLED,
    obs_tier=OBSERVATION_TIER,
    obs_format=OBSERVATION_FORMAT
  )
  genv = GameEnv(game, rng)
  # The mover was selected above, so this is a fresh standard 21-outcome roll.
  BackgammonNet.sample_chance!(game, rng)
  return genv
end

GI.init(gspec::GameSpec) = init_with_rng(gspec, MersenneTwister())

function GI.game_terminated(g::GameEnv)
  return BackgammonNet.game_terminated(g.game)
end

function GI.white_reward(g::GameEnv)
  # BackgammonNet: g.reward > 0 means P0 (white) won, < 0 means P1 (black) won
  return Float64(g.game.reward)
end

function GI.game_outcome(g::GameEnv)
  BackgammonNet.game_terminated(g.game) || return nothing
  white_won = g.game.reward > 0

  # Cube drops terminate before a checker bearoff outcome exists. They are
  # single wins for the probability heads, regardless of board contact/off state.
  winner_off = white_won ?
      BackgammonNet.get_count(g.game.p0, 25) :
      BackgammonNet.get_count(g.game.p1, 0)
  if winner_off != 15
    return GI.GameOutcome(white_won, false, false)
  end

  heads = BackgammonNet.terminal_heads_target(g.game, 0)
  is_gammon = white_won ? heads.p_gammon_win > 0.5f0 : heads.p_gammon_loss > 0.5f0
  is_backgammon = white_won ? heads.p_bg_win > 0.5f0 : heads.p_bg_loss > 0.5f0
  return GI.GameOutcome(white_won, is_gammon, is_backgammon)
end

# Cubeless terminal rewards are ±1/±2/±3 (win/gammon/backgammon). MCTS divides
# rewards by this so ordinary terminal outcomes match the NN value scale
# (equity/3 in [-1, 1]). Cubed games can exceed this range; value-head targets
# still represent outcome probabilities, while cube stakes are carried by reward.
GI.reward_scale(::GameSpec) = 3.0

"""Machine-readable AlphaZero/BackgammonNet ML contract for cluster handshakes."""
function backgammon_ml_contract(::GameSpec)
  return Dict{String,Any}(
    "game" => "backgammon-deterministic",
    "state_dim" => OBS_SIZE,
    "num_actions" => NUM_ACTIONS,
    "observation_tier" => String(OBSERVATION_TIER),
    "observation_format" => String(OBSERVATION_FORMAT),
    "observation_encoding" => BackgammonNet.OBSERVATION_ENCODING_VERSION,
    "value_head_contract" => BackgammonNet.VALUE_HEAD_CONTRACT,
    "value_head_order" => String.(collect(BackgammonNet.VALUE_HEAD_ORDER)),
    "chance_outcomes" => 21,
    "opening_rule" => "uniform_first_player_fresh_21_v1",
    "reward_scale" => GI.reward_scale(GameSpec()),
    "short_game" => SHORT_GAME,
    "cube_enabled" => CUBE_ENABLED,
    "jacoby_enabled" => JACOBY_ENABLED,
    "tavla_enabled" => TAVLA_ENABLED,
    "backgammonnet_version" => string(Base.pkgversion(BackgammonNet)),
  )
end

#####
##### Chance Node Interface (stochastic — dice exposed)
#####

function GI.is_chance_node(g::GameEnv)
  return BackgammonNet.is_chance_node(g.game)
end

function GI.chance_outcomes(g::GameEnv)
  return ML_CHANCE_OUTCOMES
end

const ML_CHANCE_OUTCOMES = SVector{21,Tuple{Int,Float64}}(
  ntuple(i -> (i, BackgammonNet.DICE_PROBS[i]), 21))

const PASS_ACTION = BackgammonNet.encode_action(BackgammonNet.PASS_LOC, BackgammonNet.PASS_LOC)

"""Handle forced passes (auto-play when only move is pass|pass)."""
function _handle_forced_pass!(g::GameEnv)
  while !BackgammonNet.game_terminated(g.game) && !BackgammonNet.is_chance_node(g.game)
    legal = BackgammonNet.legal_actions(g.game)
    if length(legal) == 1 && legal[1] == PASS_ACTION
      BackgammonNet.apply_action!(g.game, PASS_ACTION)
      strict_validate_state!(g.game, "forced pass")
    else
      break
    end
  end
end

function GI.apply_chance!(g::GameEnv, outcome)
  BackgammonNet.apply_chance!(g.game, outcome)
  strict_validate_state!(g.game, "chance outcome $outcome")
  # NOTE: Do NOT call _handle_forced_pass! here. Forced passes must be visible
  # to MCTS as single-option decision nodes so that pswitch (player perspective
  # tracking) correctly handles each player switch individually. If forced passes
  # are auto-played here, MCTS sees stale pswitch when the turn bounces back
  # to the same player (value sign is flipped incorrectly).
end

#####
##### Actions
#####

function GI.actions_mask(g::GameEnv)
  mask = falses(NUM_ACTIONS)

  if BackgammonNet.game_terminated(g.game) || BackgammonNet.is_chance_node(g.game)
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

function GI.actions_mask(::GameSpec, state::BackgammonNet.BackgammonGame)
  mask = falses(NUM_ACTIONS)

  if BackgammonNet.game_terminated(state) || BackgammonNet.is_chance_node(state)
    return mask
  end

  legal = BackgammonNet.legal_actions!(legal_actions_scratch(), state)
  for action in legal
    if 1 <= action <= NUM_ACTIONS
      mask[action] = true
    end
  end

  return mask
end

# Override default available_actions to avoid expensive collect(1:680) + indexing.
# Returns sorted legal action indices directly from BackgammonNet.
function GI.available_actions(g::GameEnv)
  if BackgammonNet.game_terminated(g.game) || BackgammonNet.is_chance_node(g.game)
    return Int[]
  end
  legal = BackgammonNet.legal_actions(g.game)
  # Must be sorted to match oracle P indexing (findall order)
  return sort!(collect(Int, action for action in legal if 1 <= action <= NUM_ACTIONS))
end

function GI.available_actions(::GameSpec, state::BackgammonNet.BackgammonGame)
  if BackgammonNet.game_terminated(state) || BackgammonNet.is_chance_node(state)
    return Int[]
  end
  legal = BackgammonNet.legal_actions!(legal_actions_scratch(), state)
  # Must be sorted to match the env-based `actions_mask`/findall order.
  return sort!(collect(Int, action for action in legal if 1 <= action <= NUM_ACTIONS))
end

function GI.play!(g::GameEnv, action)
  # Apply action only (no auto-dice-roll). Leaves game at chance node or decision node.
  # NOTE: Do NOT call _handle_forced_pass! here. See apply_chance! comment.
  BackgammonNet.apply_action!(g.game, action)
  strict_validate_state!(g.game, "checker action $action")
end

#####
##### Interactive Interface
#####

function GI.render(g::GameEnv)
  game = g.game
  println("=" ^ 50)
  println("Backgammon STOCHASTIC (short_game=$SHORT_GAME, tavla=$TAVLA_ENABLED, dice_outcomes=21)")
  println("Observation type: $(game.obs_tier)_$(game.obs_format) ($(OBS_SIZE) features)")
  println("Cube: enabled=$(game.cube_enabled), value=$(game.cube_value), owner=$(game.cube_owner)")
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
  elseif BackgammonNet.is_chance_node(game)
    player = game.current_player == 0 ? "Player 0" : "Player 1"
    println("$player's turn - Waiting for dice roll (chance node)")
  else
    player = game.current_player == 0 ? "Player 0" : "Player 1"
    d1, d2 = game.dice
    remaining = game.remaining_actions
    println("$player's turn - Phase: $(game.phase), Dice: ($d1, $d2), Actions remaining: $remaining")
  end
  println("=" ^ 50)
end

function GI.action_string(::GameSpec, action)
  return BackgammonNet.action_string(action)
end

function GI.parse_action(::GameSpec, str)
  action = BackgammonNet.parse_action(str)
  return action !== nothing && 1 <= action <= NUM_ACTIONS ? action : nothing
end

#####
##### Heuristic Value
#####

function GI.heuristic_value(g::GameEnv)
  return Float64(BackgammonNet.heuristic_value(g.game))
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
