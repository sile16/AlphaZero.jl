#####
##### Unified Game Loop
##### Single play_game() function replacing 5 duplicate game loops
#####

"""
Unified game loop module for AlphaZero.jl.

Provides a single `play_game()` function that handles:
- Self-play (training data collection with trace recording)
- Evaluation vs external agents (wildbg, GnuBG)
- Value comparison recording (NN vs opponent)
- Bear-off detection and truncation
- Chance node handling (passthrough sampling)
"""
module GameLoop

using ..AlphaZero: GI, BatchedMCTS, MctsParams, Util
import Random

#####
##### Types
#####

"""Abstract type for game-playing agents."""
abstract type GameAgent end

"""
    MctsAgent <: GameAgent

Agent that uses BatchedMCTS for move selection.

Fields:
- `oracle`: Single-state neural network oracle
- `batch_oracle`: Batched oracle for GPU/efficient evaluation
- `mcts_params`: MCTS hyperparameters
- `batch_size`: Inference batch size
- `gspec`: Game specification
- `bearoff_eval`: Optional bear-off evaluator `(game_env) -> Union{Float64, Nothing}`
"""
struct MctsAgent <: GameAgent
    oracle::Any
    batch_oracle::Any
    mcts_params::MctsParams
    batch_size::Int
    gspec::Any
    bearoff_eval::Any  # Union{Nothing, Function}
end

function MctsAgent(oracle, batch_oracle, mcts_params::MctsParams, batch_size::Int, gspec;
                   bearoff_eval=nothing)
    return MctsAgent(oracle, batch_oracle, mcts_params, batch_size, gspec, bearoff_eval)
end

"""
    ExternalAgent <: GameAgent

Agent backed by an external engine (e.g., wildbg).

The `backend` must support `BackgammonNet.agent_move(BackgammonNet.BackendAgent(backend), game)`.
"""
struct ExternalAgent <: GameAgent
    backend::Any
end

"""
    TraceEntry

A single decision point recorded during game play.
"""
struct TraceEntry
    state::Any                    # Game state (typically BackgammonGame)
    player::Int                   # Current player (0 or 1)
    action::Int                   # Action taken
    legal_actions::Vector{Int}    # All legal actions at this state
    policy::Vector{Float32}       # MCTS policy distribution over legal_actions
    is_chance::Bool               # Whether this was a chance node
    is_bearoff::Bool              # Whether this position is in bear-off
    is_contact::Bool              # Whether this position has contact
end

"""
    PositionValueSample

Value comparison sample: NN prediction vs opponent's evaluation at a position.
"""
struct PositionValueSample
    nn_val::Float64
    opponent_val::Float64
    is_contact::Bool
end

"""
    GameResult

Complete result of a played game.
"""
struct GameResult
    reward::Float64                                    # White-relative reward
    trace::Vector{TraceEntry}                          # Decision trace (empty if not recorded)
    value_samples::Vector{PositionValueSample}         # Value comparisons (empty if not recorded)
    num_moves::Int                                     # Number of decision moves played
    bearoff_truncated::Bool                            # Whether game was truncated at bear-off
    first_bearoff_equity::Union{Nothing, Float64}      # Equity at first bear-off position
    first_bearoff_white_playing::Union{Nothing, Bool}  # Who was moving at first bear-off
end

#####
##### Player creation and action selection
#####

"""
    create_player(agent::MctsAgent)

Create a `BatchedMctsPlayer` for the game. Should be created once per game,
reset after the game ends.
"""
function create_player(agent::MctsAgent)
    return BatchedMCTS.BatchedMctsPlayer(
        agent.gspec, agent.oracle, agent.mcts_params;
        batch_size=agent.batch_size, batch_oracle=agent.batch_oracle,
        bearoff_evaluator=agent.bearoff_eval)
end

"""
    create_player(agent::ExternalAgent)

External agents don't need a player object; returns `nothing`.
"""
function create_player(::ExternalAgent)
    return nothing
end

"""
    select_action(agent::MctsAgent, player, env) -> (action, policy, legal_actions)

Run MCTS and return the selected action, policy distribution, and legal actions.
"""
function select_action(agent::MctsAgent, player, env)
    actions, policy = BatchedMCTS.think(player, env)
    return (actions, Float32.(policy), actions)
end

"""
    select_action(agent::ExternalAgent, ::Nothing, env) -> (action, policy, legal_actions)

Use the external backend to select an action. Returns empty policy and legal_actions
since external agents don't produce MCTS policies.
"""
function select_action(agent::ExternalAgent, ::Nothing, env)
    # Access the underlying BackgammonGame from the GI env
    game = env.game
    # Use BackgammonNet's agent_move protocol
    bg_agent = _make_backend_agent(agent.backend)
    action = _agent_move(bg_agent, game)
    return (action, Float32[], Int[])
end

# Lazy imports from BackgammonNet — these are resolved at runtime to avoid
# compile-time dependency on BackgammonNet from within the AlphaZero module.
function _make_backend_agent(backend)
    BackgammonNet = _get_backgammonnet()
    return BackgammonNet.BackendAgent(backend)
end

function _agent_move(bg_agent, game)
    BackgammonNet = _get_backgammonnet()
    return BackgammonNet.agent_move(bg_agent, game)
end

function _is_contact_position(state)
    BackgammonNet = _get_backgammonnet()
    if state isa BackgammonNet.BackgammonGame
        return BackgammonNet.is_contact_position(state)
    end
    return true  # Default: assume contact
end

function _get_backgammonnet()
    mod = Base.get_extension(parentmodule(@__MODULE__), :BackgammonNet)
    if mod !== nothing
        return mod
    end
    # Direct import fallback
    return Main.BackgammonNet
end

#####
##### Chance node sampling
#####

"""Sample a chance outcome using passthrough (weighted random selection)."""
function _sample_chance!(env, rng)
    outcomes = GI.chance_outcomes(env)
    r = rand(rng)
    acc = 0.0
    @inbounds for i in eachindex(outcomes)
        acc += outcomes[i][2]
        if r <= acc
            GI.apply_chance!(env, outcomes[i][1])
            return
        end
    end
    # Fallback to last outcome
    GI.apply_chance!(env, outcomes[end][1])
end

#####
##### Main game loop
#####

"""
    play_game(white::GameAgent, black::GameAgent, env;
              record_trace=false,
              record_value_comparison=false,
              value_oracle=nothing,
              opponent_value_fn=nothing,
              bearoff_truncation=false,
              bearoff_lookup=nothing,
              rng=Random.default_rng(),
              temperature_fn=nothing,
              action_selection_fn=nothing) -> GameResult

Play a complete game between two agents.

# Arguments
- `white`: Agent playing as white (player 0)
- `black`: Agent playing as black (player 1)
- `env`: Game environment (already initialized with starting position)

# Keyword Arguments
- `record_trace`: If true, record all decision points in the trace
- `record_value_comparison`: If true, record NN value vs opponent value at each MCTS decision
- `value_oracle`: Function `(env) -> Float64` returning NN value prediction (for value comparison)
- `opponent_value_fn`: Function `(env) -> Float64` returning opponent's evaluation (for value comparison)
- `bearoff_truncation`: If true, truncate game at first bear-off position
- `bearoff_lookup`: Function `(game) -> Union{Nothing, NamedTuple}` for bear-off equity lookup.
  Must return `(value=Float64, equity=Vector{Float32})` or `nothing`.
- `rng`: Random number generator
- `temperature_fn`: Optional `(move_num::Int) -> Float64` for temperature scheduling.
  If nothing, uses argmax (greedy) for ExternalAgent, τ=1.0 for MctsAgent.
- `action_selection_fn`: Optional `(actions, policy, rng) -> action` for custom action selection.
  If nothing, uses temperature_fn to determine selection.

# Returns
A `GameResult` with the game outcome, trace, and metadata.
"""
function play_game(white::GameAgent, black::GameAgent, env;
                   record_trace::Bool=false,
                   record_value_comparison::Bool=false,
                   value_oracle=nothing,
                   opponent_value_fn=nothing,
                   bearoff_truncation::Bool=false,
                   bearoff_lookup=nothing,
                   rng::Random.AbstractRNG=Random.default_rng(),
                   temperature_fn=nothing,
                   action_selection_fn=nothing)

    # Create MCTS players (once per game)
    white_player = create_player(white)
    black_player = create_player(black)

    trace = TraceEntry[]
    value_samples = PositionValueSample[]
    num_moves = 0
    bearoff_truncated = false
    first_bearoff_equity = nothing
    first_bearoff_wp = nothing

    try
        while !GI.game_terminated(env)
            # ── Chance node handling ──
            if GI.is_chance_node(env)
                # Bear-off detection at chance nodes (pre-dice)
                if bearoff_lookup !== nothing
                    bo = bearoff_lookup(env.game)
                    if bo !== nothing
                        # Track first bear-off position
                        if first_bearoff_equity === nothing
                            first_bearoff_equity = Float64(bo.value)
                            first_bearoff_wp = GI.white_playing(env)
                        end
                        # Optionally truncate at bear-off
                        if bearoff_truncation
                            bearoff_truncated = true
                            break
                        end
                    end
                end

                _sample_chance!(env, rng)
                continue
            end

            # ── Decision node ──
            avail = GI.available_actions(env)

            # Single-action bypass: skip MCTS for forced moves
            if length(avail) == 1
                if record_trace
                    state = GI.current_state(env)
                    wp = GI.white_playing(env)
                    is_contact = _is_contact_position(state)
                    push!(trace, TraceEntry(state, wp ? 0 : 1, avail[1], avail,
                                            Float32[1.0], false, false, is_contact))
                end
                GI.play!(env, avail[1])
                continue
            end

            wp = GI.white_playing(env)
            agent = wp ? white : black
            player = wp ? white_player : black_player
            state = GI.current_state(env)

            # Value comparison recording (at MCTS agent decision points)
            if record_value_comparison && agent isa MctsAgent &&
                    value_oracle !== nothing && opponent_value_fn !== nothing
                nn_v = Float64(value_oracle(env))
                opp_v = Float64(opponent_value_fn(env))
                is_contact = _is_contact_position(state)
                push!(value_samples, PositionValueSample(nn_v, opp_v, is_contact))
            end

            # Select action
            result = select_action(agent, player, env)

            if agent isa MctsAgent
                actions, policy, legal = result
                num_moves += 1

                # Apply temperature and select action
                action = _select_with_temperature(actions, policy, num_moves, rng,
                                                  temperature_fn, action_selection_fn)

                # Record trace
                if record_trace
                    is_contact = _is_contact_position(state)
                    is_bearoff = !is_contact  # Simplified: non-contact = bearoff/race
                    push!(trace, TraceEntry(state, wp ? 0 : 1, action, actions,
                                            policy, false, is_bearoff, is_contact))
                end
            else
                # External agent: action is a scalar
                action = result[1]
                num_moves += 1

                if record_trace
                    is_contact = _is_contact_position(state)
                    push!(trace, TraceEntry(state, wp ? 0 : 1, action, avail,
                                            Float32[], false, !is_contact, is_contact))
                end
            end

            GI.play!(env, action)
        end
    finally
        # Always reset MCTS players to free tree memory
        if white_player !== nothing
            BatchedMCTS.reset_player!(white_player)
        end
        if black_player !== nothing
            BatchedMCTS.reset_player!(black_player)
        end
    end

    # Compute reward
    reward = if bearoff_truncated && first_bearoff_equity !== nothing
        # Use bear-off equity as the reward
        first_bearoff_wp ? first_bearoff_equity : -first_bearoff_equity
    elseif GI.game_terminated(env)
        Float64(GI.white_reward(env))
    else
        0.0
    end

    return GameResult(
        reward,
        trace,
        value_samples,
        num_moves,
        bearoff_truncated,
        first_bearoff_equity,
        first_bearoff_wp)
end

#####
##### Action selection helpers
#####

"""Select action using temperature scheduling or custom function."""
function _select_with_temperature(actions, policy, move_num, rng,
                                  temperature_fn, action_selection_fn)
    # Custom action selection takes priority
    if action_selection_fn !== nothing
        return action_selection_fn(actions, policy, rng)
    end

    # Temperature-based selection
    tau = if temperature_fn !== nothing
        temperature_fn(move_num)
    else
        1.0  # Default: sample proportionally
    end

    if iszero(tau)
        return actions[argmax(policy)]
    elseif isone(tau)
        return actions[_sample_from_policy(policy, rng)]
    else
        temp_policy = Util.apply_temperature(policy, tau)
        return actions[_sample_from_policy(temp_policy, rng)]
    end
end

"""Sample an action index from a probability distribution."""
function _sample_from_policy(policy::AbstractVector{<:Real}, rng)
    r = rand(rng)
    cumsum = 0.0
    for i in 1:length(policy)
        cumsum += policy[i]
        if r <= cumsum
            return i
        end
    end
    return length(policy)
end

end  # module GameLoop
