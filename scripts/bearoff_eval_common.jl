# Shared turn-aware bear-off evaluation helpers.
#
# Include AFTER BearoffK7 and BackgammonNet are loaded (pasted into the including
# script's scope, same pattern as src/distributed/buffer.jl).
#
# THE DOUBLES PITFALL (found 2026-07-03 via eval_table_vs_wildbg.jl):
# with 2-checker action encoding, a doubles turn is TWO sequential actions
# (`remaining_actions = 2`). After applying the first action the state is still
# the SAME player's decision node — not the opponent's pre-dice chance node.
# Evaluators that score every post-move state as "opponent pre-dice, negate"
# mis-score the first half of every doubles turn (1/6 of rolls): the negation
# flips the sign, so argmax actively picks the WORST first move. The exact-table
# policy measurably LOST to wildbg (-0.022 ± 0.010 paired) before this fix.
#
# These helpers recurse through mid-turn states so callers get the true value
# of a move considering the full remaining turn.

"""
    bearoff_turn_value(table, game, mover) -> Float64

Exact value of `game` from `mover`'s perspective in RAW points ([-3,3] scale),
where `game` is a position somewhere inside/after `mover`'s turn:

- terminal: from `game.reward` (carries the gammon/backgammon multiplier)
- chance node (turn complete): pre-dice table lookup for the player on roll,
  sign-adjusted to `mover`'s perspective
- decision node (doubles mid-turn, same player to move): recursively enumerate
  the remaining actions and take the max
"""
function bearoff_turn_value(table, game, mover::Integer)::Float64
    if game.terminated
        white_r = Float64(game.reward)
        return mover == 0 ? white_r : -white_r
    elseif BackgammonNet.is_chance_node(game)
        r = BearoffK7.lookup(table, game)
        v = Float64(BearoffK7.compute_equity(r))
        # lookup() is from game.current_player's perspective
        return Int(game.current_player) == Int(mover) ? v : -v
    else
        # Doubles mid-turn: same player still to move
        best = -Inf
        work = BackgammonNet.clone(game)
        for a in BackgammonNet.legal_actions(game)
            BackgammonNet.copy_state!(work, game)
            BackgammonNet.apply_action!(work, a)
            v = bearoff_turn_value(table, work, mover)
            v > best && (best = v)
        end
        return best
    end
end

"""
    bearoff_turn_value_equity(table, game, mover) -> (value::Float64, equity::Vector{Float32})

Like `bearoff_turn_value` but also returns the 5-head joint cumulative equity
vector from `mover`'s perspective (for training targets). Value in raw points.
"""
function bearoff_turn_value_equity(table, game, mover::Integer)
    if game.terminated
        white_r = Float32(game.reward)
        mval = mover == 0 ? white_r : -white_r  # > 0: mover just won
        is_g = mval >= 2.0f0
        is_bg = mval >= 3.0f0
        eq = Float32[1.0, is_g ? 1.0 : 0.0, is_bg ? 1.0 : 0.0, 0.0, 0.0]
        return (Float64(mval), eq)
    elseif BackgammonNet.is_chance_node(game)
        r = BearoffK7.lookup(table, game)
        v = Float64(BearoffK7.compute_equity(r))
        eq = Float32[r.pW, r.pWG, 0.0f0, r.pLG, 0.0f0]
        if Int(game.current_player) == Int(mover)
            return (v, eq)
        else
            return (-v, AlphaZero.flip_equity_perspective(eq))
        end
    else
        best_v = -Inf
        best_eq = Float32[]
        work = BackgammonNet.clone(game)
        for a in BackgammonNet.legal_actions(game)
            BackgammonNet.copy_state!(work, game)
            BackgammonNet.apply_action!(work, a)
            v, eq = bearoff_turn_value_equity(table, work, mover)
            if v > best_v
                best_v = v
                best_eq = eq
            end
        end
        return (best_v, best_eq)
    end
end

"""
    bearoff_best_move_value(table, game) -> Float64

Exact post-dice Q-value of `game` (a decision node with dice rolled) from the
CURRENT PLAYER's perspective in raw points: enumerate legal moves, score each
via `bearoff_turn_value`, return the max. Handles doubles turns correctly.
"""
function bearoff_best_move_value(table, game)::Float64
    mover = Int(game.current_player)
    best = -Inf
    work = BackgammonNet.clone(game)
    for a in BackgammonNet.legal_actions(game)
        BackgammonNet.copy_state!(work, game)
        BackgammonNet.apply_action!(work, a)
        v = bearoff_turn_value(table, work, mover)
        v > best && (best = v)
    end
    return best
end
