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
#
# OBJECTIVE WEIGHTS (added 2026-07-03, backward compatible):
# `bearoff_turn_value` / `bearoff_best_move_value` take an optional `weights`
# tuple (w_pw, w_gw, w_pl, w_gl) over (plain win, gammon win, plain loss, gammon
# loss). The default `BEAROFF_MONEY_WEIGHTS = (1, 2, -1, -2)` reproduces
# `compute_equity` / the raw reward EXACTLY (money play), so every existing
# caller is unchanged. Non-money objectives (dmp/gg/gs) reweight the same
# probabilities/outcomes so callers can pick moves that optimize a match
# objective instead of money equity.

"""Money weights: reproduce `compute_equity` and raw `reward` exactly."""
const BEAROFF_MONEY_WEIGHTS = (1.0, 2.0, -1.0, -2.0)

"""Objective value of a `BearoffResult` (mover-perspective) in points.
For money weights this equals `compute_equity(r)` (bit-exact fast path)."""
@inline function _bearoff_objective_value(r, weights)::Float64
    weights == BEAROFF_MONEY_WEIGHTS && return Float64(BearoffK7.compute_equity(r))
    pW = Float64(r.pW); pWG = Float64(r.pWG); pLG = Float64(r.pLG)
    p_plain_win = pW - pWG            # win, not a gammon
    p_plain_loss = (1.0 - pW) - pLG   # loss, not a gammon
    return weights[1] * p_plain_win + weights[2] * pWG +
           weights[3] * p_plain_loss + weights[4] * pLG
end

"""Objective value of a terminal reward (already mover-relative, signed points).
For money weights this returns the raw reward (bit-exact, handles backgammon ±3).
For other objectives, classify by |reward| (1 = plain, ≥2 = gammon; bearoff has
no backgammons) and apply the weight."""
@inline function _bearoff_terminal_value(mover_r::Float64, weights)::Float64
    weights == BEAROFF_MONEY_WEIGHTS && return mover_r
    ar = abs(mover_r)
    if mover_r > 0
        return ar >= 2 ? Float64(weights[2]) : Float64(weights[1])
    elseif mover_r < 0
        return ar >= 2 ? Float64(weights[4]) : Float64(weights[3])
    else
        return 0.0
    end
end

"""
    bearoff_turn_value(table, game, mover; weights=BEAROFF_MONEY_WEIGHTS) -> Float64

Exact value of `game` from `mover`'s perspective in RAW points ([-3,3] scale),
where `game` is a position somewhere inside/after `mover`'s turn:

- terminal: from `game.reward` (carries the gammon/backgammon multiplier)
- chance node (turn complete): pre-dice table lookup for the player on roll,
  sign-adjusted to `mover`'s perspective
- decision node (doubles mid-turn, same player to move): recursively enumerate
  the remaining actions and take the max

`weights` reweights outcomes for a match objective; the default reproduces money
equity / raw reward exactly (see `BEAROFF_MONEY_WEIGHTS`).
"""
function bearoff_turn_value(table, game, mover::Integer; weights=BEAROFF_MONEY_WEIGHTS)::Float64
    if game.terminated
        white_r = Float64(game.reward)
        mover_r = mover == 0 ? white_r : -white_r
        return _bearoff_terminal_value(mover_r, weights)
    elseif BackgammonNet.is_chance_node(game)
        r = BearoffK7.lookup(table, game)
        v = _bearoff_objective_value(r, weights)
        # lookup() is from game.current_player's perspective
        return Int(game.current_player) == Int(mover) ? v : -v
    else
        # Doubles mid-turn: same player still to move
        best = -Inf
        work = BackgammonNet.clone(game)
        for a in BackgammonNet.legal_actions(game)
            BackgammonNet.copy_state!(work, game)
            BackgammonNet.apply_action!(work, a)
            v = bearoff_turn_value(table, work, mover; weights=weights)
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
    bearoff_best_move_value(table, game; weights=BEAROFF_MONEY_WEIGHTS) -> Float64

Exact post-dice Q-value of `game` (a decision node with dice rolled) from the
CURRENT PLAYER's perspective in raw points: enumerate legal moves, score each
via `bearoff_turn_value`, return the max. Handles doubles turns correctly.
`weights` selects the objective (default = money; see `BEAROFF_MONEY_WEIGHTS`).
"""
function bearoff_best_move_value(table, game; weights=BEAROFF_MONEY_WEIGHTS)::Float64
    mover = Int(game.current_player)
    best = -Inf
    work = BackgammonNet.clone(game)
    for a in BackgammonNet.legal_actions(game)
        BackgammonNet.copy_state!(work, game)
        BackgammonNet.apply_action!(work, a)
        v = bearoff_turn_value(table, work, mover; weights=weights)
        v > best && (best = v)
    end
    return best
end
