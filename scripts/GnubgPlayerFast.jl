# GnubgPlayerFast.jl - Optimized GnuBG player using native best_move
#
# Uses gnubg's native best_move function for ~40x speedup over iterating legal moves.
# Performance: ~25K moves/sec (vs ~2K with old method)
#
# Usage:
#   include("scripts/GnubgPlayerFast.jl")
#   using .GnubgPlayerFast
#   gnubg = GnubgFast(ply=0)

module GnubgPlayerFast

using BackgammonNet
using PyCall

export GnubgFast, best_move_native

# =============================================================================
# Initialization
# =============================================================================

const _gnubg = PyNULL()
const _initialized = Ref(false)

function _init()
    if !_initialized[]
        copy!(_gnubg, pyimport("gnubg"))
        _initialized[] = true
    end
end

# =============================================================================
# Board Conversion (from BackgammonNet GnubgInterface)
# =============================================================================

function _to_gnubg_board(g::BackgammonGame)
    # gnubg format (0-indexed): indices 0-23 = points (ace=0, 24-pt=23), index 24 = bar
    # Each array uses its OWN player's perspective:
    #   P0: gnubg index i → BNet point (24-i)  (P0 bears off from BNet 24 side)
    #   P1: gnubg index i → BNet point (i+1)   (P1 bears off from BNet 1 side)
    board = zeros(Int, 2, 25)
    cp = Int(g.current_player)
    p0, p1 = g.p0, g.p1

    # Julia 1-indexed: col 1-24 = points, col 25 = bar
    if cp == 0
        for col in 1:24
            board[2, col] = Int((p0 >> ((25 - col) << 2)) & 0xF)  # P0 on-roll: BNet(25-col)
            board[1, col] = Int((p1 >> (col << 2)) & 0xF)          # P1 opponent: BNet(col)
        end
        board[2, 25] = Int((p0 >> (BackgammonNet.IDX_P0_BAR << 2)) & 0xF)
        board[1, 25] = Int((p1 >> (BackgammonNet.IDX_P1_BAR << 2)) & 0xF)
    else
        for col in 1:24
            board[2, col] = Int((p1 >> (col << 2)) & 0xF)          # P1 on-roll: BNet(col)
            board[1, col] = Int((p0 >> ((25 - col) << 2)) & 0xF)  # P0 opponent: BNet(25-col)
        end
        board[2, 25] = Int((p1 >> (BackgammonNet.IDX_P1_BAR << 2)) & 0xF)
        board[1, 25] = Int((p0 >> (BackgammonNet.IDX_P0_BAR << 2)) & 0xF)
    end

    return [board[1, :], board[2, :]]
end

# =============================================================================
# Convert gnubg move to BackgammonNet action
# =============================================================================

"""
Convert gnubg move (from, to) to BackgammonNet action.

gnubg format:
- Points 1-24 (1-indexed)
- 25 = bar (entry)
- 0 = bear off

BackgammonNet format:
- 0 = bar
- 1-24 = points
- 25 = pass
- Bear off: to=0
"""
function _gnubg_move_to_action(from::Int, to::Int)
    # Convert gnubg bar (25) to BackgammonNet bar (0)
    bg_from = from == 25 ? 0 : from
    # gnubg 0 (bear off) stays as 0 in BackgammonNet
    bg_to = to

    return BackgammonNet.encode_action(bg_from, bg_to)
end

# =============================================================================
# Native Best Move
# =============================================================================

# BackgammonNet uses "combo moves" - each action applies both dice values
# remaining_actions: 1 for non-doubles, 2 for doubles
function _expected_remaining(d1, d2)
    return d1 == d2 ? 2 : 1
end

"""
    best_move_native(g::BackgammonGame; ply::Int=0) -> Int

Get best move using gnubg's native best_move function.
Returns the encoded BackgammonNet action (full combo move).

~16x faster than iterating through all legal moves at turn start.
Falls back to position evaluation for mid-turn (doubles only).
"""
function best_move_native(g::BackgammonGame; ply::Int=0)
    _init()

    # Check for terminal or no moves
    if g.terminated || g.remaining_actions == 0
        return BackgammonNet.encode_action(25, 25)  # Pass
    end

    d1, d2 = Int(g.dice[1]), Int(g.dice[2])
    expected_remaining = _expected_remaining(d1, d2)

    # For doubles mid-turn (remaining=1 when expected=2), use eval fallback
    if g.remaining_actions != expected_remaining
        return _best_move_evaluate(g; ply=ply)
    end

    # Convert board to gnubg format
    board = _to_gnubg_board(g)

    # Call native best_move (n=ply for search depth)
    moves = _gnubg.best_move(board, d1, d2, n=ply)

    # Handle no legal moves
    if isempty(moves) || moves == () || (length(moves) > 0 && moves[1] == (0, 0))
        return BackgammonNet.encode_action(25, 25)  # Pass
    end

    # Convert gnubg moves to BackgammonNet action
    # gnubg returns list of (from, to) pairs, we need to encode as single action
    return _gnubg_moves_to_action(moves, d1, d2, Int(g.current_player))
end

"""
Convert gnubg move tuples to BackgammonNet encoded action.
BackgammonNet actions encode both moves together as (loc1, loc2) where each
loc is the SOURCE point (0=bar, 1-24=points, 25=pass).

Point mapping for player 0: gnubg point N = BackgammonNet point (25-N)
Point mapping for player 1: gnubg point N = BackgammonNet point N
"""
function _gnubg_moves_to_action(moves, d1, d2, current_player::Int)
    # gnubg returns ((from1, to1), (from2, to2), ...) for each checker moved
    # For non-doubles: 2 tuples (one per die)
    # For doubles: 4 tuples (but remaining_actions=2, so 2 combo moves of 2 each)

    if length(moves) == 0 || moves[1] == (0, 0)
        return BackgammonNet.encode_action(25, 25)  # Pass
    end

    # Convert gnubg move point to BackgammonNet source point
    # gnubg moves: 25=bar, 1-24=points (1-indexed, own perspective), 0=bearoff
    # BackgammonNet: 0=bar, 1-24=points (absolute), 25=pass
    # Mapping: gnubg move point N → gnubg board idx (N-1) → BNet point
    #   P0: BNet = 24 - (N-1) = 25 - N  (P0's ace=gnubg1 → BNet24)
    #   P1: BNet = (N-1) + 1 = N        (P1's ace=gnubg1 → BNet1)
    function convert_point(gnubg_pt)
        if gnubg_pt == 25
            return 0  # bar -> 0
        elseif gnubg_pt == 0
            return 0  # bear off -> 0 (as destination)
        elseif current_player == 0
            return 25 - gnubg_pt  # Player 0: gnubg 1 → BNet 24 (ace point)
        else
            return gnubg_pt  # Player 1: gnubg N → BNet N
        end
    end

    # Get the first two moves (one combo move)
    n_moves = min(length(moves), 2)

    if n_moves == 1
        from1, _ = moves[1]
        loc1 = convert_point(from1)
        return BackgammonNet.encode_action(loc1, 25)  # Single move + pass
    end

    from1, _ = moves[1]
    from2, _ = moves[2]

    loc1 = convert_point(from1)
    loc2 = convert_point(from2)

    return BackgammonNet.encode_action(loc1, loc2)
end

"""
Fallback: evaluate all legal moves (used for mid-turn decisions)
"""
function _best_move_evaluate(g::BackgammonGame; ply::Int=0)
    actions = BackgammonNet.legal_actions(g)

    if isempty(actions)
        return BackgammonNet.encode_action(25, 25)
    end

    best_action = actions[1]
    best_equity = -Inf

    for action in actions
        # Create copy using clone() and apply action
        g2 = BackgammonNet.clone(g)
        BackgammonNet.apply_action!(g2, action)

        # Evaluate
        equity = _evaluate(g2; ply=ply)
        if g2.current_player != g.current_player
            equity = -equity
        end

        if equity > best_equity
            best_equity = equity
            best_action = action
        end
    end

    return best_action
end

"""
Evaluate position using gnubg.
"""
function _evaluate(g::BackgammonGame; ply::Int=0)
    board = _to_gnubg_board(g)
    probs = _gnubg.probabilities(board, ply)
    win = Float64(probs[1])
    win_g = Float64(probs[2])
    win_bg = Float64(probs[3])
    lose_g = Float64(probs[4])
    lose_bg = Float64(probs[5])
    return (win - (1.0 - win)) + (win_g - lose_g) + (win_bg - lose_bg)
end

# =============================================================================
# AlphaZero Player Interface
# =============================================================================

import AlphaZero: AbstractPlayer, think, reset!

"""
Fast GnuBG player using native best_move.

    GnubgFast()        # 0-ply (neural net only)
    GnubgFast(ply=1)   # 1-ply lookahead

Performance: ~25,000 moves/sec at 0-ply (vs ~2,000 with old method)
"""
struct GnubgFast <: AbstractPlayer
    ply::Int
end

GnubgFast(; ply::Int=0) = GnubgFast(ply)

function think(p::GnubgFast, game)
    bg_game = game.game

    # Use native best_move
    action = best_move_native(bg_game; ply=p.ply)

    # Create policy with all probability on chosen action
    num_actions = 676
    π = zeros(num_actions)
    if 1 <= action <= num_actions
        π[action] = 1.0
    end

    return collect(1:num_actions), π
end

reset!(::GnubgFast) = nothing

# =============================================================================
# Benchmark Interface
# =============================================================================

import AlphaZero.Benchmark

struct GnubgFastBenchmark <: Benchmark.Player
    ply::Int
end

GnubgFastBenchmark(; ply::Int=0) = GnubgFastBenchmark(ply)

Benchmark.name(p::GnubgFastBenchmark) = "GnuBG-$(p.ply)ply-fast"

function Benchmark.instantiate(p::GnubgFastBenchmark, ::Any, nn)
    return GnubgFast(p.ply)
end

end # module
