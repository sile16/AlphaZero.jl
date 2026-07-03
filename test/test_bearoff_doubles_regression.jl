# Regression test for the doubles mid-turn bear-off scoring bug (2026-07-03).
#
# With 2-checker action encoding, a doubles turn is TWO sequential actions
# (`remaining_actions = 2`). After the first action, the state is still the SAME
# player's decision node. The old evaluators scored every post-move state as
# "opponent pre-dice, negate" — flipping the sign for the first half of every
# doubles turn, so argmax picked the WORST move. Caught by
# scripts/eval_table_vs_wildbg.jl: the exact table policy measurably LOST to
# wildbg (-0.022 ± 0.010 paired) before the fix.
#
# The turn-aware helpers (scripts/bearoff_eval_common.jl) recurse through
# mid-turn states. The recursion and terminal handling are exercised here
# WITHOUT the 88GB table (only the chance-node branch needs it).

using Test
using Random
using StaticArrays

using AlphaZero
import BackgammonNet
using BackgammonNet: BackgammonGame

const BEAROFF_K7_SRC = joinpath(homedir(), "github", "BackgammonNet.jl", "src", "bearoff_k7.jl")

if !isfile(BEAROFF_K7_SRC)
    @warn "bearoff_k7.jl not found — skipping doubles regression tests"
else
    if !isdefined(Main, :BearoffK7)
        include(BEAROFF_K7_SRC)
        using .BearoffK7
    end
    if !isdefined(Main, :bearoff_turn_value)
        include(joinpath(@__DIR__, "..", "scripts", "bearoff_eval_common.jl"))
    end

    @testset "Doubles mid-turn recursion (no table needed)" begin
        # White (P0): 11 off, 4 checkers on point 24. Black (P1): 15 in home,
        # 0 off — any white win is a GAMMON (+2).
        p0 = (UInt128(11) << (25 * 4)) | (UInt128(4) << (24 * 4))
        p1 = (UInt128(5) << (1 * 4)) | (UInt128(5) << (2 * 4)) | (UInt128(5) << (3 * 4))

        # Doubles 3-3, remaining_actions = 2: white's turn takes TWO actions.
        g = BackgammonGame(p0, p1, SVector{2, Int8}(3, 3), Int8(2), Int8(0), false, 0.0f0;
                           obs_type=:minimal_flat)
        @test !BackgammonNet.is_chance_node(g)

        # Apply the first action -> mid-turn state: SAME player, remaining = 1.
        acts = BackgammonNet.legal_actions(g)
        @test !isempty(acts)
        mid = BackgammonNet.clone(g)
        BackgammonNet.apply_action!(mid, acts[1])
        @test !mid.terminated
        @test mid.current_player == 0            # still white's move
        @test mid.remaining_actions == 1
        @test !BackgammonNet.is_chance_node(mid)

        # The old code would have treated `mid` as the opponent's pre-dice node
        # and NEGATED its value. The turn-aware helper recurses: white finishes
        # bearing off with the second action -> terminal gammon (+2 raw points).
        # Table not needed: only terminal states are reached.
        v = bearoff_turn_value(nothing, mid, 0)
        @test v == 2.0

        # Same through the equity variant: gammon vector [1,1,0,0,0].
        v2, eq = bearoff_turn_value_equity(nothing, mid, 0)
        @test v2 == 2.0
        @test eq == Float32[1, 1, 0, 0, 0]

        # Terminal equity is mover-relative in both directions. The negative
        # case is rare in normal post-move scoring, but callers of the shared
        # helper should not get a win vector when the requested mover lost.
        lost = BackgammonGame(p0, p1, SVector{2, Int8}(0, 0), Int8(0), Int8(0), true, -2.0f0;
                              obs_type=:minimal_flat)
        v_loss, eq_loss = bearoff_turn_value_equity(nothing, lost, 0)
        @test v_loss == -2.0
        @test eq_loss == Float32[0, 0, 0, 1, 0]

        # Full post-dice move evaluation from the doubles start: every playout
        # bears off all 4 checkers this turn -> exact Q is the gammon win.
        @test bearoff_best_move_value(nothing, g) == 2.0

        # Sign check from black's perspective: white's gammon is -2 for black.
        @test bearoff_turn_value(nothing, mid, 1) == -2.0
    end
end
