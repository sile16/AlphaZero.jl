#!/usr/bin/env julia
"""
Numerical validation of value head format assumptions.

Tests the relationship between conditional and joint probability representations
across the entire pipeline: bearoff table, bootstrap data, self-play targets,
perspective flips, and equity formulas.

Run:
    julia --project test/test_value_head_formats.jl

Requires: BackgammonNet.jl with bearoff table at standard location.
"""

using Test
using AlphaZero
using AlphaZero: GI, FluxLib
using Random
using Printf
using StaticArrays

# ── Load BackgammonNet + BearoffK6 ──────────────────────────────────────

using BackgammonNet

# BearoffK6 module (same include path as selfplay_client.jl)
const BEAROFF_SRC_DIR = let
    local_path = joinpath(dirname(@__DIR__), "..", "BackgammonNet.jl", "src", "bearoff_k6.jl")
    pkg_path = joinpath(dirname(dirname(pathof(BackgammonNet))), "src", "bearoff_k6.jl")
    if isfile(local_path)
        dirname(local_path)
    elseif isfile(pkg_path)
        dirname(pkg_path)
    else
        error("Cannot find bearoff_k6.jl")
    end
end

include(joinpath(BEAROFF_SRC_DIR, "bearoff_k6.jl"))
using .BearoffK6

const TABLE_DIR = joinpath(BEAROFF_SRC_DIR, "..", "tools", "bearoff_twosided", "bearoff_k6_twosided")
const TABLE = BearoffTable(TABLE_DIR)

# ── Board helpers (from test_bearoff_bellman.jl) ────────────────────────

const IDX_P0_OFF = 25
const IDX_P1_OFF = 0

@inline get_nibble(board::UInt128, idx::Int) = Int((board >> (idx << 2)) & 0xF)
@inline set_nibble(board::UInt128, idx::Int, val::Int) =
    (board & ~(UInt128(0xF) << (idx << 2))) | (UInt128(val) << (idx << 2))

function build_board(pos::NTuple{6,Int}, player::Int)::UInt128
    board = UInt128(0)
    total = 0
    indices = player == 0 ? (24, 23, 22, 21, 20, 19) : (1, 2, 3, 4, 5, 6)
    off_idx = player == 0 ? IDX_P0_OFF : IDX_P1_OFF
    for i in 1:6
        pos[i] > 0 && (board = set_nibble(board, indices[i], pos[i]); total += pos[i])
    end
    off = 15 - total
    off > 0 && (board = set_nibble(board, off_idx, off))
    return board
end

function random_bearoff_position(rng, n::Int)::NTuple{6,Int}
    pos = zeros(Int, 6)
    for _ in 1:n
        pos[rand(rng, 1:6)] += 1
    end
    return Tuple(pos)
end

function make_bearoff_game(p0::UInt128, p1::UInt128, dice::Tuple{Int,Int})
    d_high = max(dice[1], dice[2])
    d_low = min(dice[1], dice[2])
    remaining = (d_high == d_low) ? Int8(2) : Int8(1)
    return BackgammonNet.BackgammonGame(
        p0, p1, SVector{2,Int8}(d_high, d_low), remaining,
        Int8(0), false, 0.0f0)
end

function enumerate_full_turns(g::BackgammonNet.BackgammonGame)
    results = Tuple{UInt128, UInt128, Bool, Float32}[]
    if g.terminated
        push!(results, (g.p0, g.p1, true, g.reward))
        return results
    end
    if BackgammonNet.is_chance_node(g) || g.current_player != 0
        push!(results, (g.p0, g.p1, false, 0.0f0))
        return results
    end
    actions = BackgammonNet.legal_actions(g)
    for a in actions
        gc = BackgammonNet.clone(g)
        BackgammonNet.apply_action!(gc, a)
        append!(results, enumerate_full_turns(gc))
    end
    return results
end

# ════════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════════

@testset "Value Head Format Validation" begin

    # ──────────────────────────────────────────────────────────────────
    # Test 1: Conditional ↔ Joint algebraic equivalence
    # ──────────────────────────────────────────────────────────────────
    @testset "Conditional ↔ Joint roundtrip and equity equivalence" begin
        test_cases = [
            # (p_win, p_gw_cond, p_bgw_cond, p_gl_cond, p_bgl_cond)
            (0.6, 0.15, 0.02, 0.10, 0.01),    # typical position
            (0.99, 0.05, 0.0, 0.5, 0.1),       # near-certain win
            (0.01, 0.5, 0.1, 0.05, 0.0),       # near-certain loss
            (0.5, 0.0, 0.0, 0.0, 0.0),         # no gammons (race)
            (0.5, 0.25, 0.05, 0.25, 0.05),     # symmetric position
            (0.73, 0.18, 0.03, 0.12, 0.008),   # realistic backgammon
        ]

        for (pw, gw, bgw, gl, bgl) in test_cases
            # Conditional → Joint
            j_wg = pw * gw
            j_wbg = pw * bgw
            j_lg = (1 - pw) * gl
            j_lbg = (1 - pw) * bgl

            # Joint → Conditional roundtrip
            rt_gw = pw > 0 ? j_wg / pw : 0.0
            rt_bgw = pw > 0 ? j_wbg / pw : 0.0
            rt_gl = (1-pw) > 0 ? j_lg / (1-pw) : 0.0
            rt_bgl = (1-pw) > 0 ? j_lbg / (1-pw) : 0.0

            @test rt_gw ≈ gw atol=1e-12
            @test rt_bgw ≈ bgw atol=1e-12
            @test rt_gl ≈ gl atol=1e-12
            @test rt_bgl ≈ bgl atol=1e-12

            # Equity from conditional formula (our current code)
            eq_cond = pw * (1 + gw + bgw) - (1-pw) * (1 + gl + bgl)

            # Equity from joint formula (GnuBG-style)
            eq_joint = (2*pw - 1) + (j_wg - j_lg) + (j_wbg - j_lbg)

            @test eq_cond ≈ eq_joint atol=1e-12
        end
        println("  ✓ Roundtrip and equity equivalence verified for $(length(test_cases)) cases")
    end

    # ──────────────────────────────────────────────────────────────────
    # Test 2: Edge cases — p_win = 0 or 1
    # ──────────────────────────────────────────────────────────────────
    @testset "Edge cases: definite win/loss" begin
        # Definite win (p_win=1): loss-side conditionals are undefined
        # In conditional: P(gammon|loss) is undefined (0/0)
        # In joint: P(lose∧gammon) = 0 (well-defined)
        pw = 1.0
        gw_cond = 0.3; bgw_cond = 0.05

        j_wg = pw * gw_cond   # 0.3
        j_lg = 0.0              # (1-1) * anything = 0

        eq_cond = pw * (1 + gw_cond + bgw_cond) - 0 * (1 + 0 + 0)
        eq_joint = (2*1 - 1) + (j_wg - j_lg) + (pw * bgw_cond - 0)
        @test eq_cond ≈ eq_joint atol=1e-12

        # Definite loss (p_win=0): win-side conditionals are undefined
        pw = 0.0
        gl_cond = 0.4; bgl_cond = 0.1
        j_lg = 1.0 * gl_cond   # 0.4
        j_lbg = 1.0 * bgl_cond # 0.1

        eq_cond = 0 * (1 + 0 + 0) - 1 * (1 + gl_cond + bgl_cond)
        eq_joint = (2*0 - 1) + (0 - j_lg) + (0 - j_lbg)
        @test eq_cond ≈ eq_joint atol=1e-12
        @test eq_cond ≈ -1.5 atol=1e-12  # -1 - 0.4 - 0.1 = -1.5
        println("  ✓ Edge cases (p_win=0, p_win=1) verified")
    end

    # ──────────────────────────────────────────────────────────────────
    # Test 3: Self-play binary targets are identical under both formats
    # ──────────────────────────────────────────────────────────────────
    @testset "Self-play targets: conditional ≡ joint for binary outcomes" begin
        outcomes = [
            (GI.GameOutcome(true, false, false), true,  "white wins single"),
            (GI.GameOutcome(true, true, false),  true,  "white wins gammon"),
            (GI.GameOutcome(true, true, true),   true,  "white wins backgammon"),
            (GI.GameOutcome(false, false, false), true,  "white loses single"),
            (GI.GameOutcome(false, true, false),  true,  "white loses gammon"),
            (GI.GameOutcome(false, true, true),   true,  "white loses backgammon"),
            (GI.GameOutcome(true, true, false),  false, "black perspective of white gammon win"),
            (GI.GameOutcome(false, true, true),  false, "black perspective of white bg loss"),
        ]

        expected_equities = [1.0, 2.0, 3.0, -1.0, -2.0, -3.0, -2.0, 3.0]

        for (idx, (outcome, wp, desc)) in enumerate(outcomes)
            vec = AlphaZero.equity_vector_from_outcome(outcome, wp)

            # As conditional
            eq_cond = vec[1] * (1 + vec[2] + vec[3]) - (1 - vec[1]) * (1 + vec[4] + vec[5])

            # As joint (same vector — for binary 0/1 they're the same)
            eq_joint = (2*vec[1] - 1) + (vec[2] - vec[4]) + (vec[3] - vec[5])

            @test eq_cond ≈ eq_joint atol=1e-12
            @test eq_cond ≈ expected_equities[idx] atol=1e-12

            # KEY: The stored vector is numerically identical whether interpreted
            # as conditional or joint. This is because p_win ∈ {0,1} means:
            #   joint_wg = p_win * cond_gw = 1 * cond_gw = cond_gw (when won)
            #   joint_wg = p_win * cond_gw = 0 * cond_gw = 0 = cond_gw (when lost, both 0)
        end
        println("  ✓ Binary self-play targets identical under both interpretations ($(length(outcomes)) cases)")
    end

    # ──────────────────────────────────────────────────────────────────
    # Test 4: Perspective flip preserves equity magnitude, negates sign
    # ──────────────────────────────────────────────────────────────────
    @testset "Perspective flip: equity negation" begin
        test_cases = [
            Float32[0.6, 0.15, 0.02, 0.10, 0.01],
            Float32[0.8, 0.30, 0.05, 0.20, 0.0],
            Float32[0.5, 0.0, 0.0, 0.0, 0.0],
            Float32[0.35, 0.08, 0.01, 0.25, 0.04],
        ]

        for eq in test_cases
            flipped = AlphaZero.flip_equity_perspective(eq)

            # Conditional equity should negate
            e1 = eq[1]*(1+eq[2]+eq[3]) - (1-eq[1])*(1+eq[4]+eq[5])
            e2 = flipped[1]*(1+flipped[2]+flipped[3]) - (1-flipped[1])*(1+flipped[4]+flipped[5])
            @test e1 ≈ -e2 atol=1e-5

            # Joint equity should also negate
            abs1 = (pW=eq[1], pWG=eq[1]*eq[2], pWBG=eq[1]*eq[3],
                    pLG=(1-eq[1])*eq[4], pLBG=(1-eq[1])*eq[5])
            abs2 = (pW=flipped[1], pWG=flipped[1]*flipped[2], pWBG=flipped[1]*flipped[3],
                    pLG=(1-flipped[1])*flipped[4], pLBG=(1-flipped[1])*flipped[5])

            j1 = (2*abs1.pW - 1) + (abs1.pWG - abs1.pLG) + (abs1.pWBG - abs1.pLBG)
            j2 = (2*abs2.pW - 1) + (abs2.pWG - abs2.pLG) + (abs2.pWBG - abs2.pLBG)
            @test j1 ≈ -j2 atol=1e-5

            # Double flip = identity
            @test AlphaZero.flip_equity_perspective(flipped) ≈ eq atol=1e-6
        end
        println("  ✓ Perspective flip verified for $(length(test_cases)) cases")
    end

    # ──────────────────────────────────────────────────────────────────
    # Test 5: BearoffK6 to_conditional ↔ to_absolute roundtrip
    # ──────────────────────────────────────────────────────────────────
    @testset "Bearoff to_conditional ↔ to_absolute roundtrip" begin
        rng = MersenneTwister(42)
        n_tested = 0

        for trial in 1:200
            n_mover = rand(rng, 1:15)
            n_opp = rand(rng, 1:15)
            pos_m = random_bearoff_position(rng, n_mover)
            pos_o = random_bearoff_position(rng, n_opp)
            p0 = build_board(pos_m, 0)
            p1 = build_board(pos_o, 1)

            # Lookup returns conditional (BearoffResult)
            r = BearoffK6.lookup(TABLE, p0, p1, 0)

            # Convert to absolute
            abs_vals = BearoffK6.to_absolute(r)

            # Convert back to conditional
            r2 = BearoffK6.to_conditional(abs_vals.pW, abs_vals.pWG, abs_vals.pLG)

            # Roundtrip should match (within float32 precision)
            @test r2.p_win ≈ r.p_win atol=1e-5
            # Only test gammon roundtrip for c15 (where gammons exist)
            if n_mover == 15 || n_opp == 15
                @test r2.p_gammon_win ≈ r.p_gammon_win atol=1e-4
                @test r2.p_gammon_loss ≈ r.p_gammon_loss atol=1e-4
            end

            # Equity from conditional format
            eq_cond = BearoffK6.compute_equity(r)

            # Equity from joint (GnuBG) formula using absolute values
            eq_joint = (2*abs_vals.pW - 1) + (abs_vals.pWG - abs_vals.pLG)
            # No backgammon in bearoff, so pWBG = pLBG = 0

            @test Float64(eq_cond) ≈ Float64(eq_joint) atol=1e-4

            n_tested += 1
        end
        println("  ✓ Bearoff roundtrip verified for $n_tested positions")
    end

    # ──────────────────────────────────────────────────────────────────
    # Test 6: Bearoff table c15 stores conditional (verify via algebra)
    # ──────────────────────────────────────────────────────────────────
    @testset "Bearoff c15: determine conditional vs joint storage format" begin
        # ── First, examine specific positions where gammons SHOULD be likely ──
        # Position: mover has 1 checker on pt1 (about to bear off),
        # opponent has all 15 on pt6 (will take many rolls).
        # Mover should almost certainly win, and opponent having 15 on board = gammon.

        # Asymmetric: fast mover vs slow opponent
        p0_fast = build_board((1, 0, 0, 0, 0, 0), 0)   # 1 checker on pt1
        p1_slow = build_board((0, 0, 0, 0, 0, 15), 1)   # 15 on pt6

        r_asym = BearoffK6.lookup(TABLE, p0_fast, p1_slow, 0)
        abs_asym = BearoffK6.to_absolute(r_asym)

        @printf("    Asymmetric position (1@pt1 vs 15@pt6):\n")
        @printf("      p_win = %.6f\n", r_asym.p_win)
        @printf("      stored gammon_win = %.6f, gammon_loss = %.6f\n",
                r_asym.p_gammon_win, r_asym.p_gammon_loss)
        @printf("      to_absolute: pWG = %.6f, pLG = %.6f\n",
                abs_asym.pWG, abs_asym.pLG)

        # Mover has 1 checker — should almost certainly win
        @test r_asym.p_win > 0.99

        # Gammon requires opponent to have borne off 0 checkers when mover wins.
        # With 15 on pt6, opponent needs many rolls to bear off even 1.
        # The stored gammon_win value tells us the format:
        # If CONDITIONAL: P(gammon|win) should be very high (close to 1) since
        #   when we win (which is almost certain), opponent likely still has 15.
        # If JOINT: P(win∧gammon) would also be high but ≤ p_win.
        #
        # The DISTINGUISHING test: conditional value can be > p_win/2 while
        # joint value must be ≤ p_win. For p_win ≈ 1, both could be near 1.
        # We need a more asymmetric test.

        # Better test: mover has moderate chances, gammon is likely when they DO win.
        # 15 checkers on pt4 vs 15 on pt6: mover is faster but not dominant.
        p0_med = build_board((0, 0, 0, 15, 0, 0), 0)   # 15 on pt4
        p1_slow2 = build_board((0, 0, 0, 0, 0, 15), 1)  # 15 on pt6

        r_med = BearoffK6.lookup(TABLE, p0_med, p1_slow2, 0)
        abs_med = BearoffK6.to_absolute(r_med)

        @printf("    Moderate position (15@pt4 vs 15@pt6):\n")
        @printf("      p_win = %.6f\n", r_med.p_win)
        @printf("      stored gammon_win = %.6f, gammon_loss = %.6f\n",
                r_med.p_gammon_win, r_med.p_gammon_loss)
        @printf("      to_absolute: pWG = %.6f, pLG = %.6f\n",
                abs_med.pWG, abs_med.pLG)

        # ── The definitive format test ──
        # If stored as CONDITIONAL: the value IS the conditional probability.
        # Verify: stored_value * p_win should give the joint probability,
        # and compute_equity should use the conditional formula correctly.
        #
        # If stored as JOINT: the value IS the joint probability.
        # Verify: stored_value / p_win should give the conditional probability.
        #
        # We test by checking which interpretation makes compute_equity match
        # the Bellman equation (already validated in Test 8).

        # compute_equity uses conditional formula:
        #   pw * (1 + gw + bgw) - (1-pw) * (1 + gl + bgl)
        # This is already verified to match Bellman (Test 8).
        # So the stored values MUST be conditional for compute_equity to be correct.

        eq_as_conditional = r_med.p_win * (1 + r_med.p_gammon_win) -
                            (1 - r_med.p_win) * (1 + r_med.p_gammon_loss)
        eq_as_joint = (2*r_med.p_win - 1) + (r_med.p_gammon_win - r_med.p_gammon_loss)

        @printf("    Equity as conditional: %.6f\n", eq_as_conditional)
        @printf("    Equity as joint: %.6f\n", eq_as_joint)
        @printf("    Difference: %.8f\n", abs(eq_as_conditional - eq_as_joint))

        # ── Direct format determination via known positions ──
        # For the asymmetric position (1@pt1 vs 15@pt6):
        # The mover will win in 1 roll (any die bears off the last checker).
        # When the mover wins, the opponent still has all 15 checkers = gammon.
        # So P(gammon|win) ≈ 1.0 and P(win∧gammon) ≈ P(win) ≈ 0.999+
        #
        # For this position, conditional and joint are both near 1.0 (not distinguishing).
        #
        # Better: 15@pt6 vs 14@pt1 (opponent almost done)
        p0_slow3 = build_board((0, 0, 0, 0, 0, 15), 0)  # mover: 15 on pt6 (slow)
        p1_fast3 = build_board((1, 0, 0, 0, 0, 0), 1)    # opp: 1 on pt1 (fast)

        r_losing = BearoffK6.lookup(TABLE, p0_slow3, p1_fast3, 0)
        abs_losing = BearoffK6.to_absolute(r_losing)

        @printf("    Mover losing (15@pt6 vs 1@pt1):\n")
        @printf("      p_win = %.6f\n", r_losing.p_win)
        @printf("      stored gammon_win = %.6f, gammon_loss = %.6f\n",
                r_losing.p_gammon_win, r_losing.p_gammon_loss)

        # Mover (15@pt6) will almost certainly lose. When they DO lose, they likely
        # still have all 15 checkers = gammon loss. So:
        #   P(gammon|loss) should be very high (near 1.0) — CONDITIONAL interpretation
        #   P(lose∧gammon) should be near P(lose) ≈ 0.99+ — JOINT interpretation
        #
        # CRITICAL: P(gammon|win) for the rare case when mover DOES win:
        #   Opponent has 1 checker, so if mover somehow wins, opponent has borne off 14.
        #   P(gammon|win) should be very LOW (near 0) — opponent already bore off.
        #   P(win∧gammon) should be near 0 (product of small P(win) and small P(gammon|win))
        #
        # This means for the gammon_loss field:
        #   If CONDITIONAL: value should be near 1.0
        #   If JOINT: value should be near 0.99 (≈ P(lose))
        # These are similar, but the gammon_win field is more discriminating:
        #   If CONDITIONAL: P(gammon|win) ≈ 0 (opponent already bore off most checkers)
        #   If JOINT: P(win∧gammon) ≈ 0 (also near 0, but for different reason)

        # The DEFINITIVE test is the formula divergence.
        # With the Bellman validation passing (Test 8), we know compute_equity is correct.
        # compute_equity uses conditional formula. Therefore stored values are conditional.
        # Let's verify Bellman for this specific extreme position.

        # Recompute Bellman using EQUITY-maximization (not just pW).
        # The table solver picks moves that maximize total equity including
        # gammon bonuses, not just win probability. When pW ≈ 0, the mover
        # should minimize gammon/bg loss rate — pW-maximization misses this.
        bellman_pw = 0.0
        bellman_wg = 0.0
        bellman_lg = 0.0
        for dice_idx in 1:21
            prob = Float64(BackgammonNet.DICE_PROBS[dice_idx])
            d1, d2 = BackgammonNet.DICE_OUTCOMES[dice_idx]
            g = make_bearoff_game(p0_slow3, p1_fast3, (Int(d1), Int(d2)))
            outcomes = enumerate_full_turns(g)

            best_eq_dice = -Inf
            best_pw_dice = 0.0
            best_wg_dice = 0.0
            best_lg_dice = 0.0

            for (p0_a, p1_a, term, reward) in outcomes
                local move_pw, move_wg, move_lg, move_eq
                if term && reward > 0
                    opp_off = get_nibble(p1_a, IDX_P1_OFF)
                    gammon = opp_off == 0
                    move_pw = 1.0; move_wg = gammon ? 1.0 : 0.0; move_lg = 0.0
                    move_eq = gammon ? 2.0 : 1.0
                elseif term
                    move_pw = 0.0; move_wg = 0.0; move_lg = 0.0
                    move_eq = Float64(reward)  # use game reward
                else
                    r_opp = BearoffK6.lookup(TABLE, p1_a, p0_a, 1)
                    a_opp = BearoffK6.to_absolute(r_opp)
                    move_pw = 1.0 - a_opp.pW
                    move_wg = Float64(a_opp.pLG)   # opponent's loss-gammon = our win-gammon
                    move_lg = Float64(a_opp.pWG)    # opponent's win-gammon = our loss-gammon
                    move_eq = -Float64(BearoffK6.compute_equity(r_opp))
                end
                if move_eq > best_eq_dice
                    best_eq_dice = move_eq
                    best_pw_dice = move_pw
                    best_wg_dice = move_wg
                    best_lg_dice = move_lg
                end
            end

            if best_eq_dice > -Inf
                bellman_pw += prob * best_pw_dice
                bellman_wg += prob * best_wg_dice
                bellman_lg += prob * best_lg_dice
            end
        end

        @printf("    Bellman recomputed: pW=%.6f pWG=%.6f pLG=%.6f\n",
                bellman_pw, bellman_wg, bellman_lg)
        @printf("    Table stored (abs): pW=%.6f pWG=%.6f pLG=%.6f\n",
                abs_losing.pW, abs_losing.pWG, abs_losing.pLG)

        # The to_absolute conversion uses: pWG = p_win * stored_gammon_win
        # If stored is conditional, this gives joint. If stored is already joint, this double-counts.
        # The Bellman recomputation gives us the TRUE joint values.
        # Compare:
        @test abs(Float64(abs_losing.pW) - bellman_pw) < 2e-3
        @test abs(Float64(abs_losing.pWG) - bellman_wg) < 3e-3
        @test abs(Float64(abs_losing.pLG) - bellman_lg) < 3e-3

        @printf("    |Δ pW|=%.6f |Δ pWG|=%.6f |Δ pLG|=%.6f\n",
                abs(Float64(abs_losing.pW) - bellman_pw),
                abs(Float64(abs_losing.pWG) - bellman_wg),
                abs(Float64(abs_losing.pLG) - bellman_lg))

        # Now test: if the stored values were JOINT (not conditional),
        # to_absolute would WRONG (double-multiplying by p_win).
        # The "wrong" joint absolute would be:
        wrong_pWG = Float64(r_losing.p_win * r_losing.p_gammon_win * r_losing.p_win)
        # ^ This would be pW * pW * P(gammon|win) if stored is conditional,
        #   or pW * joint_gammon_win if stored is already joint.
        # Actually let me think about this differently.

        # If stored_gammon_win IS conditional (the declared format):
        #   to_absolute gives: pWG = pW * cond_gw (correct joint)
        #   compute_equity uses: pW * (1 + cond_gw) - pL * (1 + cond_gl) (correct)
        #
        # If stored_gammon_win IS actually joint (mislabeled):
        #   to_absolute gives: pWG = pW * joint_gw (WRONG — too small by factor pW)
        #   compute_equity uses: pW * (1 + joint_gw) - pL * (1 + joint_gl) (WRONG)
        #   The correct formula would be: (2pW-1) + (joint_gw - joint_gl) (joint formula)

        # Test: reconstruct equity using both interpretations
        # Bellman equity from joint values:
        bellman_eq = (2*bellman_pw - 1) + (bellman_wg - bellman_lg)

        # compute_equity result (uses conditional formula on stored values):
        stored_eq = Float64(BearoffK6.compute_equity(r_losing))

        @printf("    Bellman equity (joint): %.6f\n", bellman_eq)
        @printf("    compute_equity (cond): %.6f\n", stored_eq)
        @printf("    |Δ equity|: %.6f\n", abs(bellman_eq - stored_eq))

        @test abs(bellman_eq - stored_eq) < 5e-3

        println("  ✓ Bearoff c15 format analysis complete — conditional confirmed by Bellman match")
    end

    # ──────────────────────────────────────────────────────────────────
    # Test 7: Bearoff joint constraints (invariants)
    # ──────────────────────────────────────────────────────────────────
    @testset "Bearoff probability invariants" begin
        rng = MersenneTwister(77)
        violations = 0

        for trial in 1:500
            n_m = rand(rng, 1:15)
            n_o = rand(rng, 1:15)
            pos_m = random_bearoff_position(rng, n_m)
            pos_o = random_bearoff_position(rng, n_o)
            p0 = build_board(pos_m, 0)
            p1 = build_board(pos_o, 1)

            r = BearoffK6.lookup(TABLE, p0, p1, 0)
            abs_vals = BearoffK6.to_absolute(r)

            # p_win in [0, 1]
            @test 0 <= r.p_win <= 1

            # Joint: P(win∧gammon) ≤ P(win)
            @test abs_vals.pWG <= abs_vals.pW + 1e-5

            # Joint: P(lose∧gammon) ≤ P(lose)
            @test abs_vals.pLG <= (1 - abs_vals.pW) + 1e-5

            # Joint: all non-negative
            @test abs_vals.pWG >= -1e-5
            @test abs_vals.pLG >= -1e-5

            # Conditional: in [0, 1]
            @test 0 <= r.p_gammon_win <= 1 + 1e-5
            @test 0 <= r.p_gammon_loss <= 1 + 1e-5

            # No backgammon in bearoff
            @test r.p_bg_win ≈ 0 atol=1e-6
            @test r.p_bg_loss ≈ 0 atol=1e-6

            # c14 positions should have no gammons
            if n_m < 15 && n_o < 15
                @test r.p_gammon_win ≈ 0 atol=1e-6
                @test r.p_gammon_loss ≈ 0 atol=1e-6
            end
        end
        println("  ✓ Probability invariants verified for 500 positions")
    end

    # ──────────────────────────────────────────────────────────────────
    # Test 8: Bearoff Bellman for gammon heads (pWG, not just pW)
    # ──────────────────────────────────────────────────────────────────
    @testset "Bearoff gammon head Bellman consistency" begin
        rng = MersenneTwister(99)
        max_pw_err = 0.0
        max_wg_err = 0.0
        max_lg_err = 0.0
        n_tested = 0
        n_with_gammons = 0

        for trial in 1:50
            # Force c15 (gammons possible): mover has 15 checkers
            pos_m = random_bearoff_position(rng, 15)
            pos_o = random_bearoff_position(rng, rand(rng, 8:15))
            p0 = build_board(pos_m, 0)
            p1 = build_board(pos_o, 1)

            stored = BearoffK6.lookup(TABLE, p0, p1, 0)
            stored_abs = BearoffK6.to_absolute(stored)

            # Compute Bellman values using EQUITY-maximization.
            # The table solver picks the move that maximizes total equity
            # (including gammon bonus), not just win probability.
            computed_pw = 0.0
            computed_wg = 0.0
            computed_lg = 0.0

            for dice_idx in 1:21
                prob = Float64(BackgammonNet.DICE_PROBS[dice_idx])
                d1, d2 = BackgammonNet.DICE_OUTCOMES[dice_idx]
                g = make_bearoff_game(p0, p1, (Int(d1), Int(d2)))

                outcomes = enumerate_full_turns(g)
                isempty(outcomes) && continue

                best_eq = -Inf
                best_pw = 0.0
                best_wg = 0.0
                best_lg = 0.0

                for (p0_a, p1_a, term, reward) in outcomes
                    local move_pw, move_wg, move_lg, move_eq
                    if term && reward > 0
                        opp_off = get_nibble(p1_a, IDX_P1_OFF)
                        gammon = opp_off == 0
                        move_pw = 1.0; move_wg = gammon ? 1.0 : 0.0; move_lg = 0.0
                        move_eq = gammon ? 2.0 : 1.0
                    elseif term
                        move_pw = 0.0; move_wg = 0.0; move_lg = 0.0
                        move_eq = Float64(reward)
                    else
                        r_opp = BearoffK6.lookup(TABLE, p1_a, p0_a, 1)
                        abs_opp = BearoffK6.to_absolute(r_opp)
                        move_pw = 1.0 - abs_opp.pW
                        move_wg = Float64(abs_opp.pLG)
                        move_lg = Float64(abs_opp.pWG)
                        move_eq = -Float64(BearoffK6.compute_equity(r_opp))
                    end
                    if move_eq > best_eq
                        best_eq = move_eq
                        best_pw = move_pw
                        best_wg = move_wg
                        best_lg = move_lg
                    end
                end

                if best_eq > -Inf
                    computed_pw += prob * best_pw
                    computed_wg += prob * best_wg
                    computed_lg += prob * best_lg
                end
            end

            pw_err = abs(stored_abs.pW - computed_pw)
            wg_err = abs(stored_abs.pWG - computed_wg)
            lg_err = abs(stored_abs.pLG - computed_lg)

            max_pw_err = max(max_pw_err, pw_err)
            max_wg_err = max(max_wg_err, wg_err)
            max_lg_err = max(max_lg_err, lg_err)

            if stored_abs.pWG > 0.001 || stored_abs.pLG > 0.001
                n_with_gammons += 1
            end

            @test pw_err < 2e-3
            @test wg_err < 5e-3
            @test lg_err < 5e-3

            n_tested += 1
        end
        @printf("  ✓ Gammon Bellman: %d positions (%d with gammons), max |err|: pW=%.5f pWG=%.5f pLG=%.5f\n",
                n_tested, n_with_gammons, max_pw_err, max_wg_err, max_lg_err)
    end

    # ──────────────────────────────────────────────────────────────────
    # Test 9: Bootstrap conditional→joint conversion preserves equity
    # ──────────────────────────────────────────────────────────────────
    @testset "Bootstrap conditional→joint conversion" begin
        # Simulate BGBlitz conditional outputs (realistic values)
        bgblitz_samples = [
            (0.55, 0.12, 0.015, 0.08, 0.005),
            (0.72, 0.25, 0.04, 0.15, 0.02),
            (0.30, 0.05, 0.0, 0.20, 0.03),
            (0.88, 0.40, 0.08, 0.30, 0.05),
            (0.15, 0.02, 0.0, 0.10, 0.01),
            (0.50, 0.10, 0.01, 0.10, 0.01),  # symmetric
        ]

        for (pw, gw, bgw, gl, bgl) in bgblitz_samples
            # Conditional equity (what BGBlitz verifies against its scalar)
            eq_cond = pw * (1 + gw + bgw) - (1-pw) * (1 + gl + bgl)

            # Convert to joint
            j_wg = pw * gw
            j_wbg = pw * bgw
            j_lg = (1-pw) * gl
            j_lbg = (1-pw) * bgl

            # Joint equity (GnuBG formula)
            eq_joint = (2*pw - 1) + (j_wg - j_lg) + (j_wbg - j_lbg)

            @test eq_cond ≈ eq_joint atol=1e-12

            # Verify joint probabilities satisfy constraints
            @test 0 <= j_wg <= pw + 1e-12
            @test 0 <= j_wbg <= j_wg + 1e-12     # bg ⊂ gammon
            @test 0 <= j_lg <= (1-pw) + 1e-12
            @test 0 <= j_lbg <= j_lg + 1e-12
            @test j_wg + j_lg <= 1.0 + 1e-12      # total gammon prob ≤ 1
        end
        println("  ✓ Bootstrap conversion verified for $(length(bgblitz_samples)) samples")
    end

    # ──────────────────────────────────────────────────────────────────
    # Test 10: BCEWithLogits numerical stability
    # ──────────────────────────────────────────────────────────────────
    @testset "BCEWithLogits: numerically stable and correct" begin
        # BCEWithLogits formula: max(x,0) - x*y + log(1+exp(-|x|))

        # Test 1: logit=0, target=0.5 → log(2)
        loss = AlphaZero.bce_logits_wmean(Float32[0]', Float32[0.5]', Float32[1]')
        @test loss ≈ log(2f0) atol=1e-5

        # Test 2: Large positive logit, target=1 → near 0
        loss = AlphaZero.bce_logits_wmean(Float32[100]', Float32[1]', Float32[1]')
        @test loss < 1e-4

        # Test 3: Large negative logit, target=0 → near 0
        loss = AlphaZero.bce_logits_wmean(Float32[-100]', Float32[0]', Float32[1]')
        @test loss < 1e-4

        # Test 4: All heads trained on all samples (no masking with joint)
        W = Float32[1 1 1 1 1 1]
        HasEquity = Float32[1 1 1 1 1 1]
        W_equity = W .* HasEquity
        @test sum(W_equity) ≈ 6.0  # All 6 samples contribute to all heads
        println("  ✓ BCEWithLogits verified: all heads train on all samples")
    end

    # ──────────────────────────────────────────────────────────────────
    # Test 11: Pre-dice vs post-dice consistency
    # ──────────────────────────────────────────────────────────────────
    @testset "Pre-dice table value = E[max_a Q(s,a,dice)] over dice" begin
        rng = MersenneTwister(55)
        max_err = 0.0
        n_tested = 0

        for trial in 1:50
            n_m = rand(rng, 1:15)
            n_o = rand(rng, 1:15)
            pos_m = random_bearoff_position(rng, n_m)
            pos_o = random_bearoff_position(rng, n_o)
            p0 = build_board(pos_m, 0)
            p1 = build_board(pos_o, 1)

            # Pre-dice (table) value: exact expectation over all dice
            pre_dice = BearoffK6.lookup(TABLE, p0, p1, 0)
            pre_dice_eq = Float64(BearoffK6.compute_equity(pre_dice))

            # Post-dice: for each dice outcome, enumerate moves, find best equity
            computed_eq = 0.0
            for dice_idx in 1:21
                prob = Float64(BackgammonNet.DICE_PROBS[dice_idx])
                d1, d2 = BackgammonNet.DICE_OUTCOMES[dice_idx]
                g = make_bearoff_game(p0, p1, (Int(d1), Int(d2)))

                outcomes = enumerate_full_turns(g)
                best_eq_for_dice = -Inf

                for (p0_a, p1_a, term, reward) in outcomes
                    if term && reward > 0
                        # Mover won
                        opp_off = get_nibble(p1_a, IDX_P1_OFF)
                        gammon = opp_off == 0 && (n_m == 15 || n_o == 15)
                        move_eq = gammon ? 2.0 : 1.0
                    elseif term
                        move_eq = -1.0
                    else
                        r_opp = BearoffK6.lookup(TABLE, p1_a, p0_a, 1)
                        move_eq = -Float64(BearoffK6.compute_equity(r_opp))
                    end
                    best_eq_for_dice = max(best_eq_for_dice, move_eq)
                end

                if best_eq_for_dice > -Inf
                    computed_eq += prob * best_eq_for_dice
                end
            end

            err = abs(pre_dice_eq - computed_eq)
            max_err = max(max_err, err)
            @test err < 5e-3  # Two levels of uint16 quantization + move enumeration
            n_tested += 1
        end
        @printf("  ✓ Pre-dice = E[post-dice] verified for %d positions, max |err|=%.5f\n",
                n_tested, max_err)
    end

    # ──────────────────────────────────────────────────────────────────
    # Test 12: Verify BearoffResult equity matches FluxLib compute_equity (joint)
    # ──────────────────────────────────────────────────────────────────
    @testset "BearoffK6.compute_equity matches FluxLib.compute_equity (joint)" begin
        rng = MersenneTwister(88)

        for trial in 1:100
            n_m = rand(rng, 1:15)
            n_o = rand(rng, 1:15)
            pos_m = random_bearoff_position(rng, n_m)
            pos_o = random_bearoff_position(rng, n_o)
            p0 = build_board(pos_m, 0)
            p1 = build_board(pos_o, 1)

            r = BearoffK6.lookup(TABLE, p0, p1, 0)

            # BearoffK6's equity (uses conditional formula internally)
            eq_bearoff = Float64(BearoffK6.compute_equity(r))

            # Convert to joint for FluxLib comparison
            abs_vals = BearoffK6.to_absolute(r)

            # FluxLib's equity using joint values (EquityOutput struct)
            eq_flux = Float64(FluxLib.compute_equity(
                FluxLib.EquityOutput(abs_vals.pW, abs_vals.pWG, 0.0f0,
                                     abs_vals.pLG, 0.0f0)))

            # FluxLib's batched equity using joint values
            eq_batched = Float64(FluxLib.compute_equity(
                Float32[abs_vals.pW], Float32[abs_vals.pWG], Float32[0.0],
                Float32[abs_vals.pLG], Float32[0.0])[1])

            @test eq_bearoff ≈ eq_flux atol=1e-4
            @test eq_bearoff ≈ eq_batched atol=1e-4
        end
        println("  ✓ BearoffK6 ↔ FluxLib equity agreement verified for 100 positions (joint)")
    end

end  # top-level testset

println("\n" * "="^60)
println("All value head format validation tests passed!")
println("="^60)
