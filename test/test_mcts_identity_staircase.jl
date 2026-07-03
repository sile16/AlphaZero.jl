# ============================================================================
# MCTS DETERMINISTIC IDENTITY STAIRCASE — Rungs 1-3
# ============================================================================
#
# PHILOSOPHY (why this file exists, separate from the play-strength eval scripts):
#
# There are two fundamentally different things a test can measure about MCTS:
#
#   (1) CORRECTNESS errors — sign flips, perspective (mover vs white) confusion,
#       reward-scale (÷3) mistakes, backprop arithmetic, virtual-loss leakage,
#       chance-node probability weighting. These MUST be IDENTICALLY zero. They
#       are checkable on the tree's INTERNAL statistics (Q = W/N per action) with
#       machine-epsilon tolerance, deterministically, on tiny hand-built trees.
#
#   (2) finite-budget SUBOPTIMALITY — at low simulation counts MCTS allocates
#       visits imperfectly; the argmax can miss the true-best move by a hair.
#       This is EXPECTED, is a smooth function of the sim budget, and is measured
#       elsewhere as convergence curves (notes/mcts_convergence_sweep_20260703.md).
#
# Behavioral / play-strength tests (eval_table_vs_wildbg.jl etc.) conflate the
# two: they catch bugs only STATISTICALLY and cannot distinguish "the arithmetic
# is perfect" from "two sign errors happened to cancel over 500 games". Identity
# tests on tree internals CAN: a single mis-signed backup makes an exact equality
# fail by O(1), far outside any rounding tolerance.
#
# This file proves the MCTS engines introduce ZERO correctness error on positions
# where the ground truth is known exactly:
#
#   Rung 1 — evaluator/wrapper identities (sign, perspective, normalization, and
#            the chance-node = probability-weighted-average-of-children identity).
#   Rung 2 — DEPTH-1 search identity: every visit of a root action terminates at
#            the SAME exactly-evaluated leaf, so Q(a) = W/N must equal that exact
#            constant to machine epsilon. Any deviation is a backprop / virtual-
#            loss / sign bug. (Verified batch-size-invariant → virtual loss is
#            fully unwound.)
#   Rung 3 — MULTI-LEVEL backprop identity: on tiny full game trees that bottom
#            out at TRUE terminals, classic MCTS with exact chance averaging
#            (:full) converges to the exact expectimax value. Exercises reward
#            recording (÷3), pswitch sign flips, chance-node probability weighting
#            and multi-level accumulation — including the ±2 gammon reward path.
#
# Ground truth = the exact k=7 two-sided bear-off table (Rungs 1-2) and a direct
# pure-Julia expectimax recursion over the same game (Rung 3, cross-checked
# against the table where positions are in range).
#
# The tests SKIP gracefully (with @warn) if the k=7 table is absent.
# ============================================================================

using Test
using Random
using StaticArrays

using AlphaZero
using AlphaZero: GI, MCTS, BatchedMCTS
import BackgammonNet
using BackgammonNet: BackgammonGame

# ── Optional dependency discovery (table-optional guard) ────────────────────

const _K7_SRC = joinpath(homedir(), "github", "BackgammonNet.jl", "src", "bearoff_k7.jl")

function _find_k7_table_dir()
    for d in [
        joinpath(dirname(_K7_SRC), "..", "tools", "bearoff_twosided", "bearoff_k7_twosided"),
        joinpath(homedir(), "bearoff_k7_twosided"),
        "/homeshare/projects/AlphaZero.jl/eval_data/bearoff_k7_twosided",
    ]
        isdir(d) && isfile(joinpath(d, "bearoff_k7_c14.bin")) && return d
    end
    return nothing
end

const _K7_TABLE_DIR = isfile(_K7_SRC) ? _find_k7_table_dir() : nothing

if _K7_TABLE_DIR === nothing
    @warn "k=7 bear-off table (or bearoff_k7.jl) not found — SKIPPING MCTS identity staircase (Rungs 1-3). This is a hard SKIP, not a pass."
    @testset "MCTS Identity Staircase (SKIPPED — no k=7 table)" begin
        @test_skip true
    end
else
    # ── Load table + turn-aware helpers + game wrapper (mirror existing tests) ─
    if !isdefined(Main, :BearoffK7)
        include(_K7_SRC)
        using .BearoffK7
    end
    if !isdefined(Main, :bearoff_turn_value)
        include(joinpath(@__DIR__, "..", "scripts", "bearoff_eval_common.jl"))
    end
    if !isdefined(Main, :BackgammonDeterministic)
        include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "main.jl"))
    end
    const BGD = Main.BackgammonDeterministic
    const GSPEC_ID = BGD.GameSpec()

    const TABLE_ID = BearoffK7.BearoffTable(_K7_TABLE_DIR)

    # ── Bit-board helpers ─────────────────────────────────────────────────────
    # Layout (4-bit nibbles at idx<<2): IDX_P1_OFF=0, points 1-24, IDX_P0_OFF=25.
    # k=7 home board indices (from BackgammonNet.jl bearoff_k7.jl):
    #   P0 points 1..7 -> board indices (24,23,22,21,20,19,18)
    #   P1 points 1..7 -> board indices (1,2,3,4,5,6,7)
    _nib(b, i) = Int((b >> (i << 2)) & 0xF)
    const _P0_IDX = (24, 23, 22, 21, 20, 19, 18)
    const _P1_IDX = (1, 2, 3, 4, 5, 6, 7)

    """Build a pre-dice CHANCE-node BackgammonGame (dice unrolled) for player `cp`."""
    _chance_game(p0, p1, cp) =
        BackgammonGame(p0, p1, SVector{2, Int8}(0, 0), Int8(0), Int8(cp), false, 0.0f0;
                       obs_type = :minimal_flat)

    """Random mutual bear-off position: P0 checkers on points 1-7 (home), P1
    likewise, remainder borne off. Returns (p0, p1) UInt128 bit-boards."""
    function _rand_bearoff(rng; maxch = 6)
        while true
            c0 = rand(rng, 1:maxch); c1 = rand(rng, 1:maxch)
            cnt0 = zeros(Int, 7); cnt1 = zeros(Int, 7)
            for _ in 1:c0; cnt0[rand(rng, 1:7)] += 1; end
            for _ in 1:c1; cnt1[rand(rng, 1:7)] += 1; end
            (any(>(15), cnt0) || any(>(15), cnt1)) && continue   # nibble overflow
            p0 = UInt128(0); p1 = UInt128(0)
            for i in 1:7
                p0 |= UInt128(cnt0[i]) << (_P0_IDX[i] << 2)
                p1 |= UInt128(cnt1[i]) << (_P1_IDX[i] << 2)
            end
            p0 |= UInt128(15 - c0) << (25 << 2)   # P0 off (idx 25)
            p1 |= UInt128(15 - c1) << 0           # P1 off (idx 0)
            return p0, p1
        end
    end

    """Physical player-role swap with point-number reflection: the position where
    P0 has P1's old race and vice versa (same point numbers, off counts swapped).
    A position with P0-on-roll and its mirror with P1-on-roll are the SAME physical
    game seen from opposite seats, so their WHITE-relative values must negate.
    Verified below on hand-checked positions before mass use."""
    function _mirror(p0, p1)
        m0 = UInt128(0); m1 = UInt128(0)
        for i in 1:7
            m0 |= UInt128(_nib(p1, _P1_IDX[i])) << (_P0_IDX[i] << 2)  # new P0 pt i = old P1 pt i
            m1 |= UInt128(_nib(p0, _P0_IDX[i])) << (_P1_IDX[i] << 2)  # new P1 pt i = old P0 pt i
        end
        m0 |= UInt128(_nib(p1, 0)) << (25 << 2)   # new P0 off = old P1 off
        m1 |= UInt128(_nib(p0, 25)) << 0          # new P1 off = old P0 off
        return m0, m1
    end

    """Exact MCTS-facing bear-off evaluator (money weights), mirroring
    selfplay_client.jl / eval_table_vs_wildbg.jl: chance node -> pre-dice table
    lookup; decision node -> exact best-move value. Returns WHITE-relative equity
    NORMALIZED /3 (documented [-1,1] scale), or `nothing` if not a bear-off state.
    The BatchedMCTS engine then converts white-relative -> player-relative."""
    function _make_evaluator(table)
        return function (genv)
            bg = genv.game
            BearoffK7.is_bearoff_position(bg.p0, bg.p1) || return nothing
            if BackgammonNet.is_chance_node(bg)
                eq = Float64(BearoffK7.compute_equity(BearoffK7.lookup(table, bg))) / 3.0
                return bg.current_player == 0 ? eq : -eq
            end
            acts = BackgammonNet.legal_actions(bg)
            isempty(acts) && return nothing
            best = bearoff_best_move_value(table, bg) / 3.0
            return bg.current_player == 0 ? best : -best
        end
    end

    """Uniform-prior, V=0 oracle. Under the exact bear-off evaluator its V is never
    used (every node is a bear-off leaf); the uniform prior only seeds exploration."""
    function _uniform_oracle(state)
        n = max(1, length(GI.available_actions(GI.init(GSPEC_ID, state))))
        return (fill(Float32(1 / n), n), 0.0f0)
    end

    """Draw a random post-dice bear-off DECISION node (dice rolled, >=1 legal move)."""
    function _rand_decision(rng)
        while true
            p0, p1 = _rand_bearoff(rng)
            BearoffK7.is_bearoff_position(p0, p1) || continue
            cp = rand(rng, 0:1)
            g = _chance_game(p0, p1, cp)
            BackgammonNet.is_chance_node(g) || continue
            BackgammonNet.sample_chance!(g, rng)
            (g.terminated || BackgammonNet.is_chance_node(g)) && continue
            isempty(BackgammonNet.legal_actions(g)) && continue
            return g
        end
    end

    _hexdump(g) = "p0=0x$(string(g.p0, base=16)) p1=0x$(string(g.p1, base=16)) " *
                  "dice=$(Tuple(g.dice)) rem=$(g.remaining_actions) cp=$(g.current_player)"

    # ========================================================================
    # RUNG 1 — evaluator / wrapper identities
    # ========================================================================
    @testset "Rung 1: evaluator identities (sign, perspective, normalization, chance-avg)" begin
        rng = MersenneTwister(0xA1)
        ev = _make_evaluator(TABLE_ID)

        # --- 1.0 Hand-checked mirror: P0 far ahead <-> P1 far ahead ------------
        # P0 has 1 checker on point 1 (14 off) — nearly done; P1 has 5 on point 6
        # (10 off) — behind. P0-to-move white-relative value must be strongly +.
        hp0 = (UInt128(14) << (25 << 2)) | (UInt128(1) << (24 << 2))
        hp1 = (UInt128(10) << 0) | (UInt128(5) << (6 << 2))
        @test BearoffK7.is_bearoff_position(hp0, hp1)
        gh = _chance_game(hp0, hp1, 0)
        Eh = Float64(BearoffK7.compute_equity(BearoffK7.lookup(TABLE_ID, gh)))  # white-rel (cp=0)
        @test Eh > 0.5                                   # P0 clearly ahead
        mh0, mh1 = _mirror(hp0, hp1)
        @test BearoffK7.is_bearoff_position(mh0, mh1)
        ghm = _chance_game(mh0, mh1, 1)                  # same race, seats swapped, P1 on roll
        Ehm = Float64(BearoffK7.compute_equity(BearoffK7.lookup(TABLE_ID, ghm)))  # mover(P1)-rel
        @test isapprox(Eh, Ehm; atol = 1e-6)             # racer's equity is seat-invariant

        # --- Generate ~200 random turn-complete (chance-node) bear-off states --
        states = BackgammonGame[]
        while length(states) < 200
            p0, p1 = _rand_bearoff(rng)
            BearoffK7.is_bearoff_position(p0, p1) || continue
            cp = rand(rng, 0:1)
            g = _chance_game(p0, p1, cp)
            BackgammonNet.is_chance_node(g) || continue
            push!(states, g)
        end

        # --- 1a. Internal-consistency sign/perspective negation (EXACT) --------
        # For a turn-complete state, mover=0 and mover=1 evaluations are the SAME
        # physical equity seen from opposite seats -> exact fp negations. (Only
        # asserted on chance/terminal states; a mid-turn decision node's value is
        # NOT a pure sign flip because each mover maximizes their OWN outcome.)
        maxerr_neg = 0.0
        for g in states
            v0 = bearoff_turn_value(TABLE_ID, g, 0)
            v1 = bearoff_turn_value(TABLE_ID, g, 1)
            maxerr_neg = max(maxerr_neg, abs(v0 + v1))
        end
        @test maxerr_neg == 0.0

        # --- 1b. Physical-mirror perspective symmetry --------------------------
        # White-relative value of a position == -(white-relative value of its
        # player-swapped mirror). Uses the reflection verified in 1.0.
        maxerr_mir = 0.0
        for g in states
            mp0, mp1 = _mirror(g.p0, g.p1)
            gm = _chance_game(mp0, mp1, 1 - Int(g.current_player))
            wr   = bearoff_turn_value(TABLE_ID, g,  0)      # white-relative
            wr_m = bearoff_turn_value(TABLE_ID, gm, 0)      # white-relative of mirror
            maxerr_mir = max(maxerr_mir, abs(wr + wr_m))
        end
        @test maxerr_mir < 1e-6

        # --- 1c. Normalization: evaluator == bearoff_turn_value/3, and |v|<=1 ---
        # Chance nodes (pre-dice) and decision nodes (post-dice) both checked.
        maxerr_norm = 0.0; maxabs = 0.0
        for g in states
            cp = Int(g.current_player)
            # chance node
            out_c = ev(BGD.GameEnv(BackgammonNet.clone(g), MersenneTwister(0)))
            ref_c = bearoff_turn_value(TABLE_ID, g, 0) / 3.0     # white-relative /3
            maxerr_norm = max(maxerr_norm, abs(out_c - ref_c))
            maxabs = max(maxabs, abs(out_c))
            # decision node (roll dice, then evaluator uses best-move value)
            gd = BackgammonNet.clone(g)
            BackgammonNet.sample_chance!(gd, MersenneTwister(hash(g.p0) % 100000))
            if !gd.terminated && !BackgammonNet.is_chance_node(gd) &&
               !isempty(BackgammonNet.legal_actions(gd))
                out_d = ev(BGD.GameEnv(BackgammonNet.clone(gd), MersenneTwister(0)))
                best = bearoff_best_move_value(TABLE_ID, gd) / 3.0
                ref_d = Int(gd.current_player) == 0 ? best : -best   # white-relative
                maxerr_norm = max(maxerr_norm, abs(out_d - ref_d))
                maxabs = max(maxabs, abs(out_d))
            end
        end
        @test maxerr_norm == 0.0
        @test maxabs <= 1.0 + 1e-12

        # --- 1d. Chance-node value == probability-weighted avg of per-dice best -
        # The "sum of stochastic children, averaged" identity: the pre-dice table
        # value equals Σ P(dice) · (exact best post-dice move value). Enumerate all
        # 21 dice with weights 1/18 (non-doubles) and 1/36 (doubles). Agreement is
        # bounded by the table's 16-bit quantization (~1.5e-5 per probability).
        maxerr_avg = 0.0; nchk = 0
        for g in states
            cp = Int(g.current_player)
            pre = bearoff_turn_value(TABLE_ID, g, cp)   # mover-perspective raw points
            acc = 0.0; wsum = 0.0
            genv = BGD.GameEnv(g, MersenneTwister(0))
            for (oc, prob) in GI.chance_outcomes(genv)
                gg = BackgammonNet.clone(g)
                BackgammonNet.apply_chance!(gg, oc)
                acc += prob * bearoff_best_move_value(TABLE_ID, gg)  # same mover
                wsum += prob
            end
            @test isapprox(wsum, 1.0; atol = 1e-5)      # weights form a distribution (dice probs are Float32)
            maxerr_avg = max(maxerr_avg, abs(pre - acc)); nchk += 1
        end
        # Tolerance justification: probabilities are stored as UInt16/65535 (quantum
        # ~1.5e-5); compute_equity sums a few, per-dice takes a max over children ->
        # empirically <3e-5 across states. 1e-3 is a ~30x safety margin.
        @test maxerr_avg < 1e-3

        @info "Rung 1 summary" states=length(states) maxerr_neg maxerr_mir maxerr_norm maxabs maxerr_avg nchk
    end

    # ========================================================================
    # RUNG 2 — depth-1 search identity (BatchedMCTS)
    # ========================================================================
    #
    # Setup: uniform oracle (P=1/n, V=0) + exact bear-off evaluator, cpuct=2.0,
    # no Dirichlet noise, sims = 4·num_actions (guarantees every root action is
    # visited). At a post-dice bear-off decision node EVERY leaf reached is either
    # a terminal or a bear-off chance/decision node the evaluator scores exactly.
    #
    # A root action `a` is TURN-COMPLETING if the post-move state is terminal or a
    # chance node (opponent to roll). Every visit of such an `a` re-terminates at
    # the SAME exactly-evaluated leaf (chance-node bear-off values are recomputed
    # each visit, never cached as an expandable subtree), so:
    #
    #        Q(a) = W/N  ==  bearoff_turn_value(post_move, root_mover)/3   EXACTLY.
    #
    # ANY deviation is a backprop / virtual-loss / reward-scale / sign bug.
    #
    # For DOUBLES roots, the first action can land on a same-player MID-TURN node
    # which IS expanded and re-descended; its Q is then a convex average of exactly-
    # evaluated descendants, all <= the exact optimum. There we assert the hard
    # bound Q(a) <= exact + eps (a sign/scale bug would blow past it or go negative).
    # ------------------------------------------------------------------------
    function _run_rung2(nstates, batch_size, sims_mult, seed; assert_sumN = true)
        rng = MersenneTwister(seed)
        ev = _make_evaluator(TABLE_ID)
        maxerr_exact = 0.0; n_exact = 0
        maxviol_bound = 0.0; n_bound = 0
        n_depth1 = 0; n_states = 0
        argmax_fail_states = BackgammonGame[]
        exact_fail_states = Tuple{BackgammonGame, Int, Float64, Float64}[]

        for _ in 1:nstates
            g = _rand_decision(rng)
            mover = Int(g.current_player)
            acts = BackgammonNet.legal_actions(g)
            nact = length(acts)

            # classify: is every legal action turn-completing? (pure depth-1)
            depth1 = true
            for a in acts
                w = BackgammonNet.clone(g); BackgammonNet.apply_action!(w, a)
                if !(w.terminated || BackgammonNet.is_chance_node(w)); depth1 = false; break; end
            end

            nsims = max(8, sims_mult * nact)
            params = MctsParams(num_iters_per_turn = nsims, cpuct = 2.0, gamma = 1.0,
                temperature = ConstSchedule(0.0), dirichlet_noise_ϵ = 0.0,
                dirichlet_noise_α = 0.3, prior_temperature = 1.0, chance_mode = :passthrough)
            player = BatchedMCTS.BatchedMctsPlayer(GSPEC_ID, _uniform_oracle, params;
                        batch_size = batch_size, bearoff_evaluator = ev)
            genv = BGD.GameEnv(BackgammonNet.clone(g), MersenneTwister(7))
            BatchedMCTS.batched_explore!(player.benv, genv, nsims)
            info = player.benv.env.tree[BackgammonNet.clone(g)]

            # exact per-action value (root-mover perspective, /3 scale)
            exact = Dict{Int, Float64}()
            for a in info.actions
                w = BackgammonNet.clone(g); BackgammonNet.apply_action!(w, a)
                exact[a] = bearoff_turn_value(TABLE_ID, w, mover) / 3.0
            end

            # virtual-loss bookkeeping: each simulation selects exactly one root
            # action and increments its N by net +1 (traverse applies VL: N+1;
            # backprop restores W but LEAVES N — so the count is exact, never
            # leaked). The first simulation only CREATES the root: for a multi-
            # action root it stops at the bear-off evaluator (no action selected)
            # -> sum(N) == nsims-1; for a single-action root it descends through
            # the forced action -> sum(N) == nsims. Any other value would mean a
            # leaked/duplicated virtual-loss visit. (The stronger proof that VL is
            # fully unwound is the exact Q=W/N identity above — a stuck -VL in W
            # would corrupt Q, which we assert to 1e-9 at every batch size.)
            if assert_sumN
                @test all(s.N >= 0 for s in info.stats)
                @test (nsims - 1) <= sum(s.N for s in info.stats) <= nsims
            end

            for (k, a) in enumerate(info.actions)
                s = info.stats[k]; s.N >= 1 || continue
                Q = s.W / s.N
                w = BackgammonNet.clone(g); BackgammonNet.apply_action!(w, a)
                if w.terminated || BackgammonNet.is_chance_node(w)
                    e = abs(Q - exact[a])
                    if e > 1e-9; push!(exact_fail_states, (g, a, Q, exact[a])); end
                    maxerr_exact = max(maxerr_exact, e); n_exact += 1
                else
                    maxviol_bound = max(maxviol_bound, Q - exact[a]); n_bound += 1
                end
            end

            if depth1
                n_depth1 += 1
                vis = [(info.actions[k], info.stats[k].W / info.stats[k].N)
                       for k in eachindex(info.actions) if info.stats[k].N >= 1]
                qbest = vis[argmax([q for (_, q) in vis])][1]
                emax = maximum(values(exact))
                # allow any action whose EXACT value ties the max (within 1e-12)
                exact[qbest] >= emax - 1e-12 || push!(argmax_fail_states, g)
            end
            n_states += 1
        end

        # Actionable counterexample dumps
        for (g, a, Q, E) in first(exact_fail_states, 5)
            @error "Rung 2 EXACT identity violated" state=_hexdump(g) action=a Q=Q exact=E Δ=(Q - E)
        end
        for g in first(argmax_fail_states, 5)
            @error "Rung 2 argmax != table optimum" state=_hexdump(g)
        end

        return (; maxerr_exact, n_exact, maxviol_bound, n_bound, n_depth1, n_states,
                  argmax_fails = length(argmax_fail_states), exact_fails = length(exact_fail_states))
    end

    @testset "Rung 2: depth-1 search identity (Q = exact leaf value)" begin
        # Main sweep: batch_size = 1 (each simulation self-contained).
        r = _run_rung2(300, 1, 4, 0xB2)
        @test r.maxerr_exact < 1e-9          # turn-completing actions: EXACT identity
        @test r.exact_fails == 0
        @test r.argmax_fails == 0            # argmax-Q == exact table argmax (depth-1)
        @test r.maxviol_bound < 1e-9         # doubles mid-turn: Q never exceeds exact
        @test r.n_exact > 500                # coverage sanity
        @test r.n_depth1 > 0
        @info "Rung 2 batch=1" summary=r

        # Virtual-loss unwinding: SAME exact identity must hold at larger batch
        # sizes, where many simulations traverse concurrently applying/removing
        # virtual loss. If VL were not fully unwound, Q would be corrupted.
        for bs in (8, 16)
            rv = _run_rung2(60, bs, 4, 0xB2 + bs)
            @test rv.maxerr_exact < 1e-9
            @test rv.exact_fails == 0
            @test rv.argmax_fails == 0
            @test rv.maxviol_bound < 1e-9
            @info "Rung 2 batch=$bs virtual-loss unwind" summary=rv
        end
    end

    # ========================================================================
    # RUNG 3 — multi-level backprop identity (classic MCTS, :full expectimax)
    # ========================================================================
    #
    # Tiny FULL game trees that bottom out at TRUE terminals (no evaluator, no
    # table — only rewards + backprop). Classic MCTS (src/mcts.jl) with
    # chance_mode=:full does exact chance-node averaging. We compare each root
    # action's Q to the exact expectimax value from a direct pure-Julia recursion
    # over the same game. This exercises the mechanisms Rung 2 cannot reach:
    # reward recording (÷3), pswitch sign flips across control changes, chance-node
    # probability weighting, and multi-level value accumulation — including the ±2
    # GAMMON reward path.
    #
    # :full MCTS is DETERMINISTIC (all-outcome expansion + deficit selection, no
    # sampling), so the residual error is a fixed function of the sim budget, not
    # noise. With uniform V=0 interior priors, un-expanded leaves are truncated at
    # 0 until the search reaches real terminals, leaving a small O(0.05) downward
    # bias on deeper trees at finite sims. A sign / perspective / scale bug instead
    # produces errors of O(0.5-2.0), so a 0.1 tolerance cleanly separates the two.
    # ------------------------------------------------------------------------

    # Exact expectimax over the real game, memoized. Values in RAW points.
    const _R3_MEMO = Dict{BackgammonGame, Float64}()
    function _exact_value(g)::Float64       # from g.current_player's perspective
        haskey(_R3_MEMO, g) && return _R3_MEMO[g]
        local v
        if g.terminated
            v = g.current_player == 0 ? Float64(g.reward) : -Float64(g.reward)
        elseif BackgammonNet.is_chance_node(g)
            cp = Int(g.current_player); acc = 0.0
            genv = BGD.GameEnv(g, MersenneTwister(0))
            for (oc, prob) in GI.chance_outcomes(genv)
                gg = BackgammonNet.clone(g); BackgammonNet.apply_chance!(gg, oc)
                acc += prob * _value_to(gg, cp)
            end
            v = acc
        else
            mover = Int(g.current_player); best = -Inf
            for a in BackgammonNet.legal_actions(g)
                gg = BackgammonNet.clone(g); BackgammonNet.apply_action!(gg, a)
                best = max(best, _value_to(gg, mover))
            end
            v = best
        end
        _R3_MEMO[g] = v; return v
    end
    _value_to(g, mover)::Float64 =
        g.terminated ? (mover == 0 ? Float64(g.reward) : -Float64(g.reward)) :
        (Int(g.current_player) == mover ? _exact_value(g) : -_exact_value(g))

    """Constructed near-terminal positions (each side few checkers, ends in a few
    plies). Returned as (name, post-dice decision-node game, has_real_argmax)."""
    function _rung3_positions()
        ps = Tuple{String, BackgammonGame, Bool}[]
        # A: genuine 3-vs-3 race, both in home. Mixed action values -> real argmax.
        pA0 = (UInt128(12) << (25 << 2)) | (UInt128(1) << (24 << 2)) |
              (UInt128(1) << (23 << 2)) | (UInt128(1) << (22 << 2))       # pts 1,2,3
        pA1 = (UInt128(12) << 0) | (UInt128(1) << (1 << 2)) |
              (UInt128(1) << (2 << 2)) | (UInt128(1) << (3 << 2))
        gA = _chance_game(pA0, pA1, 0); BackgammonNet.sample_chance!(gA, MersenneTwister(11))
        push!(ps, ("A_race_3v3", gA, true))
        # C: 2-vs-3 race, BLACK to move (losing) — exercises mover=1 sign paths.
        pC0 = (UInt128(13) << (25 << 2)) | (UInt128(1) << (24 << 2)) | (UInt128(1) << (23 << 2))
        pC1 = (UInt128(12) << 0) | (UInt128(1) << (1 << 2)) |
              (UInt128(1) << (2 << 2)) | (UInt128(1) << (3 << 2))
        gC = _chance_game(pC0, pC1, 1); BackgammonNet.sample_chance!(gC, MersenneTwister(9))
        push!(ps, ("C_race_2v3_black", gC, false))
        # B: GUARANTEED GAMMON, multi-level. White 1 checker on point 6, dice (2,1)
        # cannot bear off this turn -> post-move is a BLACK chance node (turn ends).
        # Black has 15 checkers on point 7 (outside home) and 0 borne off, so black
        # can NEVER bear off before white finishes -> every line is a +2 white
        # gammon. Exercises the ±2 reward backed up through a chance node + pswitch.
        pB0 = (UInt128(14) << (25 << 2)) | (UInt128(1) << (19 << 2))      # 1 on point 6
        pB1 = UInt128(15) << (7 << 2)                                     # 15 on point 7, 0 off
        gB = BackgammonGame(pB0, pB1, SVector{2, Int8}(2, 1), Int8(1), Int8(0), false, 0.0f0;
                            obs_type = :minimal_flat)
        push!(ps, ("B_gammon_multilevel", gB, false))
        return ps
    end

    @testset "Rung 3: multi-level backprop identity (:full expectimax)" begin
        R3_SIMS = 20000
        R3_TOL = 0.1                            # raw points; see header derivation
        saw_gammon = false
        for (name, g, has_argmax) in _rung3_positions()
            empty!(_R3_MEMO)
            @test !g.terminated && !BackgammonNet.is_chance_node(g)
            mover = Int(g.current_player)
            acts = BackgammonNet.legal_actions(g)

            # exact expectimax value of each root action (raw points, mover-rel)
            exact = Dict{Int, Float64}()
            for a in acts
                gg = BackgammonNet.clone(g); BackgammonNet.apply_action!(gg, a)
                exact[a] = _value_to(gg, mover)
            end
            emax = maximum(values(exact))
            if any(≥(1.5), abs.(values(exact))); saw_gammon = true; end   # ±2 path present

            env = MCTS.Env(GSPEC_ID, _uniform_oracle; gamma = 1.0, cpuct = 2.0,
                           noise_ϵ = 0.0, noise_α = 1.0, chance_mode = :full)
            MCTS.explore!(env, BGD.GameEnv(BackgammonNet.clone(g), MersenneTwister(1)), R3_SIMS)
            info = env.tree[BackgammonNet.clone(g)]

            maxerr = 0.0
            for (k, a) in enumerate(info.actions)
                s = info.stats[k]; s.N >= 1 || continue
                Q_raw = (s.W / s.N) * GI.reward_scale(GSPEC_ID)   # undo ÷3 -> raw points
                e = abs(Q_raw - exact[a])
                if e > R3_TOL
                    @error "Rung 3 value identity violated" position=name action=a Q_raw=Q_raw exact=exact[a] Δ=e state=_hexdump(g)
                end
                maxerr = max(maxerr, e)
            end
            @test maxerr < R3_TOL

            if has_argmax
                vis = [(info.actions[k], info.stats[k].W / info.stats[k].N)
                       for k in eachindex(info.actions) if info.stats[k].N >= 1]
                qbest = vis[argmax([q for (_, q) in vis])][1]
                @test exact[qbest] >= emax - 1e-9          # MCTS argmax == expectimax argmax
            end
            @info "Rung 3 $name" nact=length(acts) emax=round(emax, digits=3) maxerr=round(maxerr, digits=4) tree=length(env.tree) chance=length(env.chance_tree)
        end
        @test saw_gammon        # the ±2 gammon reward path was exercised
    end
end
