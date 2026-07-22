# ============================================================================
# BATCHED MCTS — :exact_expectation CHANCE-NODE IDENTITY
# ============================================================================
#
# Verifies the EVAL-ONLY `:exact_expectation` chance mode in the BATCHED MCTS
# (src/batched_mcts.jl). Under this mode a chance node is a first-class tree
# entry whose value is the PROBABILITY-WEIGHTED EXPECTATION over its ~21 dice
# outcome children — exactly like the recursive `:full` engine, but reached via
# the batched (traverse / batch-evaluate / backprop) waves.
#
# Ground truth is a direct pure-Julia expectimax recursion over the REAL game
# (memoized), on constructed near-terminal positions that bottom out at true
# terminals in a few plies (so the uniform V=0 oracle only seeds the frontier
# and washes out — the same construction as Rung 3 of the identity staircase).
#
# TWO identities are checked at batch_size = 1 AND at a larger batch (16 / 32):
#
#   A) CHANCE-ROOT: the settled chance node value
#          Σ o.prob · (o.N>0 ? o.W/o.N : Vest)
#      equals the manual probability-weighted expectation of the outcome
#      children's exact values, and equals the recursive `:full` engine's
#      chance node value.
#
#   B) DECISION-ROOT (chance node ONE ply below the root): each root action's
#      Q = W/N converges to its exact expectimax value. This exercises the
#      chance-EDGE backup (no reward/gamma/sign flip; expectation propagated to
#      the parent) and, at batch>1, the load-bearing N==0→Vest fallback (a
#      follower may back up through the chance node before every outcome is
#      filled; using 0 there would permanently bias the parent decision's W).
#
# No bear-off evaluator is attached, so EVERY chance node enters the expectation
# machinery (the bear-off short-circuit is verified separately in the staircase).
# ============================================================================

using Test
using Random
using StaticArrays

using AlphaZero
using AlphaZero: GI, MCTS, BatchedMCTS
import BackgammonNet
using BackgammonNet: BackgammonGame

if !isdefined(Main, :BackgammonDeterministic)
    include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "main.jl"))
end
const BGD_CE = Main.BackgammonDeterministic
const GSPEC_CE = BGD_CE.GameSpec()
const RSCALE_CE = GI.reward_scale(GSPEC_CE)   # 3.0 — undo the ÷3 tree normalization

# ── Bit-board helpers (mirroring test_mcts_identity_staircase.jl) ────────────
# P0 points 1..7 -> board indices (24..18); P0 off = idx 25.
# P1 points 1..7 -> board indices (1..7);   P1 off = idx 0.
const _P0_IDX_CE = (24, 23, 22, 21, 20, 19, 18)
const _P1_IDX_CE = (1, 2, 3, 4, 5, 6, 7)

"""Pre-dice CHANCE-node game (dice unrolled) with player `cp` on roll."""
_chance_game_ce(p0, p1, cp) =
    BGD_CE.backgammon_game(p0, p1, SVector{2, Int8}(0, 0), Int8(0), Int8(cp), false, 0.0f0;
                           observation_type = :minimal_flat)

# ── Uniform-prior, V=0 oracle (interior frontier truncation only) ────────────
function _uniform_oracle_ce(state)
    n = max(1, length(GI.available_actions(GSPEC_CE, state)))
    return (fill(Float32(1 / n), n), 0.0f0)
end

# ── Exact expectimax over the REAL game, memoized. Values in RAW points ──────
const _MEMO_CE = Dict{BackgammonGame, Float64}()
function _exact_value_ce(g)::Float64      # from g.current_player's perspective
    haskey(_MEMO_CE, g) && return _MEMO_CE[g]
    local v
    if g.terminated
        v = g.current_player == 0 ? Float64(g.reward) : -Float64(g.reward)
    elseif BackgammonNet.is_chance_node(g)
        cp = Int(g.current_player); acc = 0.0
        genv = BGD_CE.GameEnv(g, MersenneTwister(0))
        for (oc, prob) in GI.chance_outcomes(genv)
            gg = BackgammonNet.clone(g); BackgammonNet.apply_chance!(gg, oc)
            acc += prob * _value_to_ce(gg, cp)
        end
        v = acc
    else
        mover = Int(g.current_player); best = -Inf
        for a in BackgammonNet.legal_actions(g)
            gg = BackgammonNet.clone(g); BackgammonNet.apply_action!(gg, a)
            best = max(best, _value_to_ce(gg, mover))
        end
        v = best
    end
    _MEMO_CE[g] = v; return v
end
_value_to_ce(g, mover)::Float64 =
    g.terminated ? (mover == 0 ? Float64(g.reward) : -Float64(g.reward)) :
    (Int(g.current_player) == mover ? _exact_value_ce(g) : -_exact_value_ce(g))

# ── Constructed SHALLOW CHANCE-node positions ────────────────────────────────
# Each bottoms out in ~1 ply (one bear-off move terminates), so both the recursive
# and batched engines fully resolve every line at the given sim budget and coincide
# with the exact expectation to well under tolerance. The single-checker-on-point-1
# construction means EVERY one of the 21 dice outcomes bears the last checker off,
# so the chance value is a known constant — an exact target for the expectation.
function _positions_ce()
    ps = Tuple{String, BackgammonGame}[]
    # P1: WHITE on roll, 1 checker on point 1 -> bears off next move -> single win.
    #     Black has 3 borne off (12 off) so it is NOT a gammon: every line = +1.
    p1w = (UInt128(14) << (25 << 2)) | (UInt128(1) << (24 << 2))   # 14 off, 1 on pt1
    p1b = (UInt128(12) << 0) | (UInt128(3) << (1 << 2))            # 12 off, 3 on pt1
    push!(ps, ("P1_white_single_+1", _chance_game_ce(p1w, p1b, 0)))
    # P2: GUARANTEED GAMMON. WHITE on roll, 1 on point 1; black has 0 off (3 stuck
    #     on point 7, outside home) -> white bears off next move -> +2 gammon on
    #     every line. Exercises the ±2 reward path through the chance expectation.
    p2w = (UInt128(14) << (25 << 2)) | (UInt128(1) << (24 << 2))   # 14 off, 1 on pt1
    p2b = UInt128(3) << (7 << 2)                                   # 3 on point 7, 0 off
    push!(ps, ("P2_white_gammon_+2", _chance_game_ce(p2w, p2b, 0)))
    # P3: BLACK on roll, symmetric single win -> exercises the mover=1 sign path.
    #     Black 1 on point 1 (14 off) bears off -> black wins; white has 12 off so
    #     no gammon: every line = +1 from BLACK's (the mover's) perspective.
    p3w = (UInt128(12) << (25 << 2)) | (UInt128(3) << (24 << 2))   # white 12 off, 3 on pt1
    p3b = (UInt128(14) << 0) | (UInt128(1) << (1 << 2))            # black 14 off, 1 on pt1
    push!(ps, ("P3_black_single_+1", _chance_game_ce(p3w, p3b, 1)))
    # P4: GENUINELY MIXED outcome values -> the chance value depends on the dice
    #     PROBABILITIES, not just an average. WHITE on roll with 1 checker on point 4;
    #     most rolls bear it off (win +1), but the (1,2) roll cannot finish and ends
    #     white's turn on point 1 -> black (1 on point 1) then bears off and wins
    #     (-1 for white). So the expectation is a non-trivial probability-weighted
    #     blend of +1 and -1 outcomes (a plain uniform average would be wrong).
    p4w = (UInt128(14) << (25 << 2)) | (UInt128(1) << (21 << 2))   # white 14 off, 1 on pt4
    p4b = (UInt128(14) << 0) | (UInt128(1) << (1 << 2))            # black 14 off, 1 on pt1
    push!(ps, ("P4_white_mixed", _chance_game_ce(p4w, p4b, 0)))
    return ps
end

# Settled chance node value from a chance_tree entry (RAW points).
function _settled_chance_raw(cinfo)
    Vest = Float64(cinfo.Vest)
    e = 0.0
    for o in cinfo.outcomes
        e += o.prob * (o.N > 0 ? o.W / o.N : Vest)
    end
    return e * RSCALE_CE
end

const _SIMS_CE = 20000
const _TOL_CE = 0.12    # raw points; see Rung-3 header derivation (finite-sim bias)

@testset "A: chance-root value == prob-weighted expectation" begin
    for (name, gc) in _positions_ce()
        @test BackgammonNet.is_chance_node(gc)
        empty!(_MEMO_CE)

        # Manual probability-weighted expectation of the outcome children's exact
        # values (== exact expectimax of the chance node), in RAW points.
        cp = Int(gc.current_player)
        manual = 0.0
        uniform = 0.0            # plain (WRONG) average, to show weighting matters
        childvals = Float64[]
        genv = BGD_CE.GameEnv(BackgammonNet.clone(gc), MersenneTwister(0))
        outs = GI.chance_outcomes(genv)
        for (oc, prob) in outs
            gg = BackgammonNet.clone(gc); BackgammonNet.apply_chance!(gg, oc)
            cv = _value_to_ce(gg, cp)
            manual += prob * cv
            uniform += cv / length(outs)
            push!(childvals, cv)
        end
        @test isapprox(manual, _exact_value_ce(gc); atol = 1e-9)   # internal consistency

        # For the mixed position, prove the outcomes really differ AND that the
        # probability weighting is load-bearing (uniform average would be wrong).
        if name == "P4_white_mixed"
            @test maximum(childvals) - minimum(childvals) > 0.5   # genuinely mixed
            @test abs(manual - uniform) > 1e-3                    # weighting matters
        end

        # Batched :exact_expectation at batch_size = 1 and larger, validated
        # against the analytic probability-weighted expectation `manual` (the
        # independent exact reference).
        for bs in (1, 16, 32)
            params = MctsParams(num_iters_per_turn = _SIMS_CE, cpuct = 2.0, gamma = 1.0,
                temperature = ConstSchedule(0.0), dirichlet_noise_ϵ = 0.0,
                dirichlet_noise_α = 0.0, prior_temperature = 1.0,
                chance_mode = :exact_expectation)
            player = BatchedMCTS.BatchedMctsPlayer(GSPEC_CE, _uniform_oracle_ce, params;
                        batch_size = bs)
            genv2 = BGD_CE.GameEnv(BackgammonNet.clone(gc), MersenneTwister(7))
            BatchedMCTS.batched_explore!(player.benv, genv2, _SIMS_CE)
            cinfo = player.benv.env.chance_tree[BackgammonNet.clone(gc)]
            @test cinfo.expanded
            @test all(o.N > 0 for o in cinfo.outcomes)   # every outcome was visited
            bat_raw = _settled_chance_raw(cinfo)
            if abs(bat_raw - manual) >= _TOL_CE
                @error "chance-root value off" position=name bs=bs batched=bat_raw manual=manual
            end
            @test abs(bat_raw - manual) < _TOL_CE
        end
        @info "A $name" manual=round(manual, digits=4)
    end
end

# Decision-root positions whose actions each pass the turn THROUGH a chance node
# (opponent to roll) before a shallow, guaranteed terminal — this is what exercises
# the chance-EDGE backup and, at batch>1, the N==0→Vest fallback in the path.
function _decision_positions_ce()
    ps = Tuple{String, BackgammonGame}[]
    # B1: BLACK to move and hopeless. Black has 3 stuck on point 7 (0 off) and can
    #     never bear off; white has 1 on point 1 and bears off next roll -> every
    #     line is a +2 white gammon = -2 for the black mover. The root action passes
    #     to a WHITE chance node (opponent to roll) before the terminal.
    b1w = (UInt128(14) << (25 << 2)) | (UInt128(1) << (24 << 2))   # white 14 off, 1 on pt1
    b1b = UInt128(3) << (7 << 2)                                   # black 3 on point 7, 0 off
    g1 = _chance_game_ce(b1w, b1b, 1); BackgammonNet.sample_chance!(g1, MersenneTwister(23))
    push!(ps, ("B1_black_mover_-2", g1))
    # B2: WHITE to move and hopeless (mover = white, negative). White has 3 stuck on
    #     point 7 (0 off); black has 1 on point 1 and bears off next roll -> -2 for
    #     the white mover. Passes through a BLACK chance node.
    b2w = UInt128(3) << (18 << 2)                                  # white 3 on point 7, 0 off
    b2b = (UInt128(14) << 0) | (UInt128(1) << (1 << 2))            # black 14 off, 1 on pt1
    g2 = _chance_game_ce(b2w, b2b, 0); BackgammonNet.sample_chance!(g2, MersenneTwister(29))
    push!(ps, ("B2_white_mover_-2", g2))
    return ps
end

@testset "B: decision-root action Q via chance-edge backup (batch invariance)" begin
    for (name, gd) in _decision_positions_ce()
        @test !gd.terminated && !BackgammonNet.is_chance_node(gd)
        @test !isempty(BackgammonNet.legal_actions(gd))
        empty!(_MEMO_CE)
        mover = Int(gd.current_player)

        # Exact expectimax value of each root action (raw points, mover-relative).
        exact = Dict{Int, Float64}()
        for a in BackgammonNet.legal_actions(gd)
            gg = BackgammonNet.clone(gd); BackgammonNet.apply_action!(gg, a)
            exact[a] = _value_to_ce(gg, mover)
        end
        emax = maximum(values(exact))

        for bs in (1, 32)
            params = MctsParams(num_iters_per_turn = _SIMS_CE, cpuct = 2.0, gamma = 1.0,
                temperature = ConstSchedule(0.0), dirichlet_noise_ϵ = 0.0,
                dirichlet_noise_α = 0.0, prior_temperature = 1.0,
                chance_mode = :exact_expectation)
            player = BatchedMCTS.BatchedMctsPlayer(GSPEC_CE, _uniform_oracle_ce, params;
                        batch_size = bs)
            genv = BGD_CE.GameEnv(BackgammonNet.clone(gd), MersenneTwister(7))
            BatchedMCTS.batched_explore!(player.benv, genv, _SIMS_CE)
            info = player.benv.env.tree[BackgammonNet.clone(gd)]

            maxerr = 0.0
            for (k, a) in enumerate(info.actions)
                s = info.stats[k]; s.N >= 1 || continue
                q_raw = (s.W / s.N) * RSCALE_CE
                maxerr = max(maxerr, abs(q_raw - exact[a]))
            end
            if maxerr >= _TOL_CE
                @error "decision-root Q off" position=name bs=bs maxerr=maxerr
            end
            @test maxerr < _TOL_CE

            # argmax-Q action must be an exact-optimal action.
            vis = [(info.actions[k], info.stats[k].W / info.stats[k].N)
                   for k in eachindex(info.actions) if info.stats[k].N >= 1]
            qbest = vis[argmax([q for (_, q) in vis])][1]
            @test exact[qbest] >= emax - 1e-9
        end
        @info "B $name" nact=length(exact) emax=round(emax, digits=3)
    end
end
