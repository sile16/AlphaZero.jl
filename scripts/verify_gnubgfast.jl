#!/usr/bin/env julia
# Verify GnubgPlayerFast move conversion against GnubgPlayer (reference)
#
# Plays random positions and compares the action chosen by both implementations.
# GnubgPlayer is known-correct (evaluates all legal actions via gnubg equity).
# GnubgPlayerFast uses gnubg's native best_move and converts the move.
#
# Usage: julia --threads 2 --project scripts/verify_gnubgfast.jl [num_positions]

using BackgammonNet
using Random

include(joinpath(@__DIR__, "GnubgPlayer.jl"))
include(joinpath(@__DIR__, "GnubgPlayerFast.jl"))

using .GnubgPlayer: best_move as gnubg_best_move
using .GnubgPlayerFast: best_move_native, _to_gnubg_board, _gnubg_moves_to_action

const PLY = 0
const NUM_POSITIONS = parse(Int, get(ARGS, 1, "200"))

function generate_random_position(rng)
    g = BackgammonNet.initial_state(; short_game=true, doubles_only=false, obs_type=:minimal_flat)
    # Play random moves to get diverse positions
    n_moves = rand(rng, 1:30)
    for _ in 1:n_moves
        if BackgammonNet.game_terminated(g)
            break
        end
        if BackgammonNet.is_chance_node(g)
            BackgammonNet.sample_chance!(g, rng)
            continue
        end
        actions = BackgammonNet.legal_actions(g)
        if isempty(actions)
            break
        end
        BackgammonNet.apply_action!(g, rand(rng, actions))
    end
    return g
end

function main()
    rng = MersenneTwister(42)

    n_tested = 0
    n_match = 0
    n_mismatch = 0
    n_fast_illegal = 0
    n_skipped = 0
    mismatches = []

    println("Verifying GnubgPlayerFast vs GnubgPlayer on $NUM_POSITIONS positions ($(PLY)-ply)...")
    println()

    for i in 1:NUM_POSITIONS * 3  # Generate extra since some will be skipped
        n_tested >= NUM_POSITIONS && break

        g = generate_random_position(rng)

        # Skip terminated, chance nodes, or no legal moves
        if BackgammonNet.game_terminated(g) || BackgammonNet.is_chance_node(g)
            n_skipped += 1
            continue
        end

        legal = BackgammonNet.legal_actions(g)
        if isempty(legal) || (length(legal) == 1 && legal[1] == BackgammonNet.encode_action(25, 25))
            n_skipped += 1
            continue
        end

        n_tested += 1

        # Reference: GnubgPlayer (evaluates all legal actions)
        ref_action, ref_equity = gnubg_best_move(g; ply=PLY)

        # Fast: GnubgPlayerFast (converts gnubg native best_move)
        fast_action = best_move_native(g; ply=PLY)

        # Check if fast action is legal
        is_legal = fast_action in legal

        if !is_legal
            n_fast_illegal += 1
            d1, d2 = Int(g.dice[1]), Int(g.dice[2])
            println("  ILLEGAL #$n_tested: fast=$fast_action not in legal_actions, dice=($d1,$d2), player=$(g.current_player), remaining=$(g.remaining_actions)")
            push!(mismatches, (i=n_tested, type=:illegal, ref=ref_action, fast=fast_action,
                               dice=(Int(g.dice[1]), Int(g.dice[2])), player=Int(g.current_player),
                               remaining=Int(g.remaining_actions), n_legal=length(legal)))
        elseif fast_action != ref_action
            n_mismatch += 1
            # Check if fast action has same equity (might be equally good)
            g2 = BackgammonNet.clone(g)
            BackgammonNet.apply_action!(g2, fast_action)
            fast_equity = GnubgPlayer.evaluate_position(g2; ply=PLY)
            if g2.current_player != g.current_player
                fast_equity = -fast_equity
            end

            equity_diff = abs(fast_equity - ref_equity)
            is_equiv = equity_diff < 0.001

            d1, d2 = Int(g.dice[1]), Int(g.dice[2])
            status = is_equiv ? "EQUIV" : "DIFF"
            println("  $status #$n_tested: ref=$ref_action (eq=$(round(ref_equity, digits=4))), fast=$fast_action (eq=$(round(fast_equity, digits=4))), dice=($d1,$d2), player=$(g.current_player), remaining=$(g.remaining_actions)")
            push!(mismatches, (i=n_tested, type=is_equiv ? :equiv : :diff, ref=ref_action, fast=fast_action,
                               ref_eq=ref_equity, fast_eq=fast_equity,
                               dice=(Int(g.dice[1]), Int(g.dice[2])), player=Int(g.current_player),
                               remaining=Int(g.remaining_actions), n_legal=length(legal)))
        else
            n_match += 1
        end

        if n_tested % 50 == 0
            println("  Progress: $n_tested tested, $n_match match, $n_mismatch mismatch, $n_fast_illegal illegal")
        end
    end

    println()
    println("=" ^ 60)
    println("Results: $n_tested positions tested ($n_skipped skipped)")
    println("  Match:   $n_match ($(round(100*n_match/n_tested, digits=1))%)")
    println("  Mismatch: $n_mismatch ($(round(100*n_mismatch/n_tested, digits=1))%)")
    println("  Illegal: $n_fast_illegal ($(round(100*n_fast_illegal/n_tested, digits=1))%)")
    println("=" ^ 60)

    if n_fast_illegal > 0
        println("\nCRITICAL: GnubgPlayerFast produced $n_fast_illegal illegal actions!")
    end

    if n_mismatch > 0
        n_equiv = count(m -> m.type == :equiv, mismatches)
        n_diff = count(m -> m.type == :diff, mismatches)
        println("\nMismatches: $n_equiv equivalent (same equity), $n_diff different (worse move)")
    end

    if n_match == n_tested
        println("\nAll actions match - GnubgPlayerFast conversion is correct!")
    end
end

main()
