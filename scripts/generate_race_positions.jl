#!/usr/bin/env julia
"""
Generate 2000 beginning-of-race positions for fixed evaluation.

Plays full games from opening, records the FIRST board state where
is_race_position() becomes true. These are realistic "race entry" positions
with most checkers still on the board (~20+ moves remaining).

Output: Serialized file at /homeshare/projects/AlphaZero.jl/eval_data/race_eval_2000.jls
"""

using Random
using Serialization
using Statistics
using StaticArrays

# Load BackgammonNet
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using BackgammonNet
using BackgammonNet: BackgammonGame, get_count, is_race_position

"""
Play a random game from opening, return the first position where race begins.
Returns nothing if game ends without entering race (rare, ~1.5%).
Uses random moves (no NN needed — just need realistic board positions).
"""
function find_race_entry(rng::AbstractRNG)
    g = BackgammonNet.initial_state(; short_game=true, obs_type=:minimal_flat)

    for _ in 1:500  # safety limit
        BackgammonNet.game_terminated(g) && return nothing

        if BackgammonNet.is_chance_node(g)
            # Check race BEFORE rolling dice (race entry is pre-dice)
            if is_race_position(g)
                return g
            end
            BackgammonNet.sample_chance!(g, rng)
            continue
        end

        # Random legal move
        actions = BackgammonNet.legal_actions(g)
        isempty(actions) && return nothing
        action = actions[rand(rng, 1:length(actions))]
        BackgammonNet.apply_action!(g, action)
    end
    return nothing
end

"""Compute pip count for P0 (moves 1→24→off)."""
function pip_count_p0(p0::UInt128)
    total = 0
    for pt in 1:24
        total += (25 - pt) * Int(get_count(p0, pt))
    end
    return total
end

"""Compute pip count for P1 (moves 24→1→off)."""
function pip_count_p1(p1::UInt128)
    total = 0
    for pt in 1:24
        total += pt * Int(get_count(p1, pt))
    end
    return total
end

function checkers_on_board(board::UInt128, pts::UnitRange{Int})
    total = 0
    for pt in pts
        total += Int(get_count(board, pt))
    end
    return total
end

function main()
    rng = MersenneTwister(42)  # Fixed seed for reproducibility
    N = 2000

    println("Playing random games to find $N beginning-of-race positions...")
    positions = BackgammonGame[]
    games_played = 0
    no_race_count = 0

    while length(positions) < N
        g = find_race_entry(rng)
        games_played += 1
        if g === nothing
            no_race_count += 1
            continue
        end
        push!(positions, BackgammonNet.clone(g))

        if length(positions) % 500 == 0
            println("  $(length(positions))/$N positions found ($games_played games played)")
        end
    end

    race_pct = 100 * (games_played - no_race_count) / games_played
    println("  Games played: $games_played ($(round(race_pct, digits=1))% entered race)")

    # Verify all are race positions
    @assert all(is_race_position, positions) "Some positions are not race!"
    @assert all(!BackgammonNet.game_terminated(g) for g in positions) "Some positions are terminated!"

    # Statistics
    p0_pips = [pip_count_p0(g.p0) for g in positions]
    p1_pips = [pip_count_p1(g.p1) for g in positions]
    p0_on = [checkers_on_board(g.p0, 1:24) for g in positions]
    p1_on = [checkers_on_board(g.p1, 1:24) for g in positions]

    println("\n=== Beginning-of-Race Position Statistics (N=$N) ===")
    println("P0 pips: mean=$(round(mean(p0_pips), digits=1)), min=$(minimum(p0_pips)), max=$(maximum(p0_pips))")
    println("P1 pips: mean=$(round(mean(p1_pips), digits=1)), min=$(minimum(p1_pips)), max=$(maximum(p1_pips))")
    println("P0 on board: mean=$(round(mean(p0_on), digits=1)), min=$(minimum(p0_on)), max=$(maximum(p0_on))")
    println("P1 on board: mean=$(round(mean(p1_on), digits=1)), min=$(minimum(p1_on)), max=$(maximum(p1_on))")

    # Bearoff-eligible count (both players in home board)
    n_bearoff = count(positions) do g
        checkers_on_board(g.p0, 1:18) == 0 && checkers_on_board(g.p1, 7:24) == 0
    end
    println("Bearoff-eligible: $n_bearoff / $N ($(round(100*n_bearoff/N, digits=1))%)")

    # Save as serialized bitboard pairs
    save_data = [(g.p0, g.p1, g.current_player) for g in positions]

    output_dir = "/homeshare/projects/AlphaZero.jl/eval_data"
    mkpath(output_dir)
    output_path = joinpath(output_dir, "race_eval_2000.jls")
    serialize(output_path, save_data)
    println("\nSaved to: $output_path")
    println("File size: $(round(filesize(output_path)/1024, digits=1)) KB")
end

main()
