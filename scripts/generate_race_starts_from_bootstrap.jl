#!/usr/bin/env julia
"""
Extract race starting positions from bootstrap wildbg games.

Scans 1M bootstrap games (10 part files × 100K games), finds the first
position in each game where is_race_position() is true. Saves as portable
(p0, p1, current_player) tuples.

Usage:
    julia --project scripts/generate_race_starts_from_bootstrap.jl [--eval-exclude race_eval_2000.jls]
"""

using Serialization
using Random
using Statistics

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using BackgammonNet
using BackgammonNet: is_race_position, get_count

# --- Config ---

BOOTSTRAP_DIR = joinpath(homedir(), "github", "BackgammonNet.jl", "data", "bootstrap")
OUTPUT_DIR = joinpath(@__DIR__, "..", "eval_data")
mkpath(OUTPUT_DIR)

# Parse args
EVAL_EXCLUDE_FILE = ""
for arg in ARGS
    if startswith(arg, "--eval-exclude=")
        EVAL_EXCLUDE_FILE = split(arg, "=", limit=2)[2]
    end
end

# --- Helpers ---

function pip_count(p0::UInt128, p1::UInt128)
    pips0 = sum((25 - pt) * Int(get_count(p0, pt)) for pt in 1:24)
    pips1 = sum(pt * Int(get_count(p1, pt)) for pt in 1:24)
    return pips0, pips1
end

function checkers_on_board(board::UInt128)
    sum(Int(get_count(board, pt)) for pt in 1:24)
end

# --- Main ---

function main()
    parts = sort(filter(f -> startswith(f, "bootstrap_wildbg_100k_part") && endswith(f, ".jls"),
                        readdir(BOOTSTRAP_DIR)))

    if isempty(parts)
        error("No bootstrap part files found in $BOOTSTRAP_DIR")
    end
    println("Found $(length(parts)) bootstrap part files")

    all_positions = Tuple{UInt128, UInt128, Int8}[]
    total_games = 0
    total_race = 0

    for (i, part) in enumerate(parts)
        path = joinpath(BOOTSTRAP_DIR, part)
        println("\n[$i/$(length(parts))] Loading $part...")
        flush(stdout)

        t0 = time()
        data = Serialization.deserialize(path)
        t_load = time() - t0
        println("  Loaded in $(round(t_load, digits=1))s")

        states = data.states
        n_games = 0
        n_race = 0

        # Group consecutive positions into games by tracking game boundaries.
        # Each game's positions are sequential; a new game starts when the
        # position looks like an opening (or we just scan all positions and
        # take distinct race entries).
        #
        # Actually: bootstrap states are individual positions from games, not
        # grouped by game. Just scan each position and collect race ones.
        # Deduplicate by (p0, p1, cp) to get unique starting positions.

        seen = Set{Tuple{UInt128, UInt128, Int8}}()
        for state in states
            if is_race_position(state)
                key = (state.p0, state.p1, state.current_player)
                if key ∉ seen
                    push!(seen, key)
                    push!(all_positions, key)
                    n_race += 1
                end
            end
        end

        println("  $(length(states)) positions → $n_race unique race positions")
        total_games += length(states)
        total_race += n_race

        # Free memory
        data = nothing
        GC.gc()
    end

    println("\n=== Summary ===")
    println("Total positions scanned: $total_games")
    println("Unique race positions: $(length(all_positions))")

    # Deduplicate globally (across parts)
    unique_positions = unique(all_positions)
    println("After global dedup: $(length(unique_positions))")

    # Shuffle for training randomness
    shuffle!(MersenneTwister(42), unique_positions)

    # Stats
    pips0 = [pip_count(p[1], p[2])[1] for p in unique_positions]
    pips1 = [pip_count(p[1], p[2])[2] for p in unique_positions]
    on0 = [checkers_on_board(p[1]) for p in unique_positions]
    on1 = [checkers_on_board(p[2]) for p in unique_positions]

    println("\n=== Position Statistics ===")
    println("P0 pips: mean=$(round(mean(pips0), digits=1)), min=$(minimum(pips0)), max=$(maximum(pips0))")
    println("P1 pips: mean=$(round(mean(pips1), digits=1)), min=$(minimum(pips1)), max=$(maximum(pips1))")
    println("P0 checkers: mean=$(round(mean(on0), digits=1)), min=$(minimum(on0)), max=$(maximum(on0))")
    println("P1 checkers: mean=$(round(mean(on1), digits=1)), min=$(minimum(on1)), max=$(maximum(on1))")

    # Save full set
    output_path = joinpath(OUTPUT_DIR, "race_starts_tuples_bootstrap.jls")
    Serialization.serialize(output_path, unique_positions)
    println("\nSaved $(length(unique_positions)) positions to: $output_path")
    println("File size: $(round(filesize(output_path)/1e6, digits=1)) MB")

    # Optionally create training set excluding eval positions
    if !isempty(EVAL_EXCLUDE_FILE)
        eval_path = joinpath(OUTPUT_DIR, EVAL_EXCLUDE_FILE)
        if isfile(eval_path)
            eval_pos = Set(Serialization.deserialize(eval_path))
            training = filter(p -> p ∉ eval_pos, unique_positions)
            train_path = joinpath(OUTPUT_DIR, "race_starts_tuples_bootstrap_no_eval.jls")
            Serialization.serialize(train_path, training)
            println("Training set (eval excluded): $(length(training)) positions → $train_path")
        else
            println("WARNING: Eval exclude file not found: $eval_path")
        end
    end
end

main()
