#!/usr/bin/env julia
"""
Convert wildbg bootstrap files (columnar NamedTuples with BackgammonGame states)
into per-sample training format for training_server.jl.

Input: ~/github/BackgammonNet.jl/data/bootstrap/bootstrap_wildbg_100k_part{0-9}.jls
Output: /homeshare/projects/AlphaZero.jl/eval_data/bootstrap_wildbg_1M.jls

Each output sample is a NamedTuple with:
  state::Vector{Float32}   (344-dim minimal_flat)
  policy::Vector{Float32}  (676-dim)
  value::Float32
  equity::Vector{Float32}  (5-element joint cumulative)
  has_equity::Bool
  is_chance::Bool
  is_contact::Bool
  is_bearoff::Bool
"""

using Serialization
using Random

# Set up game environment for vectorization
ENV["BACKGAMMON_OBS_TYPE"] = "minimal_flat"

# Load AlphaZero + BackgammonNet
println("Loading packages..."); flush(stdout)
t0 = time()
using AlphaZero
using AlphaZero: GI
import BackgammonNet

# Include game.jl for GameSpec and vectorize_state
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
const gspec = GameSpec()
println("Packages loaded in $(round(time()-t0, digits=1))s"); flush(stdout)

# Load k=7 bearoff module for is_bearoff_position
const BEAROFF_K7_PATH = joinpath(homedir(), "github", "BackgammonNet.jl", "src", "bearoff_k7.jl")
isfile(BEAROFF_K7_PATH) || error("bearoff_k7.jl not found at: $BEAROFF_K7_PATH")
include(BEAROFF_K7_PATH)
using .BearoffK7

const INPUT_DIR = joinpath(homedir(), "github", "BackgammonNet.jl", "data", "bootstrap")
const OUTPUT_PATH = "/homeshare/projects/AlphaZero.jl/eval_data/bootstrap_wildbg_1M.jls"
const NUM_PARTS = 10
const NUM_ACTIONS_CONST = 676
const TARGET_TOTAL = 1_000_000
const PER_PART = TARGET_TOTAL ÷ NUM_PARTS

Random.seed!(42)

function convert_part(part::Int, data, sample_indices)
    n_contact = 0; n_race = 0; n_bearoff = 0
    samples = Vector{NamedTuple{(:state, :policy, :value, :equity, :has_equity, :is_chance, :is_contact, :is_bearoff),
                                 Tuple{Vector{Float32}, Vector{Float32}, Float32, Vector{Float32}, Bool, Bool, Bool, Bool}}}()
    sizehint!(samples, length(sample_indices))

    for (count, i) in enumerate(sample_indices)
        game = data.states[i]

        # Vectorize state
        state_vec = Vector{Float32}(vec(GI.vectorize_state(gspec, game)))

        # Policy
        raw_policy = data.policies[i]
        policy = Vector{Float32}(undef, NUM_ACTIONS_CONST)
        len = min(length(raw_policy), NUM_ACTIONS_CONST)
        for j in 1:len
            @inbounds policy[j] = Float32(raw_policy[j])
        end
        for j in (len+1):NUM_ACTIONS_CONST
            @inbounds policy[j] = 0.0f0
        end

        # Value
        value = Float32(data.values[i])

        # Equity (from NTuple{5,Float64})
        eq_tuple = data.equity[i]
        equity = Float32[eq_tuple[1], eq_tuple[2], eq_tuple[3], eq_tuple[4], eq_tuple[5]]

        # Classification
        is_contact = BackgammonNet.is_contact_position(game)
        is_bearoff = BearoffK7.is_bearoff_position(game.p0, game.p1)

        if is_contact
            n_contact += 1
        elseif is_bearoff
            n_bearoff += 1
        else
            n_race += 1
        end

        push!(samples, (
            state = state_vec,
            policy = policy,
            value = value,
            equity = equity,
            has_equity = true,
            is_chance = false,
            is_contact = is_contact,
            is_bearoff = is_bearoff,
        ))

        if count % 10000 == 0
            print("    $(count)/$(length(sample_indices))...\r")
            flush(stdout)
        end
    end
    println("    $(length(sample_indices))/$(length(sample_indices)) done [contact=$n_contact, race=$n_race, bearoff=$n_bearoff]")
    flush(stdout)
    return samples, n_contact, n_race, n_bearoff
end

println("\nConverting bootstrap data (subsampling to ~$(TARGET_TOTAL) samples)...")
println("Input: $INPUT_DIR")
println("Output: $OUTPUT_PATH")
flush(stdout)

all_samples = NamedTuple[]
sizehint!(all_samples, TARGET_TOTAL)
total_contact = 0
total_race = 0
total_bearoff = 0

for part in 0:(NUM_PARTS - 1)
    part_path = joinpath(INPUT_DIR, "bootstrap_wildbg_100k_part$part.jls")
    if !isfile(part_path)
        println("  WARNING: Part $part not found, skipping"); flush(stdout)
        continue
    end

    t1 = time()
    println("  Loading part $part..."); flush(stdout)
    data = Serialization.deserialize(part_path)
    n = length(data.states)
    t_load = round(time() - t1, digits=1)
    println("  Part $part: $n positions ($(t_load)s)"); flush(stdout)

    # Subsample
    sample_indices = sort(randperm(n)[1:min(PER_PART, n)])
    println("    Converting $(length(sample_indices)) samples..."); flush(stdout)

    t2 = time()
    samples, nc, nr, nb = convert_part(part, data, sample_indices)
    t_conv = round(time() - t2, digits=1)
    println("    Converted in $(t_conv)s"); flush(stdout)

    append!(all_samples, samples)
    global total_contact += nc
    global total_race += nr
    global total_bearoff += nb

    # Free memory
    data = nothing
    samples = nothing
    GC.gc()
end

println("\nTotal samples: $(length(all_samples))")
println("  Contact: $total_contact")
println("  Race (non-bearoff): $total_race")
println("  Bearoff (k=7): $total_bearoff")
flush(stdout)

# Save
println("\nSaving to $OUTPUT_PATH ...")
flush(stdout)
t3 = time()
Serialization.serialize(OUTPUT_PATH, all_samples)
file_size = filesize(OUTPUT_PATH) / 1e9
println("Saved $(length(all_samples)) samples ($(round(file_size, digits=2)) GB) in $(round(time()-t3, digits=1))s")
println("Done!")
flush(stdout)
