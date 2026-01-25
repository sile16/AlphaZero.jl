#!/usr/bin/env julia
#####
##### Evaluate current iteration of a training session
##### Same game counts as baseline benchmark (50 games per position)
#####

using Pkg
Pkg.activate(dirname(@__DIR__))

using AlphaZero
using AlphaZero.FluxLib
using Printf
using Dates
using Serialization

# Configuration - matching baseline benchmark
const GAMES_PER_POSITION = 50  # 50 games as AZ first, 50 as random first = 100 total
const MCTS_ITERS = 100

# Use deterministic backgammon
const GAMES_DIR = joinpath(@__DIR__, "..", "games")
include(joinpath(GAMES_DIR, "backgammon-deterministic", "main.jl"))
using .BackgammonDeterministic
const GameModule = BackgammonDeterministic

# Reward histogram
mutable struct RewardHistogram
    counts::Dict{Int, Int}
    total_games::Int
    total_reward::Float64
    RewardHistogram() = new(Dict(-3=>0, -2=>0, -1=>0, 1=>0, 2=>0, 3=>0), 0, 0.0)
end

function record!(hist::RewardHistogram, reward::Float64)
    r = clamp(round(Int, reward), -3, 3)
    r = r == 0 ? (reward > 0 ? 1 : -1) : r
    hist.counts[r] = get(hist.counts, r, 0) + 1
    hist.total_games += 1
    hist.total_reward += reward
end

function print_histogram(hist::RewardHistogram, label::String)
    println("\n$label:")
    total = hist.total_games
    total == 0 && return

    for (r, name) in [(-3,"BG Loss"), (-2,"G Loss"), (-1,"Loss"), (1,"Win"), (2,"G Win"), (3,"BG Win")]
        count = get(hist.counts, r, 0)
        pct = 100.0 * count / total
        bar = "█" ^ round(Int, pct / 2)
        @printf("  %8s (%+d): %3d (%5.1f%%) %s\n", name, r, count, pct, bar)
    end
    @printf("  Avg reward: %+.3f\n", hist.total_reward / total)
end

function main(session_dir::String)
    println("=" ^ 60)
    println("EVALUATION OF CURRENT ITERATION")
    println("=" ^ 60)
    println("Session: $session_dir")
    println("Time: $(now())")

    # Get current iteration
    iter_file = joinpath(session_dir, "iter.txt")
    current_iter = parse(Int, strip(read(iter_file, String)))
    println("Current iteration: $current_iter")

    # Load network directly from BSON
    gspec = GameModule.GameSpec()

    nn_path = joinpath(session_dir, "bestnn.data")
    println("Loading network from: $nn_path")

    nn = deserialize(nn_path)
    nn = Network.copy(nn, on_gpu=true, test_mode=true)
    println("Network loaded: $(Network.num_parameters(nn)) parameters")

    # Create MCTS player
    mcts_params = MctsParams(
        num_iters_per_turn = MCTS_ITERS,
        cpuct = 1.5,
        temperature = ConstSchedule(0.2),
        dirichlet_noise_ϵ = 0.0,
        dirichlet_noise_α = 0.3
    )

    az_player = MctsPlayer(gspec, nn, mcts_params)
    random_player = GameModule.RandomPlayer()

    # Histograms
    hist_az_first = RewardHistogram()
    hist_random_first = RewardHistogram()

    println("\n" * "-" ^ 60)
    println("Benchmark 1: AZ plays first ($GAMES_PER_POSITION games)")
    println("-" ^ 60)

    for i in 1:GAMES_PER_POSITION
        trace = play_game(gspec, TwoPlayers(az_player, random_player), flip_probability=0.0)
        r = total_reward(trace)
        record!(hist_az_first, r)
        if i % 10 == 0
            @printf("  Progress: %d/%d, avg reward: %+.3f\n",
                    i, GAMES_PER_POSITION, hist_az_first.total_reward / i)
        end
    end

    println("\n" * "-" ^ 60)
    println("Benchmark 2: Random plays first ($GAMES_PER_POSITION games)")
    println("-" ^ 60)

    for i in 1:GAMES_PER_POSITION
        trace = play_game(gspec, TwoPlayers(random_player, az_player), flip_probability=0.0)
        r = total_reward(trace)
        # Negate reward since we want AZ's perspective
        record!(hist_random_first, -r)
        if i % 10 == 0
            @printf("  Progress: %d/%d, avg reward: %+.3f\n",
                    i, GAMES_PER_POSITION, hist_random_first.total_reward / i)
        end
    end

    # Results
    println("\n" * "=" ^ 60)
    println("RESULTS - Iteration $current_iter")
    println("=" ^ 60)

    avg_az_first = hist_az_first.total_reward / GAMES_PER_POSITION
    avg_random_first = hist_random_first.total_reward / GAMES_PER_POSITION
    combined_avg = (hist_az_first.total_reward + hist_random_first.total_reward) / (2 * GAMES_PER_POSITION)

    @printf("\nAZ plays first:     avg reward = %+.3f\n", avg_az_first)
    @printf("Random plays first: avg reward = %+.3f\n", avg_random_first)
    @printf("Combined average:   avg reward = %+.3f\n", combined_avg)

    print_histogram(hist_az_first, "AZ First - Reward Distribution")
    print_histogram(hist_random_first, "Random First - Reward Distribution")

    # Combined histogram
    combined = RewardHistogram()
    for r in [-3, -2, -1, 1, 2, 3]
        combined.counts[r] = get(hist_az_first.counts, r, 0) + get(hist_random_first.counts, r, 0)
    end
    combined.total_games = hist_az_first.total_games + hist_random_first.total_games
    combined.total_reward = hist_az_first.total_reward + hist_random_first.total_reward
    print_histogram(combined, "Combined - Reward Distribution")

    # Comparison with baseline
    println("\n" * "-" ^ 60)
    println("COMPARISON WITH SIMPLENET BASELINE (iter 128)")
    println("-" ^ 60)
    println("                    SimpleNet    MultiHead (iter $current_iter)")
    @printf("AZ first:           %+.2f         %+.2f\n", 0.46, avg_az_first)
    @printf("Random first:       %+.2f         %+.2f\n", 1.76, avg_random_first)
    @printf("Combined:           %+.2f         %+.2f\n", 1.11, combined_avg)

    println("\n" * "=" ^ 60)
    println("Evaluation complete: $(now())")
    println("=" ^ 60)
end

# Run
if length(ARGS) < 1
    # Default to current multihead session
    sessions = filter(s -> startswith(s, "bg-multihead-baseline"), readdir("sessions"))
    if !isempty(sessions)
        session_dir = joinpath("sessions", sort(sessions)[end])
        main(session_dir)
    else
        println("Usage: julia eval_current_iteration.jl <session_dir>")
    end
else
    main(ARGS[1])
end
