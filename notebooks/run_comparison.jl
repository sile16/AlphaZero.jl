#=
Stochastic AlphaZero Comparison Script

Compares three approaches on the Game of Pig:
1. Standard AlphaZero - Hidden stochasticity (dice in environment)
2. Stochastic AlphaZero - Explicit chance nodes with expectimax
3. Hold20 Baseline - Heuristic strategy

Usage:
    julia --project=. notebooks/run_comparison.jl

For Colab/JuliaHub, first install:
    using Pkg
    Pkg.add(url="https://github.com/sile16/AlphaZero.jl", rev="stochastic-mcts")
=#

using AlphaZero
using Random
using Statistics
using Printf
using Dates

#####
##### Deterministic Pig (hidden stochasticity)
#####

module DeterministicPig

import AlphaZero.GI
using Random

const TARGET_SCORE = 100
const ROLL = 1
const HOLD = 2
const WHITE = true
const BLACK = false

const State = @NamedTuple{
    p1_score::Int,
    p2_score::Int,
    turn_total::Int,
    curplayer::Bool
}

struct GameSpec <: GI.AbstractGameSpec end

GI.two_players(::GameSpec) = true
GI.actions(::GameSpec) = [ROLL, HOLD]
GI.num_chance_outcomes(::GameSpec) = 0

function GI.vectorize_state(::GameSpec, state)
    return Float32[
        state.p1_score / TARGET_SCORE,
        state.p2_score / TARGET_SCORE,
        state.turn_total / TARGET_SCORE,
        state.curplayer ? 1f0 : 0f0
    ]
end

mutable struct GameEnv <: GI.AbstractGameEnv
    state::State
end

GI.spec(::GameEnv) = GameSpec()
GI.current_state(g::GameEnv) = g.state
GI.set_state!(g::GameEnv, s) = (g.state = s)
GI.white_playing(g::GameEnv) = g.state.curplayer
GI.init(::GameSpec) = GameEnv(State((0, 0, 0, WHITE)))
GI.init(::GameSpec, s) = GameEnv(s)

function GI.game_terminated(g::GameEnv)
    return g.state.p1_score >= TARGET_SCORE || g.state.p2_score >= TARGET_SCORE
end

function GI.white_reward(g::GameEnv)
    g.state.p1_score >= TARGET_SCORE && return 1.0
    g.state.p2_score >= TARGET_SCORE && return -1.0
    return 0.0
end

GI.is_chance_node(::GameEnv) = false
GI.actions_mask(::GameEnv) = [true, true]

function GI.play!(g::GameEnv, action)
    s = g.state
    if action == ROLL
        die = rand(1:6)
        if die == 1
            g.state = State((s.p1_score, s.p2_score, 0, !s.curplayer))
        else
            g.state = State((s.p1_score, s.p2_score, s.turn_total + die, s.curplayer))
        end
    else  # HOLD
        if s.curplayer
            g.state = State((s.p1_score + s.turn_total, s.p2_score, 0, false))
        else
            g.state = State((s.p1_score, s.p2_score + s.turn_total, 0, true))
        end
    end
end

function GI.heuristic_value(g::GameEnv)
    s = g.state
    diff = s.curplayer ? (s.p1_score - s.p2_score) : (s.p2_score - s.p1_score)
    return (diff + s.turn_total) / TARGET_SCORE
end

end # module DeterministicPig

#####
##### Hold20 Player
#####

struct Hold20Player <: AbstractPlayer
    threshold::Int
end
Hold20Player() = Hold20Player(20)

function AlphaZero.think(p::Hold20Player, game)
    s = GI.current_state(game)
    turn_total = hasproperty(s, :turn_total) ? s.turn_total : s[3]
    π = turn_total >= p.threshold ? [0.0, 1.0] : [1.0, 0.0]
    return [1, 2], π
end
AlphaZero.reset!(::Hold20Player) = nothing

#####
##### Evaluation
#####

function evaluate_vs_hold20(gspec, player, num_games)
    hold20 = Hold20Player()
    wins = 0
    for i in 1:num_games
        if i % 2 == 1
            trace = play_game(gspec, TwoPlayers(player, hold20))
            final = GI.init(gspec, trace.states[end])
            wins += GI.white_reward(final) > 0 ? 1 : 0
        else
            trace = play_game(gspec, TwoPlayers(hold20, player))
            final = GI.init(gspec, trace.states[end])
            wins += GI.white_reward(final) < 0 ? 1 : 0
        end
    end
    return wins / num_games
end

#####
##### Training Loop
#####

function train_until_beats_hold20(gspec, name;
        max_iters=30,
        mcts_iters=50,
        self_play_games=50,
        eval_games=50,
        win_threshold=0.55)

    println("\n" * "="^60)
    println("Training: $name")
    println("="^60)

    results = (
        win_rates = Float64[],
        total_sims = Int[],
        iterations = Int[]
    )

    # Network
    netparams = NetLib.SimpleNetHP(
        width=64, depth_common=4,
        use_batch_norm=true, batch_norm_momentum=1.0
    )
    nn = NetLib.SimpleNet(gspec, netparams)

    # MCTS params
    mcts_params = MctsParams(
        num_iters_per_turn=mcts_iters,
        cpuct=1.0,
        temperature=ConstSchedule(1.0),
        dirichlet_noise_ϵ=0.25,
        dirichlet_noise_α=1.0
    )

    total_sims = 0

    for iter in 1:max_iters
        println("\nIteration $iter/$max_iters")

        # Self-play
        print("  Self-play ($self_play_games games)... ")
        flush(stdout)
        mcts_env = MCTS.Env(gspec, nn, cpuct=1.0, noise_ϵ=0.25, noise_α=1.0)
        player = MctsPlayer(mcts_env, niters=mcts_iters, τ=ConstSchedule(1.0))

        for _ in 1:self_play_games
            trace = play_game(gspec, TwoPlayers(player, player))
            total_sims += length(trace) * mcts_iters
        end
        println("done")

        # Evaluation
        print("  Evaluating vs Hold20 ($eval_games games)... ")
        flush(stdout)
        eval_player = MctsPlayer(mcts_env, niters=mcts_iters, τ=ConstSchedule(0.1))
        win_rate = evaluate_vs_hold20(gspec, eval_player, eval_games)
        @printf("%.1f%% win rate\n", win_rate * 100)

        # Record
        push!(results.win_rates, win_rate)
        push!(results.total_sims, total_sims)
        push!(results.iterations, iter)

        if win_rate >= win_threshold
            println("\n✓ Beat Hold20 at iteration $iter!")
            break
        end
    end

    return results
end

#####
##### Main
#####

function main()
    println("="^60)
    println("Stochastic AlphaZero Comparison")
    println("Started: $(now())")
    println("="^60)

    # Include stochastic Pig
    include(joinpath(@__DIR__, "..", "games", "pig", "main.jl"))

    # Configuration
    config = (
        max_iters = 20,
        mcts_iters = 50,
        self_play_games = 50,
        eval_games = 50,
        win_threshold = 0.55
    )

    println("\nConfiguration:")
    println("  Max iterations: $(config.max_iters)")
    println("  MCTS iterations/turn: $(config.mcts_iters)")
    println("  Self-play games/iter: $(config.self_play_games)")
    println("  Eval games: $(config.eval_games)")
    println("  Win threshold: $(config.win_threshold * 100)%")

    # Train Standard AlphaZero
    results_std = train_until_beats_hold20(
        DeterministicPig.GameSpec(),
        "Standard AlphaZero (hidden stochasticity)";
        config...
    )

    # Train Stochastic AlphaZero
    results_stoch = train_until_beats_hold20(
        Pig.GameSpec(),
        "Stochastic AlphaZero (explicit chance nodes)";
        config...
    )

    # Summary
    println("\n" * "="^60)
    println("RESULTS SUMMARY")
    println("="^60)

    println("\nStandard AlphaZero:")
    println("  Final win rate: $(round(results_std.win_rates[end] * 100, digits=1))%")
    println("  Iterations: $(length(results_std.iterations))")
    println("  Total simulations: $(results_std.total_sims[end])")

    println("\nStochastic AlphaZero:")
    println("  Final win rate: $(round(results_stoch.win_rates[end] * 100, digits=1))%")
    println("  Iterations: $(length(results_stoch.iterations))")
    println("  Total simulations: $(results_stoch.total_sims[end])")

    # Output CSV for plotting
    open("comparison_results.csv", "w") do f
        println(f, "type,iteration,win_rate,total_sims")
        for (i, (wr, ts)) in enumerate(zip(results_std.win_rates, results_std.total_sims))
            println(f, "standard,$i,$wr,$ts")
        end
        for (i, (wr, ts)) in enumerate(zip(results_stoch.win_rates, results_stoch.total_sims))
            println(f, "stochastic,$i,$wr,$ts")
        end
    end
    println("\nResults saved to: comparison_results.csv")

    println("\nCompleted: $(now())")
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
