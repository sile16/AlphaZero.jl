#!/usr/bin/env julia
# Compare 1-ply vs 2-ply move choices in a game to verify they differ

ENV["BACKGAMMON_OBS_TYPE"] = "minimal"
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
using BackgammonNet
include(joinpath(@__DIR__, "GnubgPlayer.jl"))
using .GnubgPlayer: GnubgInterface
using AlphaZero: GI
using Random

GnubgInterface._init()
gspec = GameSpec()

# Play a game with both plies making decisions, compare
function play_and_compare(seed)
    rng = MersenneTwister(seed)
    env = GI.init(gspec)
    env.rng = rng

    differences = 0
    total_decisions = 0

    while !GI.game_terminated(env)
        if GI.is_chance_node(env)
            outcomes = GI.chance_outcomes(env)
            r = rand(rng)
            acc = 0.0
            idx = length(outcomes)
            for i in eachindex(outcomes)
                acc += outcomes[i][2]
                if r <= acc; idx = i; break; end
            end
            GI.apply_chance!(env, outcomes[idx][1])
            continue
        end

        actions = GI.available_actions(env)
        if length(actions) <= 1
            GI.play!(env, actions[1])
            continue
        end

        game = env.game
        total_decisions += 1

        # Get best move at both plies
        action_1ply, eq_1ply = GnubgInterface.best_move(game; ply=1)
        action_2ply, eq_2ply = GnubgInterface.best_move(game; ply=2)

        if action_1ply != action_2ply
            differences += 1
            if differences <= 3  # Print first 3 differences
                println("  Diff at decision $total_decisions: 1ply=$action_1ply (eq=$(round(eq_1ply, digits=4))), 2ply=$action_2ply (eq=$(round(eq_2ply, digits=4)))")
            end
        end

        # Play the 1-ply move (doesn't matter which, just need to advance)
        GI.play!(env, action_1ply)
    end

    return total_decisions, differences
end

println("Comparing 1-ply vs 2-ply move choices...\n")
total_d = 0
total_diff = 0
for seed in 1:5
    println("Game $seed (seed=$seed):")
    decisions, diffs = play_and_compare(seed)
    println("  $diffs/$decisions decisions differed ($(round(100*diffs/max(decisions,1), digits=1))%)\n")
    total_d += decisions
    total_diff += diffs
end
println("Overall: $total_diff/$total_d decisions differed ($(round(100*total_diff/max(total_d,1), digits=1))%)")
