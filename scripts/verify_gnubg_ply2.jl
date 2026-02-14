#!/usr/bin/env julia
# Verify gnubg board encoding differs per position

ENV["BACKGAMMON_OBS_TYPE"] = "minimal"
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
using BackgammonNet
include(joinpath(@__DIR__, "GnubgPlayer.jl"))
using .GnubgPlayer: GnubgInterface
using AlphaZero: GI
using Random

GnubgInterface._init()

gspec = GameSpec()

for seed in [42, 123, 456]
    rng = MersenneTwister(seed)
    env = GI.init(gspec)
    env.rng = rng

    # Advance to decision state
    while GI.is_chance_node(env)
        outcomes = GI.chance_outcomes(env)
        r = rand(rng)
        acc = 0.0
        idx = length(outcomes)
        for i in eachindex(outcomes)
            acc += outcomes[i][2]
            if r <= acc; idx = i; break; end
        end
        GI.apply_chance!(env, outcomes[idx][1])
    end

    # Make a move, then advance again to get a different position
    actions = GI.available_actions(env)
    GI.play!(env, actions[1])
    while GI.is_chance_node(env)
        outcomes = GI.chance_outcomes(env)
        r = rand(rng)
        acc = 0.0
        idx = length(outcomes)
        for i in eachindex(outcomes)
            acc += outcomes[i][2]
            if r <= acc; idx = i; break; end
        end
        GI.apply_chance!(env, outcomes[idx][1])
    end

    game = env.game
    board = GnubgInterface._to_gnubg_board(game)
    println("Seed=$seed, Dice=$(game.dice), Player=$(game.current_player)")
    println("  Board[opp]: $(board[1])")
    println("  Board[onr]: $(board[2])")

    # Call probabilities directly
    probs = GnubgInterface._gnubg.probabilities(board, 0)
    println("  0-ply probs: $(Float64.([probs[i] for i in 1:5]))")

    # Try calling with board as tuple of tuples (python native)
    probs2 = GnubgInterface._gnubg.probabilities(board, 1)
    println("  1-ply probs: $(Float64.([probs2[i] for i in 1:5]))")
    println()
end
