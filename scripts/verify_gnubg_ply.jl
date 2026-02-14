#!/usr/bin/env julia
# Verify GnuBG ply levels are working correctly
# Compares evaluations at 0-ply, 1-ply, 2-ply for the same position

ENV["BACKGAMMON_OBS_TYPE"] = "minimal"
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))

using BackgammonNet

include(joinpath(@__DIR__, "GnubgPlayer.jl"))
using .GnubgPlayer: GnubgInterface

# Initialize gnubg
println("Initializing GnuBG...")
GnubgInterface._init()
println("GnuBG initialized\n")

gspec = GameSpec()

# Create a position and roll dice
using AlphaZero: GI
using Random

for seed in [42, 123, 456, 789, 1001]
    rng = MersenneTwister(seed)
    env = GI.init(gspec)
    env.rng = rng

    # Roll dice to get a decision state
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
    println("=" ^ 60)
    println("Seed=$seed, Player=$(game.current_player), Dice=$(game.dice)")

    # Get legal actions
    actions = BackgammonNet.legal_actions(game)
    println("Legal actions: $(length(actions))")

    # Evaluate position at different plies
    board = GnubgInterface._to_gnubg_board(game)

    for ply in [0, 1, 2]
        t0 = time()
        probs = GnubgInterface._gnubg.probabilities(board, ply)
        elapsed = time() - t0

        win = Float64(probs[1])
        wg = Float64(probs[2])
        wbg = Float64(probs[3])
        lg = Float64(probs[4])
        lbg = Float64(probs[5])
        equity = (win - (1.0 - win)) + (wg - lg) + 2.0 * (wbg - lbg)

        println("  $(ply)-ply: win=$(round(win, digits=4)), wg=$(round(wg, digits=4)), wbg=$(round(wbg, digits=4)), lg=$(round(lg, digits=4)), lbg=$(round(lbg, digits=4)), equity=$(round(equity, digits=4)) ($(round(elapsed*1000, digits=1))ms)")
    end

    # Also compare best_move at each ply
    println("\n  Best moves:")
    for ply in [0, 1, 2]
        t0 = time()
        action, eq = GnubgInterface.best_move(game; ply=ply)
        elapsed = time() - t0
        println("  $(ply)-ply: action=$action, equity=$(round(eq, digits=4)) ($(round(elapsed*1000, digits=1))ms)")
    end
    println()
end
