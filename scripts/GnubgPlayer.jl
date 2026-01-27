# GnubgPlayer.jl - AlphaZero-compatible GnuBG player
#
# This wraps the validated GnubgInterface from BackgammonNet.jl
# which uses PyCall for fast gnubg integration (~48k evals/sec at 0-ply).
#
# Usage:
#   include("scripts/GnubgPlayer.jl")
#   using .GnubgPlayer
#
#   gnubg = GnubgBaseline()  # or GnubgBaseline(ply=1) for 1-ply search
#   action, equity = best_move(game)

module GnubgPlayer

using BackgammonNet

# Include the validated GnubgInterface from BackgammonNet.jl
const BACKGAMMON_NET_PATH = dirname(dirname(pathof(BackgammonNet)))
include(joinpath(BACKGAMMON_NET_PATH, "test", "GnubgInterface.jl"))
using .GnubgInterface

export evaluate_position, best_move, GnubgBaseline
export set_default_ply!, get_default_ply

# Re-export key functions from GnubgInterface
const evaluate_position = GnubgInterface.evaluate
const best_move = GnubgInterface.best_move

# AlphaZero player interface
import AlphaZero: AbstractPlayer, think, reset!

"""
AlphaZero-compatible player that uses GnuBG's neural network.

    GnubgBaseline()        # 0-ply (neural net only, fastest)
    GnubgBaseline(ply=1)   # 1-ply lookahead
    GnubgBaseline(ply=2)   # 2-ply lookahead (slower but stronger)

Performance (approximate):
- 0-ply: ~48,000 evals/sec
- 1-ply: ~35,000 evals/sec
- 2-ply: ~35 evals/sec
"""
struct GnubgBaseline <: AbstractPlayer
    ply::Int
end

GnubgBaseline(; ply::Int=0) = GnubgBaseline(ply)

function think(p::GnubgBaseline, game)
    # Get the underlying BackgammonGame
    bg_game = game.game

    # Use gnubg to get the best move
    action, _ = best_move(bg_game; ply=p.ply)

    # Create policy with all probability on the chosen action
    num_actions = 676  # Backgammon action space
    π = zeros(num_actions)
    if 1 <= action <= num_actions
        π[action] = 1.0
    end

    return collect(1:num_actions), π
end

reset!(::GnubgBaseline) = nothing

# Benchmark player interface
import AlphaZero.Benchmark

struct GnubgBenchmark <: Benchmark.Player
    ply::Int
end

GnubgBenchmark(; ply::Int=0) = GnubgBenchmark(ply)

Benchmark.name(p::GnubgBenchmark) = "GnuBG-$(p.ply)ply"

function Benchmark.instantiate(p::GnubgBenchmark, ::Any, nn)
    return GnubgBaseline(p.ply)
end

end # module
