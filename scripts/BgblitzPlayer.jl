# BgblitzPlayer.jl - AlphaZero-compatible BGBlitz player
#
# This wraps the validated BgblitzInterface from BackgammonNet.jl
# which manages the BGBlitz TCP server, XGID conversion, and move parsing.
#
# Server mode (recommended for eval):
#   BgblitzPlayer.start_server(slots=4, ply=0)
#   conn = BgblitzPlayer.take_connection()
#   action = BgblitzPlayer.best_move(conn, game_env)
#   BgblitzPlayer.return_connection(conn)
#   BgblitzPlayer.stop_server()
#
# Subprocess mode (backward compat):
#   player = BgblitzFast(ply=0)
#   AlphaZero.think(player, env)

module BgblitzPlayer

using BackgammonNet

# Include the validated BgblitzInterface from BackgammonNet.jl
const BACKGAMMON_NET_PATH = dirname(dirname(pathof(BackgammonNet)))
include(joinpath(BACKGAMMON_NET_PATH, "test", "BgblitzInterface.jl"))
using .BgblitzInterface

export BgblitzFast

# Re-export key functions from BgblitzInterface
const start_server = BgblitzInterface.start_server
const stop_server = BgblitzInterface.stop_server
const take_connection = BgblitzInterface.take_connection
const return_connection = BgblitzInterface.return_connection

"""
    best_move(conn::TCPSocket, env) -> Int

Get BGBlitz's best move for the current position. Returns an action index.
Unwraps the AlphaZero game env to get the underlying BackgammonGame.
"""
best_move(conn, env) = BgblitzInterface.best_move(conn, env.game)

# =============================================================================
# AlphaZero Player Interface (subprocess mode)
# =============================================================================

import AlphaZero: AbstractPlayer, think, reset!

"""
BGBlitz player using TachiAI neural net evaluator.

    BgblitzFast()        # 0-ply (neural net only)
    BgblitzFast(ply=2)   # 2-ply lookahead
"""
struct BgblitzFast <: AbstractPlayer
    ply::Int
end

BgblitzFast(; ply::Int=0) = BgblitzFast(ply)

function think(p::BgblitzFast, game)
    bg_game = game.game
    action = BgblitzInterface.best_move_subprocess(bg_game; ply=p.ply)

    num_actions = 676
    π = zeros(num_actions)
    if 1 <= action <= num_actions
        π[action] = 1.0
    end

    return collect(1:num_actions), π
end

reset!(::BgblitzFast) = nothing

# =============================================================================
# Benchmark Interface
# =============================================================================

import AlphaZero.Benchmark

struct BgblitzFastBenchmark <: Benchmark.Player
    ply::Int
end

BgblitzFastBenchmark(; ply::Int=0) = BgblitzFastBenchmark(ply)

Benchmark.name(p::BgblitzFastBenchmark) = "BGBlitz-$(p.ply)ply"

function Benchmark.instantiate(p::BgblitzFastBenchmark, ::Any, nn)
    return BgblitzFast(p.ply)
end

end # module
