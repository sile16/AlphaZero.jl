# Benchmark script to test GnuBG evaluation speed
# Tests: gnubg CLI calls per second, games per second

using Printf
using Statistics
using Random

# Load GnubgPlayer
include(joinpath(@__DIR__, "GnubgPlayer.jl"))
using .GnubgPlayer
using BackgammonNet

println("=" ^ 60)
println("GnuBG CLI Interface Benchmark")
println("=" ^ 60)

#####
##### Test 1: Raw CLI call speed (best_move)
#####

println("\n### Test 1: best_move() calls per second ###")

g = BackgammonNet.initial_state(; short_game=true, doubles_only=true)
BackgammonNet.apply_chance!(g, 1)  # Roll 1-1

# Warmup
GnubgPlayer.best_move(g)

# Benchmark
n_calls = 20
start = time()
for i in 1:n_calls
    GnubgPlayer.best_move(g)
end
elapsed = time() - start

calls_per_sec = n_calls / elapsed
@printf("  %d calls in %.2f seconds\n", n_calls, elapsed)
@printf("  Rate: %.2f calls/sec (%.1f ms/call)\n", calls_per_sec, 1000/calls_per_sec)

#####
##### Test 2: Play games with GnuBG vs Random
#####

println("\n### Test 2: Games per second (GnuBG vs Random) ###")

function play_game_gnubg_vs_random(; seed=1)
    rng = Random.MersenneTwister(seed)
    g = BackgammonNet.initial_state(; short_game=true, doubles_only=true)
    moves = 0

    while !BackgammonNet.game_terminated(g)
        if BackgammonNet.is_chance_node(g)
            BackgammonNet.sample_chance!(g, rng)
        else
            cp = Int(g.current_player)
            if cp == 0  # GnuBG plays as P0
                action, _ = GnubgPlayer.best_move(g)
            else  # Random plays as P1
                actions = BackgammonNet.legal_actions(g)
                action = actions[rand(rng, 1:length(actions))]
            end
            BackgammonNet.apply_action!(g, action)
            moves += 1
        end
    end

    return g.reward, moves
end

# Warmup
play_game_gnubg_vs_random(seed=0)

# Benchmark
n_games = 5
total_moves = 0
start = time()
for i in 1:n_games
    _, moves = play_game_gnubg_vs_random(seed=i)
    total_moves += moves
end
elapsed = time() - start

games_per_sec = n_games / elapsed
moves_per_game = total_moves / n_games
@printf("  %d games in %.2f seconds\n", n_games, elapsed)
@printf("  Rate: %.3f games/sec (%.1f sec/game)\n", games_per_sec, 1/games_per_sec)
@printf("  Avg moves per game: %.1f\n", moves_per_game)
@printf("  GnuBG calls per game: ~%.1f (one per P0 turn)\n", moves_per_game / 2)

#####
##### Test 3: Play games with Random vs Random (baseline)
#####

println("\n### Test 3: Games per second (Random vs Random - baseline) ###")

function play_game_random_vs_random(; seed=1)
    rng = Random.MersenneTwister(seed)
    g = BackgammonNet.initial_state(; short_game=true, doubles_only=true)
    moves = 0

    while !BackgammonNet.game_terminated(g)
        if BackgammonNet.is_chance_node(g)
            BackgammonNet.sample_chance!(g, rng)
        else
            actions = BackgammonNet.legal_actions(g)
            action = actions[rand(rng, 1:length(actions))]
            BackgammonNet.apply_action!(g, action)
            moves += 1
        end
    end

    return g.reward, moves
end

# Benchmark
n_games_rand = 100
start = time()
for i in 1:n_games_rand
    play_game_random_vs_random(seed=i)
end
elapsed = time() - start

games_per_sec_rand = n_games_rand / elapsed
@printf("  %d games in %.2f seconds\n", n_games_rand, elapsed)
@printf("  Rate: %.1f games/sec\n", games_per_sec_rand)

#####
##### Estimate: Time for 1000 games evaluation
#####

println("\n### Estimates for Evaluation ###")
println("-" ^ 40)

# GnuBG is only called for half the moves (one player)
# But if both players use MCTS, no gnubg calls needed
gnubg_time_per_1000 = 1000 / games_per_sec
random_time_per_1000 = 1000 / games_per_sec_rand

@printf("GnuBG vs Random (1000 games): %.1f minutes\n", gnubg_time_per_1000 / 60)
@printf("Random vs Random (1000 games): %.1f seconds\n", random_time_per_1000)
@printf("\nGnuBG slowdown factor: %.0fx\n", games_per_sec_rand / games_per_sec)

println("\n### Recommendations ###")
println("-" ^ 40)
println("1. GnuBG CLI is slow (~0.5-1 games/sec)")
println("2. For statistical significance with 1000+ games,")
println("   GnuBG eval will take 15-30+ minutes per matchup")
println("3. Consider reducing GnuBG eval games or running overnight")
println("4. AlphaZero vs Random is much faster (use MCTS)")
