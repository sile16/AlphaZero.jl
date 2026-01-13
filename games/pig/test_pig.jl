# Test script for Pig game with stochastic MCTS
# Run from the AlphaZero.jl root directory:
#   julia --project=. games/pig/test_pig.jl

using AlphaZero

# Include the Pig game
include("main.jl")
using .Pig

println("=" ^ 60)
println("Testing Pig Game with Stochastic MCTS")
println("=" ^ 60)

# Create game
gspec = GameSpec()
game = GI.init(gspec)

println("\n1. Initial state:")
GI.render(game)

println("\n2. Testing actions mask (should be [true, true] - roll and hold):")
println("   Actions mask: ", GI.actions_mask(game))

println("\n3. Testing ROLL action (should transition to chance node):")
GI.play!(game, Pig.ROLL)
GI.render(game)
println("   Is chance node: ", GI.is_chance_node(game))
println("   Chance outcomes: ", GI.chance_outcomes(game))

println("\n4. Applying chance outcome (die roll = 4):")
GI.apply_chance!(game, 4)
GI.render(game)
println("   Is chance node: ", GI.is_chance_node(game))
println("   Turn total should be 4")

println("\n5. Testing HOLD action:")
GI.play!(game, Pig.HOLD)
GI.render(game)
println("   Player 1 score should be 4, turn switches to Player 2")

println("\n6. Simulating a few more turns...")
# Player 2 rolls
GI.play!(game, Pig.ROLL)
GI.apply_chance!(game, 6)
GI.play!(game, Pig.ROLL)
GI.apply_chance!(game, 5)
GI.play!(game, Pig.HOLD)
GI.render(game)

println("\n7. Testing MCTS with random oracle:")
game2 = GI.init(gspec)
oracle = MCTS.RolloutOracle(gspec)
mcts_env = MCTS.Env(gspec, oracle, cpuct=1.0, noise_ϵ=0.0)

println("   Running 100 MCTS simulations...")
MCTS.explore!(mcts_env, game2, 100)

println("   Decision tree size: ", length(mcts_env.tree))
println("   Chance tree size: ", length(mcts_env.chance_tree))
println("   Total chance nodes expanded: ", mcts_env.total_chance_nodes_expanded)

actions, policy = MCTS.policy(mcts_env, game2)
println("   MCTS policy: ROLL=$(round(policy[1], digits=3)), HOLD=$(round(policy[2], digits=3))")

println("\n8. Playing a full game with random MCTS player...")
player = MctsPlayer(mcts_env, niters=50, τ=ConstSchedule(0.5))
trace = play_game(gspec, TwoPlayers(player, player))
println("   Game length: ", length(trace), " transitions")
println("   Final state:")
final_game = GI.init(gspec, trace.states[end])
GI.render(final_game)

# Count chance vs decision nodes in trace
num_chance = sum(trace.is_chance)
num_decision = length(trace.is_chance) - num_chance
println("   Chance nodes in trace: ", num_chance)
println("   Decision nodes in trace: ", num_decision)

println("\n" * "=" ^ 60)
println("All tests passed!")
println("=" ^ 60)
