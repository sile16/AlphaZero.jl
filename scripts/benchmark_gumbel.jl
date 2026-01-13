"""
Benchmark script comparing standard MCTS vs Gumbel MCTS.

This script compares the two search algorithms on Connect Four at different
simulation budgets (50, 100, 200 simulations per move).

Run with: julia --project scripts/benchmark_gumbel.jl
"""

using AlphaZero
using Statistics: mean, std

# Load Connect Four
const GAME = "connect-four"
gspec = Examples.games[GAME]

println("=" ^ 60)
println("Gumbel MCTS vs Standard MCTS Benchmark")
println("Game: $GAME")
println("=" ^ 60)
println()

# Simulation budgets to test
const BUDGETS = [50, 100, 200]

# Number of games per comparison
const NUM_GAMES = 100

# Common simulation parameters
const NUM_WORKERS = 32
const BATCH_SIZE = 32

# Create a rollout oracle for testing without a neural network
oracle = MCTS.RolloutOracle(gspec)

function run_duel(player1, player2, num_games)
  """Run games between two players and return win statistics."""
  wins_p1 = 0
  wins_p2 = 0
  draws = 0

  for i in 1:num_games
    # Alternate who plays first
    if i % 2 == 1
      white, black = player1, player2
    else
      white, black = player2, player1
    end

    combined = TwoPlayers(white, black)
    trace = play_game(gspec, combined)
    reward = total_reward(trace)

    if i % 2 == 1
      # player1 was white
      if reward > 0
        wins_p1 += 1
      elseif reward < 0
        wins_p2 += 1
      else
        draws += 1
      end
    else
      # player2 was white
      if reward > 0
        wins_p2 += 1
      elseif reward < 0
        wins_p1 += 1
      else
        draws += 1
      end
    end

    # Reset players for next game
    reset_player!(player1)
    reset_player!(player2)
  end

  return (wins_p1=wins_p1, wins_p2=wins_p2, draws=draws)
end

function benchmark_budget(budget::Int)
  println("-" ^ 60)
  println("Simulation Budget: $budget")
  println("-" ^ 60)

  # Create MCTS player
  mcts_params = MctsParams(
    num_iters_per_turn=budget,
    cpuct=1.0,
    temperature=ConstSchedule(0.1),  # Low temperature for strong play
    dirichlet_noise_ϵ=0.0,  # No noise for evaluation
    dirichlet_noise_α=1.0
  )
  mcts_player = MctsPlayer(gspec, oracle, mcts_params)

  # Create Gumbel MCTS player
  gumbel_params = GumbelMctsParams(
    num_simulations=budget,
    max_considered_actions=7,  # Connect Four has 7 columns
    temperature=ConstSchedule(0.1),
    c_scale=1.0,
    c_visit=50.0
  )
  gumbel_player = GumbelMctsPlayer(gspec, oracle, gumbel_params)

  # Run head-to-head comparison
  println("\nHead-to-head: MCTS vs Gumbel")
  println("Running $NUM_GAMES games...")

  t_start = time()
  results = run_duel(mcts_player, gumbel_player, NUM_GAMES)
  elapsed = time() - t_start

  println("Results:")
  println("  MCTS wins:   $(results.wins_p1) ($(round(100*results.wins_p1/NUM_GAMES, digits=1))%)")
  println("  Gumbel wins: $(results.wins_p2) ($(round(100*results.wins_p2/NUM_GAMES, digits=1))%)")
  println("  Draws:       $(results.draws) ($(round(100*results.draws/NUM_GAMES, digits=1))%)")
  println("  Time:        $(round(elapsed, digits=1))s")
  println("  Games/sec:   $(round(NUM_GAMES/elapsed, digits=2))")

  return results
end

# Run benchmarks at each simulation budget
all_results = Dict{Int, NamedTuple}()

for budget in BUDGETS
  all_results[budget] = benchmark_budget(budget)
end

# Summary
println()
println("=" ^ 60)
println("Summary")
println("=" ^ 60)
println()
println("Budget  | MCTS Win% | Gumbel Win% | Draw%")
println("-" ^ 50)
for budget in BUDGETS
  r = all_results[budget]
  mcts_pct = round(100*r.wins_p1/NUM_GAMES, digits=1)
  gumbel_pct = round(100*r.wins_p2/NUM_GAMES, digits=1)
  draw_pct = round(100*r.draws/NUM_GAMES, digits=1)
  println("$budget     | $mcts_pct%      | $gumbel_pct%       | $draw_pct%")
end
println()

# Create MinMax baseline comparison if time permits
println("Note: This benchmark uses a rollout oracle (random playouts)")
println("for evaluation. For full evaluation, train a neural network")
println("and use the trained network as the oracle.")
