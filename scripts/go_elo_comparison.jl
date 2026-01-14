"""
Elo Comparison: Standard MCTS vs Gumbel MCTS on 9x9 Go

Both methods use only 4 simulations per move to test sample efficiency.
Players play against each other to determine relative Elo.

Run with: julia --project scripts/go_elo_comparison.jl
"""

using AlphaZero
using Statistics: mean, std

const GAME = "go-9x9"
const NUM_SIMULATIONS = 4  # Very low - tests sample efficiency
const NUM_GAMES = 100  # Games per matchup

println("=" ^ 70)
println("9x9 Go: Elo Comparison")
println("Standard MCTS vs Gumbel MCTS")
println("Simulations per move: $NUM_SIMULATIONS")
println("Games per matchup: $NUM_GAMES")
println("=" ^ 70)
println()

gspec = Examples.games[GAME]

# Use rollout oracle (no neural network - pure MCTS comparison)
oracle = MCTS.RolloutOracle(gspec)

#####
##### Create Players
#####

# Standard MCTS with Dirichlet noise
standard_params = MctsParams(
  num_iters_per_turn=NUM_SIMULATIONS,
  cpuct=2.0,
  temperature=ConstSchedule(0.5),
  dirichlet_noise_ϵ=0.25,
  dirichlet_noise_α=0.03)  # 10/num_actions for Go

# Gumbel MCTS
gumbel_params = GumbelMctsParams(
  num_simulations=NUM_SIMULATIONS,
  max_considered_actions=20,  # Consider top 20 moves
  temperature=ConstSchedule(0.5),
  c_scale=1.0,
  c_visit=50.0)

#####
##### Elo Calculation
#####

"""
Estimate Elo difference from win rate.
win_rate is from player A's perspective.
Returns estimated Elo(A) - Elo(B).
"""
function elo_difference(win_rate::Float64)
  if win_rate <= 0.0
    return -400.0  # Cap at large negative
  elseif win_rate >= 1.0
    return 400.0   # Cap at large positive
  else
    return -400.0 * log10(1.0 / win_rate - 1.0)
  end
end

"""
Calculate 95% confidence interval for win rate.
"""
function win_rate_ci(wins::Int, total::Int)
  p = wins / total
  se = sqrt(p * (1 - p) / total)
  return (p - 1.96 * se, p + 1.96 * se)
end

#####
##### Run matches
#####

function run_match(player1, player2, num_games::Int; name1="P1", name2="P2")
  println("\nMatch: $name1 vs $name2 ($num_games games)")

  wins1 = 0
  wins2 = 0
  draws = 0

  for i in 1:num_games
    # Alternate colors
    if i % 2 == 1
      white, black = player1, player2
      p1_is_white = true
    else
      white, black = player2, player1
      p1_is_white = false
    end

    combined = TwoPlayers(white, black)
    trace = play_game(gspec, combined)
    reward = total_reward(trace)  # From white's perspective

    if p1_is_white
      if reward > 0
        wins1 += 1
      elseif reward < 0
        wins2 += 1
      else
        draws += 1
      end
    else
      if reward < 0
        wins1 += 1
      elseif reward > 0
        wins2 += 1
      else
        draws += 1
      end
    end

    reset_player!(player1)
    reset_player!(player2)

    # Progress indicator
    if i % 10 == 0
      print(".")
    end
  end
  println()

  win_rate1 = (wins1 + 0.5 * draws) / num_games
  elo_diff = elo_difference(win_rate1)
  ci_low, ci_high = win_rate_ci(wins1, num_games)

  println("Results:")
  println("  $name1: $wins1 wins ($(round(100*wins1/num_games, digits=1))%)")
  println("  $name2: $wins2 wins ($(round(100*wins2/num_games, digits=1))%)")
  println("  Draws: $draws ($(round(100*draws/num_games, digits=1))%)")
  println("  Win rate ($name1): $(round(100*win_rate1, digits=1))%")
  println("  95% CI: [$(round(100*ci_low, digits=1))%, $(round(100*ci_high, digits=1))%]")
  println("  Elo difference ($name1 - $name2): $(round(elo_diff, digits=0))")

  return (
    wins1=wins1, wins2=wins2, draws=draws,
    win_rate=win_rate1, elo_diff=elo_diff
  )
end

#####
##### Main comparison
#####

println("\nCreating players with $NUM_SIMULATIONS simulations each...")

# Create standard MCTS player
standard_player = MctsPlayer(gspec, oracle, standard_params)
println("  Standard MCTS player created")

# Create Gumbel MCTS player
gumbel_player = GumbelMctsPlayer(gspec, oracle, gumbel_params)
println("  Gumbel MCTS player created")

# Also create a random baseline
random_player = RandomPlayer()
println("  Random baseline created")

println("\n" * "=" ^ 70)
println("RUNNING MATCHES")
println("=" ^ 70)

# Match 1: Standard vs Gumbel
result_sg = run_match(standard_player, gumbel_player, NUM_GAMES,
                      name1="Standard", name2="Gumbel")

# Recreate players (reset state)
standard_player = MctsPlayer(gspec, oracle, standard_params)
gumbel_player = GumbelMctsPlayer(gspec, oracle, gumbel_params)

# Match 2: Standard vs Random
result_sr = run_match(standard_player, random_player, NUM_GAMES,
                      name1="Standard", name2="Random")

# Recreate players
standard_player = MctsPlayer(gspec, oracle, standard_params)
gumbel_player = GumbelMctsPlayer(gspec, oracle, gumbel_params)

# Match 3: Gumbel vs Random
result_gr = run_match(gumbel_player, random_player, NUM_GAMES,
                      name1="Gumbel", name2="Random")

#####
##### Summary
#####

println("\n" * "=" ^ 70)
println("ELO SUMMARY")
println("=" ^ 70)

# Use Random as baseline (Elo = 0)
println("\nRelative Elo ratings (Random = 0):")

elo_standard = result_sr.elo_diff
elo_gumbel = result_gr.elo_diff

println("  Random:   0")
println("  Standard: $(round(elo_standard, digits=0))")
println("  Gumbel:   $(round(elo_gumbel, digits=0))")

println("\nHead-to-head (Standard vs Gumbel):")
println("  Elo difference: $(round(result_sg.elo_diff, digits=0)) (positive = Standard stronger)")

if result_sg.elo_diff > 50
  println("\n  → Standard MCTS is significantly stronger")
elseif result_sg.elo_diff < -50
  println("\n  → Gumbel MCTS is significantly stronger")
else
  println("\n  → Both methods perform similarly")
end

println("\n" * "=" ^ 70)
println("Note: With only $NUM_SIMULATIONS simulations, both methods are weak.")
println("This test measures relative sample efficiency, not absolute strength.")
println("=" ^ 70)
