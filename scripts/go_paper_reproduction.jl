"""
Reproduce Gumbel AlphaZero paper results on 9x9 Go

Paper: "Policy improvement by planning with Gumbel" (ICLR 2022)

Paper settings (Section 5.1, 9x9 Go experiments):
- Training simulations: n ∈ {2, 4, 16, 200}
- Evaluation simulations: 800
- Sampled actions: m = min(n, 16)
- c_visit = 50, c_scale = 1.0
- Key finding: Gumbel learns even with 2-4 sims, Standard fails with ≤16 sims

This script reproduces these exact conditions.

Run with: julia --project scripts/go_paper_reproduction.jl
"""

using AlphaZero
using AlphaZero: Handlers, Report, resize_memory!, self_play_step!, memory_report, learning_step!
using Statistics: mean, std

const GAME = "go-9x9"

#####
##### Paper Parameters (DO NOT SCALE DOWN)
#####

# Training simulations - paper tests {2, 4, 16, 200}
# We use 4 as the key test case (where Gumbel succeeds, Standard fails)
const TRAIN_SIMS = 4

# Evaluation simulations - paper uses 800
const EVAL_SIMS = 800

# Gumbel parameters from paper
const SAMPLED_ACTIONS = 4      # m = min(n, 16) when n=4
const C_VISIT = 50.0           # Paper value
const C_SCALE = 1.0            # Paper value

# Training scale - paper likely uses thousands of iterations
# We use a smaller but still substantial amount
const NUM_ITERATIONS = 50
const NUM_GAMES_PER_ITER = 200
const NUM_EVAL_GAMES = 100

println("=" ^ 70)
println("Gumbel AlphaZero Paper Reproduction")
println("=" ^ 70)
println("Paper: 'Policy improvement by planning with Gumbel' (ICLR 2022)")
println()
println("Game: 9x9 Go")
println("Training simulations: $TRAIN_SIMS (paper tests 2,4,16,200)")
println("Evaluation simulations: $EVAL_SIMS (matches paper)")
println("Sampled actions (Gumbel): $SAMPLED_ACTIONS (m = min(n, 16))")
println("c_visit = $C_VISIT, c_scale = $C_SCALE (paper values)")
println()
println("Training:")
println("  Iterations: $NUM_ITERATIONS")
println("  Games/iteration: $NUM_GAMES_PER_ITER")
println("  Evaluation games: $NUM_EVAL_GAMES")
println("=" ^ 70)
println()

gspec = Examples.games[GAME]

# Network architecture - ResNet similar to paper
# Paper uses larger networks; this is a reasonable approximation
Network = NetLib.ResNet
netparams = NetLib.ResNetHP(
  num_filters=128,      # Substantial capacity
  num_blocks=8,         # 8 residual blocks
  conv_kernel_size=(3, 3),
  num_policy_head_filters=64,
  num_value_head_filters=64,
  batch_norm_momentum=0.1)

# Standard AlphaZero with Dirichlet noise
standard_self_play = SelfPlayParams(
  sim=SimParams(
    num_games=NUM_GAMES_PER_ITER,
    num_workers=64,
    batch_size=64,
    use_gpu=false,  # Set to true if CUDA available
    reset_every=2,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=TRAIN_SIMS,
    cpuct=2.0,
    temperature=ConstSchedule(1.0),
    dirichlet_noise_ϵ=0.25,
    dirichlet_noise_α=0.03))  # ~10/num_actions for Go

# Gumbel AlphaZero with paper settings
gumbel_self_play = SelfPlayParams(
  sim=SimParams(
    num_games=NUM_GAMES_PER_ITER,
    num_workers=64,
    batch_size=64,
    use_gpu=false,  # Set to true if CUDA available
    reset_every=2,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=TRAIN_SIMS,
    cpuct=2.0,
    temperature=ConstSchedule(1.0),
    dirichlet_noise_ϵ=0.25,
    dirichlet_noise_α=0.03),
  gumbel_mcts=GumbelMctsParams(
    num_simulations=TRAIN_SIMS,
    max_considered_actions=SAMPLED_ACTIONS,
    temperature=ConstSchedule(1.0),
    c_scale=C_SCALE,
    c_visit=C_VISIT))

# Learning parameters
learning = LearningParams(
  use_gpu=false,  # Set to true if CUDA available
  samples_weighing_policy=LOG_WEIGHT,
  l2_regularization=1e-4,
  optimiser=Adam(lr=2e-3),
  batch_size=256,
  loss_computation_batch_size=2048,
  nonvalidity_penalty=1.,
  min_checkpoints_per_epoch=0,
  max_batches_per_checkpoint=2000,
  num_checkpoints=1)

make_standard_params(n) = Params(
  arena=nothing,
  self_play=standard_self_play,
  learning=learning,
  num_iters=n,
  ternary_outcome=true,
  use_symmetries=true,  # Use symmetries for Go
  mem_buffer_size=PLSchedule(100_000))

make_gumbel_params(n) = Params(
  arena=nothing,
  self_play=gumbel_self_play,
  learning=learning,
  num_iters=n,
  ternary_outcome=true,
  use_symmetries=true,  # Use symmetries for Go
  mem_buffer_size=PLSchedule(100_000))

# Evaluation with 800 sims against random baseline
function evaluate_vs_random(env, num_games=NUM_EVAL_GAMES)
  eval_params = MctsParams(
    num_iters_per_turn=EVAL_SIMS,
    cpuct=2.0,
    temperature=ConstSchedule(0.1),  # Low temp for evaluation
    dirichlet_noise_ϵ=0.0,           # No noise in eval
    dirichlet_noise_α=0.03)

  net = AlphaZero.Network.copy(env.bestnn, on_gpu=false, test_mode=true)
  trained_player = MctsPlayer(gspec, net, eval_params)
  random_player = RandomPlayer()

  wins = 0
  for i in 1:num_games
    white = i % 2 == 1 ? trained_player : random_player
    black = i % 2 == 1 ? random_player : trained_player
    trained_is_white = i % 2 == 1

    trace = play_game(gspec, TwoPlayers(white, black))
    reward = total_reward(trace)

    if (trained_is_white && reward > 0) || (!trained_is_white && reward < 0)
      wins += 1
    end
    reset_player!(trained_player)

    # Progress indicator every 10 games
    if i % 10 == 0
      print(".")
    end
  end
  println()
  return wins / num_games
end

# Head-to-head evaluation between two trained networks
function evaluate_head_to_head(env1, env2, num_games=NUM_EVAL_GAMES)
  eval_params = MctsParams(
    num_iters_per_turn=EVAL_SIMS,
    cpuct=2.0,
    temperature=ConstSchedule(0.1),
    dirichlet_noise_ϵ=0.0,
    dirichlet_noise_α=0.03)

  net1 = AlphaZero.Network.copy(env1.bestnn, on_gpu=false, test_mode=true)
  net2 = AlphaZero.Network.copy(env2.bestnn, on_gpu=false, test_mode=true)
  player1 = MctsPlayer(gspec, net1, eval_params)
  player2 = MctsPlayer(gspec, net2, eval_params)

  wins1 = 0
  for i in 1:num_games
    white = i % 2 == 1 ? player1 : player2
    black = i % 2 == 1 ? player2 : player1
    p1_is_white = i % 2 == 1

    trace = play_game(gspec, TwoPlayers(white, black))
    reward = total_reward(trace)

    if (p1_is_white && reward > 0) || (!p1_is_white && reward < 0)
      wins1 += 1
    end
    reset_player!(player1)
    reset_player!(player2)

    if i % 10 == 0
      print(".")
    end
  end
  println()
  return wins1 / num_games
end

struct TrainingResult
  name::String
  win_rates::Vector{Float64}
  times::Vector{Float64}
end

function train_agent(name::String, make_params::Function, num_iters::Int)
  println("\n" * "=" ^ 70)
  println("Training: $name")
  println("=" ^ 70)

  nn = Network(gspec, netparams)
  params = make_params(num_iters)
  env = Env(gspec, params, nn)

  win_rates = Float64[]
  times = Float64[]

  # Initial evaluation
  println("\nIteration 0: Evaluating ($EVAL_SIMS sims, $NUM_EVAL_GAMES games)...")
  wr = evaluate_vs_random(env)
  push!(win_rates, wr)
  push!(times, 0.0)
  println("  Win rate vs Random: $(round(100*wr, digits=1))%")

  for iter in 1:num_iters
    println("\nIteration $iter:")
    t_start = time()

    print("  Self-play ($NUM_GAMES_PER_ITER games, $TRAIN_SIMS sims)... ")
    Handlers.iteration_started(nothing)
    resize_memory!(env, env.params.mem_buffer_size[env.itc])
    Report.@timed self_play_step!(env, nothing)
    println("done")

    print("  Memory report... ")
    Report.@timed memory_report(env, nothing)
    println("done")

    print("  Learning... ")
    Report.@timed learning_step!(env, nothing)
    println("done")

    env.itc += 1
    elapsed = time() - t_start

    println("  Evaluating ($EVAL_SIMS sims, $NUM_EVAL_GAMES games)...")
    wr = evaluate_vs_random(env)
    push!(win_rates, wr)
    push!(times, elapsed)
    println("  Win rate: $(round(100*wr, digits=1))% | Training time: $(round(elapsed/60, digits=1)) min")
  end

  return env, TrainingResult(name, win_rates, times)
end

# Elo estimation
function elo_diff(win_rate)
  win_rate = clamp(win_rate, 0.01, 0.99)
  return -400 * log10(1/win_rate - 1)
end

#####
##### Main Training
#####

println("\n" * "=" ^ 70)
println("TRAINING PHASE")
println("Expected duration: Several hours per method")
println("=" ^ 70)

standard_env, standard_result = train_agent("Standard AlphaZero", make_standard_params, NUM_ITERATIONS)
gumbel_env, gumbel_result = train_agent("Gumbel AlphaZero", make_gumbel_params, NUM_ITERATIONS)

#####
##### Head-to-head evaluation
#####

println("\n" * "=" ^ 70)
println("HEAD-TO-HEAD EVALUATION")
println("=" ^ 70)
println("\nPlaying Standard vs Gumbel ($NUM_EVAL_GAMES games, $EVAL_SIMS sims each)...")

h2h_win_rate = evaluate_head_to_head(standard_env, gumbel_env, NUM_EVAL_GAMES)
println("Standard win rate vs Gumbel: $(round(100*h2h_win_rate, digits=1))%")
println("Gumbel win rate vs Standard: $(round(100*(1-h2h_win_rate), digits=1))%")

elo = elo_diff(1 - h2h_win_rate)  # Gumbel's perspective
println("Estimated Elo difference (Gumbel - Standard): $(round(elo, digits=0))")

#####
##### Results Summary
#####

println("\n" * "=" ^ 70)
println("RESULTS SUMMARY")
println("=" ^ 70)

println("\nWin Rate vs Random (with $EVAL_SIMS eval sims):")
println("-" ^ 60)
println("Iter | Standard AlphaZero | Gumbel AlphaZero | Δ")
println("-" ^ 60)

for i in 0:NUM_ITERATIONS
  std_wr = round(100 * standard_result.win_rates[i+1], digits=1)
  gum_wr = round(100 * gumbel_result.win_rates[i+1], digits=1)
  delta = round(gum_wr - std_wr, digits=1)
  delta_str = delta >= 0 ? "+$delta" : "$delta"
  println("$(lpad(i, 3))  |      $(lpad(std_wr, 5))%       |      $(lpad(gum_wr, 5))%      | $delta_str")
end
println("-" ^ 60)

total_std_time = sum(standard_result.times)
total_gum_time = sum(gumbel_result.times)

println("\nFinal Results:")
println("  Standard vs Random: $(round(100*standard_result.win_rates[end], digits=1))%")
println("  Gumbel vs Random:   $(round(100*gumbel_result.win_rates[end], digits=1))%")
println("  Gumbel vs Standard: $(round(100*(1-h2h_win_rate), digits=1))%")
println("  Elo(Gumbel) - Elo(Standard): $(round(elo, digits=0))")
println()
println("  Total training time (Standard): $(round(total_std_time/3600, digits=1)) hours")
println("  Total training time (Gumbel):   $(round(total_gum_time/3600, digits=1)) hours")

println("\n" * "=" ^ 70)
if elo > 100
  println("✓ PAPER FINDING CONFIRMED: Gumbel significantly outperforms Standard")
  println("  with only $TRAIN_SIMS training simulations!")
elseif elo > 50
  println("✓ Gumbel shows advantage over Standard with $TRAIN_SIMS training sims")
elseif elo < -50
  println("✗ Unexpected: Standard outperforms Gumbel")
else
  println("≈ Both methods perform similarly with $TRAIN_SIMS training sims")
end
println("=" ^ 70)
