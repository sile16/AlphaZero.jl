"""
Compare training speed: Standard AlphaZero (with Dirichlet noise) vs Gumbel AlphaZero

This script trains both algorithms on Tic-Tac-Toe and compares learning speed
by measuring win rate against a MinMax baseline at each iteration.

Run with: julia --project scripts/compare_training.jl
"""

using AlphaZero
using AlphaZero: Handlers, Report, resize_memory!, self_play_step!, memory_report, learning_step!
using Statistics: mean, std

const GAME = "tictactoe"
const NUM_ITERATIONS = 10
const NUM_EVAL_GAMES = 100

# Reduced simulation budget for faster comparison
const NUM_SIMS = 100
const NUM_GAMES_PER_ITER = 300

println("=" ^ 70)
println("Training Comparison: Standard AlphaZero vs Gumbel AlphaZero")
println("Game: $GAME")
println("Iterations: $NUM_ITERATIONS")
println("Simulations per move: $NUM_SIMS")
println("Games per iteration: $NUM_GAMES_PER_ITER")
println("=" ^ 70)
println()

# Get game spec
gspec = Examples.games[GAME]

# Network configuration (same for both)
Network = NetLib.SimpleNet
netparams = NetLib.SimpleNetHP(
  width=100,
  depth_common=4,
  use_batch_norm=true,
  batch_norm_momentum=1.)

# Standard AlphaZero parameters (with Dirichlet noise)
standard_self_play = SelfPlayParams(
  sim=SimParams(
    num_games=NUM_GAMES_PER_ITER,
    num_workers=32,
    batch_size=32,
    use_gpu=false,
    reset_every=2,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=NUM_SIMS,
    cpuct=1.0,
    temperature=ConstSchedule(1.0),
    dirichlet_noise_ϵ=0.25,
    dirichlet_noise_α=1.0))

# Gumbel AlphaZero parameters
# Note: mcts is still required for the struct, but gumbel_mcts takes precedence in self-play
gumbel_self_play = SelfPlayParams(
  sim=SimParams(
    num_games=NUM_GAMES_PER_ITER,
    num_workers=32,
    batch_size=32,
    use_gpu=false,
    reset_every=2,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=NUM_SIMS,
    cpuct=1.0,
    temperature=ConstSchedule(1.0),
    dirichlet_noise_ϵ=0.25,
    dirichlet_noise_α=1.0),
  gumbel_mcts=GumbelMctsParams(
    num_simulations=NUM_SIMS,
    max_considered_actions=9,  # Tic-Tac-Toe has 9 cells
    temperature=ConstSchedule(1.0),
    c_scale=1.0,
    c_visit=50.0))

# Learning parameters (same for both)
learning = LearningParams(
  use_gpu=false,
  samples_weighing_policy=LOG_WEIGHT,
  l2_regularization=1e-4,
  optimiser=CyclicNesterov(
    lr_base=1e-3,
    lr_high=1e-2,
    lr_low=1e-3,
    momentum_high=0.9,
    momentum_low=0.8),
  batch_size=32,
  loss_computation_batch_size=1024,
  nonvalidity_penalty=1.,
  min_checkpoints_per_epoch=0,
  max_batches_per_checkpoint=2000,
  num_checkpoints=1)

make_standard_params(num_iters) = Params(
  arena=nothing,  # Skip arena for faster training
  self_play=standard_self_play,
  learning=learning,
  num_iters=num_iters,
  ternary_outcome=true,
  use_symmetries=true,
  mem_buffer_size=PLSchedule(20_000))

make_gumbel_params(num_iters) = Params(
  arena=nothing,
  self_play=gumbel_self_play,
  learning=learning,
  num_iters=num_iters,
  ternary_outcome=true,
  use_symmetries=true,
  mem_buffer_size=PLSchedule(20_000))

# Evaluation function: play against random player
function evaluate_against_random(env, num_games=NUM_EVAL_GAMES)
  # Create player from current best network (with fewer sims for speed)
  mcts_params = MctsParams(
    num_iters_per_turn=30,
    cpuct=1.0,
    temperature=ConstSchedule(0.1),
    dirichlet_noise_ϵ=0.0,
    dirichlet_noise_α=1.0)

  net = AlphaZero.Network.copy(env.bestnn, on_gpu=false, test_mode=true)
  az_player = MctsPlayer(gspec, net, mcts_params)

  # Random opponent
  random_player = RandomPlayer()

  wins = 0
  losses = 0
  draws = 0

  for i in 1:num_games
    # Alternate colors
    if i % 2 == 1
      white, black = az_player, random_player
      az_is_white = true
    else
      white, black = random_player, az_player
      az_is_white = false
    end

    combined = TwoPlayers(white, black)
    trace = play_game(gspec, combined)
    reward = total_reward(trace)

    if az_is_white
      if reward > 0
        wins += 1
      elseif reward < 0
        losses += 1
      else
        draws += 1
      end
    else
      if reward < 0
        wins += 1
      elseif reward > 0
        losses += 1
      else
        draws += 1
      end
    end

    reset_player!(az_player)
  end

  win_rate = wins / num_games
  non_loss_rate = (wins + draws) / num_games
  return (wins=wins, losses=losses, draws=draws, win_rate=win_rate, non_loss_rate=non_loss_rate)
end

# Training results storage
struct TrainingResult
  name::String
  win_rates::Vector{Float64}
  times::Vector{Float64}
end

function train_and_evaluate(name::String, make_params::Function, num_iters::Int)
  println("\n" * "-" ^ 70)
  println("Training: $name")
  println("-" ^ 70)

  # Create network and environment with all iterations
  nn = Network(gspec, netparams)
  full_params = make_params(num_iters)
  env = Env(gspec, full_params, nn)

  win_rates = Float64[]
  times = Float64[]
  cumulative_time = 0.0

  # Initial evaluation (before any training)
  print("Iteration 0: Evaluating... ")
  eval_result = evaluate_against_random(env)
  push!(win_rates, eval_result.win_rate)
  push!(times, 0.0)
  println("Win rate: $(round(100*eval_result.win_rate, digits=1))%")

  # Training loop - run one iteration at a time and evaluate
  for iter in 1:num_iters
    print("Iteration $iter: Training... ")
    t_start = time()

    # Run a single training iteration (self-play + learning)
    Handlers.iteration_started(nothing)
    resize_memory!(env, env.params.mem_buffer_size[env.itc])
    sprep, spperfs = Report.@timed self_play_step!(env, nothing)
    mrep, mperfs = Report.@timed memory_report(env, nothing)
    lrep, lperfs = Report.@timed learning_step!(env, nothing)
    rep = Report.Iteration(spperfs, mperfs, lperfs, sprep, mrep, lrep)
    env.itc += 1
    Handlers.iteration_finished(nothing, rep)

    elapsed = time() - t_start
    cumulative_time += elapsed

    print("Evaluating... ")
    eval_result = evaluate_against_random(env)
    push!(win_rates, eval_result.win_rate)
    push!(times, elapsed)

    println("Win rate: $(round(100*eval_result.win_rate, digits=1))% ($(round(elapsed, digits=1))s)")
  end

  return TrainingResult(name, win_rates, times)
end

# Run both training experiments
println("\nStarting training comparison...\n")

standard_result = train_and_evaluate("Standard AlphaZero (Dirichlet)", make_standard_params, NUM_ITERATIONS)
gumbel_result = train_and_evaluate("Gumbel AlphaZero", make_gumbel_params, NUM_ITERATIONS)

# Summary
println("\n" * "=" ^ 70)
println("RESULTS SUMMARY")
println("=" ^ 70)

println("\nWin Rate vs Random Player after each iteration:")
println("-" ^ 50)
println("Iter | Standard AlphaZero | Gumbel AlphaZero")
println("-" ^ 50)

for i in 0:NUM_ITERATIONS
  std_wr = round(100 * standard_result.win_rates[i+1], digits=1)
  gum_wr = round(100 * gumbel_result.win_rates[i+1], digits=1)
  println("  $i  |      $std_wr%         |     $gum_wr%")
end

println("-" ^ 50)

# Final comparison
final_std = standard_result.win_rates[end]
final_gum = gumbel_result.win_rates[end]

println("\nFinal Results:")
println("  Standard AlphaZero: $(round(100*final_std, digits=1))% win rate")
println("  Gumbel AlphaZero:   $(round(100*final_gum, digits=1))% win rate")

if final_gum > final_std
  improvement = round(100 * (final_gum - final_std), digits=1)
  println("\n  → Gumbel shows +$(improvement)% improvement")
elseif final_std > final_gum
  improvement = round(100 * (final_std - final_gum), digits=1)
  println("\n  → Standard shows +$(improvement)% improvement")
else
  println("\n  → Both methods perform equally")
end

# Time comparison
total_std_time = sum(standard_result.times)
total_gum_time = sum(gumbel_result.times)
println("\nTotal training time:")
println("  Standard: $(round(total_std_time, digits=1))s")
println("  Gumbel:   $(round(total_gum_time, digits=1))s")

println("\n" * "=" ^ 70)
