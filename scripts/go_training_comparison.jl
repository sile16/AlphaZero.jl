"""
Reproduce Gumbel AlphaZero paper results on 9x9 Go

Key insight from paper: Gumbel can learn with very few simulations (even 2-4),
while standard MCTS/MuZero fails with <16 simulations.

This script:
1. Trains Standard AlphaZero with 4 sims during self-play
2. Trains Gumbel AlphaZero with 4 sims during self-play
3. Evaluates both with MORE sims (like the paper does with 800)
4. Compares learning curves

Run with: julia --project scripts/go_training_comparison.jl
"""

using AlphaZero
using AlphaZero: Handlers, Report, resize_memory!, self_play_step!, memory_report, learning_step!
using Statistics: mean

const GAME = "go-9x9"
const TRAIN_SIMS = 4  # Simulations during self-play (like paper)
const EVAL_SIMS = 50  # Simulations during evaluation (paper uses 800, we use less for speed)
const NUM_ITERATIONS = 5
const NUM_GAMES_PER_ITER = 100
const NUM_EVAL_GAMES = 30

println("=" ^ 70)
println("9x9 Go: Gumbel AlphaZero Training Comparison")
println("=" ^ 70)
println("Training simulations: $TRAIN_SIMS (like Gumbel paper)")
println("Evaluation: $EVAL_SIMS sims vs Random baseline")
println("Iterations: $NUM_ITERATIONS")
println("Games per iteration: $NUM_GAMES_PER_ITER")
println("=" ^ 70)
println()

gspec = Examples.games[GAME]

# Smaller network for faster training
Network = NetLib.ResNet
netparams = NetLib.ResNetHP(
  num_filters=32,
  num_blocks=3,
  conv_kernel_size=(3, 3),
  num_policy_head_filters=16,
  num_value_head_filters=16,
  batch_norm_momentum=0.1)

# Standard AlphaZero: 4 sims with Dirichlet noise
standard_self_play = SelfPlayParams(
  sim=SimParams(
    num_games=NUM_GAMES_PER_ITER,
    num_workers=16,
    batch_size=16,
    use_gpu=false,
    reset_every=2,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=TRAIN_SIMS,
    cpuct=2.0,
    temperature=ConstSchedule(1.0),
    dirichlet_noise_ϵ=0.25,
    dirichlet_noise_α=0.03))  # ~10/num_actions for Go

# Gumbel AlphaZero: 4 sims
gumbel_self_play = SelfPlayParams(
  sim=SimParams(
    num_games=NUM_GAMES_PER_ITER,
    num_workers=16,
    batch_size=16,
    use_gpu=false,
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
    max_considered_actions=20,
    temperature=ConstSchedule(1.0),
    c_scale=1.0,
    c_visit=50.0))

# Learning parameters
learning = LearningParams(
  use_gpu=false,
  samples_weighing_policy=LOG_WEIGHT,
  l2_regularization=1e-4,
  optimiser=Adam(lr=1e-3),
  batch_size=32,
  loss_computation_batch_size=512,
  nonvalidity_penalty=1.,
  min_checkpoints_per_epoch=0,
  max_batches_per_checkpoint=500,
  num_checkpoints=1)

make_standard_params(n) = Params(
  arena=nothing,
  self_play=standard_self_play,
  learning=learning,
  num_iters=n,
  ternary_outcome=true,
  use_symmetries=false,
  mem_buffer_size=PLSchedule(20_000))

make_gumbel_params(n) = Params(
  arena=nothing,
  self_play=gumbel_self_play,
  learning=learning,
  num_iters=n,
  ternary_outcome=true,
  use_symmetries=false,
  mem_buffer_size=PLSchedule(20_000))

# Evaluation: trained network vs random baseline
function evaluate_vs_random(env, num_games=NUM_EVAL_GAMES)
  # Create player with trained network
  eval_params = MctsParams(
    num_iters_per_turn=EVAL_SIMS,
    cpuct=2.0,
    temperature=ConstSchedule(0.3),
    dirichlet_noise_ϵ=0.0,
    dirichlet_noise_α=0.03)

  net = AlphaZero.Network.copy(env.bestnn, on_gpu=false, test_mode=true)
  trained_player = MctsPlayer(gspec, net, eval_params)
  random_player = RandomPlayer()

  wins = 0
  for i in 1:num_games
    if i % 2 == 1
      white, black = trained_player, random_player
      trained_is_white = true
    else
      white, black = random_player, trained_player
      trained_is_white = false
    end

    trace = play_game(gspec, TwoPlayers(white, black))
    reward = total_reward(trace)

    if (trained_is_white && reward > 0) || (!trained_is_white && reward < 0)
      wins += 1
    end

    reset_player!(trained_player)
  end

  return wins / num_games
end

struct TrainingResult
  name::String
  win_rates::Vector{Float64}
  times::Vector{Float64}
end

function train_and_evaluate(name::String, make_params::Function, num_iters::Int)
  println("\n" * "-" ^ 70)
  println("Training: $name ($(TRAIN_SIMS) sims during self-play)")
  println("-" ^ 70)

  nn = Network(gspec, netparams)
  params = make_params(num_iters)
  env = Env(gspec, params, nn)

  win_rates = Float64[]
  times = Float64[]

  # Initial evaluation
  print("Iteration 0: Evaluating... ")
  wr = evaluate_vs_random(env)
  push!(win_rates, wr)
  push!(times, 0.0)
  println("Win rate vs MCTS-rollouts: $(round(100*wr, digits=1))%")

  for iter in 1:num_iters
    print("Iteration $iter: Training... ")
    t_start = time()

    Handlers.iteration_started(nothing)
    resize_memory!(env, env.params.mem_buffer_size[env.itc])
    sprep, _ = Report.@timed self_play_step!(env, nothing)
    mrep, _ = Report.@timed memory_report(env, nothing)
    lrep, _ = Report.@timed learning_step!(env, nothing)
    env.itc += 1

    elapsed = time() - t_start

    print("Evaluating... ")
    wr = evaluate_vs_random(env)
    push!(win_rates, wr)
    push!(times, elapsed)

    println("Win rate: $(round(100*wr, digits=1))% ($(round(elapsed, digits=1))s)")
  end

  return TrainingResult(name, win_rates, times)
end

# Run training
println("\nStarting training comparison...\n")

standard_result = train_and_evaluate("Standard AlphaZero", make_standard_params, NUM_ITERATIONS)
gumbel_result = train_and_evaluate("Gumbel AlphaZero", make_gumbel_params, NUM_ITERATIONS)

# Summary
println("\n" * "=" ^ 70)
println("RESULTS SUMMARY")
println("=" ^ 70)
println("\nTraining: $(TRAIN_SIMS) simulations during self-play")
println("Evaluation: $(EVAL_SIMS) simulations vs MCTS-rollouts baseline")

println("\nWin Rate after each iteration:")
println("-" ^ 50)
println("Iter | Standard AlphaZero | Gumbel AlphaZero")
println("-" ^ 50)

for i in 0:NUM_ITERATIONS
  std_wr = round(100 * standard_result.win_rates[i+1], digits=1)
  gum_wr = round(100 * gumbel_result.win_rates[i+1], digits=1)
  println("  $i  |      $std_wr%         |     $gum_wr%")
end
println("-" ^ 50)

final_std = standard_result.win_rates[end]
final_gum = gumbel_result.win_rates[end]

println("\nFinal win rates:")
println("  Standard: $(round(100*final_std, digits=1))%")
println("  Gumbel:   $(round(100*final_gum, digits=1))%")

if final_gum > final_std + 0.1
  println("\n  → Gumbel learns better with $(TRAIN_SIMS) sims (as predicted by paper)")
elseif final_std > final_gum + 0.1
  println("\n  → Standard learns better (unexpected)")
else
  println("\n  → Both learn similarly")
end

println("\n" * "=" ^ 70)
println("Paper finding: Gumbel can learn from 2-4 sims while standard fails with <16")
println("=" ^ 70)
