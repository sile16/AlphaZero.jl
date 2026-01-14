"""
Compare training: Standard AlphaZero (100 sims) vs Gumbel AlphaZero (50 sims)

Test if Gumbel with HALF the simulations can match standard AlphaZero on Connect Four.

Run with: julia --project scripts/compare_connect4.jl
"""

using AlphaZero
using AlphaZero: Handlers, Report, resize_memory!, self_play_step!, memory_report, learning_step!
using Statistics: mean

const GAME = "connect-four"
const NUM_ITERATIONS = 8
const NUM_EVAL_GAMES = 50

# Standard AlphaZero: 100 simulations
const STANDARD_SIMS = 100
# Gumbel: 50 simulations (half!)
const GUMBEL_SIMS = 50

const NUM_GAMES_PER_ITER = 200

println("=" ^ 70)
println("Connect Four: Standard AlphaZero vs Gumbel AlphaZero")
println("=" ^ 70)
println("Standard AlphaZero: $STANDARD_SIMS simulations per move")
println("Gumbel AlphaZero:   $GUMBEL_SIMS simulations per move (50% less!)")
println("Iterations: $NUM_ITERATIONS")
println("Games per iteration: $NUM_GAMES_PER_ITER")
println("=" ^ 70)
println()

# Get game spec
gspec = Examples.games[GAME]

# Smaller network for faster training
Network = NetLib.ResNet
netparams = NetLib.ResNetHP(
  num_filters=64,
  num_blocks=3,
  conv_kernel_size=(3, 3),
  num_policy_head_filters=16,
  num_value_head_filters=16,
  batch_norm_momentum=0.1)

# Standard AlphaZero parameters (100 sims, Dirichlet noise)
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
    num_iters_per_turn=STANDARD_SIMS,
    cpuct=2.0,
    temperature=ConstSchedule(1.0),
    dirichlet_noise_ϵ=0.25,
    dirichlet_noise_α=1.0))

# Gumbel AlphaZero parameters (50 sims - half the budget!)
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
    num_iters_per_turn=GUMBEL_SIMS,
    cpuct=2.0,
    temperature=ConstSchedule(1.0),
    dirichlet_noise_ϵ=0.25,
    dirichlet_noise_α=1.0),
  gumbel_mcts=GumbelMctsParams(
    num_simulations=GUMBEL_SIMS,
    max_considered_actions=7,  # Connect Four has 7 columns
    temperature=ConstSchedule(1.0),
    c_scale=1.0,
    c_visit=50.0))

# Learning parameters
learning = LearningParams(
  use_gpu=false,
  samples_weighing_policy=LOG_WEIGHT,
  l2_regularization=1e-4,
  optimiser=Adam(lr=1e-3),
  batch_size=64,
  loss_computation_batch_size=1024,
  nonvalidity_penalty=1.,
  min_checkpoints_per_epoch=0,
  max_batches_per_checkpoint=1000,
  num_checkpoints=1)

make_standard_params(num_iters) = Params(
  arena=nothing,
  self_play=standard_self_play,
  learning=learning,
  num_iters=num_iters,
  ternary_outcome=true,
  use_symmetries=true,
  mem_buffer_size=PLSchedule(50_000))

make_gumbel_params(num_iters) = Params(
  arena=nothing,
  self_play=gumbel_self_play,
  learning=learning,
  num_iters=num_iters,
  ternary_outcome=true,
  use_symmetries=true,
  mem_buffer_size=PLSchedule(50_000))

# Evaluation: play against MCTS with rollouts
function evaluate_against_mcts_rollouts(env, num_games=NUM_EVAL_GAMES)
  # Player from current network
  mcts_params = MctsParams(
    num_iters_per_turn=30,
    cpuct=2.0,
    temperature=ConstSchedule(0.1),
    dirichlet_noise_ϵ=0.0,
    dirichlet_noise_α=1.0)

  net = AlphaZero.Network.copy(env.bestnn, on_gpu=false, test_mode=true)
  az_player = MctsPlayer(gspec, net, mcts_params)

  # MCTS with random rollouts as baseline
  rollout_oracle = MCTS.RolloutOracle(gspec)
  rollout_params = MctsParams(
    num_iters_per_turn=100,
    cpuct=1.0,
    temperature=ConstSchedule(0.1),
    dirichlet_noise_ϵ=0.0,
    dirichlet_noise_α=1.0)
  baseline = MctsPlayer(gspec, rollout_oracle, rollout_params)

  wins = 0
  losses = 0
  draws = 0

  for i in 1:num_games
    if i % 2 == 1
      white, black = az_player, baseline
      az_is_white = true
    else
      white, black = baseline, az_player
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
    reset_player!(baseline)
  end

  win_rate = wins / num_games
  non_loss_rate = (wins + draws) / num_games
  return (wins=wins, losses=losses, draws=draws, win_rate=win_rate, non_loss_rate=non_loss_rate)
end

struct TrainingResult
  name::String
  sims::Int
  win_rates::Vector{Float64}
  times::Vector{Float64}
end

function train_and_evaluate(name::String, sims::Int, make_params::Function, num_iters::Int)
  println("\n" * "-" ^ 70)
  println("Training: $name ($sims sims/move)")
  println("-" ^ 70)

  nn = Network(gspec, netparams)
  full_params = make_params(num_iters)
  env = Env(gspec, full_params, nn)

  win_rates = Float64[]
  times = Float64[]

  # Initial evaluation
  print("Iteration 0: Evaluating... ")
  eval_result = evaluate_against_mcts_rollouts(env)
  push!(win_rates, eval_result.win_rate)
  push!(times, 0.0)
  println("Win rate: $(round(100*eval_result.win_rate, digits=1))%")

  for iter in 1:num_iters
    print("Iteration $iter: Training... ")
    t_start = time()

    Handlers.iteration_started(nothing)
    resize_memory!(env, env.params.mem_buffer_size[env.itc])
    sprep, spperfs = Report.@timed self_play_step!(env, nothing)
    mrep, mperfs = Report.@timed memory_report(env, nothing)
    lrep, lperfs = Report.@timed learning_step!(env, nothing)
    rep = Report.Iteration(spperfs, mperfs, lperfs, sprep, mrep, lrep)
    env.itc += 1
    Handlers.iteration_finished(nothing, rep)

    elapsed = time() - t_start

    print("Evaluating... ")
    eval_result = evaluate_against_mcts_rollouts(env)
    push!(win_rates, eval_result.win_rate)
    push!(times, elapsed)

    println("Win rate: $(round(100*eval_result.win_rate, digits=1))% ($(round(elapsed, digits=1))s)")
  end

  return TrainingResult(name, sims, win_rates, times)
end

# Run training
println("\nStarting training comparison...\n")

standard_result = train_and_evaluate("Standard AlphaZero", STANDARD_SIMS, make_standard_params, NUM_ITERATIONS)
gumbel_result = train_and_evaluate("Gumbel AlphaZero", GUMBEL_SIMS, make_gumbel_params, NUM_ITERATIONS)

# Summary
println("\n" * "=" ^ 70)
println("RESULTS SUMMARY")
println("=" ^ 70)
println("\nStandard AlphaZero: $STANDARD_SIMS sims/move")
println("Gumbel AlphaZero:   $GUMBEL_SIMS sims/move (50% reduction)")

println("\nWin Rate vs MCTS Rollouts (100 sims) after each iteration:")
println("-" ^ 55)
println("Iter | Standard ($STANDARD_SIMS sims) | Gumbel ($GUMBEL_SIMS sims)")
println("-" ^ 55)

for i in 0:NUM_ITERATIONS
  std_wr = round(100 * standard_result.win_rates[i+1], digits=1)
  gum_wr = round(100 * gumbel_result.win_rates[i+1], digits=1)
  diff = round(gum_wr - std_wr, digits=1)
  diff_str = diff >= 0 ? "+$diff" : "$diff"
  println("  $i  |      $std_wr%          |     $gum_wr%  ($diff_str)")
end

println("-" ^ 55)

final_std = standard_result.win_rates[end]
final_gum = gumbel_result.win_rates[end]

println("\nFinal Results:")
println("  Standard AlphaZero ($STANDARD_SIMS sims): $(round(100*final_std, digits=1))% win rate")
println("  Gumbel AlphaZero ($GUMBEL_SIMS sims):   $(round(100*final_gum, digits=1))% win rate")

if final_gum >= final_std
  println("\n  ✓ Gumbel with 50% fewer sims MATCHES or BEATS standard!")
else
  gap = round(100 * (final_std - final_gum), digits=1)
  println("\n  Standard ahead by $gap% (Gumbel uses 50% fewer sims)")
end

total_std_time = sum(standard_result.times)
total_gum_time = sum(gumbel_result.times)
println("\nTotal training time:")
println("  Standard: $(round(total_std_time, digits=1))s")
println("  Gumbel:   $(round(total_gum_time, digits=1))s")

println("\n" * "=" ^ 70)
