#####
##### Training hyperparameters for 9x9 Go
#####

Network = NetLib.ResNet

netparams = NetLib.ResNetHP(
  num_filters=64,
  num_blocks=5,
  conv_kernel_size=(3, 3),
  num_policy_head_filters=32,
  num_value_head_filters=32,
  batch_norm_momentum=0.1)

self_play = SelfPlayParams(
  sim=SimParams(
    num_games=500,
    num_workers=64,
    batch_size=64,
    use_gpu=true,
    reset_every=2,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=400,
    cpuct=2.0,
    prior_temperature=1.0,
    temperature=PLSchedule([0, 30, 50], [1.0, 1.0, 0.3]),
    dirichlet_noise_ϵ=0.25,
    dirichlet_noise_α=0.03))  # α = 10/num_actions ≈ 10/82

arena = ArenaParams(
  sim=SimParams(
    num_games=64,
    num_workers=64,
    batch_size=64,
    use_gpu=true,
    reset_every=2,
    flip_probability=0.5,
    alternate_colors=true),
  mcts=MctsParams(
    self_play.mcts,
    temperature=ConstSchedule(0.2),
    dirichlet_noise_ϵ=0.05),
  update_threshold=0.05)

learning = LearningParams(
  use_gpu=true,
  use_position_averaging=true,
  samples_weighing_policy=LOG_WEIGHT,
  batch_size=256,
  loss_computation_batch_size=1024,
  optimiser=Adam(lr=1e-3),
  l2_regularization=1e-4,
  nonvalidity_penalty=1.,
  min_checkpoints_per_epoch=1,
  max_batches_per_checkpoint=2000,
  num_checkpoints=1)

params = Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=20,
  ternary_outcome=true,
  use_symmetries=false,  # Symmetries not fully implemented yet
  memory_analysis=nothing,
  mem_buffer_size=PLSchedule(
    [     0,       20],
    [200_000, 500_000]))

#####
##### Evaluation benchmark
#####

mcts_baseline = Benchmark.MctsRollouts(
  MctsParams(
    arena.mcts,
    num_iters_per_turn=200,
    cpuct=1.))

alphazero_player = Benchmark.Full(arena.mcts)
network_player = Benchmark.NetworkOnly(τ=0.5)

benchmark_sim = SimParams(
  arena.sim;
  num_games=100,
  num_workers=50,
  batch_size=50,
  alternate_colors=true)

benchmark = [
  Benchmark.Duel(alphazero_player, mcts_baseline, benchmark_sim),
  Benchmark.Duel(network_player, mcts_baseline, benchmark_sim)
]

#####
##### Experiment definition
#####

experiment = Experiment("go-9x9",
  GameSpec(), params, Network, netparams, benchmark)
