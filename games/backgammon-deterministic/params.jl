#####
##### Random Benchmark Player
#####

struct RandomBaseline <: Benchmark.Player end

Benchmark.name(p::RandomBaseline) = "Random"

function Benchmark.instantiate(p::RandomBaseline, ::AbstractGameSpec, nn)
  return RandomPlayer()
end

#####
##### Network Configuration
#####

# Same network architecture as stochastic version for fair comparison
Network = NetLib.SimpleNet

netparams = NetLib.SimpleNetHP(
  width=128,
  depth_common=6,
  use_batch_norm=true,
  batch_norm_momentum=1.)

self_play = SelfPlayParams(
  sim=SimParams(
    num_games=100,
    num_workers=16,
    batch_size=16,
    use_gpu=true,
    reset_every=4,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=100,
    cpuct=1.5,
    temperature=ConstSchedule(1.0),
    dirichlet_noise_ϵ=0.25,
    dirichlet_noise_α=0.3))  # No chance_mode - standard MCTS (hidden stochasticity)

arena = ArenaParams(
  sim=SimParams(
    num_games=20,
    num_workers=12,
    batch_size=12,
    use_gpu=true,
    reset_every=1,
    flip_probability=0.,
    alternate_colors=true),
  mcts = MctsParams(
    self_play.mcts,
    temperature=ConstSchedule(0.2),
    dirichlet_noise_ϵ=0.05),
  update_threshold=0.00)

learning = LearningParams(
  use_gpu=true,
  samples_weighing_policy=LOG_WEIGHT,
  l2_regularization=1e-4,
  optimiser=CyclicNesterov(
    lr_base=1e-3,
    lr_high=1e-2,
    lr_low=1e-3,
    momentum_high=0.9,
    momentum_low=0.8),
  batch_size=64,
  loss_computation_batch_size=2048,
  nonvalidity_penalty=1.,
  min_checkpoints_per_epoch=0,
  max_batches_per_checkpoint=1_000,
  num_checkpoints=1)

params = Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=10,
  memory_analysis=MemAnalysisParams(
    num_game_stages=4),
  ternary_outcome=false,
  use_symmetries=false,
  mem_buffer_size=PLSchedule(100_000))

# Benchmark sim params
benchmark_sim_agent_first = SimParams(
  arena.sim;
  num_games=200,
  num_workers=16,
  batch_size=16,
  alternate_colors=false)

benchmark_sim_random_first = SimParams(
  arena.sim;
  num_games=200,
  num_workers=16,
  batch_size=16,
  alternate_colors=false)

benchmark = [
  Benchmark.Duel(
    Benchmark.Full(self_play.mcts),
    RandomBaseline(),
    benchmark_sim_agent_first),
  Benchmark.Duel(
    RandomBaseline(),
    Benchmark.Full(self_play.mcts),
    benchmark_sim_random_first)]

experiment = Experiment(
  "backgammon-deterministic", GameSpec(), params, Network, netparams, benchmark)
