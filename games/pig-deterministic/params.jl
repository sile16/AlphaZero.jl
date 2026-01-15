#####
##### Hold20 Benchmark Player
#####

struct Hold20Baseline <: Benchmark.Player
  threshold::Int
end

Hold20Baseline() = Hold20Baseline(20)

Benchmark.name(p::Hold20Baseline) = "Hold$(p.threshold)"

function Benchmark.instantiate(p::Hold20Baseline, ::AbstractGameSpec, nn)
  return Hold20Player(p.threshold)
end

#####
##### Network Configuration
#####

Network = NetLib.SimpleNet

netparams = NetLib.SimpleNetHP(
  width=64,
  depth_common=4,
  use_batch_norm=true,
  batch_norm_momentum=1.)

self_play = SelfPlayParams(
  sim=SimParams(
    num_games=200,          # More games per iteration for better learning signal
    num_workers=16,
    batch_size=16,
    use_gpu=true,           # GPU for batched NN inference
    reset_every=4,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=50,  # More MCTS simulations for stronger play
    cpuct=1.0,
    temperature=ConstSchedule(1.0),
    dirichlet_noise_ϵ=0.25,
    dirichlet_noise_α=1.0))

arena = ArenaParams(
  sim=SimParams(
    num_games=25,
    num_workers=12,
    batch_size=12,
    use_gpu=true,           # GPU for batched NN inference
    reset_every=1,
    flip_probability=0.,
    alternate_colors=true),
  mcts = MctsParams(
    self_play.mcts,
    temperature=ConstSchedule(0.2),
    dirichlet_noise_ϵ=0.05),
  update_threshold=0.00)

learning = LearningParams(
  use_gpu=true,             # GPU for training
  samples_weighing_policy=LOG_WEIGHT,
  l2_regularization=1e-4,
  optimiser=CyclicNesterov(
    lr_base=1e-3,
    lr_high=1e-2,
    lr_low=1e-3,
    momentum_high=0.9,
    momentum_low=0.8),
  batch_size=32,
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
  ternary_outcome=true,
  use_symmetries=false,
  mem_buffer_size=PLSchedule(100_000))  # Large buffer so we don't hit limits

# Benchmark sim params - no color alternation so we can track first-player advantage
# 600 games per direction (1200 total) for statistical significance (SE ≈ 1.4%)
benchmark_sim_agent_first = SimParams(
  arena.sim;
  num_games=600,
  num_workers=16,
  batch_size=16,
  alternate_colors=false)

benchmark_sim_hold20_first = SimParams(
  arena.sim;
  num_games=600,
  num_workers=16,
  batch_size=16,
  alternate_colors=false)

# Benchmark against Hold20 strategy - track both directions
# Duel 1: Agent goes first (is WHITE), reports agent's win rate
# Duel 2: Hold20 goes first (is WHITE), reports Hold20's win rate (agent's loss rate)
benchmark = [
  Benchmark.Duel(
    Benchmark.Full(self_play.mcts),
    Hold20Baseline(),
    benchmark_sim_agent_first),
  Benchmark.Duel(
    Hold20Baseline(),
    Benchmark.Full(self_play.mcts),
    benchmark_sim_hold20_first)]

experiment = Experiment(
  "pig-deterministic", GameSpec(), params, Network, netparams, benchmark)
