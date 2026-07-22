#####
##### AlphaZero.jl
##### Jonathan Laurent, Carnegie Mellon University (2019-2021)
#####

module AlphaZero

  import Distributed
  import Random
  using Format
  using Base: @kwdef
  using DataStructures: CircularBuffer
  using Distributions: Categorical, Dirichlet
  using Statistics: mean
  using Requires


  # Even when using the Knet backend, we use utilities from Flux such as
  # `Flux.batch` and `Flux.DataLoader`
  import Flux

  # When running on a CPU, having multiple threads does not play
  # well with BLAS multithreading
  import LinearAlgebra
  LinearAlgebra.BLAS.set_num_threads(1)

  # Internal helper functions
  include("util.jl")
  using .Util
  export Util
  export apply_temperature
  include("prof_utils.jl")
  using .ProfUtils

  # A generic interface for single-player or zero-sum two-players games.
  include("game.jl")
  using .GameInterface
  const GI = GameInterface
  export GameInterface, GI
  export AbstractGameEnv
  export AbstractGameSpec

  # A standalone, generic MCTS implementation
  include("mcts.jl")
  using .MCTS
  export MCTS

  # Batched MCTS for improved GPU utilization
  include("batched_mcts.jl")
  using .BatchedMCTS
  export BatchedMCTS

  # A generic network interface
  include("networks/network.jl")
  using .Network
  export Network
  export AbstractNetwork
  export OptimiserSpec
  export CyclicNesterov, Adam

  # Schedules
  include("schedule.jl")
  export AbstractSchedule
  export ConstSchedule, PLSchedule, StepSchedule, CyclicSchedule

  # Training params
  include("params.jl")
  export Params
  export MctsParams
  export SimParams
  export SelfPlayParams
  export LearningParams
  export ArenaParams
  export MemAnalysisParams
  export ProgressiveSimParams, compute_sim_budget
  export TurnProgressiveSimParams, compute_turn_sim_budget, compute_ramp_turns
  export SamplesWeighingPolicy, CONSTANT_WEIGHT, LOG_WEIGHT, LINEAR_WEIGHT

  # Unified game loop (replaces duplicate play loops in scripts)
  include("game_loop.jl")
  using .GameLoop
  export GameLoop

  # Stats about training
  include("report.jl")
  export Report

  # Game traces
  include("trace.jl")
  export Trace
  export total_reward

  # Memory buffer to hold samples generated during self-play
  include("memory.jl")
  export MemoryBuffer, get_experience
  export VALUE_HEAD_CONTRACT, VALUE_HEAD_ORDER, VALUE_HEAD_STRICT_TOL
  export EquityTargets, equity_targets_from_outcome
  export equity_vector, equity_vector_from_outcome, flip_equity_perspective

  # Utilities to train the neural network based on collected samples
  include("learning.jl")
  export split_equity_targets

  # We provide a library of standard network, both in Knet and Flux.
  # Which backend is used to implement this library is determined during precompilation
  # based on the value of the ALPHAZERO_DEFAULT_DL_FRAMEWORK environment variable.
  const DEFAULT_DL_FRAMEWORK = get(ENV, "ALPHAZERO_DEFAULT_DL_FRAMEWORK", "FLUX")

  if DEFAULT_DL_FRAMEWORK == "FLUX"
    # @info "Using the Flux implementation of AlphaZero.NetLib."
    @eval begin
      include("networks/flux.jl")
      const NetLib = FluxLib
    end
  elseif DEFAULT_DL_FRAMEWORK == "KNET"
    error("The Knet backend is currently not available, due to Knet not supporting the latest versions of Julia and CUDA.")
    @info "Using the Knet implementation of AlphaZero.NetLib."
    @eval begin
      include("networks/knet.jl")
      const NetLib = KnetLib
    end
  else
    error("Unknown DL framework: $(DEFAULT_DL_FRAMEWORK)")
  end

  using .NetLib
  export NetLib
  export SimpleNet, SimpleNetHP, ResNet, ResNetHP, FCResNet, FCResNetHP

  # Inference backends used by self-play and evaluation scripts
  include("inference/fast_weights.jl")
  using .FastInference
  export FastInference
  include("inference/backgammon_oracles.jl")
  using .BackgammonInference
  export BackgammonInference

end
