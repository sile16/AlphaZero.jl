module ConnectFour
  export GameSpec, GameEnv, Board
  include("game.jl")
  module Training
    using AlphaZero
    import ..GameSpec
    include("params.jl")
  end
  module TrainingProgressive
    using AlphaZero
    import ..GameSpec
    include("params_progressive.jl")
  end
  module TrainingTurnProgressive
    using AlphaZero
    import ..GameSpec
    include("params_turn_progressive.jl")
  end
  include("solver.jl")
end