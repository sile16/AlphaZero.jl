module BackgammonDeterministic
  export GameEnv, GameSpec, RandomPlayer
  include("game.jl")
  module Training
    using AlphaZero
    import ..GameSpec, ..RandomPlayer
    include("params.jl")
  end
end
