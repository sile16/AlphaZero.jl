module BackgammonDeterministic
  export GameEnv, GameSpec, RandomPlayer, backgammon_game, init_with_rng,
         backgammon_ml_contract
  include("game.jl")
  module Training
    using AlphaZero
    import ..GameSpec, ..RandomPlayer
    include("params.jl")
  end
end
