module BackgammonDeterministic
  export GameEnv, GameSpec, backgammon_game, init_with_rng,
         backgammon_ml_contract
  include("game.jl")
end
