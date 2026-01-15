module PigDeterministic
  export GameEnv, GameSpec, Hold20Player
  include("game.jl")
  module Training
    using AlphaZero
    import ..GameSpec, ..Hold20Player
    include("params.jl")
  end
end
