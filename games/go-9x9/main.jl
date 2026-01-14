module Go9x9
  export GameSpec, GameEnv
  include("game.jl")
  module Training
    using AlphaZero
    import ..GameSpec
    include("params.jl")
  end
end
