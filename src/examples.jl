module Examples

  using ..AlphaZero

  include("../games/tictactoe/main.jl")
  export Tictactoe

  const games = Dict(
    "tictactoe" => Tictactoe.GameSpec())

  const experiments = Dict(
    "tictactoe" => Tictactoe.Training.experiment)

end