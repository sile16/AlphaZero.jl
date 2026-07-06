const ALPHAZERO_REPO_ROOT = dirname(@__DIR__)
const ALPHAZERO_PARENT_DIR = dirname(ALPHAZERO_REPO_ROOT)

function backgammonnet_repo_root()
    return get(ENV, "BACKGAMMONNET_REPO",
               joinpath(ALPHAZERO_PARENT_DIR, "BackgammonNet.jl"))
end

function backgammonnet_data_dir()
    return get(ENV, "BACKGAMMONNET_DATA_DIR",
               joinpath(backgammonnet_repo_root(), "data"))
end

function backgammonnet_bootstrap_dir()
    return get(ENV, "BACKGAMMONNET_BOOTSTRAP_DIR",
               joinpath(backgammonnet_data_dir(), "bootstrap"))
end

function backgammonnet_eval_data_dir()
    return get(ENV, "BACKGAMMONNET_EVAL_DATA_DIR",
               joinpath(backgammonnet_data_dir(), "eval"))
end

backgammonnet_bootstrap_file(name::AbstractString) =
    joinpath(backgammonnet_bootstrap_dir(), name)

backgammonnet_eval_data_file(name::AbstractString) =
    joinpath(backgammonnet_eval_data_dir(), name)
