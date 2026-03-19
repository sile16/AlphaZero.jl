module BackgammonInference

using Base.Threads

using ..AlphaZero: GI, Network
using ..FastInference: FastWeights, FastBuffers, fast_forward_normalized!, extract_fast_weights

import BackgammonNet

export OracleConfig, InputBuffers
export normalize_cpu_backend, resolve_cpu_backend, cpu_backend_summary
export make_cpu_oracles

struct OracleConfig{VF, RF}
    state_dim::Int
    num_actions::Int
    gspec::Any
    vectorize_state!::VF
    route_state::RF
end

function OracleConfig(state_dim::Int, num_actions::Int, gspec;
                      vectorize_state! = (dest, gspec_, state) -> (dest .= vec(GI.vectorize_state(gspec_, state))),
                      route_state = _ -> 1)
    OracleConfig(state_dim, num_actions, gspec, vectorize_state!, route_state)
end

struct InputBuffers
    X::Matrix{Float32}
    A::Matrix{Float32}
end

InputBuffers(state_dim::Int, num_actions::Int, max_batch::Int) =
    InputBuffers(zeros(Float32, state_dim, max_batch), zeros(Float32, num_actions, max_batch))

struct OracleScratch
    primary_input::InputBuffers
    secondary_input::Union{InputBuffers, Nothing}
    primary_idxs::Vector{Int}
    secondary_idxs::Vector{Int}
    policy_bufs::Vector{Vector{Float32}}
    results::Vector{Tuple{Vector{Float32}, Float32}}
end

function OracleScratch(state_dim::Int, num_actions::Int, max_batch::Int; dual::Bool)
    OracleScratch(
        InputBuffers(state_dim, num_actions, max_batch),
        dual ? InputBuffers(state_dim, num_actions, max_batch) : nothing,
        Vector{Int}(undef, max_batch),
        Vector{Int}(undef, max_batch),
        [Vector{Float32}(undef, num_actions) for _ in 1:max_batch],
        Vector{Tuple{Vector{Float32}, Float32}}(undef, max_batch))
end

struct FastWorkerBuffers
    scratch::OracleScratch
    primary_fast::FastBuffers
    secondary_fast::Union{FastBuffers, Nothing}
end

function normalize_cpu_backend(name::Union{Symbol, AbstractString})
    backend = Symbol(lowercase(String(name)))
    backend in (:auto, :fast, :flux) ||
        error("Unsupported inference backend: $name (expected auto, fast, or flux)")
    return backend
end

default_cpu_backend() = :fast

function resolve_cpu_backend(name::Union{Symbol, AbstractString}=:auto)
    backend = normalize_cpu_backend(name)
    return backend == :auto ? default_cpu_backend() : backend
end

function cpu_backend_summary(backend::Union{Symbol, AbstractString})
    resolved = resolve_cpu_backend(backend)
    if resolved == :fast
        return "FastWeights"
    else
        return "Flux/BLAS"
    end
end

function _worker_slot(buffers)
    return min(threadid(), length(buffers))
end

function _populate_column!(cfg::OracleConfig, X::Matrix{Float32}, A::Matrix{Float32}, col::Int, state)
    cfg.vectorize_state!(@view(X[:, col]), cfg.gspec, state)
    a_col = @view(A[:, col])
    fill!(a_col, 0.0f0)
    if !BackgammonNet.game_terminated(state) && !BackgammonNet.is_chance_node(state)
        actions = GI.available_actions(GI.init(cfg.gspec, state))
        @inbounds for action in actions
            if 1 <= action <= cfg.num_actions
                a_col[action] = 1.0f0
            end
        end
    end
end

function _pack_states!(scratch::OracleScratch, states, cfg::OracleConfig)
    n_primary = 0
    n_secondary = 0
    dual = scratch.secondary_input !== nothing
    primary_input = scratch.primary_input
    secondary_input = scratch.secondary_input

    @inbounds for (idx, state) in enumerate(states)
        route = dual ? cfg.route_state(state) : 1
        if route == 2 && dual
            n_secondary += 1
            scratch.secondary_idxs[n_secondary] = idx
            _populate_column!(cfg, secondary_input.X, secondary_input.A, n_secondary, state)
        else
            n_primary += 1
            scratch.primary_idxs[n_primary] = idx
            _populate_column!(cfg, primary_input.X, primary_input.A, n_primary, state)
        end
    end

    return n_primary, n_secondary
end

function _store_results!(scratch::OracleScratch, idxs::Vector{Int}, A::Matrix{Float32},
                         P::AbstractMatrix, V, n::Int, num_actions::Int)
    @inbounds for j in 1:n
        idx = idxs[j]
        pv = scratch.policy_bufs[idx]
        k = 0
        for a in 1:num_actions
            if A[a, j] > 0.0f0
                k += 1
                pv[k] = P[a, j]
            end
        end
        resize!(pv, k)
        v = V isa AbstractVector ? V[j] : V[1, j]
        scratch.results[idx] = (pv, v)
    end
end

function _forward_fast!(scratch::OracleScratch, fw::FastWeights, fb::FastBuffers,
                        input::InputBuffers, idxs::Vector{Int}, n::Int, cfg::OracleConfig)
    n == 0 && return
    P, V, _ = fast_forward_normalized!(fw, fb, input.X, input.A, n)
    _store_results!(scratch, idxs, input.A, P, V, n, cfg.num_actions)
end

function _make_fast_buffers(cfg::OracleConfig, batch_size::Int, nslots::Int,
                            primary_width::Int; secondary_width::Union{Int, Nothing}=nothing)
    max_batch = batch_size + 1
    dual = secondary_width !== nothing
    [FastWorkerBuffers(
        OracleScratch(cfg.state_dim, cfg.num_actions, max_batch; dual),
        FastBuffers(primary_width, cfg.num_actions, max_batch),
        dual ? FastBuffers(secondary_width, cfg.num_actions, max_batch) : nothing
    ) for _ in 1:nslots]
end

function make_cpu_oracles(backend::Union{Symbol, AbstractString},
                          primary_net,
                          cfg::OracleConfig;
                          secondary_net=nothing,
                          batch_size::Int,
                          primary_fw::Union{FastWeights, Nothing}=nothing,
                          secondary_fw::Union{FastWeights, Nothing}=nothing,
                          nslots::Int=Threads.nthreads())
    resolved = resolve_cpu_backend(backend)
    dual = secondary_net !== nothing || secondary_fw !== nothing

    if resolved == :flux
        function flux_batch_oracle(states::Vector)
            n = length(states)
            n == 0 && return Tuple{Vector{Float32}, Float32}[]
            if !dual
                return Network.evaluate_batch(primary_net, states)
            end
            primary_states = eltype(states)[]
            secondary_states = eltype(states)[]
            primary_idxs = Int[]
            secondary_idxs = Int[]
            for (idx, state) in enumerate(states)
                if cfg.route_state(state) == 2
                    push!(secondary_states, state)
                    push!(secondary_idxs, idx)
                else
                    push!(primary_states, state)
                    push!(primary_idxs, idx)
                end
            end
            results = Vector{Tuple{Vector{Float32}, Float32}}(undef, n)
            if !isempty(primary_states)
                evals = Network.evaluate_batch(primary_net, primary_states)
                for (j, idx) in enumerate(primary_idxs)
                    results[idx] = evals[j]
                end
            end
            if !isempty(secondary_states)
                evals = Network.evaluate_batch(secondary_net, secondary_states)
                for (j, idx) in enumerate(secondary_idxs)
                    results[idx] = evals[j]
                end
            end
            return results
        end
        flux_single_oracle(state) = flux_batch_oracle([state])[1]
        return flux_single_oracle, flux_batch_oracle
    end

    primary_fw = isnothing(primary_fw) ? extract_fast_weights(primary_net) : primary_fw
    if dual
        secondary_fw = isnothing(secondary_fw) ? extract_fast_weights(secondary_net) : secondary_fw
    end
    primary_width = size(primary_fw.W_in, 1)
    secondary_width = dual ? size(secondary_fw.W_in, 1) : nothing
    buffers = _make_fast_buffers(cfg, batch_size, nslots, primary_width; secondary_width)

    function fast_batch_oracle(states::Vector)
        n = length(states)
        n == 0 && return Tuple{Vector{Float32}, Float32}[]
        worker = buffers[_worker_slot(buffers)]
        scratch = worker.scratch
        n_primary, n_secondary = _pack_states!(scratch, states, cfg)
        _forward_fast!(scratch, primary_fw, worker.primary_fast, scratch.primary_input, scratch.primary_idxs, n_primary, cfg)
        if dual
            _forward_fast!(scratch, secondary_fw, worker.secondary_fast, scratch.secondary_input, scratch.secondary_idxs, n_secondary, cfg)
        end
        return @view(scratch.results[1:n])
    end

    fast_single_oracle(state) = fast_batch_oracle([state])[1]
    return fast_single_oracle, fast_batch_oracle
end

end
