module BackgammonInference

using Base.Threads

using ..AlphaZero: GI, Network
using ..FastInference: FastWeights, FastBuffers, fast_forward_normalized!, extract_fast_weights

import BackgammonNet

export OracleConfig, InputBuffers
export normalize_cpu_backend, resolve_cpu_backend, cpu_backend_summary
export make_cpu_oracles, make_gpu_oracles, make_gpu_server_oracles

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
    results::Vector{Tuple{Vector{Float32}, Float32}}
end

function OracleScratch(state_dim::Int, num_actions::Int, max_batch::Int; dual::Bool)
    OracleScratch(
        InputBuffers(state_dim, num_actions, max_batch),
        dual ? InputBuffers(state_dim, num_actions, max_batch) : nothing,
        Vector{Int}(undef, max_batch),
        Vector{Int}(undef, max_batch),
        Vector{Tuple{Vector{Float32}, Float32}}(undef, max_batch))
end

struct FastWorkerBuffers
    scratch::OracleScratch
    primary_fast::FastBuffers
    secondary_fast::Union{FastBuffers, Nothing}
end

struct GpuWorkerBuffers
    scratch::OracleScratch
end

struct GpuOracleRequest
    states::Vector
    answer_channel::Channel{Vector{Tuple{Vector{Float32}, Float32}}}
end

struct GpuOracleServer
    request_channel::Channel{GpuOracleRequest}
    task::Task
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

function _populate_column!(cfg::OracleConfig, X::Matrix{Float32}, A::Matrix{Float32}, col::Int, state)
    actions = Int[]
    if !BackgammonNet.game_terminated(state) && !BackgammonNet.is_chance_node(state)
        actions = GI.available_actions(GI.init(cfg.gspec, state))
    end
    cfg.vectorize_state!(@view(X[:, col]), cfg.gspec, state)
    a_col = @view(A[:, col])
    fill!(a_col, 0.0f0)
    @inbounds for action in actions
        if 1 <= action <= cfg.num_actions
            a_col[action] = 1.0f0
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
        k = 0
        for a in 1:num_actions
            if A[a, j] > 0.0f0
                k += 1
            end
        end
        pv = Vector{Float32}(undef, k)
        k = 0
        for a in 1:num_actions
            if A[a, j] > 0.0f0
                k += 1
                pv[k] = P[a, j]
            end
        end
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

function _new_fast_worker_buffers(cfg::OracleConfig, batch_size::Int,
                                  primary_width::Int; secondary_width::Union{Int, Nothing}=nothing)
    max_batch = batch_size + 1
    dual = secondary_width !== nothing
    FastWorkerBuffers(
        OracleScratch(cfg.state_dim, cfg.num_actions, max_batch; dual),
        FastBuffers(primary_width, cfg.num_actions, max_batch),
        dual ? FastBuffers(secondary_width, cfg.num_actions, max_batch) : nothing
    )
end

function _new_gpu_worker_buffers(cfg::OracleConfig, batch_size::Int;
                                 secondary::Bool=false)
    max_batch = batch_size + 1
    GpuWorkerBuffers(OracleScratch(cfg.state_dim, cfg.num_actions, max_batch; dual=secondary))
end

function _task_buffer_resolver(initfn)
    buffers_lock = ReentrantLock()
    buffers_by_task = IdDict{Task, Any}()

    function task_buffers()
        task = current_task()
        lock(buffers_lock) do
            return get!(buffers_by_task, task) do
                initfn()
            end
        end
    end

    return task_buffers
end

function _forward_gpu!(scratch::OracleScratch, net_gpu,
                       input::InputBuffers, idxs::Vector{Int}, n::Int,
                       cfg::OracleConfig, gpu_array_fn, sync_fn, gpu_lock)
    n == 0 && return
    local P_cpu, V_cpu
    X_active = @view(input.X[:, 1:n])
    A_active = @view(input.A[:, 1:n])
    lock(gpu_lock) do
        X_g = gpu_array_fn(X_active)
        A_g = gpu_array_fn(A_active)
        result = Network.forward_normalized(net_gpu, X_g, A_g)
        sync_fn()
        P_cpu = Array(result[1])
        V_cpu = Array(result[2])
    end
    _store_results!(scratch, idxs, input.A, P_cpu, V_cpu, n, cfg.num_actions)
end

function _gpu_server_loop!(server_scratch::OracleScratch, request_channel::Channel{GpuOracleRequest},
                           primary_net_gpu, secondary_net_gpu, cfg::OracleConfig,
                           max_batch_size::Int, max_wait_ns::Int,
                           gpu_array_fn, sync_fn, gpu_lock)
    dual = secondary_net_gpu !== nothing
    pending = GpuOracleRequest[]

    while true
        while isready(request_channel)
            try
                push!(pending, take!(request_channel))
            catch err
                err isa InvalidStateException || rethrow()
                break
            end
        end

        if isempty(pending)
            if !isopen(request_channel)
                break
            end
            try
                push!(pending, take!(request_channel))
            catch err
                if err isa InvalidStateException
                    break
                end
                rethrow()
            end
        end

        t0 = time_ns()
        total_states = sum(length(req.states) for req in pending)
        while total_states < max_batch_size && isopen(request_channel)
            if isready(request_channel)
                try
                    req = take!(request_channel)
                    push!(pending, req)
                    total_states += length(req.states)
                catch err
                    err isa InvalidStateException || rethrow()
                    break
                end
            elseif time_ns() - t0 >= max_wait_ns
                break
            else
                yield()
            end
        end

        isempty(pending) && continue

        used = 0
        nreq = 0
        while nreq < length(pending)
            next_count = length(pending[nreq + 1].states)
            if nreq > 0 && used + next_count > max_batch_size
                break
            end
            nreq += 1
            used += next_count
        end

        batch_requests = pending[1:nreq]
        deleteat!(pending, 1:nreq)

        first_states = batch_requests[1].states
        combined_states = Vector{eltype(first_states)}(undef, used)
        offsets = Vector{Int}(undef, nreq + 1)
        offsets[1] = 1
        pos = 1
        for i in 1:nreq
            states_i = batch_requests[i].states
            copyto!(combined_states, pos, states_i, 1, length(states_i))
            pos += length(states_i)
            offsets[i + 1] = pos
        end

        n_primary, n_secondary = _pack_states!(server_scratch, combined_states, cfg)
        _forward_gpu!(server_scratch, primary_net_gpu, server_scratch.primary_input, server_scratch.primary_idxs, n_primary,
                      cfg, gpu_array_fn, sync_fn, gpu_lock)
        if dual
            _forward_gpu!(server_scratch, secondary_net_gpu, server_scratch.secondary_input, server_scratch.secondary_idxs, n_secondary,
                          cfg, gpu_array_fn, sync_fn, gpu_lock)
        end

        for i in 1:nreq
            lo = offsets[i]
            hi = offsets[i + 1] - 1
            put!(batch_requests[i].answer_channel, copy(server_scratch.results[lo:hi]))
        end
    end
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
    task_buffers = _task_buffer_resolver() do
        _new_fast_worker_buffers(cfg, batch_size, primary_width; secondary_width)
    end

    function fast_batch_oracle(states::Vector)
        n = length(states)
        n == 0 && return Tuple{Vector{Float32}, Float32}[]
        worker = task_buffers()
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

function make_gpu_oracles(primary_net_gpu,
                          cfg::OracleConfig;
                          secondary_net_gpu=nothing,
                          batch_size::Int,
                          gpu_array_fn,
                          sync_fn,
                          gpu_lock=ReentrantLock(),
                          nslots::Int=Threads.nthreads())
    dual = secondary_net_gpu !== nothing
    task_buffers = _task_buffer_resolver() do
        _new_gpu_worker_buffers(cfg, batch_size; secondary=dual)
    end

    function gpu_batch_oracle(states::Vector)
        n = length(states)
        n == 0 && return Tuple{Vector{Float32}, Float32}[]
        worker = task_buffers()
        scratch = worker.scratch
        n_primary, n_secondary = _pack_states!(scratch, states, cfg)
        _forward_gpu!(scratch, primary_net_gpu, scratch.primary_input, scratch.primary_idxs, n_primary,
                      cfg, gpu_array_fn, sync_fn, gpu_lock)
        if dual
            _forward_gpu!(scratch, secondary_net_gpu, scratch.secondary_input, scratch.secondary_idxs, n_secondary,
                          cfg, gpu_array_fn, sync_fn, gpu_lock)
        end
        return @view(scratch.results[1:n])
    end

    gpu_single_oracle(state) = gpu_batch_oracle([state])[1]
    return gpu_single_oracle, gpu_batch_oracle
end

function make_gpu_server_oracles(primary_net_gpu,
                                 cfg::OracleConfig;
                                 secondary_net_gpu=nothing,
                                 batch_size::Int,
                                 num_workers::Int,
                                 gpu_array_fn,
                                 sync_fn,
                                 gpu_lock=ReentrantLock(),
                                 max_wait_ns::Int=1_000_000)
    dual = secondary_net_gpu !== nothing
    max_batch_size = max(batch_size, batch_size * max(num_workers, 1))
    server_buffers = _new_gpu_worker_buffers(cfg, max_batch_size; secondary=dual)
    request_channel = Channel{GpuOracleRequest}(max(num_workers * 2, 1))
    server_task = Threads.@spawn _gpu_server_loop!(
        server_buffers.scratch, request_channel,
        primary_net_gpu, secondary_net_gpu, cfg,
        max_batch_size, max_wait_ns,
        gpu_array_fn, sync_fn, gpu_lock)
    server = GpuOracleServer(request_channel, server_task)

    function gpu_batch_oracle(states::Vector)
        isempty(states) && return Tuple{Vector{Float32}, Float32}[]
        answer_channel = Channel{Vector{Tuple{Vector{Float32}, Float32}}}(1)
        put!(request_channel, GpuOracleRequest(states, answer_channel))
        return take!(answer_channel)
    end

    gpu_single_oracle(state) = gpu_batch_oracle([state])[1]
    return gpu_single_oracle, gpu_batch_oracle, server
end

function Base.close(server::GpuOracleServer)
    close(server.request_channel)
    wait(server.task)
    return nothing
end

end
