#####
##### TensorBoard Logging Integration
#####
##### Pure Julia implementation using TensorBoardLogger.jl
##### No Python dependencies - works alongside PyCall (GnuBG)
#####

module TensorBoard

using TensorBoardLogger
using TensorBoardLogger: with_logger

# Global logger reference
const _tb_logger = Ref{Union{TBLogger, Nothing}}(nothing)

"""
    tb_init(; logdir, run_name=nothing)

Initialize TensorBoard logging.

# Arguments
- `logdir::String`: Directory to write TensorBoard logs
- `run_name::String=nothing`: Optional run name (appended to logdir)

# Example
```julia
tb_init(logdir="runs", run_name="experiment-001")
# View with: tensorboard --logdir=runs/experiment-001
```
"""
function tb_init(; logdir::String, run_name::Union{String, Nothing}=nothing)
    if !isnothing(run_name)
        logdir = joinpath(logdir, run_name)
    end
    mkpath(logdir)
    _tb_logger[] = TBLogger(logdir, tb_append)
    @info "TensorBoard initialized: $logdir"
    @info "View with: tensorboard --logdir=$logdir"
    return _tb_logger[]
end

"""
    tb_log(metrics::Dict; step=nothing)

Log metrics to TensorBoard.

# Arguments
- `metrics::Dict`: Dictionary of metric names to values
- `step::Int=nothing`: Optional step number (uses internal counter if not provided)
"""
function tb_log(metrics::Dict; step::Union{Int, Nothing}=nothing)
    logger = _tb_logger[]
    if isnothing(logger)
        @warn "TensorBoard not initialized, call tb_init() first"
        return
    end

    with_logger(logger) do
        for (name, value) in metrics
            if value isa Number && isfinite(value)
                if isnothing(step)
                    @info name value
                else
                    @info name value log_step_increment=0
                    TensorBoardLogger.set_step!(logger, step)
                    @info name value
                end
            end
        end
    end
end

"""
    tb_log_config(config::Dict)

Log configuration/hyperparameters as text.
"""
function tb_log_config(config::Dict)
    logger = _tb_logger[]
    if isnothing(logger)
        return
    end

    # Format config as markdown table
    lines = ["# Configuration", "", "| Parameter | Value |", "|-----------|-------|"]
    for (k, v) in sort(collect(config), by=first)
        push!(lines, "| $k | $v |")
    end
    text = join(lines, "\n")

    with_logger(logger) do
        @info "config" text=text
    end
end

"""
    tb_finish()

Finish TensorBoard logging (close the logger).
"""
function tb_finish()
    logger = _tb_logger[]
    if !isnothing(logger)
        close(logger)
        _tb_logger[] = nothing
    end
end

"""
    tb_available() -> Bool

Check if TensorBoard logger is initialized.
"""
tb_available() = !isnothing(_tb_logger[])

#####
##### System metrics collection (copied from wandb.jl)
#####

"""
    system_metrics(; prefix="system", host_id=nothing) -> Dict

Collect system metrics: CPU load, memory usage.
"""
function system_metrics(; prefix::String="system", host_id::Union{String, Nothing}=nothing)
    p = isnothing(host_id) ? prefix : "$(prefix)/$(host_id)"
    metrics = Dict{String, Any}()

    try
        # CPU load (1-minute average on Linux)
        if Sys.islinux()
            loadavg = open("/proc/loadavg") do f
                parse(Float64, split(readline(f))[1])
            end
            metrics["$(p)/cpu_load_1m"] = loadavg
        end

        # Memory usage
        if Sys.islinux()
            meminfo = Dict{String,Int}()
            open("/proc/meminfo") do f
                for line in eachline(f)
                    parts = split(line)
                    if length(parts) >= 2
                        key = replace(parts[1], ":" => "")
                        val = tryparse(Int, parts[2])
                        if !isnothing(val)
                            meminfo[key] = val  # in KB
                        end
                    end
                end
            end

            total = get(meminfo, "MemTotal", 0) / 1024 / 1024  # GB
            available = get(meminfo, "MemAvailable", 0) / 1024 / 1024
            used = total - available
            metrics["$(p)/ram_total_gb"] = total
            metrics["$(p)/ram_used_gb"] = used
            metrics["$(p)/ram_used_pct"] = total > 0 ? (used / total) * 100 : 0.0
        end
    catch e
        # Silently ignore errors - system metrics are optional
    end

    return metrics
end

"""
    gpu_metrics(; prefix="system", host_id=nothing, cuda_module=nothing) -> Dict

Collect GPU metrics: memory usage, utilization.
"""
function gpu_metrics(; prefix::String="system", host_id::Union{String, Nothing}=nothing, cuda_module=nothing)
    p = isnothing(host_id) ? prefix : "$(prefix)/$(host_id)"
    metrics = Dict{String, Any}()

    try
        # Try using provided CUDA module first
        if !isnothing(cuda_module) && cuda_module.functional()
            total_mem = cuda_module.total_memory()
            free_mem = cuda_module.available_memory()
            used_mem = total_mem - free_mem

            metrics["$(p)/gpu_mem_total_gb"] = total_mem / 1e9
            metrics["$(p)/gpu_mem_used_gb"] = used_mem / 1e9
            metrics["$(p)/gpu_mem_free_gb"] = free_mem / 1e9
            metrics["$(p)/gpu_mem_used_pct"] = (used_mem / total_mem) * 100
        end

        # GPU utilization requires nvidia-smi
        try
            result = read(`nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits`, String)
            util = parse(Float64, strip(result))
            metrics["$(p)/gpu_utilization_pct"] = util
        catch
        end
    catch e
        # Silently ignore - GPU metrics are optional
    end

    return metrics
end

"""
    all_system_metrics(; host_id=nothing, cuda_module=nothing) -> Dict

Collect all system metrics (CPU, RAM, GPU).
"""
function all_system_metrics(; host_id::Union{String, Nothing}=nothing, cuda_module=nothing)
    metrics = system_metrics(host_id=host_id)
    merge!(metrics, gpu_metrics(host_id=host_id, cuda_module=cuda_module))
    return metrics
end

export tb_init, tb_log, tb_log_config, tb_finish, tb_available
export system_metrics, gpu_metrics, all_system_metrics

end # module TensorBoard
