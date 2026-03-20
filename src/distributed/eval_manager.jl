"""
Eval job manager for distributed evaluation.

Manages chunked eval work: creating jobs, checking out chunks to clients,
collecting results, and aggregating final statistics. Standalone module
with no HTTP or TensorBoard dependencies — the server wraps this with
endpoints and locking.
"""
module EvalManager

using Statistics: mean, cor

export EvalChunk, EvalChunkResult, EvalJob,
       create_eval_job, checkout_chunk!, submit_chunk!, extend_lease!,
       expire_stale_checkouts!, is_complete, status, finalize_eval

mutable struct EvalChunk
    chunk_id::Int
    position_range::UnitRange{Int}    # indices into eval positions array
    az_is_white::Bool                 # true = AZ plays white, false = AZ plays black
    checked_out_by::Union{Nothing, String}  # client name or nothing
    checkout_time::Float64            # time() when checked out
    completed::Bool
end

struct EvalChunkResult
    chunk_id::Int
    az_is_white::Bool
    rewards::Vector{Float64}          # one per game in the chunk
    value_nn::Vector{Float64}         # NN value predictions
    value_opp::Vector{Float64}        # opponent (wildbg) values
    value_is_contact::Vector{Bool}    # whether each value sample was from a contact position
end

mutable struct EvalJob
    iter::Int
    weights_version::Int
    chunks::Vector{EvalChunk}
    results::Dict{Int, EvalChunkResult}  # chunk_id → result
    created_at::Float64
end

"""
    create_eval_job(iter, num_positions, weights_version; chunk_size=50) -> EvalJob

Create an eval job that plays each position from both sides (white and black).
Chunks are numbered sequentially: first half for az_is_white=true, second half
for az_is_white=false.
"""
function create_eval_job(iter::Int, num_positions::Int, weights_version::Int;
                         chunk_size::Int=50)
    chunks = EvalChunk[]
    chunk_id = 0

    # White chunks
    for start in 1:chunk_size:num_positions
        stop = min(start + chunk_size - 1, num_positions)
        chunk_id += 1
        push!(chunks, EvalChunk(chunk_id, start:stop, true, nothing, 0.0, false))
    end

    # Black chunks
    for start in 1:chunk_size:num_positions
        stop = min(start + chunk_size - 1, num_positions)
        chunk_id += 1
        push!(chunks, EvalChunk(chunk_id, start:stop, false, nothing, 0.0, false))
    end

    EvalJob(iter, weights_version, chunks, Dict{Int, EvalChunkResult}(), time())
end

"""
    checkout_chunk!(job, client_name) -> Union{EvalChunk, Nothing}

Find the first available (not completed, not checked out) chunk, mark it as
checked out by `client_name`, and return it. Returns `nothing` if no chunks
are available.
"""
function checkout_chunk!(job::EvalJob, client_name::String)
    for chunk in job.chunks
        if !chunk.completed && chunk.checked_out_by === nothing
            chunk.checked_out_by = client_name
            chunk.checkout_time = time()
            return chunk
        end
    end
    return nothing
end

"""
    submit_chunk!(job, result) -> Bool

Store a completed chunk result and mark the chunk as done. Returns true on
success, false if the chunk_id is invalid.
"""
function submit_chunk!(job::EvalJob, result::EvalChunkResult)
    idx = findfirst(c -> c.chunk_id == result.chunk_id, job.chunks)
    idx === nothing && return false
    chunk = job.chunks[idx]
    chunk.completed = true
    chunk.checked_out_by = nothing
    job.results[result.chunk_id] = result
    return true
end

"""
    extend_lease!(job, chunk_id, client_name) -> Bool

Reset the checkout_time for an active checkout. Returns false if the chunk is
not checked out by the given client.
"""
function extend_lease!(job::EvalJob, chunk_id::Int, client_name::String)
    idx = findfirst(c -> c.chunk_id == chunk_id, job.chunks)
    idx === nothing && return false
    chunk = job.chunks[idx]
    if chunk.checked_out_by == client_name && !chunk.completed
        chunk.checkout_time = time()
        return true
    end
    return false
end

"""
    expire_stale_checkouts!(job; lease_seconds=300.0) -> Int

Release chunks that have been checked out longer than `lease_seconds`.
Returns the number of chunks expired.
"""
function expire_stale_checkouts!(job::EvalJob; lease_seconds::Float64=300.0)
    now = time()
    expired = 0
    for chunk in job.chunks
        if !chunk.completed && chunk.checked_out_by !== nothing
            if (now - chunk.checkout_time) > lease_seconds
                chunk.checked_out_by = nothing
                chunk.checkout_time = 0.0
                expired += 1
            end
        end
    end
    return expired
end

"""
    is_complete(job) -> Bool

Returns true if every chunk in the job has been completed.
"""
function is_complete(job::EvalJob)
    all(c -> c.completed, job.chunks)
end

"""
    status(job) -> NamedTuple

Returns a summary of the job's progress.
"""
function status(job::EvalJob)
    total = length(job.chunks)
    completed = count(c -> c.completed, job.chunks)
    checked_out = count(c -> !c.completed && c.checked_out_by !== nothing, job.chunks)
    available = total - completed - checked_out
    (eval_iter=job.iter, total_chunks=total,
     completed=completed, checked_out=checked_out, available=available)
end

"""
    finalize_eval(job) -> NamedTuple

Aggregate all chunk results into final eval statistics.

Returns a NamedTuple with:
- equity, win_pct: overall stats
- white_equity, black_equity: per-side stats
- value_mse, value_corr: NN vs opponent value accuracy
- num_games: total games played
"""
function finalize_eval(job::EvalJob)
    white_rewards = Float64[]
    black_rewards = Float64[]
    all_value_nn = Float64[]
    all_value_opp = Float64[]

    for (_, result) in job.results
        if result.az_is_white
            append!(white_rewards, result.rewards)
        else
            append!(black_rewards, result.rewards)
        end
        append!(all_value_nn, result.value_nn)
        append!(all_value_opp, result.value_opp)
    end

    all_rewards = vcat(white_rewards, black_rewards)
    equity = isempty(all_rewards) ? 0.0 : mean(all_rewards)
    win_pct = isempty(all_rewards) ? 0.0 : mean(r -> r > 0 ? 1.0 : 0.0, all_rewards)
    white_equity = isempty(white_rewards) ? 0.0 : mean(white_rewards)
    black_equity = isempty(black_rewards) ? 0.0 : mean(black_rewards)

    # Value accuracy
    if length(all_value_nn) >= 2
        diffs = all_value_nn .- all_value_opp
        value_mse = mean(diffs .^ 2)
        value_corr = cor(all_value_nn, all_value_opp)
    else
        value_mse = 0.0
        value_corr = 0.0
    end

    (equity=equity, win_pct=win_pct,
     white_equity=white_equity, black_equity=black_equity,
     value_mse=value_mse, value_corr=value_corr,
     num_games=length(all_rewards))
end

end # module
