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
    reward_is_contact::Vector{Bool}   # class of each fixed starting position/game
end

EvalChunkResult(chunk_id::Integer, az_is_white::Bool, rewards::AbstractVector,
                value_nn::AbstractVector, value_opp::AbstractVector,
                value_is_contact::AbstractVector) =
    EvalChunkResult(Int(chunk_id), az_is_white, Float64.(rewards),
                    Float64.(value_nn), Float64.(value_opp),
                    Bool.(value_is_contact), fill(false, length(rewards)))

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
success, false if the chunk_id is invalid or the chunk was already completed.
"""
function submit_chunk!(job::EvalJob, result::EvalChunkResult)
    length(result.value_nn) == length(result.value_opp) ==
        length(result.value_is_contact) || throw(ArgumentError(
            "eval value arrays must have identical lengths"))
    length(result.reward_is_contact) == length(result.rewards) || throw(ArgumentError(
        "eval reward classification must match rewards"))
    idx = findfirst(c -> c.chunk_id == result.chunk_id, job.chunks)
    idx === nothing && return false
    chunk = job.chunks[idx]
    chunk.completed && return false
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
- value_mse, value_corr: NN vs opponent value accuracy overall
- contact_value_*, race_value_*: the same metrics split by position class
- num_games: total games played
"""
function finalize_eval(job::EvalJob)
    white_rewards = Float64[]
    black_rewards = Float64[]
    white_reward_is_contact = Bool[]
    black_reward_is_contact = Bool[]
    all_value_nn = Float64[]
    all_value_opp = Float64[]
    all_value_is_contact = Bool[]

    for (_, result) in job.results
        if result.az_is_white
            append!(white_rewards, result.rewards)
            append!(white_reward_is_contact, result.reward_is_contact)
        else
            append!(black_rewards, result.rewards)
            append!(black_reward_is_contact, result.reward_is_contact)
        end
        append!(all_value_nn, result.value_nn)
        append!(all_value_opp, result.value_opp)
        append!(all_value_is_contact, result.value_is_contact)
    end

    all_rewards = vcat(white_rewards, black_rewards)
    all_reward_is_contact = vcat(white_reward_is_contact, black_reward_is_contact)
    equity = isempty(all_rewards) ? 0.0 : mean(all_rewards)
    win_pct = isempty(all_rewards) ? 0.0 : mean(r -> r > 0 ? 1.0 : 0.0, all_rewards)
    white_equity = isempty(white_rewards) ? 0.0 : mean(white_rewards)
    black_equity = isempty(black_rewards) ? 0.0 : mean(black_rewards)

    function value_stats(nn, opp)
        n = length(nn)
        if n >= 2
            diffs = nn .- opp
            correlation = cor(nn, opp)
            return (mse=mean(diffs .^ 2),
                    corr=isfinite(correlation) ? correlation : 0.0, n=n)
        end
        return (mse=0.0, corr=0.0, n=n)
    end

    overall = value_stats(all_value_nn, all_value_opp)
    contact_mask = findall(all_value_is_contact)
    race_mask = findall(!, all_value_is_contact)
    contact = value_stats(all_value_nn[contact_mask], all_value_opp[contact_mask])
    race = value_stats(all_value_nn[race_mask], all_value_opp[race_mask])
    function outcome_stats(mask)
        selected = all_rewards[mask]
        isempty(selected) && return (equity=0.0, win_pct=0.0, n=0)
        return (equity=mean(selected),
                win_pct=mean(r -> r > 0 ? 1.0 : 0.0, selected),
                n=length(selected))
    end
    contact_outcomes = outcome_stats(findall(all_reward_is_contact))
    race_outcomes = outcome_stats(findall(!, all_reward_is_contact))

    (equity=equity, win_pct=win_pct,
     white_equity=white_equity, black_equity=black_equity,
     value_mse=overall.mse, value_corr=overall.corr,
     contact_value_mse=contact.mse, contact_value_corr=contact.corr,
     contact_value_n=contact.n,
     race_value_mse=race.mse, race_value_corr=race.corr,
     race_value_n=race.n,
     contact_equity=contact_outcomes.equity,
     contact_win_pct=contact_outcomes.win_pct,
     contact_games=contact_outcomes.n,
     race_equity=race_outcomes.equity,
     race_win_pct=race_outcomes.win_pct,
     race_games=race_outcomes.n,
     num_games=length(all_rewards))
end

end # module
