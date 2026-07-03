"""
Thread-safe Prioritized Experience Replay (PER) buffer.

Columnar storage with circular write semantics:
- Pre-allocated matrices (no per-sample allocation)
- O(1) append via circular write pointer (no deleteat! shifts)
- Samples ingested directly from columnar SampleBatch wire format
- Lock held only during memcpy, not during deserialization
"""

"""Columnar PER buffer with circular write semantics.

Pre-allocates all storage at construction time. New samples overwrite
the oldest entries via a circular write pointer — no memory shifting."""
mutable struct PERBuffer
    # Columnar sample storage (pre-allocated to capacity)
    states::Matrix{Float32}       # (state_dim, capacity)
    policies::Matrix{Float32}     # (num_actions, capacity)
    values::Vector{Float32}       # (capacity,)
    equities::Matrix{Float32}     # (5, capacity)
    has_equity::Vector{Bool}      # (capacity,)
    is_chance::Vector{Bool}       # (capacity,)
    is_contact::Vector{Bool}      # (capacity,)
    is_bearoff::Vector{Bool}      # (capacity,)

    # PER priorities
    priorities::Vector{Float32}   # (capacity,)

    # Per-slot write generation: incremented every time a slot is (re)written by
    # per_add_batch!. Lets reanalyze detect a slot that was overwritten between
    # its extract and its (slow, unlocked) NN inference, so it does not blend a
    # stale prediction into a newer sample.
    generation::Vector{UInt32}    # (capacity,)

    # Circular buffer state
    capacity::Int
    size::Int                     # Current number of valid samples (≤ capacity)
    write_pos::Int                # Next write position (1-based, wraps around)

    # PER annealing
    beta::Float32
    beta_init::Float32
    beta_annealing_iters::Int
    current_iter::Int

    # Thread safety
    lock::ReentrantLock
end

function PERBuffer(capacity::Int, state_dim::Int, num_actions::Int;
                   beta_init=0.4f0, annealing_iters=200)
    PERBuffer(
        zeros(Float32, state_dim, capacity),
        zeros(Float32, num_actions, capacity),
        zeros(Float32, capacity),
        zeros(Float32, 5, capacity),
        fill(false, capacity),
        fill(false, capacity),
        fill(false, capacity),
        fill(false, capacity),
        ones(Float32, capacity),     # Initial priority = 1.0
        zeros(UInt32, capacity),     # generation (0 = never written)
        capacity, 0, 1,
        beta_init, beta_init, annealing_iters, 0,
        ReentrantLock(),
    )
end

"""Add columnar samples directly from a SampleBatch (no NamedTuple allocation).

Copies columns from the batch into the circular buffer. O(n) memcpy, O(1) amortized
per sample (no shifting). Lock is held only during the copy."""
function per_add_batch!(buf::PERBuffer, batch_states::AbstractMatrix{Float32},
                        batch_policies::AbstractMatrix{Float32},
                        batch_values::AbstractVector{Float32},
                        batch_equities::AbstractMatrix{Float32},
                        batch_has_equity::AbstractVector{Bool},
                        batch_is_chance::AbstractVector{Bool},
                        batch_is_contact::AbstractVector{Bool},
                        batch_is_bearoff::AbstractVector{Bool};
                        initial_priority::Float32=1.0f0)
    n = length(batch_values)
    n == 0 && return

    lock(buf.lock) do
        for i in 1:n
            pos = buf.write_pos

            # Copy single column — tight inner loop, no allocation
            @inbounds begin
                for j in axes(buf.states, 1)
                    buf.states[j, pos] = batch_states[j, i]
                end
                for j in axes(buf.policies, 1)
                    buf.policies[j, pos] = batch_policies[j, i]
                end
                buf.values[pos] = batch_values[i]
                for j in 1:5
                    buf.equities[j, pos] = batch_equities[j, i]
                end
                buf.has_equity[pos] = batch_has_equity[i]
                buf.is_chance[pos] = batch_is_chance[i]
                buf.is_contact[pos] = batch_is_contact[i]
                buf.is_bearoff[pos] = batch_is_bearoff[i]
                buf.priorities[pos] = initial_priority
                buf.generation[pos] += UInt32(1)
            end

            # Advance circular pointer
            buf.write_pos = (pos % buf.capacity) + 1
            buf.size = min(buf.size + 1, buf.capacity)
        end
    end
end

function per_anneal_beta!(buf::PERBuffer)
    buf.current_iter += 1
    frac = min(1.0f0, Float32(buf.current_iter) / Float32(buf.beta_annealing_iters))
    buf.beta = buf.beta_init + (1.0f0 - buf.beta_init) * frac
end

"""Sample from PER buffer. Returns (indices, importance_weights).
Uses precomputed cumulative sum + binary search for O(batch_size * log(n)) sampling.

Indices are into the buffer's column positions (1:buf.size)."""
function per_sample(buf::PERBuffer, batch_size::Int, alpha::Float32, epsilon::Float32)
    lock(buf.lock) do
        n = buf.size
        n < batch_size && error("Buffer too small ($n < $batch_size)")

        # Compute priorities^alpha and cumulative sum
        cumsum_pα = Vector{Float32}(undef, n)
        @inbounds begin
            cumsum_pα[1] = (buf.priorities[1] + epsilon) ^ alpha
            for i in 2:n
                cumsum_pα[i] = cumsum_pα[i-1] + (buf.priorities[i] + epsilon) ^ alpha
            end
        end
        total = cumsum_pα[n]

        # Proportional sampling via binary search on cumsum
        indices = Vector{Int}(undef, batch_size)
        for j in 1:batch_size
            r = rand(Float32) * total
            lo, hi = 1, n
            while lo < hi
                mid = (lo + hi) >>> 1
                if cumsum_pα[mid] < r
                    lo = mid + 1
                else
                    hi = mid
                end
            end
            indices[j] = lo
        end

        # Importance sampling weights: w_i = (N * P(i))^(-beta) / max(w)
        inv_total = 1.0f0 / total
        weights = Vector{Float32}(undef, batch_size)
        @inbounds for j in 1:batch_size
            idx = indices[j]
            pα_i = idx == 1 ? cumsum_pα[1] : cumsum_pα[idx] - cumsum_pα[idx-1]
            weights[j] = (Float32(n) * pα_i * inv_total) ^ (-buf.beta)
        end
        max_w = maximum(weights)
        weights ./= max_w

        return (indices, weights)
    end
end

"""Sample from a SUBSET of buffer indices using PER priorities.

Given `subset_indices` (e.g. contact-only or race-only indices from partition_indices),
samples `batch_size` entries proportional to their PER priorities.
Returns (sampled_buf_indices, importance_weights)."""
function per_sample_partition(buf::PERBuffer, subset_indices::Vector{Int},
                              batch_size::Int, alpha::Float32, epsilon::Float32)
    n = length(subset_indices)
    n < batch_size && error("Subset too small ($n < $batch_size)")

    # Compute priorities^alpha and cumulative sum over the subset. The priority
    # snapshot is taken UNDER the buffer lock (per_update_priorities! /
    # per_add_batch! mutate priorities concurrently); the sampling math below runs
    # on the local snapshot outside the lock. buf.lock is reentrant, so nesting
    # under a caller that already holds it is safe.
    cumsum_pα = Vector{Float32}(undef, n)
    lock(buf.lock) do
        @inbounds begin
            cumsum_pα[1] = (buf.priorities[subset_indices[1]] + epsilon) ^ alpha
            for i in 2:n
                cumsum_pα[i] = cumsum_pα[i-1] + (buf.priorities[subset_indices[i]] + epsilon) ^ alpha
            end
        end
    end
    total = cumsum_pα[n]

    # Proportional sampling via binary search on cumsum
    local_indices = Vector{Int}(undef, batch_size)
    for j in 1:batch_size
        r = rand(Float32) * total
        lo, hi = 1, n
        while lo < hi
            mid = (lo + hi) >>> 1
            if cumsum_pα[mid] < r
                lo = mid + 1
            else
                hi = mid
            end
        end
        local_indices[j] = lo
    end

    # Map back to buffer indices
    buf_indices = Vector{Int}(undef, batch_size)
    @inbounds for j in 1:batch_size
        buf_indices[j] = subset_indices[local_indices[j]]
    end

    # Importance sampling weights: w_i = (N * P(i))^(-beta) / max(w)
    inv_total = 1.0f0 / total
    weights = Vector{Float32}(undef, batch_size)
    @inbounds for j in 1:batch_size
        idx = local_indices[j]
        pα_i = idx == 1 ? cumsum_pα[1] : cumsum_pα[idx] - cumsum_pα[idx-1]
        weights[j] = (Float32(n) * pα_i * inv_total) ^ (-buf.beta)
    end
    max_w = maximum(weights)
    weights ./= max_w

    return (buf_indices, weights)
end

"""Update priorities for given indices with new TD-errors."""
function per_update_priorities!(buf::PERBuffer, indices::Vector{Int}, td_errors::Vector{Float32})
    lock(buf.lock) do
        @inbounds for (idx, td) in zip(indices, td_errors)
            if 1 <= idx <= buf.size
                buf.priorities[idx] = abs(td)
            end
        end
    end
end

"""Buffer length helper (thread-safe)."""
function buf_length(buf::PERBuffer)
    lock(buf.lock) do
        buf.size
    end
end

"""Extract a training batch from buffer at given indices.

Returns columnar data ready for prepare_batch — no NamedTuple allocation.
Holds the buffer lock during the copy: once the circular buffer is full,
`per_add_batch!` OVERWRITES old entries in place, so a lock-free read could
see a torn sample (state from the old entry, policy from the new one). The
copy is O(batch) and cheap relative to a training step."""
function extract_batch(buf::PERBuffer, indices::Vector{Int})
    lock(buf.lock) do
        states = buf.states[:, indices]
        policies = buf.policies[:, indices]
        values = buf.values[indices]
        equities = buf.equities[:, indices]
        has_equity = buf.has_equity[indices]
        is_chance = buf.is_chance[indices]
        is_contact = buf.is_contact[indices]
        is_bearoff = buf.is_bearoff[indices]
        generations = buf.generation[indices]
        (; states, policies, values, equities, has_equity, is_chance, is_contact, is_bearoff, generations)
    end
end

"""Get indices for contact vs race samples.
Holds the buffer lock: once the circular buffer is full, `per_add_batch!`
overwrites entries (including `is_contact`) in place, so a lock-free scan
could partition an index on a stale flag and later train the wrong model
on that sample."""
function partition_indices(buf::PERBuffer)
    lock(buf.lock) do
        n = buf.size
        contact_idx = Int[]
        race_idx = Int[]
        for i in 1:n
            if buf.is_contact[i]
                push!(contact_idx, i)
            else
                push!(race_idx, i)
            end
        end
        (contact=contact_idx, race=race_idx, all=collect(1:n))
    end
end

"""Vectorized reanalyze update — blends new NN values into buffer at given indices.

Holds the buffer lock: once the circular buffer is full, `per_add_batch!`
overwrites entries in place, so an unlocked blend could mix stale NN outputs
(computed from the OLD sample at that index) into a NEWER sample's targets.
The lock ensures the blend applies atomically w.r.t. concurrent uploads.

`expected_generations` is the per-slot generation captured at `extract_batch`
time. The NN inference between extract and this write is slow and unlocked; a
slot overwritten in between now carries a DIFFERENT sample, and the blend value
was computed from the OLD one. Under the lock we skip any index whose current
generation no longer matches its expectation. Returns the number skipped."""
function reanalyze_update!(buf::PERBuffer, indices::Vector{Int},
                           expected_generations::Vector{UInt32},
                           new_values::Vector{Float32},
                           new_eq_win::Vector{Float32},
                           new_eq_gw::Vector{Float32},
                           new_eq_bgw::Vector{Float32},
                           new_eq_gl::Vector{Float32},
                           new_eq_bgl::Vector{Float32};
                           α_blend::Float32=0.5f0)
    n = length(indices)
    skipped = 0

    lock(buf.lock) do
        @inbounds for k in 1:n
            idx = indices[k]
            # Skip slots overwritten since extraction — the NN blend is stale.
            if buf.generation[idx] != expected_generations[k]
                skipped += 1
                continue
            end
            old_val = buf.values[idx]
            buf.values[idx] = (1f0 - α_blend) * old_val + α_blend * new_values[k]

            # Blend equity heads
            if buf.has_equity[idx]
                buf.equities[1, idx] = (1f0 - α_blend) * buf.equities[1, idx] + α_blend * new_eq_win[k]
                buf.equities[2, idx] = (1f0 - α_blend) * buf.equities[2, idx] + α_blend * new_eq_gw[k]
                buf.equities[3, idx] = (1f0 - α_blend) * buf.equities[3, idx] + α_blend * new_eq_bgw[k]
                buf.equities[4, idx] = (1f0 - α_blend) * buf.equities[4, idx] + α_blend * new_eq_gl[k]
                buf.equities[5, idx] = (1f0 - α_blend) * buf.equities[5, idx] + α_blend * new_eq_bgl[k]
            end

            # Update PER priority
            buf.priorities[idx] = abs(new_values[k] - old_val)
        end
    end
    return skipped
end

"""Save buffer state to disk. Only saves the valid portion (buf.size samples)."""
function save_buffer(buf::PERBuffer, path::String)
    lock(buf.lock) do
        n = buf.size
        data = Dict{String, Any}(
            "states"     => buf.states[:, 1:n],
            "policies"   => buf.policies[:, 1:n],
            "values"     => buf.values[1:n],
            "equities"   => buf.equities[:, 1:n],
            "has_equity"  => buf.has_equity[1:n],
            "is_chance"   => buf.is_chance[1:n],
            "is_contact"  => buf.is_contact[1:n],
            "is_bearoff"  => buf.is_bearoff[1:n],
            "priorities"  => buf.priorities[1:n],
            "size"        => n,
            "write_pos"   => buf.write_pos,
            "beta"        => buf.beta,
            "beta_init"   => buf.beta_init,
            "beta_annealing_iters" => buf.beta_annealing_iters,
            "current_iter" => buf.current_iter,
        )
        Serialization.serialize(path, data)
    end
    return nothing
end

"""Load buffer state from disk, restoring into an existing PERBuffer."""
function load_buffer!(buf::PERBuffer, path::String)
    data = Serialization.deserialize(path)
    n = data["size"]::Int
    @assert n <= buf.capacity "Saved buffer size ($n) exceeds capacity ($(buf.capacity))"

    lock(buf.lock) do
        buf.states[:, 1:n]    .= data["states"]
        buf.policies[:, 1:n]  .= data["policies"]
        buf.values[1:n]       .= data["values"]
        buf.equities[:, 1:n]  .= data["equities"]
        buf.has_equity[1:n]   .= data["has_equity"]
        buf.is_chance[1:n]    .= data["is_chance"]
        buf.is_contact[1:n]   .= data["is_contact"]
        buf.is_bearoff[1:n]   .= data["is_bearoff"]
        buf.priorities[1:n]   .= data["priorities"]
        buf.size = n
        buf.write_pos = data["write_pos"]::Int
        buf.beta = data["beta"]
        buf.beta_init = data["beta_init"]
        buf.current_iter = data["current_iter"]
        buf.beta_annealing_iters = data["beta_annealing_iters"]
    end
    @info "Loaded buffer: $n samples, write_pos=$(buf.write_pos), beta=$(buf.beta)"
    return nothing
end
