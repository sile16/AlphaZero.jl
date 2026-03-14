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
                buf.is_chance[pos] = false
                buf.is_contact[pos] = batch_is_contact[i]
                buf.is_bearoff[pos] = batch_is_bearoff[i]
                buf.priorities[pos] = initial_priority
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
Uses views where possible to avoid copies."""
function extract_batch(buf::PERBuffer, indices::Vector{Int})
    lock(buf.lock) do
        n = length(indices)
        states = buf.states[:, indices]
        policies = buf.policies[:, indices]
        values = buf.values[indices]
        equities = buf.equities[:, indices]
        has_equity = buf.has_equity[indices]
        is_chance = buf.is_chance[indices]
        is_contact = buf.is_contact[indices]
        is_bearoff = buf.is_bearoff[indices]
        return (; states, policies, values, equities, has_equity, is_chance, is_contact, is_bearoff)
    end
end

"""Get indices for contact vs race samples."""
function partition_indices(buf::PERBuffer)
    lock(buf.lock) do
        contact_idx = Int[]
        race_idx = Int[]
        for i in 1:buf.size
            if buf.is_contact[i]
                push!(contact_idx, i)
            else
                push!(race_idx, i)
            end
        end
        return (contact=contact_idx, race=race_idx, all=collect(1:buf.size))
    end
end

"""Vectorized reanalyze update — blends new NN values into buffer at given indices.

Uses broadcasting for bulk updates instead of per-sample loops."""
function reanalyze_update!(buf::PERBuffer, indices::Vector{Int},
                           new_values::Vector{Float32},
                           new_eq_win::Vector{Float32},
                           new_eq_gw::Vector{Float32},
                           new_eq_bgw::Vector{Float32},
                           new_eq_gl::Vector{Float32},
                           new_eq_bgl::Vector{Float32};
                           α_blend::Float32=0.5f0)
    lock(buf.lock) do
        n = length(indices)

        # Vectorized value blending
        @inbounds for k in 1:n
            idx = indices[k]
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
end
