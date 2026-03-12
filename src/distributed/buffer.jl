"""
Thread-safe Prioritized Experience Replay (PER) buffer.

Extracted from train_distributed.jl for use by both the training server
and the single-machine training script.
"""

"""Prioritized Experience Replay buffer.
Wraps a flat buffer with per-sample priorities for proportional sampling."""
mutable struct PERBuffer
    samples::Vector{Any}       # The actual samples
    priorities::Vector{Float32} # Priority for each sample (|TD-error| + ε)
    capacity::Int
    beta::Float32              # Current IS beta (anneals to 1.0)
    beta_init::Float32
    beta_annealing_iters::Int
    current_iter::Int
    lock::ReentrantLock        # Thread safety for concurrent add/sample
end

function PERBuffer(capacity::Int; beta_init=0.4f0, annealing_iters=200)
    PERBuffer(Any[], Float32[], capacity, beta_init, beta_init, annealing_iters, 0, ReentrantLock())
end

function per_add!(buf::PERBuffer, samples, initial_priority::Float32=1.0f0)
    lock(buf.lock) do
        append!(buf.samples, samples)
        append!(buf.priorities, fill(initial_priority, length(samples)))
        if length(buf.samples) > buf.capacity
            excess = length(buf.samples) - buf.capacity
            deleteat!(buf.samples, 1:excess)
            deleteat!(buf.priorities, 1:excess)
        end
    end
end

function per_anneal_beta!(buf::PERBuffer)
    buf.current_iter += 1
    frac = min(1.0f0, Float32(buf.current_iter) / Float32(buf.beta_annealing_iters))
    buf.beta = buf.beta_init + (1.0f0 - buf.beta_init) * frac
end

"""Sample from PER buffer. Returns (indices, samples, importance_weights).
Uses precomputed cumulative sum + binary search for O(batch_size * log(n)) sampling."""
function per_sample(buf::PERBuffer, batch_size::Int, alpha::Float32, epsilon::Float32)
    lock(buf.lock) do
        n = length(buf.samples)
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
        weights ./= max_w  # Normalize so max weight = 1.0

        samples = [buf.samples[i] for i in indices]
        return (indices, samples, weights)
    end
end

"""Update priorities for given indices with new TD-errors."""
function per_update_priorities!(buf::PERBuffer, indices::Vector{Int}, td_errors::Vector{Float32})
    lock(buf.lock) do
        @inbounds for (idx, td) in zip(indices, td_errors)
            if 1 <= idx <= length(buf.priorities)
                buf.priorities[idx] = abs(td)
            end
        end
    end
end

"""Buffer length helper (thread-safe)."""
function buf_length(buf::PERBuffer)
    lock(buf.lock) do
        length(buf.samples)
    end
end
buf_length(buf::Vector) = length(buf)
