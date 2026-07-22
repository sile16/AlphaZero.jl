#####
##### Reanalyze Module
#####
##### MuZero-style reanalysis for improved sample efficiency.
##### Re-evaluates positions in the replay buffer with the latest network.
#####
##### This module provides generic reanalysis logic that works with any sample type
##### that has the required fields (state, value, priority, reanalyze_count, etc.).
#####

using Statistics: mean
import Flux

"""
    ReanalyzeConfig

Configuration for MuZero-style reanalysis.

| Field | Description |
|:------|:------------|
| `enabled` | Whether reanalysis is enabled |
| `batch_size` | Positions per reanalyze step |
| `update_interval` | Run every N training steps |
| `reanalyze_alpha` | Blend factor (0=keep old, 1=use new) |
| `max_reanalyze_count` | Max times to reanalyze same position |
| `prioritize_high_td` | Prioritize high TD-error for reanalysis |
| `log_interval` | Log stats every N reanalyze steps |
"""
@kwdef struct ReanalyzeConfig
    enabled::Bool = true
    batch_size::Int = 256
    update_interval::Int = 1
    reanalyze_alpha::Float32 = 0.5f0
    max_reanalyze_count::Int = 5
    prioritize_high_td::Bool = true
    log_interval::Int = 10
end

"""
    ReanalyzeStats

Statistics from reanalysis for logging.
"""
mutable struct ReanalyzeStats
    total_reanalyzed::Int
    total_steps::Int
    avg_td_error::Float64
    max_td_error::Float64
    avg_value_change::Float64

    function ReanalyzeStats()
        new(0, 0, 0.0, 0.0, 0.0)
    end
end

"""
    sample_for_reanalysis(buffer, batch_size, prioritize_high_td, current_step, max_reanalyze_count)

Sample positions from buffer for reanalysis.

If `prioritize_high_td=true`, samples positions with higher TD-error more frequently.
Also prioritizes positions that haven't been reanalyzed recently.

The sample type must have fields: `priority`, `last_reanalyze_step`, `reanalyze_count`.
"""
function sample_for_reanalysis(
    buffer::Vector,
    batch_size::Int,
    prioritize_high_td::Bool,
    current_step::Int,
    max_reanalyze_count::Int
)
    n = length(buffer)
    if n == 0
        return Int[]
    end

    actual_batch = min(batch_size, n)

    if !prioritize_high_td
        # Uniform random sampling
        return rand(1:n, actual_batch)
    end

    # Priority-based sampling
    # Higher priority = more likely to be reanalyzed
    priorities = Float64[]
    eligible_indices = Int[]

    for (i, sample) in enumerate(buffer)
        # Skip if already reanalyzed too many times
        if sample.reanalyze_count >= max_reanalyze_count
            continue
        end

        push!(eligible_indices, i)

        # Priority components:
        # 1. TD-error (higher = more important to learn from)
        td_priority = sample.priority + 0.01  # Add small constant to avoid zero

        # 2. Staleness (not reanalyzed recently = higher priority)
        staleness = current_step - sample.last_reanalyze_step + 1
        staleness_priority = min(staleness / 100.0, 2.0)  # Cap at 2x

        # Combined priority
        push!(priorities, td_priority * staleness_priority)
    end

    if isempty(eligible_indices)
        # All samples already at max reanalyze count, sample uniformly
        return rand(1:n, actual_batch)
    end

    # Normalize priorities to probabilities
    total = sum(priorities)
    probs = priorities ./ total

    # Sample according to priorities (with replacement for simplicity)
    selected = Int[]
    for _ in 1:min(actual_batch, length(eligible_indices))
        r = rand()
        cumsum_p = 0.0
        for (j, p) in enumerate(probs)
            cumsum_p += p
            if r <= cumsum_p
                push!(selected, eligible_indices[j])
                break
            end
        end
    end

    return selected
end

"""
    sample_for_smart_reanalysis(buffer, batch_size, current_model_iter)

Sample positions for smart reanalysis based on model iteration staleness.

Returns indices of samples that need reanalysis (model_iter_reanalyzed < current_model_iter),
prioritizing the most stale (oldest model iteration).

Returns empty array if all samples are up-to-date with current model.
"""
function sample_for_smart_reanalysis(
    buffer::Vector,
    batch_size::Int,
    current_model_iter::Int
)
    n = length(buffer)
    if n == 0
        return Int[]
    end

    # Find all samples that need reanalysis (not up-to-date with current model)
    stale_samples = Tuple{Int, Int}[]  # (index, model_iter_reanalyzed)
    for (i, sample) in enumerate(buffer)
        if sample.model_iter_reanalyzed < current_model_iter
            push!(stale_samples, (i, sample.model_iter_reanalyzed))
        end
    end

    if isempty(stale_samples)
        return Int[]  # All samples are up-to-date
    end

    # Sort by staleness (oldest model iteration first)
    sort!(stale_samples, by = x -> x[2])

    # Take the most stale samples up to batch_size
    actual_batch = min(batch_size, length(stale_samples))
    return [stale_samples[i][1] for i in 1:actual_batch]
end

"""
    count_stale_samples(buffer, current_model_iter)

Count samples that need reanalysis (not up-to-date with current model).
"""
function count_stale_samples(buffer::Vector, current_model_iter::Int)
    count = 0
    for sample in buffer
        if sample.model_iter_reanalyzed < current_model_iter
            count += 1
        end
    end
    return count
end

"""
    get_reanalyze_metrics(stats)

Get reanalysis metrics for TensorBoard logging.
"""
function get_reanalyze_metrics(stats::ReanalyzeStats)
    return Dict{String, Float64}(
        "reanalyze/total_reanalyzed" => Float64(stats.total_reanalyzed),
        "reanalyze/total_steps" => Float64(stats.total_steps),
        "reanalyze/avg_td_error" => stats.avg_td_error,
        "reanalyze/max_td_error" => stats.max_td_error,
        "reanalyze/avg_value_change" => stats.avg_value_change
    )
end
