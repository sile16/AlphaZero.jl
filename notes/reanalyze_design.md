# Reanalyze and Smart Buffer Design

## Overview

MuZero-style reanalysis to improve sample efficiency by re-evaluating stored positions with the latest neural network.

## Current Buffer Implementation

### What We Have
- **Structure**: `CircularBuffer{TrainingSample}` (FIFO)
- **Capacity**: 100K samples
- **Eviction**: Oldest samples removed when full (no priority)
- **Sampling**: Uniform random for training batches
- **Per-sample data**:
  - `s::State` - Game state
  - `π::Vector{Float64}` - MCTS policy (visit counts)
  - `z::Float64` - Final game outcome (discounted)
  - `t::Float64` - Turns remaining estimate
  - `n::Int` - Merge count
  - `equity::EquityTargets` - Multi-head targets (5 values)

### Limitations
1. Values `z` are from game outcome at time of play - not updated as network improves
2. Policy `π` from old MCTS with old network - may be suboptimal
3. FIFO eviction loses potentially valuable diverse positions
4. No mechanism to incorporate external knowledge (bear-off DB, curated positions)

---

## Reanalyze Design

### Core Idea
After each training step, a background thread re-evaluates positions in the buffer with the latest network to get updated value estimates.

### What Gets Updated

| Field | Update Method | Notes |
|-------|---------------|-------|
| `z` (value) | Fresh network evaluation | Key improvement - better targets |
| `π` (policy) | Optional: mini-MCTS or network prior | Expensive if full MCTS |
| `equity` | Fresh network evaluation | All 5 heads updated |
| `priority` | NEW: TD-error based | For smart sampling |

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Main Training Loop                       │
├─────────────────────────────────────────────────────────────┤
│  Self-Play Workers  →  Buffer  ←  Reanalyze Worker          │
│       (add new)           ↑           (update old)          │
│                           │                                  │
│                    Training Thread                           │
│                    (sample & learn)                          │
└─────────────────────────────────────────────────────────────┘
```

### Reanalyze Worker

```julia
struct ReanalyzeWorker
    buffer::ReplayBuffer
    network_ref::Ref{Network}  # Latest network (updated after training)
    batch_size::Int            # Positions per reanalyze batch
    update_interval::Int       # How often to run (every N training steps)
end

function reanalyze_step!(worker::ReanalyzeWorker)
    # 1. Sample batch of positions (prioritize high TD-error or old)
    indices = sample_for_reanalysis(worker.buffer, worker.batch_size)

    # 2. Batch evaluate with latest network
    states = [worker.buffer[i].s for i in indices]
    new_values, new_policies, new_equities = batch_evaluate(worker.network_ref[], states)

    # 3. Update samples in buffer
    for (idx, v, π, eq) in zip(indices, new_values, new_policies, new_equities)
        sample = worker.buffer[idx]

        # Compute TD-error before update (for priority)
        td_error = abs(v - sample.z)

        # Update value (blend old and new, or replace)
        sample.z = lerp(sample.z, v, reanalyze_alpha)

        # Update equity targets
        sample.equity = eq

        # Update priority for smart sampling
        sample.priority = td_error

        # Track reanalysis count
        sample.reanalyze_count += 1
    end
end
```

### Configuration Options

```julia
@kwdef struct ReanalyzeConfig
    enabled::Bool = true
    batch_size::Int = 256           # Positions per reanalyze step
    update_interval::Int = 1        # Run every N training steps
    reanalyze_alpha::Float32 = 0.5  # Blend factor (0=keep old, 1=use new)
    max_reanalyze_count::Int = 5    # Max times to reanalyze same position
    prioritize_high_td::Bool = true # Prioritize high TD-error for reanalysis
    update_policy::Bool = false     # Also update π (expensive)
    mini_mcts_iters::Int = 10       # If updating policy, use mini-MCTS
end
```

---

## Smart Eviction Policy

### Current: FIFO (age-based)
When buffer full, remove oldest samples.

### Proposed: Priority-based Eviction

Keep samples that are:
1. **High TD-error** - Network disagrees with stored value (still learning)
2. **Diverse** - Unique positions not similar to others
3. **Recently reanalyzed** - Fresh values are more accurate
4. **From wins/losses** - Clear signal (not draws)

### Eviction Priority Score

```julia
function eviction_priority(sample::TrainingSample, buffer_stats)
    # Lower score = more likely to evict

    td_error = sample.priority  # From reanalysis
    age = buffer_stats.current_step - sample.added_step
    reanalyze_freshness = buffer_stats.current_step - sample.last_reanalyze_step

    # Score components (higher = keep)
    td_score = td_error * 10.0           # High TD-error = valuable
    freshness_score = 1.0 / (age + 1)    # Newer = slightly better
    reanalyze_score = 1.0 / (reanalyze_freshness + 1)  # Recently updated = better
    outcome_score = abs(sample.z) > 0.5 ? 1.0 : 0.5    # Clear outcomes = better

    return td_score + freshness_score + reanalyze_score + outcome_score
end

function evict_samples!(buffer, n_to_remove)
    # Compute priorities
    priorities = [eviction_priority(s, stats) for s in buffer]

    # Remove lowest priority samples
    indices_to_remove = partialsortperm(priorities, 1:n_to_remove)
    deleteat!(buffer, sort(indices_to_remove))
end
```

### Alternative: Reservoir Sampling
Keep diverse samples using locality-sensitive hashing on state features.

---

## Extended Sample Structure

```julia
mutable struct TrainingSample{State}
    # Existing fields
    s::State
    π::Vector{Float64}
    z::Float64
    t::Float64
    n::Int
    is_chance::Bool
    equity::Union{Nothing, EquityTargets}

    # NEW: Reanalysis tracking
    priority::Float32              # TD-error based priority
    added_step::Int                # Training step when added
    last_reanalyze_step::Int       # Last reanalysis step
    reanalyze_count::Int           # Times reanalyzed

    # NEW: Source tracking
    source::SampleSource           # :self_play, :bearoff_db, :curated_library
    confidence::Float32            # How confident is the value (1.0 for DB lookups)
end

@enum SampleSource begin
    SELF_PLAY       # From self-play games
    BEAROFF_DB      # From bear-off database (exact values)
    CURATED_LIBRARY # From curated position library
    REANALYZED      # Value updated via reanalysis
end
```

---

## Future: External Data Sources

### 1. Bear-off Database

For endgame positions (all checkers in home board), exact equity can be computed.

```julia
struct BearoffDB
    db::Dict{BearoffPosition, BearoffEntry}
end

struct BearoffEntry
    equity::Float32           # Exact cubeless equity
    p_win::Float32            # Exact P(win)
    p_gammon_win::Float32     # Exact P(gammon|win)
    # etc.
end

function lookup_or_evaluate(db::BearoffDB, state::State, network)
    if is_bearoff_position(state)
        entry = db[to_bearoff_key(state)]
        return (value=entry.equity, source=BEAROFF_DB, confidence=1.0)
    else
        v = evaluate(network, state)
        return (value=v, source=SELF_PLAY, confidence=0.8)
    end
end
```

### 2. Curated Position Library

High-quality positions with known values from:
- Rollout analysis (10K+ games)
- Expert annotations
- Puzzle positions with known best moves

```julia
struct CuratedLibrary
    positions::Vector{CuratedPosition}
end

struct CuratedPosition
    state::State
    target_value::Float32
    target_policy::Vector{Float32}  # Best move(s) marked
    confidence::Float32
    source::String  # "gnubg_rollout", "expert", "puzzle"
end

function inject_curated_samples!(buffer, library, n_samples)
    # Periodically inject curated samples into training
    samples = rand(library.positions, n_samples)
    for pos in samples
        push!(buffer, TrainingSample(
            s = pos.state,
            π = pos.target_policy,
            z = pos.target_value,
            source = CURATED_LIBRARY,
            confidence = pos.confidence
        ))
    end
end
```

---

## Implementation Plan

### Phase 1: Basic Reanalyze (Immediate)
- [ ] Add `priority`, `added_step`, `reanalyze_count` to TrainingSample
- [ ] Implement ReanalyzeWorker with value-only updates
- [ ] Run reanalyze after each training step
- [ ] Log reanalysis stats to TensorBoard

### Phase 2: Smart Eviction
- [ ] Implement priority-based eviction
- [ ] Add `eviction_priority()` scoring function
- [ ] Replace FIFO with priority-based removal
- [ ] Benchmark impact on training

### Phase 3: External Data (Future)
- [ ] Bear-off database integration
- [ ] Curated position library format
- [ ] Mixed training with external data

---

## Expected Benefits

1. **Better sample efficiency** - Each position trained on better targets
2. **Faster convergence** - Network improvements propagate to old samples
3. **Less overfitting** - Diverse buffer with smart eviction
4. **Stronger endgame** - Bear-off DB gives exact values
5. **Targeted improvement** - Curated positions fix known weaknesses

## References

- MuZero paper: Reanalyze section
- Prioritized Experience Replay (PER): Schaul et al.
- TD-Gammon: Bear-off database integration
