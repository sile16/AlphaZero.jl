#####
##### Memory Buffer:
##### Datastructure to collect self-play experience
#####

"""
    EquityTargets

Targets for multi-head equity training (backgammon-style games).
Uses joint cumulative semantics — all 5 heads trained on all samples.

| Field | Description |
|:------|:------------|
| `p_win` | P(win) — 1.0 if won, 0.0 if lost |
| `p_gammon_win` | P(win ∧ gammon+) — 1.0 if won by gammon or backgammon |
| `p_bg_win` | P(win ∧ backgammon) — 1.0 if won by backgammon |
| `p_gammon_loss` | P(lose ∧ gammon+) — 1.0 if lost by gammon or backgammon |
| `p_bg_loss` | P(lose ∧ backgammon) — 1.0 if lost by backgammon |

For self-play binary targets, the vectors are numerically identical to the
old conditional representation (zeros on the non-applicable side are valid
joint probabilities, not masked placeholders).
"""
struct EquityTargets
  p_win :: Float64
  p_gammon_win :: Float64
  p_bg_win :: Float64
  p_gammon_loss :: Float64
  p_bg_loss :: Float64
end

"""
    equity_targets_from_outcome(outcome::GI.GameOutcome, white_perspective::Bool)

Create equity targets from a game outcome using joint cumulative semantics.

For a won game:
- p_win = 1.0
- p_gammon_win = 1.0 if gammon or backgammon, 0.0 otherwise
- p_bg_win = 1.0 if backgammon, 0.0 otherwise
- p_gammon_loss = 0.0 (valid joint probability — did not lose)
- p_bg_loss = 0.0

For a lost game:
- p_win = 0.0
- p_gammon_win = 0.0 (valid joint probability — did not win)
- p_bg_win = 0.0
- p_gammon_loss = 1.0 if opponent won by gammon or backgammon, 0.0 otherwise
- p_bg_loss = 1.0 if opponent won by backgammon, 0.0 otherwise
"""
function equity_targets_from_outcome(outcome::GI.GameOutcome, white_perspective::Bool)
  won = outcome.white_won == white_perspective

  if won
    p_win = 1.0
    p_gammon_win = outcome.is_gammon ? 1.0 : 0.0
    p_bg_win = outcome.is_backgammon ? 1.0 : 0.0
    p_gammon_loss = 0.0
    p_bg_loss = 0.0
  else
    p_win = 0.0
    p_gammon_win = 0.0
    p_bg_win = 0.0
    p_gammon_loss = outcome.is_gammon ? 1.0 : 0.0
    p_bg_loss = outcome.is_backgammon ? 1.0 : 0.0
  end

  return EquityTargets(p_win, p_gammon_win, p_bg_win, p_gammon_loss, p_bg_loss)
end

"""
    equity_vector(targets::EquityTargets, [T=Float32]) -> Vector{T}

Materialize an `EquityTargets` struct into the canonical 5-head vector order:

`[P(win), P(win∧gammon+), P(win∧bg), P(lose∧gammon+), P(lose∧bg)]`
"""
function equity_vector(targets::EquityTargets, ::Type{T}=Float32) where T <: AbstractFloat
  return T[
    targets.p_win,
    targets.p_gammon_win,
    targets.p_bg_win,
    targets.p_gammon_loss,
    targets.p_bg_loss,
  ]
end

"""
    equity_vector_from_outcome(outcome, white_perspective, [T=Float32]) -> Vector{T}

Convenience wrapper around `equity_targets_from_outcome` for callers that need
the packed 5-head vector rather than the struct representation.
"""
function equity_vector_from_outcome(
    outcome::GI.GameOutcome,
    white_perspective::Bool,
    ::Type{T}=Float32) where T <: AbstractFloat
  return equity_vector(equity_targets_from_outcome(outcome, white_perspective), T)
end

"""
    flip_equity_perspective(eq::AbstractVector{<:Real}) -> Vector

Flip a canonical 5-head equity vector to the opponent's perspective.
Works identically for both joint and conditional semantics: negate p_win,
swap the win-side and loss-side heads.

If `eq` is:
`[P(win), P(win∧gammon+), P(win∧bg), P(lose∧gammon+), P(lose∧bg)]`

then the returned vector is:
`[P(lose), P(lose∧gammon+), P(lose∧bg), P(win∧gammon+), P(win∧bg)]`
"""
function flip_equity_perspective(eq::AbstractVector{T}) where T <: Real
  length(eq) == 5 || throw(ArgumentError("expected 5 equity heads, got $(length(eq))"))
  return T[
    one(T) - eq[1],
    eq[4],
    eq[5],
    eq[2],
    eq[3],
  ]
end

"""
    TrainingSample{State}

Type of a training sample. A sample features the following fields:
- `s::State` is the state
- `π::Vector{Float64}` is the recorded MCTS policy for this position
- `z::Float64` is the discounted reward cumulated from state `s`
- `t::Float64` is the (average) number of moves remaining before the end of the game
- `n::Int` is the number of times the state `s` was recorded
- `is_chance::Bool` is true if this is a chance node sample (no policy target)
- `equity::Union{Nothing, EquityTargets}` is optional multi-head equity targets

As revealed by the last field `n`, several samples that correspond to the
same state can be merged, in which case the `π`, `z` and `t`
fields are averaged together.

For chance nodes (`is_chance=true`), the policy `π` should be empty and the
sample should only contribute to value training, not policy training.
"""
struct TrainingSample{State}
  s :: State
  π :: Vector{Float64}
  z :: Float64
  t :: Float64
  n :: Int
  is_chance :: Bool
  equity :: Union{Nothing, EquityTargets}
end

# Backward compatibility constructors
TrainingSample(s, π, z, t, n) = TrainingSample(s, π, z, t, n, false, nothing)
TrainingSample(s, π, z, t, n, is_chance) = TrainingSample(s, π, z, t, n, is_chance, nothing)

sample_state_type(::Type{<:TrainingSample{S}}) where S = S

"""
    MemoryBuffer(game_spec, size, experience=[])

A circular buffer to hold memory samples.
"""
mutable struct MemoryBuffer{GameSpec, State}
  gspec :: GameSpec
  buf :: CircularBuffer{TrainingSample{State}}
  cur_batch_size :: Int
  function MemoryBuffer(gspec, size, experience=[])
    State = GI.state_type(gspec)
    buf = CircularBuffer{TrainingSample{State}}(size)
    append!(buf, experience)
    new{typeof(gspec), State}(gspec, buf, 0)
  end
end

"""
    get_experience(::MemoryBuffer) :: Vector{<:TrainingSample}

Return all samples in the memory buffer.
"""
get_experience(mem::MemoryBuffer) = mem.buf[:]

last_batch(mem::MemoryBuffer) = mem.buf[end-cur_batch_size(mem)+1:end]

cur_batch_size(mem::MemoryBuffer) = min(mem.cur_batch_size, length(mem))

new_batch!(mem::MemoryBuffer) = (mem.cur_batch_size = 0)

function Base.empty!(mem::MemoryBuffer)
  empty!(mem.buf)
  mem.cur_batch_size = 0
end

Base.length(mem::MemoryBuffer) = length(mem.buf)

"""
    push_trace!(mem::MemoryBuffer, trace::Trace, gamma)

Collect samples out of a game trace and add them to the memory buffer.

Here, `gamma` is the reward discount factor.

For stochastic games, chance node samples are marked with `is_chance=true`.

For games with equity targets (backgammon-style), samples include multi-head targets.
"""
function push_trace!(mem::MemoryBuffer, trace, gamma)
  n = length(trace)
  wr = 0.

  # Compute equity targets from game outcome if available
  has_outcome = !isnothing(trace.outcome)

  for i in reverse(1:n)
    wr = gamma * wr + trace.rewards[i]
    s = trace.states[i]
    π = trace.policies[i]
    is_chance = trace.is_chance[i]
    # Use direct state access to avoid potential side effects from GI.init/set_state!
    # This ensures consistent perspective computation for canonical observations
    wp = GI.white_playing(mem.gspec, s)
    z = wp ? wr : -wr
    t = float(n - i + 1)

    # Compute equity targets if game outcome is available
    equity = if has_outcome
      equity_targets_from_outcome(trace.outcome, wp)
    else
      nothing
    end

    push!(mem.buf, TrainingSample(s, π, z, t, 1, is_chance, equity))
  end
  mem.cur_batch_size += n
end

function merge_samples(samples)
  s = samples[1].s
  π = mean(e.π for e in samples)
  z = mean(e.z for e in samples)
  n = sum(e.n for e in samples)
  t = mean(e.t for e in samples)
  is_chance = samples[1].is_chance  # All merged samples should have same is_chance

  # Average equity targets if present (they should all be the same for same state)
  equity = if !isnothing(samples[1].equity)
    # Average the equity targets
    p_win = mean(e.equity.p_win for e in samples)
    p_gw = mean(e.equity.p_gammon_win for e in samples)
    p_bgw = mean(e.equity.p_bg_win for e in samples)
    p_gl = mean(e.equity.p_gammon_loss for e in samples)
    p_bgl = mean(e.equity.p_bg_loss for e in samples)
    EquityTargets(p_win, p_gw, p_bgw, p_gl, p_bgl)
  else
    nothing
  end

  return eltype(samples)(s, π, z, t, n, is_chance, equity)
end

# Merge samples that correspond to identical states
function merge_by_state(samples)
  Sample = eltype(samples)
  State = sample_state_type(Sample)
  dict = Dict{State, Vector{Sample}}()
  sizehint!(dict, length(samples))
  for s in samples
    if haskey(dict, s.s)
      push!(dict[s.s], s)
    else
      dict[s.s] = [s]
    end
  end
  return [merge_samples(ss) for ss in values(dict)]
end

function apply_symmetry(gspec, sample, (symstate, aperm))
  mask = GI.actions_mask(GI.init(gspec, sample.s))
  symmask = GI.actions_mask(GI.init(gspec, symstate))
  π = zeros(eltype(sample.π), length(mask))
  π[mask] = sample.π
  π = π[aperm]
  @assert iszero(π[.~symmask])
  π = π[symmask]
  return typeof(sample)(
    symstate, π, sample.z, sample.t, sample.n, sample.is_chance, sample.equity)
end

function augment_with_symmetries(gspec, samples)
  symsamples = [apply_symmetry(gspec, s, sym)
    for s in samples for sym in GI.symmetries(gspec, s.s)]
  return [samples ; symsamples]
end
