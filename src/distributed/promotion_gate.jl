"""
Weight promotion gate — pure decision logic for gating weight PUBLICATION on a
regression metric. No server / GPU / network dependencies so it is unit-testable
in isolation (see `test/test_promotion_gate.jl`).

Included top-level (no module) by `scripts/training_server.jl`, the same way
`src/distributed/buffer.jl` is.

## Why a gate

The training server trains every iteration and, unconditionally, PUBLISHES the
fresh weights: it bumps the served weight version (clients pull and self-play
with them) and overwrites `race_latest.data`. A regressed model therefore feeds
its own regression back into the replay buffer. The gate makes PUBLICATION
conditional on a cheap, synchronous quality signal while never blocking or
stalling training — training always proceeds; only publication is held back.

## Gated metric (lower is better)

We gate on the fixed-set bearoff eval's **value MAE** — the race value head's
mean absolute error against the exact k=7 bearoff table, in normalized equity
units (both sides are `v/3 ∈ [-1,1]`). Rationale:

- It is a *smooth, continuous* regression signal. Move-accuracy metrics
  (policy_top1, nn_top1) are coarse step functions of many discrete argmaxes —
  they can stay flat across a real value-head regression and jump noisily on a
  single position, making a tolerance band hard to set. MAE moves monotonically
  with value-head quality, so a percentage tolerance is meaningful.
- It measures exactly the failure mode we fear: the value head drifting away
  from ground truth during self-play (observed in v11: corr 0.974→0.875 while
  loss fell). That drift is precisely what poisons the buffer.
- It is already computed synchronously in-server every `--bearoff-eval-interval`
  iters at negligible cost.

A combined MAE-plus-move-accuracy rule was considered; MAE-primary was chosen for
smoothness and to keep exactly one tunable tolerance. Move-accuracy remains
logged to TB for human oversight.

## Rule

For best-so-far metric `best` (lowest MAE seen) and this eval's `metric`:

    threshold = best * (1 + tol_frac) + tol_abs
    publish   = metric <= threshold

- `tol_frac` (`--gate-tolerance`, default 0.10): the metric may be up to 10%
  worse than the best-so-far and still publish. Self-play evals are noisy; a
  strict "never worse" rule would block on sampling noise.
- `tol_abs` (`GATE_TOL_ABS`, 0.003 normalized ≈ 0.009 points): an absolute floor
  so the band never collapses to zero width when `best` is tiny.
- The first finite eval always publishes and seeds `best`.
- `best` only decreases (best-so-far), updated on any publishing improvement.

The decision PERSISTS between eval iterations: intermediate iterations reuse the
last decision (publish if the last eval passed, hold if it failed). A later
passing eval resumes publication with the then-current weights.
"""

import JSON

"""Persistent gate state. `best_metric` is the lowest gated metric seen so far
(`Inf` until the first eval). `last_published` is the most recent decision, reused
by iterations between evals."""
struct GateState
    best_metric::Float64
    last_published::Bool
    n_evals::Int
    n_blocked::Int
end

GateState() = GateState(Inf, true, 0, 0)

"""Result of one gate evaluation. `state` is the next `GateState` to carry forward."""
struct GateDecision
    publish::Bool        # may weights be published this eval?
    improved::Bool       # did this eval set a new best (→ save race_best.data)?
    metric::Float64      # the metric evaluated
    best_metric::Float64 # best-so-far after this eval
    threshold::Float64   # publication ceiling used
    state::GateState     # next state
end

"""
    gate_threshold(best, tol_frac, tol_abs)

Publication ceiling for a lower-is-better metric. `Inf` when no baseline exists
yet (first eval always passes).
"""
gate_threshold(best::Real, tol_frac::Real, tol_abs::Real) =
    isfinite(best) ? Float64(best) * (1 + Float64(tol_frac)) + Float64(tol_abs) : Inf

"""
    gate_evaluate(state, metric; tol_frac, tol_abs) -> GateDecision

Pure gate step. `metric` is lower-is-better (bearoff value MAE, normalized).

- First finite eval (`best == Inf`) always publishes and seeds `best`.
- Otherwise publishes iff `metric <= gate_threshold(best, tol_frac, tol_abs)`.
- `best` decreases to `min(best, metric)` on any publishing improvement.
- A non-finite metric (broken eval) never publishes and never touches `best`.
"""
function gate_evaluate(state::GateState, metric::Real; tol_frac::Real, tol_abs::Real)
    m = Float64(metric)
    if !isfinite(m)
        ns = GateState(state.best_metric, false, state.n_evals + 1, state.n_blocked + 1)
        return GateDecision(false, false, m, state.best_metric,
                            gate_threshold(state.best_metric, tol_frac, tol_abs), ns)
    end
    thr = gate_threshold(state.best_metric, tol_frac, tol_abs)
    publish = m <= thr
    improved = publish && m < state.best_metric
    new_best = publish ? min(state.best_metric, m) : state.best_metric
    n_blocked = state.n_blocked + (publish ? 0 : 1)
    ns = GateState(new_best, publish, state.n_evals + 1, n_blocked)
    return GateDecision(publish, improved, m, new_best, thr, ns)
end

# ── JSON sidecar (resume) ──────────────────────────────────────────────────

"""Plain Dict for JSON serialization. `best_metric == Inf` is stored as `null`."""
function gate_state_to_dict(state::GateState; metric_name::String="value_mae",
                            tol_frac::Real=NaN, tol_abs::Real=NaN)
    Dict{String,Any}(
        "best_metric"    => (isfinite(state.best_metric) ? state.best_metric : nothing),
        "last_published" => state.last_published,
        "n_evals"        => state.n_evals,
        "n_blocked"      => state.n_blocked,
        "metric_name"    => metric_name,
        "tol_frac"       => (isfinite(tol_frac) ? tol_frac : nothing),
        "tol_abs"        => (isfinite(tol_abs) ? tol_abs : nothing),
    )
end

"""Reconstruct a `GateState` from a parsed dict. Missing/`null` fields fall back
to fresh-state defaults (best = Inf, published = true)."""
function gate_state_from_dict(d::AbstractDict)
    bm = get(d, "best_metric", nothing)
    best = (bm === nothing) ? Inf : Float64(bm)
    GateState(best,
              Bool(get(d, "last_published", true)),
              Int(get(d, "n_evals", 0)),
              Int(get(d, "n_blocked", 0)))
end

"""Write the gate state to `path` (JSON). Overwrites atomically enough for a
once-per-checkpoint sidecar."""
function save_gate_state(path::String, state::GateState; metric_name::String="value_mae",
                         tol_frac::Real=NaN, tol_abs::Real=NaN)
    open(path, "w") do io
        JSON.print(io, gate_state_to_dict(state; metric_name=metric_name,
                                          tol_frac=tol_frac, tol_abs=tol_abs))
    end
    return path
end

"""Load a gate state from `path`, or `nothing` if absent/unreadable (graceful
resume — a missing sidecar just starts the gate fresh)."""
function load_gate_state(path::String)
    isfile(path) || return nothing
    try
        return gate_state_from_dict(JSON.parsefile(path))
    catch e
        @warn "Could not parse gate state sidecar; starting gate fresh" path=path exception=e
        return nothing
    end
end
