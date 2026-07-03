# Unit tests for the weight promotion gate's PURE decision logic
# (src/distributed/promotion_gate.jl). The gate holds weight publication when the
# race value MAE (lower is better) regresses beyond tolerance vs best-so-far.
#
# These tests exercise the pure function + JSON sidecar with no server/GPU/network
# dependency — the whole point of factoring the decision out of training_server.jl.

using Test

# promotion_gate.jl is a plain include (no module), same as the server includes it.
include(joinpath(@__DIR__, "..", "src", "distributed", "promotion_gate.jl"))

const TOL_FRAC = 0.10   # 10% fractional tolerance (matches server default)
const TOL_ABS = 0.003   # absolute floor (matches server GATE_TOL_ABS)

step(state, metric) = gate_evaluate(state, metric; tol_frac=TOL_FRAC, tol_abs=TOL_ABS)

@testset "fresh state defaults" begin
    s = GateState()
    @test !isfinite(s.best_metric)     # no baseline yet
    @test s.last_published             # publish until first eval (bootstrap)
    @test s.n_evals == 0
    @test s.n_blocked == 0
end

@testset "first eval always passes and seeds best" begin
    d = step(GateState(), 0.05)
    @test d.publish
    @test d.improved
    @test d.best_metric == 0.05
    @test d.state.best_metric == 0.05
    @test d.state.last_published
    @test d.state.n_evals == 1
    @test d.state.n_blocked == 0
    # First eval passes regardless of how large the metric is (no baseline).
    d2 = step(GateState(), 12.3)
    @test d2.publish
    @test d2.best_metric == 12.3
end

@testset "improvement lowers best and flags improved" begin
    s = step(GateState(), 0.05).state
    d = step(s, 0.03)
    @test d.publish
    @test d.improved
    @test d.best_metric == 0.03
end

@testset "small regression within tolerance publishes, best unchanged" begin
    s = step(GateState(), 0.10).state          # best = 0.10
    # threshold = 0.10*1.10 + 0.003 = 0.113
    d = step(s, 0.112)
    @test d.publish
    @test !d.improved                          # worse than best → not an improvement
    @test d.best_metric == 0.10                # best does not rise
    @test d.threshold ≈ 0.113 atol=1e-9
end

@testset "exactly-at-threshold publishes (inclusive)" begin
    s = step(GateState(), 0.10).state
    d = step(s, 0.113)                         # == threshold
    @test d.publish
    @test d.best_metric == 0.10
end

@testset "big regression blocks and does not touch best" begin
    s = step(GateState(), 0.10).state
    d = step(s, 0.20)                          # 0.20 > 0.113 threshold
    @test !d.publish
    @test !d.improved
    @test d.best_metric == 0.10                # best preserved for rollback
    @test !d.state.last_published
    @test d.state.n_blocked == 1
end

@testset "absolute floor lets a tiny best tolerate small noise" begin
    s = step(GateState(), 0.0).state           # best = 0.0 (fractional band is zero-width)
    # threshold = 0.0*1.10 + 0.003 = 0.003 → small noise still publishes
    @test step(s, 0.002).publish
    @test !step(s, 0.004).publish              # beyond the absolute floor → block
end

@testset "recovery after a block resumes publication" begin
    s0 = step(GateState(), 0.10).state         # best 0.10, published
    blocked = step(s0, 0.30)                   # BLOCK
    @test !blocked.publish
    @test !blocked.state.last_published
    # A later eval that recovers within tolerance publishes again.
    recovered = step(blocked.state, 0.11)      # 0.11 <= 0.113 threshold
    @test recovered.publish
    @test recovered.state.last_published
    @test recovered.state.n_blocked == 1       # block count carried forward, not reset
    @test recovered.best_metric == 0.10
end

@testset "non-finite metric is defensive (blocks, keeps best)" begin
    s = step(GateState(), 0.10).state
    d = step(s, NaN)
    @test !d.publish
    @test !d.improved
    @test d.best_metric == 0.10
    @test d.state.n_blocked == 1
end

@testset "persisted decision drives between-eval iterations" begin
    # Between evals the server reuses state.last_published; assert it reflects the
    # last decision so intermediate iterations publish/hold correctly.
    passed = step(GateState(), 0.05).state
    @test passed.last_published                 # → intermediate iters publish
    blocked = step(passed, 0.50).state
    @test !blocked.last_published               # → intermediate iters hold
end

@testset "eval-error cold start fails OPEN (never calibrated)" begin
    # No eval has ever succeeded (best == Inf): a signal failure must not stall the
    # run before a baseline exists, so publish (fail-open).
    s = GateState()
    @test !isfinite(s.best_metric)
    d = gate_on_eval_error(s)
    @test d.publish                                # fail-open
    @test d.state.last_published
    @test d.state.n_eval_failures == 1
    @test d.state.n_blocked == 0                    # a publish is not a block
    # Consecutive failures accumulate.
    d2 = gate_on_eval_error(d.state)
    @test d2.publish
    @test d2.state.n_eval_failures == 2
end

@testset "eval-error after calibration fails CLOSED" begin
    # At least one eval has succeeded (finite best): a broken signal must hold the
    # last-good published weights rather than publish untested ones.
    s = step(GateState(), 0.05).state              # calibrated: best = 0.05
    @test isfinite(s.best_metric)
    d = gate_on_eval_error(s)
    @test !d.publish                               # fail-closed
    @test !d.state.last_published
    @test d.state.best_metric == 0.05              # best preserved for rollback
    @test d.state.n_eval_failures == 1
    @test d.state.n_blocked == 1
    # A later successful finite eval resets the consecutive-failure streak.
    recovered = step(d.state, 0.05)
    @test recovered.state.n_eval_failures == 0
end

@testset "non-finite metric also counts as an eval failure" begin
    s = step(GateState(), 0.10).state
    d = step(s, NaN)
    @test !d.publish
    @test d.state.n_eval_failures == 1             # non-finite → failure streak
    # Recover: finite eval resets the streak.
    @test step(d.state, 0.10).state.n_eval_failures == 0
end

@testset "JSON sidecar round-trips" begin
    # Drive a short sequence, save, reload, and confirm the state is identical.
    s = GateState()
    for m in (0.08, 0.06, 0.20, 0.061)          # pass, improve, block, pass
        s = step(s, m).state
    end
    @test isfinite(s.best_metric)
    dir = mktempdir()
    path = joinpath(dir, "gate_state.json")
    save_gate_state(path, s; metric_name="value_mae", tol_frac=TOL_FRAC, tol_abs=TOL_ABS)
    @test isfile(path)
    loaded = load_gate_state(path)
    @test loaded.best_metric == s.best_metric
    @test loaded.last_published == s.last_published
    @test loaded.n_evals == s.n_evals
    @test loaded.n_blocked == s.n_blocked
    @test loaded.n_eval_failures == s.n_eval_failures
end

@testset "sidecar requires current valid schema" begin
    @test_throws Exception load_gate_state(joinpath(mktempdir(), "absent.json"))

    dir = mktempdir()
    bad = joinpath(dir, "gate_state.json")
    open(bad, "w") do io; print(io, "{not valid json"); end
    @test_throws Exception load_gate_state(bad)

    missing_field = Dict{String,Any}(
        "best_metric" => 0.1,
        "last_published" => true,
        "n_evals" => 1,
        "n_blocked" => 0,
    )
    @test_throws KeyError gate_state_from_dict(missing_field)

    # A fresh (Inf best) state stores best as null and reloads to Inf.
    fresh_path = joinpath(dir, "fresh.json")
    save_gate_state(fresh_path, GateState())
    reloaded = load_gate_state(fresh_path)
    @test !isfinite(reloaded.best_metric)
    @test reloaded.last_published
end
