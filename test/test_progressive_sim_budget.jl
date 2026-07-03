# Regression tests for the progressive MCTS-simulation-budget schedules.
#
# The schedule functions (compute_sim_budget / compute_ramp_turns /
# compute_turn_sim_budget) were wired into the self-play client in commit
# 498f66e (mcts_budget_mode = progressive | turn_progressive). They feed
# BatchedMctsPlayer.sim_budget_fn, which think() consumes per turn:
#     niters = sim_budget_fn(turn_count)
# A bug in this math silently trains at the wrong search depth, so the
# boundary values and monotonicity are pinned here. The constant-mode path
# (sim_budget_fn === nothing) is exercised end-to-end by the eval ladder and
# self-play; these tests cover the active progressive path's arithmetic.

using Test
using AlphaZero
using AlphaZero: ProgressiveSimParams, compute_sim_budget,
                 TurnProgressiveSimParams, compute_turn_sim_budget, compute_ramp_turns

@testset "Progressive sim budget (per-iteration)" begin
    p = ProgressiveSimParams(sim_min=100, sim_max=400)
    N = 10

    # Boundary: final iteration hits sim_max exactly; every iteration in range
    # stays within [sim_min, sim_max].
    @test compute_sim_budget(p, N, N) == 400
    @test compute_sim_budget(p, 5, N) == 250          # t=0.5 → midpoint
    @test compute_sim_budget(p, 1, N) == 130          # t=0.1 → 100 + 300*0.1
    for iter in 1:N
        b = compute_sim_budget(p, iter, N)
        @test 100 <= b <= 400
    end

    # Monotonic non-decreasing in iteration.
    budgets = [compute_sim_budget(p, iter, N) for iter in 1:N]
    @test issorted(budgets)

    # Degenerate single-iteration run must not divide by zero / stay in range.
    @test compute_sim_budget(p, 1, 1) == 400          # t=1 → sim_max

    # A flat schedule (min == max) is always constant.
    flat = ProgressiveSimParams(sim_min=300, sim_max=300)
    @test all(compute_sim_budget(flat, i, N) == 300 for i in 1:N)
end

@testset "Turn-progressive ramp turns" begin
    p = TurnProgressiveSimParams(turn_sim_min=2, turn_sim_target=600,
                                 ramp_turns_initial=30, ramp_turns_final=3)
    N = 50

    # First iteration ramps slowly (30 turns); final iteration ramps fast (3).
    @test compute_ramp_turns(p, 1, N) == 30.0
    @test compute_ramp_turns(p, N, N) == 3.0
    @test compute_ramp_turns(p, 1, 1) == 3.0          # num_iters<=1 → final

    # Monotonic non-increasing in iteration (ramp gets faster as we learn).
    ramps = [compute_ramp_turns(p, iter, N) for iter in 1:N]
    @test issorted(ramps; rev=true)
    @test all(3.0 <= r <= 30.0 for r in ramps)
end

@testset "Turn-progressive sim budget (per-turn)" begin
    p = TurnProgressiveSimParams(turn_sim_min=2, turn_sim_target=600,
                                 ramp_turns_initial=30, ramp_turns_final=3)
    N = 50

    # Iteration 1 (ramp = 30 turns).
    @test compute_turn_sim_budget(p, 0, 1, N) == 2      # game start → min
    @test compute_turn_sim_budget(p, 15, 1, N) == 301   # halfway up the ramp
    @test compute_turn_sim_budget(p, 30, 1, N) == 600   # reached target
    @test compute_turn_sim_budget(p, 100, 1, N) == 600  # past ramp → capped

    # Final iteration (ramp = 3 turns) reaches target far sooner.
    @test compute_turn_sim_budget(p, 3, N, N) == 600
    @test compute_turn_sim_budget(p, 0, N, N) == 2

    # Per-turn budget is monotonic non-decreasing and bounded by the target.
    for iter in (1, 25, N)
        vals = [compute_turn_sim_budget(p, t, iter, N) for t in 0:40]
        @test issorted(vals)
        @test all(2 <= v <= 600 for v in vals)
        @test vals[end] == 600
    end
end
