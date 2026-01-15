#####
##### Pig Game Comparison: Standard vs Stochastic AlphaZero
##### Fair wall-clock time comparison
#####
##### Run with: julia --project scripts/pig_comparison.jl
#####

using AlphaZero
using Printf
using Statistics

# Load both Pig game variants
include(joinpath(@__DIR__, "..", "games", "pig", "main.jl"))
include(joinpath(@__DIR__, "..", "games", "pig-deterministic", "main.jl"))

using .Pig
using .PigDeterministic

#####
##### Configuration
#####

# Training time limit per approach (in seconds)
const TRAINING_TIME_SECONDS = 1500  # 25 minutes each

# Number of training iterations (will stop early if time limit reached)
const MAX_ITERS = 200

#####
##### Statistical Analysis
#####

"""
Calculate standard error for a proportion.
SE = sqrt(p * (1-p) / n)
"""
function standard_error(p::Float64, n::Int)
    return sqrt(p * (1 - p) / n)
end

"""
Calculate z-score for comparing two proportions.
H0: p1 = p2 (no difference)
"""
function two_proportion_z_test(p1::Float64, n1::Int, p2::Float64, n2::Int)
    # Pooled proportion under null hypothesis
    p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)

    # Standard error of the difference
    se_diff = sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))

    # Z-score
    z = (p1 - p2) / se_diff

    return z
end

"""
Calculate two-tailed p-value from z-score using normal approximation.
"""
function p_value_from_z(z::Float64)
    # Using the complementary error function for normal CDF
    # P(Z > |z|) * 2 for two-tailed test
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
end

"""
Calculate confidence interval for a proportion.
"""
function confidence_interval(p::Float64, n::Int; confidence=0.95)
    z = confidence == 0.95 ? 1.96 : (confidence == 0.99 ? 2.576 : 1.645)
    se = standard_error(p, n)
    return (max(0, p - z * se), min(1, p + z * se))
end

#####
##### Timed Training
#####

# Import the training internals we need
import AlphaZero: self_play_step!, learning_step!, memory_report, resize_memory!
import AlphaZero: Handlers, Report
import AlphaZero.UserInterface: zeroth_iteration!, missing_zeroth_iteration

"""
Run training for a fixed wall-clock time, return elapsed time and iteration count.
Calls individual training steps directly to enable time-limited iteration.
"""
function timed_training!(session, time_limit_seconds)
    start_time = time()
    iters_completed = 0
    env = session.env
    handler = session  # Session itself is the handler

    # Run zeroth iteration (initial benchmark) if needed
    if missing_zeroth_iteration(session)
        success = zeroth_iteration!(session)
        if !success
            println("ERROR: Initial checks failed")
            return time() - start_time, 0
        end
    end

    while (time() - start_time) < time_limit_seconds
        # Check if enough time remains for another iteration (estimate 10s minimum)
        elapsed = time() - start_time
        remaining = time_limit_seconds - elapsed
        if remaining < 10
            break
        end

        # Run one training iteration (following exact pattern from train! in training.jl)
        Handlers.iteration_started(handler)
        resize_memory!(env, env.params.mem_buffer_size[env.itc])

        # Self-play phase with performance tracking
        sprep, spperfs = Report.@timed self_play_step!(env, handler)

        # Memory analysis with performance tracking
        mrep, mperfs = Report.@timed memory_report(env, handler)

        # Learning phase with performance tracking
        lrep, lperfs = Report.@timed learning_step!(env, handler)

        # Create iteration report and notify handler
        rep = Report.Iteration(spperfs, mperfs, lperfs, sprep, mrep, lrep)
        env.itc += 1
        Handlers.iteration_finished(handler, rep)

        iters_completed += 1
        @printf("  Iteration %d completed (%.1fs elapsed)\n", iters_completed, time() - start_time)
    end

    Handlers.training_finished(handler)
    elapsed_time = time() - start_time
    return elapsed_time, iters_completed
end

#####
##### Main
#####

function main()
    println("="^70)
    println("Pig Game: Standard vs Stochastic MCTS Comparison")
    println("Fair wall-clock time comparison: $(TRAINING_TIME_SECONDS ÷ 60) minutes each")
    println("Benchmark: 600 games per direction (1200 total) for statistical significance")
    println("="^70)

    #####
    ##### Phase 1: Train Standard AlphaZero (hidden stochasticity)
    #####

    println("\n" * "="^70)
    println("PHASE 1: Standard AlphaZero (hidden stochasticity in play!)")
    println("="^70)

    # Get the experiment from PigDeterministic
    exp_standard = PigDeterministic.Training.experiment

    # Override to use many iterations (will be time-limited)
    new_params = Params(exp_standard.params; num_iters=MAX_ITERS)
    exp_standard = Experiment(
        exp_standard.name,
        exp_standard.gspec,
        new_params,
        exp_standard.mknet,
        exp_standard.netparams,
        exp_standard.benchmark)

    # Create session and train with time limit
    session_standard = Session(exp_standard; dir="sessions/pig-comparison-standard")

    println("\nTraining for $(TRAINING_TIME_SECONDS) seconds...")
    time_standard, iters_standard = timed_training!(session_standard, TRAINING_TIME_SECONDS)

    println("\n" * "-"^40)
    @printf("Standard AlphaZero completed:\n")
    @printf("  Time: %.1f seconds\n", time_standard)
    @printf("  Iterations: %d\n", iters_standard)
    @printf("  Avg time/iter: %.2f seconds\n", time_standard / max(iters_standard, 1))

    #####
    ##### Phase 2: Train Stochastic AlphaZero (explicit chance nodes with sampling)
    #####

    println("\n" * "="^70)
    println("PHASE 2: Stochastic AlphaZero (progressive widening for chance nodes)")
    println("="^70)

    # Get the experiment from Pig (stochastic version)
    exp_stochastic = Pig.Training.experiment

    # Override to use many iterations (will be time-limited)
    new_params = Params(exp_stochastic.params; num_iters=MAX_ITERS)
    exp_stochastic = Experiment(
        exp_stochastic.name,
        exp_stochastic.gspec,
        new_params,
        exp_stochastic.mknet,
        exp_stochastic.netparams,
        exp_stochastic.benchmark)

    # Create session and train with time limit
    session_stochastic = Session(exp_stochastic; dir="sessions/pig-comparison-stochastic")

    println("\nTraining for $(TRAINING_TIME_SECONDS) seconds...")
    time_stochastic, iters_stochastic = timed_training!(session_stochastic, TRAINING_TIME_SECONDS)

    println("\n" * "-"^40)
    @printf("Stochastic AlphaZero completed:\n")
    @printf("  Time: %.1f seconds\n", time_stochastic)
    @printf("  Iterations: %d\n", iters_stochastic)
    @printf("  Avg time/iter: %.2f seconds\n", time_stochastic / max(iters_stochastic, 1))

    #####
    ##### Final Summary
    #####

    println("\n" * "="^70)
    println("COMPARISON COMPLETE")
    println("="^70)

    println("\n### Training Summary ###")
    println("-"^40)
    @printf("%-25s %15s %15s\n", "", "Standard", "Stochastic")
    @printf("%-25s %15.1f %15.1f\n", "Training time (s):", time_standard, time_stochastic)
    @printf("%-25s %15d %15d\n", "Iterations completed:", iters_standard, iters_stochastic)
    @printf("%-25s %15.2f %15.2f\n", "Seconds per iteration:",
            time_standard/max(iters_standard,1), time_stochastic/max(iters_stochastic,1))

    println("\n### Benchmark Configuration ###")
    println("Each benchmark: 600 games agent-first + 600 games Hold20-first = 1200 games")
    println("Standard error for p=0.5: SE ≈ 1.44% (√(0.5×0.5/1200))")
    println("95% CI width: ±2.8%")

    #####
    ##### Statistical Significance Analysis
    #####

    println("\n" * "="^70)
    println("STATISTICAL SIGNIFICANCE ANALYSIS")
    println("="^70)

    # Note: To get actual win rates, user should extract from logs above
    # Here we show the formula and what would be significant

    n_games = 1200  # Total games per approach

    println("\n### How to interpret results ###")
    println("-"^40)
    @printf("With n=%d games per approach:\n", n_games)
    @printf("  Standard Error (SE) at p=0.50: %.2f%%\n", 100 * standard_error(0.5, n_games))
    @printf("  Standard Error (SE) at p=0.40: %.2f%%\n", 100 * standard_error(0.4, n_games))
    @printf("  95%% Confidence Interval: ±%.2f%%\n", 100 * 1.96 * standard_error(0.5, n_games))

    println("\n### Minimum detectable difference for significance ###")
    println("-"^40)
    # For comparing two proportions with same n
    # SE_diff = sqrt(2 * p * (1-p) / n) for pooled estimate
    se_diff_50 = sqrt(2 * 0.5 * 0.5 / n_games)
    min_diff_95 = 1.96 * se_diff_50
    min_diff_99 = 2.576 * se_diff_50
    @printf("At p ≈ 0.50: need %.1f%% difference for p<0.05\n", 100 * min_diff_95)
    @printf("At p ≈ 0.50: need %.1f%% difference for p<0.01\n", 100 * min_diff_99)

    println("\n### Example p-value calculations ###")
    println("-"^40)
    println("If Standard wins 35% and Stochastic wins 45%:")
    z = two_proportion_z_test(0.45, n_games, 0.35, n_games)
    p = p_value_from_z(z)
    @printf("  z-score = %.3f, p-value = %.6f %s\n", z, p, p < 0.05 ? "(significant at α=0.05)" : "")

    println("\nIf Standard wins 38% and Stochastic wins 42%:")
    z = two_proportion_z_test(0.42, n_games, 0.38, n_games)
    p = p_value_from_z(z)
    @printf("  z-score = %.3f, p-value = %.6f %s\n", z, p, p < 0.05 ? "(significant at α=0.05)" : "")

    println("\nResults saved to:")
    println("  Standard:   sessions/pig-comparison-standard/")
    println("  Stochastic: sessions/pig-comparison-stochastic/")

    return session_standard, session_stochastic
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
