#####
##### Compare baseline vs progressive simulation training
#####

using JSON3
using Statistics
using Printf

# Configuration
baseline_dir = "sessions/connect-four"
progressive_dir = "sessions/connect-four-progressive"

# Progressive simulation params (must match what was used)
SIM_MIN = 100
SIM_MAX = 600
NUM_ITERS = 15
BASELINE_SIMS = 600

function compute_progressive_sims(iter, num_iters)
    t = iter / num_iters
    return round(Int, SIM_MIN + (SIM_MAX - SIM_MIN) * t)
end

function load_iteration_reports(dir)
    reports = []
    iter_dirs = readdir(joinpath(dir, "iterations"))
    for iter_str in sort(iter_dirs, by=x->parse(Int, x))
        iter = parse(Int, iter_str)
        report_path = joinpath(dir, "iterations", iter_str, "report.json")
        if isfile(report_path)
            report = JSON3.read(read(report_path, String))
            push!(reports, (iter=iter, report=report))
        end
    end
    return reports
end

function load_benchmark_reports(dir)
    benchmarks = []
    iter_dirs = readdir(joinpath(dir, "iterations"))
    for iter_str in sort(iter_dirs, by=x->parse(Int, x))
        iter = parse(Int, iter_str)
        bench_path = joinpath(dir, "iterations", iter_str, "benchmark.json")
        if isfile(bench_path)
            bench = JSON3.read(read(bench_path, String))
            push!(benchmarks, (iter=iter, benchmark=bench))
        end
    end
    return benchmarks
end

function analyze_run(name, dir, get_sims_fn)
    println("\n" * "="^60)
    println("Analyzing: $name")
    println("="^60)

    reports = load_iteration_reports(dir)
    benchmarks = load_benchmark_reports(dir)

    total_self_play_time = 0.0
    total_learning_time = 0.0
    total_sims = 0

    println("\nPer-iteration stats:")
    println("-"^80)
    @printf("%-6s %10s %12s %12s %10s %10s\n",
            "Iter", "Sims/turn", "SP Time(s)", "Total Time", "Samples/s", "Benchmark")
    println("-"^80)

    for (i, (iter, report)) in enumerate(reports)
        sims = get_sims_fn(iter)
        sp_time = report.perfs_self_play.time
        learn_time = report.perfs_learning.time
        samples_speed = report.self_play.samples_gen_speed

        # Get benchmark score for this iteration
        bench_score = "N/A"
        for (bi, bench) in benchmarks
            if bi == iter && length(bench) > 0
                bench_score = @sprintf("%.2f", bench[1].avgr)
            end
        end

        total_self_play_time += sp_time
        total_learning_time += learn_time
        total_sims += sims * 5000  # 5000 games * sims/turn * ~avg_moves

        @printf("%-6d %10d %12.1f %12.1f %10.1f %10s\n",
                iter, sims, sp_time, sp_time + learn_time, samples_speed, bench_score)
    end

    println("-"^80)
    println("\nSummary:")
    println("  Total self-play time: $(round(total_self_play_time/60, digits=1)) minutes")
    println("  Total learning time: $(round(total_learning_time/60, digits=1)) minutes")
    println("  Total training time: $(round((total_self_play_time + total_learning_time)/60, digits=1)) minutes")

    # Final benchmark
    if !isempty(benchmarks)
        final_bench = last(benchmarks).benchmark
        if length(final_bench) > 0
            println("  Final benchmark (AlphaZero vs MCTS): $(final_bench[1].avgr)")
        end
    end

    return (
        total_sp_time = total_self_play_time,
        total_learn_time = total_learning_time,
        total_time = total_self_play_time + total_learning_time,
        reports = reports,
        benchmarks = benchmarks
    )
end

function main()
    println("Progressive Simulation Budget Analysis")
    println("======================================")

    # Analyze baseline (constant 600 sims)
    baseline_results = nothing
    if isdir(baseline_dir)
        baseline_results = analyze_run("Baseline (600 sims/turn)", baseline_dir,
                                       iter -> BASELINE_SIMS)
    else
        println("Baseline directory not found: $baseline_dir")
    end

    # Analyze progressive
    progressive_results = nothing
    if isdir(progressive_dir)
        progressive_results = analyze_run("Progressive ($SIM_MIN → $SIM_MAX sims/turn)",
                                          progressive_dir,
                                          iter -> compute_progressive_sims(iter, NUM_ITERS))
    else
        println("Progressive directory not found: $progressive_dir")
    end

    # Comparison
    if !isnothing(baseline_results) && !isnothing(progressive_results)
        println("\n" * "="^60)
        println("COMPARISON")
        println("="^60)

        speedup = baseline_results.total_time / progressive_results.total_time
        sp_speedup = baseline_results.total_sp_time / progressive_results.total_sp_time

        println("\nTime comparison:")
        @printf("  Baseline total time:     %.1f minutes\n", baseline_results.total_time/60)
        @printf("  Progressive total time:  %.1f minutes\n", progressive_results.total_time/60)
        @printf("  Overall speedup:         %.2fx\n", speedup)
        @printf("  Self-play speedup:       %.2fx\n", sp_speedup)

        # Compare final benchmarks
        if !isempty(baseline_results.benchmarks) && !isempty(progressive_results.benchmarks)
            base_final = last(baseline_results.benchmarks).benchmark[1].avgr
            prog_final = last(progressive_results.benchmarks).benchmark[1].avgr
            println("\nFinal benchmark comparison:")
            @printf("  Baseline:     %.3f\n", base_final)
            @printf("  Progressive:  %.3f\n", prog_final)
        end
    end
end

main()
