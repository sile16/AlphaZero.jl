#####
##### Backgammon Full Evaluation: AlphaZero vs Stochastic AlphaZero
##### Comprehensive benchmark against Random and GnuBG
#####
##### Run with: julia --project scripts/backgammon_full_evaluation.jl
#####
##### Configuration flags:
#####   --training-hours N     Training time per approach (default: 1)
#####   --eval-games N         Games per evaluation matchup (default: 1000)
#####   --skip-training        Skip training, load from existing sessions
#####   --quick                Quick test mode (5 min training, 100 eval games)
#####   --wandb                Enable wandb logging (note: disables GnuBG evaluation)
#####
##### Note: wandb (uses PythonCall) and GnuBG (uses PyCall) cannot be used together
##### in the same session due to Python runtime conflicts. Use one or the other.
#####

using Distributed
using Printf
using Statistics
using Dates
using Random

# Parse command line args early (before worker setup)
const QUICK_MODE = "--quick" in ARGS
const SKIP_TRAINING = "--skip-training" in ARGS
const WANDB_REQUESTED = "--wandb" in ARGS

function parse_arg(args, flag, default)
    idx = findfirst(==(flag), args)
    if idx !== nothing && idx < length(args)
        return parse(Float64, args[idx + 1])
    end
    return default
end

# Configuration
const TRAINING_HOURS = QUICK_MODE ? 5/60 : parse_arg(ARGS, "--training-hours", 1.0)
const TRAINING_TIME_SECONDS = TRAINING_HOURS * 3600
const EVAL_GAMES_PER_MATCHUP = Int(QUICK_MODE ? 100 : parse_arg(ARGS, "--eval-games", 1000))

# MCTS iterations - higher for training (quality), lower for eval (speed)
const TRAINING_MCTS_ITERS = Int(parse_arg(ARGS, "--training-mcts", 400))
const EVAL_MCTS_ITERS = Int(QUICK_MODE ? 50 : parse_arg(ARGS, "--eval-mcts", 100))

# GnuBG evaluation is slow (~0.5 games/sec via CLI), so use fewer games
const GNUBG_EVAL_GAMES = Int(QUICK_MODE ? 20 : parse_arg(ARGS, "--gnubg-games", 100))

#####
##### Worker and Module Setup (Julia 1.12 compatible)
#####
##### To use distributed workers, start Julia with: julia --project -p N scripts/backgammon_full_evaluation.jl
##### Workers must be spawned BEFORE CUDA is initialized for proper GPU memory sharing.
#####

# Step 1: Load AlphaZero on main process and all pre-spawned workers
@everywhere using AlphaZero
using AlphaZero: MctsPlayer, MctsParams, SimParams, TwoPlayers, play_game, Network, MCTS
using AlphaZero: ConstSchedule, SelfPlayParams

# Step 2: Define game module path (interpolate into @everywhere block)
const GAMES_DIR = joinpath(@__DIR__, "..", "games")

# Step 3: Load game modules on main process and all workers using explicit eval
# This ensures proper module binding for Julia 1.12's stricter deserialization rules
@everywhere begin
    # Use interpolation to pass the path from main process
    const _GAMES_DIR = $GAMES_DIR

    # Load modules in Main namespace explicitly
    Base.eval(Main, quote
        include(joinpath($_GAMES_DIR, "backgammon", "main.jl"))
        include(joinpath($_GAMES_DIR, "backgammon-deterministic", "main.jl"))
    end)
end

# Step 4: Now bring the modules into scope on all processes
@everywhere using .Backgammon
@everywhere using .BackgammonDeterministic

println("Workers ready: $(nprocs()) processes (1 main + $(nworkers()) workers)")

# Initialize wandb BEFORE loading GnubgPlayer if requested
# (PythonCall must be loaded before PyCall to avoid conflicts)
const USE_WANDB = if WANDB_REQUESTED
    @eval using AlphaZero.Wandb: wandb_available, wandb_init, wandb_log, wandb_finish
    wandb_available()
else
    false
end

# Stub functions if wandb not used
if !USE_WANDB
    wandb_init(; kwargs...) = nothing
    wandb_log(args...; kwargs...) = nothing
    wandb_finish() = nothing
end

# Load GnubgPlayer only if wandb is not enabled (PyCall/PythonCall conflict)
const GNUBG_AVAILABLE = if !WANDB_REQUESTED
    try
        include(joinpath(@__DIR__, "GnubgPlayer.jl"))
        @eval using .GnubgPlayer
        true
    catch e
        @warn "GnubgPlayer failed to load: $e"
        false
    end
else
    @info "Skipping GnuBG (PyCall conflicts with PythonCall/wandb)"
    false
end

#####
##### Statistical Analysis Functions
#####

"""
Calculate standard error for a proportion.
SE = sqrt(p * (1-p) / n)
"""
function standard_error(p::Float64, n::Int)
    n == 0 && return Inf
    return sqrt(p * (1 - p) / n)
end

"""
Calculate z-score for comparing two proportions.
H0: p1 = p2 (no difference)
"""
function two_proportion_z_test(p1::Float64, n1::Int, p2::Float64, n2::Int)
    (n1 == 0 || n2 == 0) && return 0.0
    # Pooled proportion under null hypothesis
    p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    p_pooled == 0 || p_pooled == 1 && return 0.0

    # Standard error of the difference
    se_diff = sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    se_diff == 0 && return 0.0

    return (p1 - p2) / se_diff
end

"""
Calculate two-tailed p-value from z-score using normal approximation.
"""
function p_value_from_z(z::Float64)
    x = abs(z) / sqrt(2)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf_approx = 1.0 - poly * exp(-x * x)
    return 2 * (1 - 0.5 * (1 + erf_approx))
end

"""
Calculate confidence interval for a proportion.
"""
function confidence_interval(p::Float64, n::Int; confidence=0.95)
    n == 0 && return (0.0, 1.0)
    z = confidence == 0.95 ? 1.96 : (confidence == 0.99 ? 2.576 : 1.645)
    se = standard_error(p, n)
    return (max(0.0, p - z * se), min(1.0, p + z * se))
end

"""
Calculate minimum sample size needed to detect a given effect size with desired power.
"""
function min_sample_size_for_significance(effect_size::Float64, alpha::Float64=0.05, power::Float64=0.80)
    z_alpha = 1.96  # for alpha=0.05, two-tailed
    z_beta = 0.84   # for power=0.80
    p1 = 0.5 + effect_size/2
    p2 = 0.5 - effect_size/2
    p_bar = (p1 + p2) / 2
    n = 2 * p_bar * (1 - p_bar) * ((z_alpha + z_beta) / effect_size)^2
    return ceil(Int, n)
end

"""
Format significance result.
"""
function significance_string(p_val::Float64)
    if p_val < 0.001
        return "***  (p < 0.001)"
    elseif p_val < 0.01
        return "**   (p < 0.01)"
    elseif p_val < 0.05
        return "*    (p < 0.05)"
    else
        return "ns   (p = $(round(p_val, digits=3)))"
    end
end

# GnubgPlayer is loaded from GnubgPlayer.jl module
# It provides the GnubgBaseline player for use in evaluations

#####
##### Timed Training
#####

import AlphaZero: self_play_step!, learning_step!, memory_report, resize_memory!
import AlphaZero: Handlers, Report
import AlphaZero.UserInterface: zeroth_iteration!, missing_zeroth_iteration

"""
Run training for a fixed wall-clock time, return elapsed time and iteration count.
"""
function timed_training!(session, time_limit_seconds; progress_interval=60)
    start_time = time()
    iters_completed = 0
    env = session.env
    handler = session

    # Run zeroth iteration (initial benchmark) if needed
    if missing_zeroth_iteration(session)
        success = zeroth_iteration!(session)
        if !success
            println("ERROR: Initial checks failed")
            return time() - start_time, 0
        end
    end

    last_progress = start_time

    while (time() - start_time) < time_limit_seconds
        elapsed = time() - start_time
        remaining = time_limit_seconds - elapsed
        if remaining < 30  # Need at least 30s for an iteration
            break
        end

        # Progress update
        if time() - last_progress > progress_interval
            @printf("    [%.1f min elapsed, %.1f min remaining, %d iters]\n",
                    elapsed/60, remaining/60, iters_completed)
            last_progress = time()
        end

        # Run one training iteration
        Handlers.iteration_started(handler)
        resize_memory!(env, env.params.mem_buffer_size[env.itc])

        sprep, spperfs = Report.@timed self_play_step!(env, handler)
        mrep, mperfs = Report.@timed memory_report(env, handler)
        lrep, lperfs = Report.@timed learning_step!(env, handler)

        rep = Report.Iteration(spperfs, mperfs, lperfs, sprep, mrep, lrep)
        env.itc += 1
        Handlers.iteration_finished(handler, rep)

        iters_completed += 1
    end

    Handlers.training_finished(handler)
    elapsed_time = time() - start_time
    return elapsed_time, iters_completed
end

#####
##### Evaluation Functions
#####

"""
Evaluate a trained network against a baseline player.
Returns (wins, losses, draws, avg_reward, results_detail).
"""
function evaluate_vs_baseline(gspec, network, n_games, baseline_player;
                               mcts_params=nothing, use_gpu=true, name="Baseline",
                               mcts_iters=EVAL_MCTS_ITERS)
    if isnothing(mcts_params)
        mcts_params = MctsParams(
            num_iters_per_turn=mcts_iters,
            cpuct=1.0,
            temperature=ConstSchedule(0.1),  # Low temp for evaluation
            dirichlet_noise_ϵ=0.0,
            dirichlet_noise_α=1.0
        )
    end

    # Create MCTS player
    net = Network.copy(network, on_gpu=use_gpu, test_mode=true)

    all_rewards = Float64[]
    games_per_color = n_games ÷ 2

    # Play games: AlphaZero as white (player 0)
    @printf("  Playing %d games (AlphaZero as P0 vs %s)...\n", games_per_color, name)
    player_white = MctsPlayer(gspec, net, mcts_params)

    for i in 1:games_per_color
        MCTS.reset!(player_white.mcts)
        AlphaZero.reset!(baseline_player)
        trace = play_game(gspec, TwoPlayers(player_white, baseline_player))
        push!(all_rewards, trace.rewards[end])
        if i % 100 == 0
            @printf("    Game %d/%d (wins so far: %d)\n",
                    i, games_per_color, count(r -> r > 0, all_rewards))
        end
    end

    # Play games: AlphaZero as black (player 1)
    @printf("  Playing %d games (AlphaZero as P1 vs %s)...\n", games_per_color, name)
    player_black = MctsPlayer(gspec, net, mcts_params)

    for i in 1:games_per_color
        MCTS.reset!(player_black.mcts)
        AlphaZero.reset!(baseline_player)
        trace = play_game(gspec, TwoPlayers(baseline_player, player_black))
        push!(all_rewards, -trace.rewards[end])  # Negate since we're black
        if i % 100 == 0
            total_played = games_per_color + i
            @printf("    Game %d/%d (wins so far: %d)\n",
                    total_played, n_games, count(r -> r > 0, all_rewards))
        end
    end

    wins = count(r -> r > 0, all_rewards)
    losses = count(r -> r < 0, all_rewards)
    draws = count(r -> r == 0, all_rewards)
    avg_reward = mean(all_rewards)

    return wins, losses, draws, avg_reward, all_rewards
end

"""
Head-to-head evaluation between two networks.
Uses deterministic game spec for fair comparison.
"""
function head_to_head(gspec, net1, net2, n_games;
                      mcts_params=nothing, use_gpu=true,
                      name1="Model1", name2="Model2",
                      mcts_iters=EVAL_MCTS_ITERS)
    if isnothing(mcts_params)
        mcts_params = MctsParams(
            num_iters_per_turn=mcts_iters,
            cpuct=1.0,
            temperature=ConstSchedule(0.1),
            dirichlet_noise_ϵ=0.0,
            dirichlet_noise_α=1.0
        )
    end

    # Create network copies
    network1 = Network.copy(net1, on_gpu=use_gpu, test_mode=true)
    network2 = Network.copy(net2, on_gpu=use_gpu, test_mode=true)

    all_rewards = Float64[]
    games_per_color = n_games ÷ 2

    # Play games: net1 as white
    @printf("  Playing %d games (%s as P0)...\n", games_per_color, name1)
    player1_white = MctsPlayer(gspec, network1, mcts_params)
    player2_black = MctsPlayer(gspec, network2, mcts_params)

    for i in 1:games_per_color
        MCTS.reset!(player1_white.mcts)
        MCTS.reset!(player2_black.mcts)
        trace = play_game(gspec, TwoPlayers(player1_white, player2_black))
        push!(all_rewards, trace.rewards[end])
        if i % 100 == 0
            @printf("    Game %d/%d\n", i, games_per_color)
        end
    end

    # Play games: net1 as black
    @printf("  Playing %d games (%s as P1)...\n", games_per_color, name1)
    player2_white = MctsPlayer(gspec, network2, mcts_params)
    player1_black = MctsPlayer(gspec, network1, mcts_params)

    for i in 1:games_per_color
        MCTS.reset!(player2_white.mcts)
        MCTS.reset!(player1_black.mcts)
        trace = play_game(gspec, TwoPlayers(player2_white, player1_black))
        push!(all_rewards, -trace.rewards[end])  # Negate since net1 is black
        if i % 100 == 0
            @printf("    Game %d/%d\n", games_per_color + i, n_games)
        end
    end

    net1_wins = count(r -> r > 0, all_rewards)
    net2_wins = count(r -> r < 0, all_rewards)
    draws = count(r -> r == 0, all_rewards)
    net1_avg_reward = mean(all_rewards)

    return net1_wins, net2_wins, draws, net1_avg_reward, all_rewards
end

#####
##### Results Storage
#####

mutable struct EvaluationResults
    # Training info
    standard_training_time::Float64
    standard_iterations::Int
    stochastic_training_time::Float64
    stochastic_iterations::Int

    # vs Random
    std_vs_random::NamedTuple
    sto_vs_random::NamedTuple

    # vs GnuBG
    std_vs_gnubg::NamedTuple
    sto_vs_gnubg::NamedTuple

    # Head-to-head
    head_to_head::NamedTuple

    # Timestamp
    timestamp::String
end

function save_results(results::EvaluationResults, filename::String)
    open(filename, "w") do f
        println(f, "=" ^ 80)
        println(f, "BACKGAMMON EVALUATION RESULTS")
        println(f, "Timestamp: $(results.timestamp)")
        println(f, "Configuration: SHORT_GAME=true, DOUBLES_ONLY=true")
        println(f, "=" ^ 80)

        println(f, "\n### TRAINING SUMMARY ###")
        println(f, "Standard AlphaZero:")
        @printf(f, "  Time: %.1f seconds (%.2f hours)\n",
                results.standard_training_time, results.standard_training_time/3600)
        @printf(f, "  Iterations: %d\n", results.standard_iterations)

        println(f, "\nStochastic AlphaZero:")
        @printf(f, "  Time: %.1f seconds (%.2f hours)\n",
                results.stochastic_training_time, results.stochastic_training_time/3600)
        @printf(f, "  Iterations: %d\n", results.stochastic_iterations)

        println(f, "\n### EVALUATION VS RANDOM ###")
        println(f, "-" ^ 60)
        @printf(f, "%-25s %15s %15s\n", "", "Standard", "Stochastic")
        @printf(f, "%-25s %15d %15d\n", "Wins:",
                results.std_vs_random.wins, results.sto_vs_random.wins)
        @printf(f, "%-25s %15d %15d\n", "Losses:",
                results.std_vs_random.losses, results.sto_vs_random.losses)
        @printf(f, "%-25s %14.1f%% %14.1f%%\n", "Win rate:",
                results.std_vs_random.win_rate * 100, results.sto_vs_random.win_rate * 100)
        @printf(f, "%-25s %15.3f %15.3f\n", "Avg reward:",
                results.std_vs_random.avg_reward, results.sto_vs_random.avg_reward)

        println(f, "\n### EVALUATION VS GNUBG (0-PLY) ###")
        println(f, "-" ^ 60)
        @printf(f, "%-25s %15s %15s\n", "", "Standard", "Stochastic")
        @printf(f, "%-25s %15d %15d\n", "Wins:",
                results.std_vs_gnubg.wins, results.sto_vs_gnubg.wins)
        @printf(f, "%-25s %15d %15d\n", "Losses:",
                results.std_vs_gnubg.losses, results.sto_vs_gnubg.losses)
        @printf(f, "%-25s %14.1f%% %14.1f%%\n", "Win rate:",
                results.std_vs_gnubg.win_rate * 100, results.sto_vs_gnubg.win_rate * 100)
        @printf(f, "%-25s %15.3f %15.3f\n", "Avg reward:",
                results.std_vs_gnubg.avg_reward, results.sto_vs_gnubg.avg_reward)

        println(f, "\n### HEAD-TO-HEAD: STANDARD VS STOCHASTIC ###")
        println(f, "-" ^ 60)
        h2h = results.head_to_head
        @printf(f, "Standard wins:   %d (%.1f%%)\n", h2h.std_wins, h2h.std_win_rate * 100)
        @printf(f, "Stochastic wins: %d (%.1f%%)\n", h2h.sto_wins, h2h.sto_win_rate * 100)
        @printf(f, "Draws:           %d\n", h2h.draws)
        @printf(f, "Z-score:         %.2f\n", h2h.z_score)
        @printf(f, "P-value:         %.4f\n", h2h.p_value)
        @printf(f, "Significance:    %s\n", h2h.significance)

        println(f, "\n### STATISTICAL ANALYSIS ###")
        println(f, "-" ^ 60)
        n = results.std_vs_random.total
        @printf(f, "Games per evaluation: %d\n", n)
        @printf(f, "Standard error at p=0.50: %.2f%%\n", 100 * standard_error(0.5, n))
        @printf(f, "95%% CI half-width: ±%.2f%%\n", 100 * 1.96 * standard_error(0.5, n))

        min_detectable = 1.96 * sqrt(2 * 0.5 * 0.5 / n) * 2
        @printf(f, "Min detectable difference (p<0.05): %.1f%%\n", min_detectable * 100)
    end
    println("Results saved to: $filename")
end

#####
##### Main Evaluation
#####

function main()
    start_time = now()

    println("=" ^ 80)
    println("BACKGAMMON FULL EVALUATION")
    println("=" ^ 80)
    println("\nConfiguration:")
    println("  - Game mode: SHORT_GAME=true, DOUBLES_ONLY=true")
    @printf("  - Training time: %.1f hours per approach\n", TRAINING_HOURS)
    @printf("  - Training MCTS iters: %d per turn\n", TRAINING_MCTS_ITERS)
    @printf("  - Eval MCTS iters: %d per turn\n", EVAL_MCTS_ITERS)
    @printf("  - Eval games (vs Random, H2H): %d per matchup\n", EVAL_GAMES_PER_MATCHUP)
    @printf("  - Eval games (vs GnuBG): %d (CLI is slow)\n", GNUBG_EVAL_GAMES)
    println("  - Skip training: $SKIP_TRAINING")
    println("  - Quick mode: $QUICK_MODE")
    println("  - WandB logging: $USE_WANDB")
    println("\nStart time: $start_time")

    # Initialize wandb if enabled
    if USE_WANDB
        wandb_init(
            project="backgammon-evaluation",
            name="standard-vs-stochastic-$(Dates.format(start_time, "yyyymmdd-HHMMSS"))",
            config=Dict(
                "training_hours" => TRAINING_HOURS,
                "training_mcts_iters" => TRAINING_MCTS_ITERS,
                "eval_mcts_iters" => EVAL_MCTS_ITERS,
                "eval_games" => EVAL_GAMES_PER_MATCHUP,
                "gnubg_games" => GNUBG_EVAL_GAMES,
                "short_game" => true,
                "doubles_only" => true,
                "quick_mode" => QUICK_MODE,
            )
        )
        println("WandB run initialized")
    end

    # Sample size analysis
    println("\n### Sample Size Analysis ###")
    for effect in [0.05, 0.10, 0.15, 0.20]
        n = min_sample_size_for_significance(effect)
        @printf("  To detect %.0f%% difference with 80%% power: %d games needed\n", effect*100, n)
    end

    println("\nWith $EVAL_GAMES_PER_MATCHUP games:")
    min_detectable = 1.96 * sqrt(2 * 0.5 * 0.5 / EVAL_GAMES_PER_MATCHUP) * 2
    @printf("  Min detectable difference (p<0.05): %.1f%%\n", min_detectable * 100)

    # Initialize results
    results = EvaluationResults(
        0.0, 0, 0.0, 0,
        (wins=0, losses=0, draws=0, avg_reward=0.0, win_rate=0.0, total=0),
        (wins=0, losses=0, draws=0, avg_reward=0.0, win_rate=0.0, total=0),
        (wins=0, losses=0, draws=0, avg_reward=0.0, win_rate=0.0, total=0),
        (wins=0, losses=0, draws=0, avg_reward=0.0, win_rate=0.0, total=0),
        (std_wins=0, sto_wins=0, draws=0, std_win_rate=0.0, sto_win_rate=0.0,
         z_score=0.0, p_value=1.0, significance=""),
        string(start_time)
    )

    #####
    ##### Phase 1: Training
    #####

    session_standard = nothing
    session_stochastic = nothing

    if SKIP_TRAINING
        println("\n" * "=" ^ 80)
        println("SKIPPING TRAINING - Loading from existing sessions")
        println("=" ^ 80)

        # Try to load existing sessions
        exp_standard = BackgammonDeterministic.Training.experiment
        exp_stochastic = Backgammon.Training.experiment

        try
            session_standard = Session(exp_standard; dir="sessions/backgammon-comparison-standard")
            session_stochastic = Session(exp_stochastic; dir="sessions/backgammon-comparison-stochastic")
            println("Loaded existing sessions successfully")
        catch e
            println("ERROR: Could not load existing sessions: $e")
            println("Please run without --skip-training first")
            return
        end
    else
        #####
        ##### Phase 1a: Train Standard AlphaZero
        #####

        println("\n" * "=" ^ 80)
        println("PHASE 1a: Training Standard AlphaZero (hidden stochasticity)")
        println("=" ^ 80)

        exp_standard = BackgammonDeterministic.Training.experiment
        # Override MCTS iterations for training
        base_mcts = exp_standard.params.self_play.mcts
        new_mcts = MctsParams(base_mcts; num_iters_per_turn=TRAINING_MCTS_ITERS)
        new_self_play = SelfPlayParams(exp_standard.params.self_play; mcts=new_mcts)
        new_params = Params(exp_standard.params; num_iters=1000, self_play=new_self_play)
        exp_standard = Experiment(
            exp_standard.name,
            exp_standard.gspec,
            new_params,
            exp_standard.mknet,
            exp_standard.netparams,
            exp_standard.benchmark)

        session_standard = Session(exp_standard; dir="sessions/backgammon-comparison-standard")

        @printf("\nTraining for %.1f hours (%.0f seconds)...\n", TRAINING_HOURS, TRAINING_TIME_SECONDS)
        time_standard, iters_standard = timed_training!(session_standard, TRAINING_TIME_SECONDS)

        results.standard_training_time = time_standard
        results.standard_iterations = iters_standard

        println("\n" * "-" ^ 40)
        @printf("Standard AlphaZero completed:\n")
        @printf("  Time: %.1f seconds (%.2f hours)\n", time_standard, time_standard/3600)
        @printf("  Iterations: %d\n", iters_standard)

        #####
        ##### Phase 1b: Train Stochastic AlphaZero
        #####

        println("\n" * "=" ^ 80)
        println("PHASE 1b: Training Stochastic AlphaZero (progressive widening)")
        println("=" ^ 80)

        exp_stochastic = Backgammon.Training.experiment
        # Override MCTS iterations for training
        base_mcts_sto = exp_stochastic.params.self_play.mcts
        new_mcts_sto = MctsParams(base_mcts_sto; num_iters_per_turn=TRAINING_MCTS_ITERS)
        new_self_play_sto = SelfPlayParams(exp_stochastic.params.self_play; mcts=new_mcts_sto)
        new_params = Params(exp_stochastic.params; num_iters=1000, self_play=new_self_play_sto)
        exp_stochastic = Experiment(
            exp_stochastic.name,
            exp_stochastic.gspec,
            new_params,
            exp_stochastic.mknet,
            exp_stochastic.netparams,
            exp_stochastic.benchmark)

        session_stochastic = Session(exp_stochastic; dir="sessions/backgammon-comparison-stochastic")

        @printf("\nTraining for %.1f hours (%.0f seconds)...\n", TRAINING_HOURS, TRAINING_TIME_SECONDS)
        time_stochastic, iters_stochastic = timed_training!(session_stochastic, TRAINING_TIME_SECONDS)

        results.stochastic_training_time = time_stochastic
        results.stochastic_iterations = iters_stochastic

        println("\n" * "-" ^ 40)
        @printf("Stochastic AlphaZero completed:\n")
        @printf("  Time: %.1f seconds (%.2f hours)\n", time_stochastic, time_stochastic/3600)
        @printf("  Iterations: %d\n", iters_stochastic)
    end

    #####
    ##### Phase 2: Evaluation vs Random
    #####

    println("\n" * "=" ^ 80)
    println("PHASE 2: Evaluation vs Random Player")
    println("=" ^ 80)

    n_eval = EVAL_GAMES_PER_MATCHUP

    # Standard vs Random
    println("\n### Standard AlphaZero vs Random ###")
    std_wins, std_losses, std_draws, std_avg, _ = evaluate_vs_baseline(
        BackgammonDeterministic.Training.experiment.gspec,
        session_standard.env.bestnn,
        n_eval,
        BackgammonDeterministic.RandomPlayer();
        name="Random"
    )
    std_total = n_eval
    std_win_rate = std_wins / std_total

    results.std_vs_random = (
        wins=std_wins, losses=std_losses, draws=std_draws,
        avg_reward=std_avg, win_rate=std_win_rate, total=std_total
    )

    @printf("\n  Results: %d wins, %d losses, %d draws\n", std_wins, std_losses, std_draws)
    @printf("  Win rate: %.1f%% (avg reward: %.3f)\n", std_win_rate * 100, std_avg)
    ci = confidence_interval(std_win_rate, std_total)
    @printf("  95%% CI: [%.1f%%, %.1f%%]\n", ci[1]*100, ci[2]*100)

    # Stochastic vs Random
    println("\n### Stochastic AlphaZero vs Random ###")
    sto_wins, sto_losses, sto_draws, sto_avg, _ = evaluate_vs_baseline(
        Backgammon.Training.experiment.gspec,
        session_stochastic.env.bestnn,
        n_eval,
        Backgammon.RandomPlayer();
        name="Random"
    )
    sto_total = n_eval
    sto_win_rate = sto_wins / sto_total

    results.sto_vs_random = (
        wins=sto_wins, losses=sto_losses, draws=sto_draws,
        avg_reward=sto_avg, win_rate=sto_win_rate, total=sto_total
    )

    @printf("\n  Results: %d wins, %d losses, %d draws\n", sto_wins, sto_losses, sto_draws)
    @printf("  Win rate: %.1f%% (avg reward: %.3f)\n", sto_win_rate * 100, sto_avg)
    ci = confidence_interval(sto_win_rate, sto_total)
    @printf("  95%% CI: [%.1f%%, %.1f%%]\n", ci[1]*100, ci[2]*100)

    # Statistical comparison
    z_random = two_proportion_z_test(std_win_rate, std_total, sto_win_rate, sto_total)
    p_random = p_value_from_z(z_random)
    println("\n### Comparison (Standard vs Stochastic against Random) ###")
    @printf("  Z-score: %.2f, p-value: %.4f %s\n", z_random, p_random, significance_string(p_random))

    # Log to wandb
    if USE_WANDB
        wandb_log(Dict(
            "vs_random/standard_win_rate" => std_win_rate,
            "vs_random/standard_avg_reward" => std_avg,
            "vs_random/stochastic_win_rate" => sto_win_rate,
            "vs_random/stochastic_avg_reward" => sto_avg,
            "vs_random/z_score" => z_random,
            "vs_random/p_value" => p_random,
        ))
    end

    #####
    ##### Phase 3: Evaluation vs GnuBG
    #####

    # Initialize GnuBG results (may be skipped if wandb is enabled)
    z_gnubg = 0.0
    p_gnubg = 1.0

    if GNUBG_AVAILABLE
        println("\n" * "=" ^ 80)
        println("PHASE 3: Evaluation vs GnuBG (0-ply neural net)")
        println("NOTE: Using fast PyCall interface (~48k evals/sec)")
        println("=" ^ 80)

        gnubg_player = GnubgBaseline()  # 0-ply = neural net only
        n_gnubg = GNUBG_EVAL_GAMES

        # Standard vs GnuBG
        println("\n### Standard AlphaZero vs GnuBG ###")
        std_gnubg_wins, std_gnubg_losses, std_gnubg_draws, std_gnubg_avg, _ = evaluate_vs_baseline(
            BackgammonDeterministic.Training.experiment.gspec,
            session_standard.env.bestnn,
            n_gnubg,
            gnubg_player;
            name="GnuBG"
        )
        std_gnubg_total = n_gnubg
        std_gnubg_win_rate = std_gnubg_wins / std_gnubg_total

        results.std_vs_gnubg = (
            wins=std_gnubg_wins, losses=std_gnubg_losses, draws=std_gnubg_draws,
            avg_reward=std_gnubg_avg, win_rate=std_gnubg_win_rate, total=std_gnubg_total
        )

        @printf("\n  Results: %d wins, %d losses, %d draws\n",
                std_gnubg_wins, std_gnubg_losses, std_gnubg_draws)
        @printf("  Win rate: %.1f%% (avg reward: %.3f)\n", std_gnubg_win_rate * 100, std_gnubg_avg)
        ci = confidence_interval(std_gnubg_win_rate, std_gnubg_total)
        @printf("  95%% CI: [%.1f%%, %.1f%%]\n", ci[1]*100, ci[2]*100)

        # Stochastic vs GnuBG
        println("\n### Stochastic AlphaZero vs GnuBG ###")
        sto_gnubg_wins, sto_gnubg_losses, sto_gnubg_draws, sto_gnubg_avg, _ = evaluate_vs_baseline(
            Backgammon.Training.experiment.gspec,
            session_stochastic.env.bestnn,
            n_gnubg,
            gnubg_player;
            name="GnuBG"
        )
        sto_gnubg_total = n_gnubg
        sto_gnubg_win_rate = sto_gnubg_wins / sto_gnubg_total

        results.sto_vs_gnubg = (
            wins=sto_gnubg_wins, losses=sto_gnubg_losses, draws=sto_gnubg_draws,
            avg_reward=sto_gnubg_avg, win_rate=sto_gnubg_win_rate, total=sto_gnubg_total
        )

        @printf("\n  Results: %d wins, %d losses, %d draws\n",
                sto_gnubg_wins, sto_gnubg_losses, sto_gnubg_draws)
        @printf("  Win rate: %.1f%% (avg reward: %.3f)\n", sto_gnubg_win_rate * 100, sto_gnubg_avg)
        ci = confidence_interval(sto_gnubg_win_rate, sto_gnubg_total)
        @printf("  95%% CI: [%.1f%%, %.1f%%]\n", ci[1]*100, ci[2]*100)

        # Statistical comparison
        z_gnubg = two_proportion_z_test(std_gnubg_win_rate, std_gnubg_total,
                                         sto_gnubg_win_rate, sto_gnubg_total)
        p_gnubg = p_value_from_z(z_gnubg)
        println("\n### Comparison (Standard vs Stochastic against GnuBG) ###")
        @printf("  Z-score: %.2f, p-value: %.4f %s\n", z_gnubg, p_gnubg, significance_string(p_gnubg))
    else
        println("\n" * "=" ^ 80)
        println("PHASE 3: Evaluation vs GnuBG - SKIPPED")
        println("(GnuBG not available - PyCall conflicts with wandb/PythonCall)")
        println("=" ^ 80)
    end

    # Log to wandb (only if both wandb and gnubg results are available)
    if USE_WANDB && GNUBG_AVAILABLE
        wandb_log(Dict(
            "vs_gnubg/standard_win_rate" => std_gnubg_win_rate,
            "vs_gnubg/standard_avg_reward" => std_gnubg_avg,
            "vs_gnubg/stochastic_win_rate" => sto_gnubg_win_rate,
            "vs_gnubg/stochastic_avg_reward" => sto_gnubg_avg,
            "vs_gnubg/z_score" => z_gnubg,
            "vs_gnubg/p_value" => p_gnubg,
        ))
    end

    #####
    ##### Phase 4: Head-to-Head
    #####

    println("\n" * "=" ^ 80)
    println("PHASE 4: Head-to-Head (Standard vs Stochastic)")
    println("=" ^ 80)

    # Use deterministic game spec for fair comparison
    h2h_gspec = BackgammonDeterministic.Training.experiment.gspec

    println("\nStandard AlphaZero vs Stochastic AlphaZero")
    h2h_std_wins, h2h_sto_wins, h2h_draws, h2h_std_avg, _ = head_to_head(
        h2h_gspec,
        session_standard.env.bestnn,
        session_stochastic.env.bestnn,
        n_eval;
        name1="Standard", name2="Stochastic"
    )
    h2h_total = n_eval
    h2h_decisive = h2h_std_wins + h2h_sto_wins

    h2h_std_win_rate = h2h_std_wins / h2h_total
    h2h_sto_win_rate = h2h_sto_wins / h2h_total

    # Z-test for head-to-head (against 50% null hypothesis)
    if h2h_decisive > 0
        p_std = h2h_std_wins / h2h_decisive
        z_h2h = (p_std - 0.5) / sqrt(0.25 / h2h_decisive)
        p_h2h = p_value_from_z(z_h2h)
    else
        z_h2h = 0.0
        p_h2h = 1.0
    end

    results.head_to_head = (
        std_wins=h2h_std_wins, sto_wins=h2h_sto_wins, draws=h2h_draws,
        std_win_rate=h2h_std_win_rate, sto_win_rate=h2h_sto_win_rate,
        z_score=z_h2h, p_value=p_h2h,
        significance=significance_string(p_h2h)
    )

    @printf("\n  Standard wins:   %d (%.1f%%)\n", h2h_std_wins, h2h_std_win_rate * 100)
    @printf("  Stochastic wins: %d (%.1f%%)\n", h2h_sto_wins, h2h_sto_win_rate * 100)
    @printf("  Draws:           %d\n", h2h_draws)
    @printf("  Standard avg reward: %.3f\n", h2h_std_avg)
    @printf("  Z-score: %.2f, p-value: %.4f %s\n", z_h2h, p_h2h, significance_string(p_h2h))

    if p_h2h < 0.05
        winner = h2h_std_wins > h2h_sto_wins ? "Standard AlphaZero" : "Stochastic AlphaZero"
        println("\n  >>> $winner is SIGNIFICANTLY BETTER (p < 0.05) <<<")
    else
        println("\n  >>> No statistically significant difference <<<")
    end

    # Log to wandb
    if USE_WANDB
        wandb_log(Dict(
            "head_to_head/standard_wins" => h2h_std_wins,
            "head_to_head/stochastic_wins" => h2h_sto_wins,
            "head_to_head/draws" => h2h_draws,
            "head_to_head/standard_win_rate" => h2h_std_win_rate,
            "head_to_head/stochastic_win_rate" => h2h_sto_win_rate,
            "head_to_head/z_score" => z_h2h,
            "head_to_head/p_value" => p_h2h,
        ))
    end

    #####
    ##### Final Summary
    #####

    end_time = now()
    total_duration = Dates.value(end_time - start_time) / 1000 / 3600  # hours

    println("\n" * "=" ^ 80)
    println("FINAL SUMMARY")
    println("=" ^ 80)

    println("\n### Experiment Duration ###")
    @printf("  Total time: %.2f hours\n", total_duration)
    @printf("  Training: %.2f hours\n", (results.standard_training_time + results.stochastic_training_time) / 3600)
    @printf("  Evaluation: %.2f hours\n", total_duration - (results.standard_training_time + results.stochastic_training_time) / 3600)

    println("\n### Key Results ###")
    println("-" ^ 60)
    @printf("%-35s %12s %12s\n", "Metric", "Standard", "Stochastic")
    println("-" ^ 60)
    @printf("%-35s %12d %12d\n", "Training iterations:",
            results.standard_iterations, results.stochastic_iterations)
    @printf("%-35s %11.1f%% %11.1f%%\n", "Win rate vs Random:",
            results.std_vs_random.win_rate * 100, results.sto_vs_random.win_rate * 100)
    if GNUBG_AVAILABLE
        @printf("%-35s %11.1f%% %11.1f%%\n", "Win rate vs GnuBG:",
                results.std_vs_gnubg.win_rate * 100, results.sto_vs_gnubg.win_rate * 100)
    else
        @printf("%-35s %12s %12s\n", "Win rate vs GnuBG:", "N/A", "N/A")
    end
    @printf("%-35s %11.1f%% %11.1f%%\n", "Head-to-head win rate:",
            results.head_to_head.std_win_rate * 100, results.head_to_head.sto_win_rate * 100)
    println("-" ^ 60)

    println("\n### Statistical Conclusions ###")

    # vs Random comparison
    if p_random < 0.05
        better_random = std_win_rate > sto_win_rate ? "Standard" : "Stochastic"
        worse_random = std_win_rate > sto_win_rate ? "Stochastic" : "Standard"
        @printf("  vs Random: %s significantly better than %s (p=%.4f)\n",
                better_random, worse_random, p_random)
    else
        println("  vs Random: No significant difference between approaches")
    end

    # vs GnuBG comparison
    if GNUBG_AVAILABLE
        if p_gnubg < 0.05
            better_gnubg = results.std_vs_gnubg.win_rate > results.sto_vs_gnubg.win_rate ? "Standard" : "Stochastic"
            worse_gnubg = results.std_vs_gnubg.win_rate > results.sto_vs_gnubg.win_rate ? "Stochastic" : "Standard"
            @printf("  vs GnuBG: %s significantly better than %s (p=%.4f)\n",
                    better_gnubg, worse_gnubg, p_gnubg)
        else
            println("  vs GnuBG: No significant difference between approaches")
        end
    else
        println("  vs GnuBG: Skipped (not available with wandb enabled)")
    end

    # Head-to-head
    if p_h2h < 0.05
        h2h_winner = h2h_std_wins > h2h_sto_wins ? "Standard" : "Stochastic"
        @printf("  Head-to-head: %s is significantly stronger (p=%.4f)\n", h2h_winner, p_h2h)
    else
        println("  Head-to-head: No significant difference")
    end

    # Save results
    results_file = "results/backgammon_eval_$(Dates.format(start_time, "yyyymmdd_HHMMSS")).txt"
    mkpath(dirname(results_file))
    save_results(results, results_file)

    # Final wandb logging and cleanup
    if USE_WANDB
        wandb_log(Dict(
            "summary/total_duration_hours" => total_duration,
            "summary/standard_iterations" => results.standard_iterations,
            "summary/stochastic_iterations" => results.stochastic_iterations,
            "summary/standard_training_hours" => results.standard_training_time / 3600,
            "summary/stochastic_training_hours" => results.stochastic_training_time / 3600,
        ))
        wandb_finish()
        println("WandB run finished")
    end

    println("\n" * "=" ^ 80)
    println("Evaluation complete!")
    println("Results saved to: $results_file")
    println("=" ^ 80)

    return results, session_standard, session_stochastic
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
