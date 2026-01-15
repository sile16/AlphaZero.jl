#####
##### Backgammon Comparison: Standard vs Stochastic AlphaZero
##### Fair wall-clock time comparison
#####
##### Run with: julia --project scripts/backgammon_comparison.jl
#####

using AlphaZero
using AlphaZero: MctsPlayer, MctsParams, SimParams, TwoPlayers, play_game, Network, MCTS
using AlphaZero: ConstSchedule
using Printf
using Statistics

# Load both Backgammon game variants
include(joinpath(@__DIR__, "..", "games", "backgammon", "main.jl"))
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "main.jl"))

using .Backgammon
using .BackgammonDeterministic

#####
##### Configuration
#####

# Training time limit per approach (in seconds)
const TRAINING_TIME_SECONDS = 1800  # 30 minutes each

# Number of training iterations (will stop early if time limit reached)
const MAX_ITERS = 100

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
    # Using polynomial approximation for normal CDF
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
    z = confidence == 0.95 ? 1.96 : (confidence == 0.99 ? 2.576 : 1.645)
    se = standard_error(p, n)
    return (max(0, p - z * se), min(1, p + z * se))
end

#####
##### Timed Training
#####

import AlphaZero: self_play_step!, learning_step!, memory_report, resize_memory!
import AlphaZero: Handlers, Report
import AlphaZero.UserInterface: zeroth_iteration!, missing_zeroth_iteration

"""
Run training for a fixed wall-clock time, return elapsed time and iteration count.
"""
function timed_training!(session, time_limit_seconds)
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

    while (time() - start_time) < time_limit_seconds
        elapsed = time() - start_time
        remaining = time_limit_seconds - elapsed
        if remaining < 30  # Need at least 30s for an iteration
            break
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
        @printf("  Iteration %d completed (%.1fs elapsed)\n", iters_completed, time() - start_time)
    end

    Handlers.training_finished(handler)
    elapsed_time = time() - start_time
    return elapsed_time, iters_completed
end

#####
##### Evaluation Functions
#####

"""
Evaluate a trained network against Random player.
Returns (wins, losses, draws, avg_reward).
"""
function evaluate_vs_random(gspec, network, n_games, random_player; mcts_params=nothing, use_gpu=true)
    if isnothing(mcts_params)
        mcts_params = MctsParams(
            num_iters_per_turn=400,
            cpuct=1.0,
            temperature=ConstSchedule(0.2),
            dirichlet_noise_ϵ=0.0,
            dirichlet_noise_α=1.0
        )
    end

    # Create MCTS player
    net = Network.copy(network, on_gpu=use_gpu, test_mode=true)

    # Play games: AlphaZero as white
    println("  Playing $n_games games (AlphaZero first)...")
    player_white = MctsPlayer(gspec, net, mcts_params)

    rewards_white = Float64[]
    for i in 1:n_games
        MCTS.reset!(player_white.mcts)
        trace = play_game(gspec, TwoPlayers(player_white, random_player))
        push!(rewards_white, trace.reward)
        if i % 50 == 0
            @printf("    Game %d/%d\n", i, n_games)
        end
    end

    # Play games: AlphaZero as black
    println("  Playing $n_games games (AlphaZero second)...")
    player_black = MctsPlayer(gspec, net, mcts_params)

    rewards_black = Float64[]
    for i in 1:n_games
        MCTS.reset!(player_black.mcts)
        trace = play_game(gspec, TwoPlayers(random_player, player_black))
        push!(rewards_black, -trace.reward)  # Negate since we're black
        if i % 50 == 0
            @printf("    Game %d/%d\n", i, n_games)
        end
    end

    all_rewards = vcat(rewards_white, rewards_black)
    wins = count(r -> r > 0, all_rewards)
    losses = count(r -> r < 0, all_rewards)
    draws = count(r -> r == 0, all_rewards)
    avg_reward = mean(all_rewards)

    return wins, losses, draws, avg_reward
end

"""
Head-to-head evaluation between two networks.
Uses deterministic game spec so both networks see standard states (no chance nodes).
Returns (net1_wins, net2_wins, draws, net1_avg_reward).
"""
function head_to_head(gspec, net1, net2, n_games; mcts_params=nothing, use_gpu=true)
    if isnothing(mcts_params)
        mcts_params = MctsParams(
            num_iters_per_turn=400,
            cpuct=1.0,
            temperature=ConstSchedule(0.2),
            dirichlet_noise_ϵ=0.0,
            dirichlet_noise_α=1.0
        )
    end

    # Create network copies
    network1 = Network.copy(net1, on_gpu=use_gpu, test_mode=true)
    network2 = Network.copy(net2, on_gpu=use_gpu, test_mode=true)

    # Play games: net1 as white
    println("  Playing $n_games games (Model 1 first)...")
    player1_white = MctsPlayer(gspec, network1, mcts_params)
    player2_black = MctsPlayer(gspec, network2, mcts_params)

    rewards_as_white = Float64[]
    for i in 1:n_games
        MCTS.reset!(player1_white.mcts)
        MCTS.reset!(player2_black.mcts)
        trace = play_game(gspec, TwoPlayers(player1_white, player2_black))
        push!(rewards_as_white, trace.reward)
        if i % 50 == 0
            @printf("    Game %d/%d\n", i, n_games)
        end
    end

    # Play games: net1 as black
    println("  Playing $n_games games (Model 1 second)...")
    player2_white = MctsPlayer(gspec, network2, mcts_params)
    player1_black = MctsPlayer(gspec, network1, mcts_params)

    rewards_as_black = Float64[]
    for i in 1:n_games
        MCTS.reset!(player2_white.mcts)
        MCTS.reset!(player1_black.mcts)
        trace = play_game(gspec, TwoPlayers(player2_white, player1_black))
        push!(rewards_as_black, -trace.reward)  # Negate since net1 is black
        if i % 50 == 0
            @printf("    Game %d/%d\n", i, n_games)
        end
    end

    all_rewards = vcat(rewards_as_white, rewards_as_black)
    net1_wins = count(r -> r > 0, all_rewards)
    net2_wins = count(r -> r < 0, all_rewards)
    draws = count(r -> r == 0, all_rewards)
    net1_avg_reward = mean(all_rewards)

    return net1_wins, net2_wins, draws, net1_avg_reward
end

#####
##### Main
#####

function main()
    println("=" ^ 70)
    println("Backgammon: Standard vs Stochastic MCTS Comparison")
    println("Configuration: short_game=true, doubles_only=true")
    println("Fair wall-clock time comparison: $(TRAINING_TIME_SECONDS ÷ 60) minutes each")
    println("=" ^ 70)

    #####
    ##### Phase 1: Train Standard AlphaZero (hidden stochasticity)
    #####

    println("\n" * "=" ^ 70)
    println("PHASE 1: Standard AlphaZero (hidden stochasticity in step!)")
    println("=" ^ 70)

    exp_standard = BackgammonDeterministic.Training.experiment

    # Override to use many iterations (will be time-limited)
    new_params = Params(exp_standard.params; num_iters=MAX_ITERS)
    exp_standard = Experiment(
        exp_standard.name,
        exp_standard.gspec,
        new_params,
        exp_standard.mknet,
        exp_standard.netparams,
        exp_standard.benchmark)

    session_standard = Session(exp_standard; dir="sessions/backgammon-comparison-standard")

    println("\nTraining for $(TRAINING_TIME_SECONDS) seconds...")
    time_standard, iters_standard = timed_training!(session_standard, TRAINING_TIME_SECONDS)

    println("\n" * "-" ^ 40)
    @printf("Standard AlphaZero completed:\n")
    @printf("  Time: %.1f seconds\n", time_standard)
    @printf("  Iterations: %d\n", iters_standard)
    @printf("  Avg time/iter: %.2f seconds\n", time_standard / max(iters_standard, 1))

    #####
    ##### Phase 2: Train Stochastic AlphaZero (explicit chance nodes)
    #####

    println("\n" * "=" ^ 70)
    println("PHASE 2: Stochastic AlphaZero (progressive widening for chance nodes)")
    println("=" ^ 70)

    exp_stochastic = Backgammon.Training.experiment

    new_params = Params(exp_stochastic.params; num_iters=MAX_ITERS)
    exp_stochastic = Experiment(
        exp_stochastic.name,
        exp_stochastic.gspec,
        new_params,
        exp_stochastic.mknet,
        exp_stochastic.netparams,
        exp_stochastic.benchmark)

    session_stochastic = Session(exp_stochastic; dir="sessions/backgammon-comparison-stochastic")

    println("\nTraining for $(TRAINING_TIME_SECONDS) seconds...")
    time_stochastic, iters_stochastic = timed_training!(session_stochastic, TRAINING_TIME_SECONDS)

    println("\n" * "-" ^ 40)
    @printf("Stochastic AlphaZero completed:\n")
    @printf("  Time: %.1f seconds\n", time_stochastic)
    @printf("  Iterations: %d\n", iters_stochastic)
    @printf("  Avg time/iter: %.2f seconds\n", time_stochastic / max(iters_stochastic, 1))

    #####
    ##### Phase 3: Evaluation against Random
    #####

    println("\n" * "=" ^ 70)
    println("PHASE 3: Evaluation against Random Player")
    println("=" ^ 70)

    n_eval_games = 200  # 200 as white + 200 as black = 400 total per model

    # Evaluate Standard model against Random
    println("\n### Standard AlphaZero vs Random ###")
    std_wins, std_losses, std_draws, std_avg = evaluate_vs_random(
        BackgammonDeterministic.Training.experiment.gspec,
        session_standard.env.bestnn,
        n_eval_games,
        BackgammonDeterministic.RandomPlayer()
    )
    std_total = 2 * n_eval_games
    std_win_rate = std_wins / std_total

    @printf("  Wins: %d, Losses: %d, Draws: %d\n", std_wins, std_losses, std_draws)
    @printf("  Win rate: %.1f%% (%.2f avg reward)\n", 100 * std_win_rate, std_avg)

    # Evaluate Stochastic model against Random
    println("\n### Stochastic AlphaZero vs Random ###")
    sto_wins, sto_losses, sto_draws, sto_avg = evaluate_vs_random(
        Backgammon.Training.experiment.gspec,
        session_stochastic.env.bestnn,
        n_eval_games,
        Backgammon.RandomPlayer()
    )
    sto_total = 2 * n_eval_games
    sto_win_rate = sto_wins / sto_total

    @printf("  Wins: %d, Losses: %d, Draws: %d\n", sto_wins, sto_losses, sto_draws)
    @printf("  Win rate: %.1f%% (%.2f avg reward)\n", 100 * sto_win_rate, sto_avg)

    #####
    ##### Phase 4: Head-to-Head Evaluation
    #####

    println("\n" * "=" ^ 70)
    println("PHASE 4: Head-to-Head Evaluation (Standard vs Stochastic)")
    println("=" ^ 70)

    n_h2h_games = 200  # 200 as white + 200 as black = 400 total

    # Use deterministic game spec for head-to-head (both networks trained on same state encoding)
    h2h_gspec = BackgammonDeterministic.Training.experiment.gspec

    println("\nStandard (Model 1) vs Stochastic (Model 2)")
    std_h2h_wins, sto_h2h_wins, h2h_draws, std_h2h_avg = head_to_head(
        h2h_gspec,
        session_standard.env.bestnn,
        session_stochastic.env.bestnn,
        n_h2h_games
    )
    h2h_total = 2 * n_h2h_games

    @printf("\n  Standard wins: %d (%.1f%%)\n", std_h2h_wins, 100 * std_h2h_wins / h2h_total)
    @printf("  Stochastic wins: %d (%.1f%%)\n", sto_h2h_wins, 100 * sto_h2h_wins / h2h_total)
    @printf("  Draws: %d\n", h2h_draws)
    @printf("  Standard avg reward: %.3f\n", std_h2h_avg)

    #####
    ##### Final Summary
    #####

    println("\n" * "=" ^ 70)
    println("FINAL RESULTS SUMMARY")
    println("=" ^ 70)

    println("\n### Training Summary ###")
    println("-" ^ 40)
    @printf("%-25s %15s %15s\n", "", "Standard", "Stochastic")
    @printf("%-25s %15.1f %15.1f\n", "Training time (s):", time_standard, time_stochastic)
    @printf("%-25s %15d %15d\n", "Iterations completed:", iters_standard, iters_stochastic)
    @printf("%-25s %15.2f %15.2f\n", "Seconds per iteration:",
            time_standard/max(iters_standard,1), time_stochastic/max(iters_stochastic,1))

    println("\n### Performance vs Random ###")
    println("-" ^ 40)
    @printf("%-25s %15s %15s\n", "", "Standard", "Stochastic")
    @printf("%-25s %15d %15d\n", "Wins:", std_wins, sto_wins)
    @printf("%-25s %15d %15d\n", "Losses:", std_losses, sto_losses)
    @printf("%-25s %14.1f%% %14.1f%%\n", "Win rate:", 100*std_win_rate, 100*sto_win_rate)
    @printf("%-25s %15.2f %15.2f\n", "Avg reward:", std_avg, sto_avg)

    println("\n### Head-to-Head Results ###")
    println("-" ^ 40)
    @printf("Standard vs Stochastic: %d - %d (%.1f%% - %.1f%%)\n",
            std_h2h_wins, sto_h2h_wins,
            100 * std_h2h_wins / h2h_total,
            100 * sto_h2h_wins / h2h_total)

    # Statistical significance for head-to-head
    if h2h_total > 0 && std_h2h_wins + sto_h2h_wins > 0
        p_std = std_h2h_wins / (std_h2h_wins + sto_h2h_wins)
        n_decisive = std_h2h_wins + sto_h2h_wins
        z = (p_std - 0.5) / sqrt(0.25 / n_decisive)
        p_val = p_value_from_z(z)
        @printf("Z-score: %.2f, p-value: %.4f\n", z, p_val)
        if p_val < 0.05
            winner = p_std > 0.5 ? "Standard" : "Stochastic"
            println("Result: $winner is significantly better (p < 0.05)")
        else
            println("Result: No statistically significant difference")
        end
    end

    #####
    ##### Statistical Analysis Notes
    #####

    println("\n" * "=" ^ 70)
    println("STATISTICAL NOTES")
    println("=" ^ 70)

    n_games = h2h_total

    println("\n### Interpretation guide ###")
    println("-" ^ 40)
    @printf("With n=%d games:\n", n_games)
    @printf("  Standard Error at p=0.50: %.2f%%\n", 100 * standard_error(0.5, n_games))
    @printf("  95%% CI: ±%.2f%%\n", 100 * 1.96 * standard_error(0.5, n_games))

    se_diff_50 = sqrt(2 * 0.5 * 0.5 / n_games)
    min_diff_95 = 1.96 * se_diff_50
    @printf("  Min detectable difference (p<0.05): %.1f%%\n", 100 * min_diff_95)

    println("\nResults saved to:")
    println("  Standard:   sessions/backgammon-comparison-standard/")
    println("  Stochastic: sessions/backgammon-comparison-stochastic/")

    return session_standard, session_stochastic
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
