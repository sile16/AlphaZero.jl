#!/usr/bin/env julia
# Plot comparison of Standard vs Stochastic AlphaZero training

using UnicodePlots

function parse_log(logfile)
    iterations = Int[]
    az_vs_random = Float64[]
    random_vs_az = Float64[]
    loss_before = Float64[]
    loss_after = Float64[]
    value_loss_before = Float64[]
    value_loss_after = Float64[]
    policy_loss_before = Float64[]
    policy_loss_after = Float64[]

    lines = readlines(logfile)
    current_iter = 0
    in_optimization = false
    opt_line_count = 0

    for (i, line) in enumerate(lines)
        # Parse iteration number
        m = match(r"Starting iteration (\d+)", line)
        if m !== nothing
            current_iter = parse(Int, m.captures[1])
            continue
        end

        # Parse benchmark results
        if occursin("AlphaZero against Random", line) && i + 2 <= length(lines)
            m = match(r"Average reward: ([+-]?\d+\.?\d*)", lines[i+2])
            if m !== nothing
                push!(iterations, current_iter)
                push!(az_vs_random, parse(Float64, m.captures[1]))
            end
        end

        if occursin("Random against AlphaZero", line) && i + 2 <= length(lines)
            m = match(r"Average reward: ([+-]?\d+\.?\d*)", lines[i+2])
            if m !== nothing
                push!(random_vs_az, parse(Float64, m.captures[1]))
            end
        end

        # Parse optimization loss
        if occursin("Optimizing the loss", line)
            in_optimization = true
            opt_line_count = 0
            continue
        end

        if in_optimization
            # Skip header line
            if occursin("Loss", line) && occursin("Lv", line)
                continue
            end
            # Parse loss values
            parts = split(strip(line))
            if length(parts) >= 3
                try
                    loss = parse(Float64, parts[1])
                    lv = parse(Float64, parts[2])
                    lp = parse(Float64, parts[3])
                    opt_line_count += 1
                    if opt_line_count == 1
                        push!(loss_before, loss)
                        push!(value_loss_before, lv)
                        push!(policy_loss_before, lp)
                    elseif opt_line_count == 2
                        push!(loss_after, loss)
                        push!(value_loss_after, lv)
                        push!(policy_loss_after, lp)
                        in_optimization = false
                    end
                catch
                    in_optimization = false
                end
            end
        end
    end

    return (
        iterations = iterations,
        az_vs_random = az_vs_random,
        random_vs_az = random_vs_az,
        loss_before = loss_before,
        loss_after = loss_after,
        value_loss_before = value_loss_before,
        value_loss_after = value_loss_after,
        policy_loss_before = policy_loss_before,
        policy_loss_after = policy_loss_after
    )
end

# Parse both logs
println("Parsing training logs...")
standard = parse_log("sessions/backgammon-comparison-standard/log.txt")
stochastic = parse_log("sessions/backgammon-comparison-stochastic/log.txt")

# Convert reward to win rate
reward_to_winrate(r) = (r + 1) / 2 * 100

println("\n" * "="^70)
println("EVALUATION VS RANDOM (Win Rate %)")
println("="^70)

# Plot AZ vs Random win rate
std_winrate = reward_to_winrate.(standard.az_vs_random)
sto_winrate = reward_to_winrate.(stochastic.az_vs_random)

plt = lineplot(0:length(std_winrate)-1, std_winrate,
    name = "Standard",
    title = "AlphaZero vs Random (Win Rate %)",
    xlabel = "Iteration",
    ylabel = "Win %",
    ylim = (0, 50),
    color = :blue,
    width = 60,
    height = 15)
lineplot!(plt, 0:length(sto_winrate)-1, sto_winrate, name = "Stochastic", color = :red)
println(plt)

# Plot Random vs AZ (defense)
std_def = reward_to_winrate.(standard.random_vs_az)
sto_def = reward_to_winrate.(stochastic.random_vs_az)

plt2 = lineplot(0:length(std_def)-1, std_def,
    name = "Standard",
    title = "Defense: Random vs AlphaZero (AZ Win Rate %)",
    xlabel = "Iteration",
    ylabel = "Win %",
    ylim = (0, 60),
    color = :blue,
    width = 60,
    height = 15)
lineplot!(plt2, 0:length(sto_def)-1, sto_def, name = "Stochastic", color = :red)
println(plt2)

# Combined win rate (average of both positions)
std_combined = (std_winrate .+ std_def[1:length(std_winrate)]) ./ 2
sto_combined = (sto_winrate .+ sto_def[1:length(sto_winrate)]) ./ 2

plt3 = lineplot(0:length(std_combined)-1, std_combined,
    name = "Standard",
    title = "Combined Win Rate vs Random (%)",
    xlabel = "Iteration",
    ylabel = "Win %",
    ylim = (0, 50),
    color = :blue,
    width = 60,
    height = 15)
lineplot!(plt3, 0:length(sto_combined)-1, sto_combined, name = "Stochastic", color = :red)
println(plt3)

println("\n" * "="^70)
println("TRAINING LOSS CURVES")
println("="^70)

# Total loss after optimization
plt4 = lineplot(1:length(standard.loss_after), standard.loss_after,
    name = "Standard",
    title = "Total Loss (after optimization)",
    xlabel = "Iteration",
    ylabel = "Loss",
    color = :blue,
    width = 60,
    height = 15)
lineplot!(plt4, 1:length(stochastic.loss_after), stochastic.loss_after, name = "Stochastic", color = :red)
println(plt4)

# Value loss
plt5 = lineplot(1:length(standard.value_loss_after), standard.value_loss_after,
    name = "Standard",
    title = "Value Loss (Lv) - after optimization",
    xlabel = "Iteration",
    ylabel = "Loss",
    color = :blue,
    width = 60,
    height = 15)
lineplot!(plt5, 1:length(stochastic.value_loss_after), stochastic.value_loss_after, name = "Stochastic", color = :red)
println(plt5)

# Policy loss
plt6 = lineplot(1:length(standard.policy_loss_after), standard.policy_loss_after,
    name = "Standard",
    title = "Policy Loss (Lp) - after optimization",
    xlabel = "Iteration",
    ylabel = "Loss",
    color = :blue,
    width = 60,
    height = 15)
lineplot!(plt6, 1:length(stochastic.policy_loss_after), stochastic.policy_loss_after, name = "Stochastic", color = :red)
println(plt6)

println("\n" * "="^70)
println("SUMMARY STATISTICS")
println("="^70)

println("\nStandard AlphaZero:")
println("  Iterations completed: $(length(standard.loss_after))")
println("  Final loss: $(round(standard.loss_after[end], digits=4))")
println("  Final value loss: $(round(standard.value_loss_after[end], digits=4))")
println("  Final policy loss: $(round(standard.policy_loss_after[end], digits=4))")
println("  Best win rate vs Random: $(round(maximum(std_combined), digits=1))%")

println("\nStochastic AlphaZero:")
println("  Iterations completed: $(length(stochastic.loss_after))")
println("  Final loss: $(round(stochastic.loss_after[end], digits=4))")
println("  Final value loss: $(round(stochastic.value_loss_after[end], digits=4))")
println("  Final policy loss: $(round(stochastic.policy_loss_after[end], digits=4))")
println("  Best win rate vs Random: $(round(maximum(sto_combined), digits=1))%")
