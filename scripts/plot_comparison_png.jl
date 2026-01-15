#!/usr/bin/env julia
# Plot comparison of Standard vs Stochastic AlphaZero training - PNG output

using Plots
gr()  # Use GR backend for file output

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
        m = match(r"Starting iteration (\d+)", line)
        if m !== nothing
            current_iter = parse(Int, m.captures[1])
            continue
        end

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

        if occursin("Optimizing the loss", line)
            in_optimization = true
            opt_line_count = 0
            continue
        end

        if in_optimization
            if occursin("Loss", line) && occursin("Lv", line)
                continue
            end
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

# Create output directory
outdir = "sessions/comparison_plots"
mkpath(outdir)

# Convert reward to win rate
reward_to_winrate(r) = (r + 1) / 2 * 100

# Prepare data
std_winrate = reward_to_winrate.(standard.az_vs_random)
sto_winrate = reward_to_winrate.(stochastic.az_vs_random)
std_def = reward_to_winrate.(standard.random_vs_az)
sto_def = reward_to_winrate.(stochastic.random_vs_az)
std_combined = (std_winrate .+ std_def[1:length(std_winrate)]) ./ 2
sto_combined = (sto_winrate .+ sto_def[1:length(sto_winrate)]) ./ 2

# Plot 1: AZ vs Random win rate
p1 = plot(0:length(std_winrate)-1, std_winrate,
    label="Standard AlphaZero",
    title="AlphaZero vs Random (Win Rate %)",
    xlabel="Iteration",
    ylabel="Win Rate (%)",
    linewidth=2,
    marker=:circle,
    markersize=4,
    legend=:bottomright,
    size=(800, 500))
plot!(p1, 0:length(sto_winrate)-1, sto_winrate,
    label="Stochastic AlphaZero",
    linewidth=2,
    marker=:diamond,
    markersize=4)
savefig(p1, joinpath(outdir, "01_az_vs_random.png"))
println("Saved: $(joinpath(outdir, "01_az_vs_random.png"))")

# Plot 2: Defense (Random vs AZ)
p2 = plot(0:length(std_def)-1, std_def,
    label="Standard AlphaZero",
    title="Defense: Random vs AlphaZero (AZ Win Rate %)",
    xlabel="Iteration",
    ylabel="Win Rate (%)",
    linewidth=2,
    marker=:circle,
    markersize=4,
    legend=:bottomright,
    size=(800, 500))
plot!(p2, 0:length(sto_def)-1, sto_def,
    label="Stochastic AlphaZero",
    linewidth=2,
    marker=:diamond,
    markersize=4)
savefig(p2, joinpath(outdir, "02_defense_random_vs_az.png"))
println("Saved: $(joinpath(outdir, "02_defense_random_vs_az.png"))")

# Plot 3: Combined win rate
p3 = plot(0:length(std_combined)-1, std_combined,
    label="Standard AlphaZero",
    title="Combined Win Rate vs Random (%)",
    xlabel="Iteration",
    ylabel="Win Rate (%)",
    linewidth=2,
    marker=:circle,
    markersize=4,
    legend=:bottomright,
    size=(800, 500))
plot!(p3, 0:length(sto_combined)-1, sto_combined,
    label="Stochastic AlphaZero",
    linewidth=2,
    marker=:diamond,
    markersize=4)
savefig(p3, joinpath(outdir, "03_combined_winrate.png"))
println("Saved: $(joinpath(outdir, "03_combined_winrate.png"))")

# Plot 4: Total Loss
p4 = plot(1:length(standard.loss_after), standard.loss_after,
    label="Standard AlphaZero",
    title="Total Loss (After Optimization)",
    xlabel="Iteration",
    ylabel="Loss",
    linewidth=2,
    marker=:circle,
    markersize=4,
    legend=:topright,
    size=(800, 500))
plot!(p4, 1:length(stochastic.loss_after), stochastic.loss_after,
    label="Stochastic AlphaZero",
    linewidth=2,
    marker=:diamond,
    markersize=4)
savefig(p4, joinpath(outdir, "04_total_loss.png"))
println("Saved: $(joinpath(outdir, "04_total_loss.png"))")

# Plot 5: Value Loss
p5 = plot(1:length(standard.value_loss_after), standard.value_loss_after,
    label="Standard AlphaZero",
    title="Value Loss (Lv) - After Optimization",
    xlabel="Iteration",
    ylabel="Loss",
    linewidth=2,
    marker=:circle,
    markersize=4,
    legend=:topright,
    size=(800, 500))
plot!(p5, 1:length(stochastic.value_loss_after), stochastic.value_loss_after,
    label="Stochastic AlphaZero",
    linewidth=2,
    marker=:diamond,
    markersize=4)
savefig(p5, joinpath(outdir, "05_value_loss.png"))
println("Saved: $(joinpath(outdir, "05_value_loss.png"))")

# Plot 6: Policy Loss
p6 = plot(1:length(standard.policy_loss_after), standard.policy_loss_after,
    label="Standard AlphaZero",
    title="Policy Loss (Lp) - After Optimization",
    xlabel="Iteration",
    ylabel="Loss",
    linewidth=2,
    marker=:circle,
    markersize=4,
    legend=:bottomright,
    size=(800, 500))
plot!(p6, 1:length(stochastic.policy_loss_after), stochastic.policy_loss_after,
    label="Stochastic AlphaZero",
    linewidth=2,
    marker=:diamond,
    markersize=4)
savefig(p6, joinpath(outdir, "06_policy_loss.png"))
println("Saved: $(joinpath(outdir, "06_policy_loss.png"))")

# Plot 7: Combined summary (2x2 grid)
p_summary = plot(p1, p3, p4, p5,
    layout=(2,2),
    size=(1200, 900),
    title="Standard vs Stochastic AlphaZero Comparison")
savefig(p_summary, joinpath(outdir, "00_summary.png"))
println("Saved: $(joinpath(outdir, "00_summary.png"))")

println("\n" * "="^60)
println("All plots saved to: $outdir")
println("="^60)

# Print summary stats
println("\nSUMMARY STATISTICS")
println("-"^40)
println("\nStandard AlphaZero:")
println("  Iterations: $(length(standard.loss_after))")
println("  Final loss: $(round(standard.loss_after[end], digits=4))")
println("  Final value loss: $(round(standard.value_loss_after[end], digits=4))")
println("  Final policy loss: $(round(standard.policy_loss_after[end], digits=4))")
println("  Best combined win rate: $(round(maximum(std_combined), digits=1))%")

println("\nStochastic AlphaZero:")
println("  Iterations: $(length(stochastic.loss_after))")
println("  Final loss: $(round(stochastic.loss_after[end], digits=4))")
println("  Final value loss: $(round(stochastic.value_loss_after[end], digits=4))")
println("  Final policy loss: $(round(stochastic.policy_loss_after[end], digits=4))")
println("  Best combined win rate: $(round(maximum(sto_combined), digits=1))%")
