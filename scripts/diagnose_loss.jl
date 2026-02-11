#!/usr/bin/env julia
"""Diagnose loss components across checkpoints."""

using Pkg; Pkg.activate(".")

ENV["BACKGAMMON_OBS_TYPE"] = "minimal"
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))

using AlphaZero
using AlphaZero.FluxLib
using AlphaZero.Network
using Statistics
using Random
using Printf

const GI2 = AlphaZero.GameInterface

session_dir = ARGS[1]
gspec = GameSpec()
num_actions = GI.num_actions(gspec)

# Generate random game positions by playing random games
println("Generating training-like data from random games...")
Random.seed!(42)
samples = []
for g in 1:200
    game = GI.init(gspec)
    states = []; is_chance_flags = Bool[]
    while !GI.game_terminated(game)
        if GI.is_chance_node(game)
            outcomes = GI.chance_outcomes(game)
            outcome = outcomes[rand(1:length(outcomes))][1]
            GI.apply_chance!(game, outcome)
            continue
        end
        push!(states, GI.current_state(game))
        push!(is_chance_flags, false)
        actions = GI.available_actions(game)
        GI.play!(game, rand(actions))
    end
    outcome = GI.game_outcome(game)
    final_reward = GI.white_reward(game)
    for i in 1:length(states)
        state = states[i]
        wp = GI.white_playing(gspec, state)
        z = wp ? final_reward : -final_reward
        state_vec = Vector{Float32}(vec(GI.vectorize_state(gspec, state)))

        # Random but valid policy (uniform over legal actions for decision nodes)
        full_policy = zeros(Float32, num_actions)
        if !is_chance_flags[i]
            game_tmp = GI.init(gspec, state)
            if !GI.game_terminated(game_tmp) && !GI.is_chance_node(game_tmp)
                legal = GI.available_actions(game_tmp)
                for a in legal
                    full_policy[a] = 1.0f0 / length(legal)
                end
            end
        end

        eq = zeros(Float32, 5)
        has_eq = false
        if !isnothing(outcome)
            has_eq = true
            won = outcome.white_won == wp
            if won
                eq[1] = 1.0f0
                eq[2] = outcome.is_gammon ? 1.0f0 : 0.0f0
                eq[3] = outcome.is_backgammon ? 1.0f0 : 0.0f0
            else
                eq[4] = outcome.is_gammon ? 1.0f0 : 0.0f0
                eq[5] = outcome.is_backgammon ? 1.0f0 : 0.0f0
            end
        end
        push!(samples, (state=state_vec, policy=full_policy, value=z,
                        equity=eq, has_equity=has_eq, is_chance=is_chance_flags[i]))
    end
end
println("Generated $(length(samples)) samples from 200 random games")

# Prepare batch
batch_size = min(512, length(samples))
indices = Random.shuffle(1:length(samples))[1:batch_size]
batch = [samples[i] for i in indices]
n = length(batch)

W = ones(Float32, 1, n)
X = hcat([s.state for s in batch]...)
P = hcat([s.policy for s in batch]...)
V = reshape(Float32[s.value for s in batch], 1, n)
A = zeros(Float32, num_actions, n)
IsChance = zeros(Float32, 1, n)
for i in 1:n
    if batch[i].is_chance
        A[:, i] .= 1.0f0
        IsChance[1, i] = 1.0f0
    else
        A[:, i] .= Float32.(batch[i].policy .> 0)
    end
end
EqWin = zeros(Float32, 1, n); EqGW = zeros(Float32, 1, n)
EqBGW = zeros(Float32, 1, n); EqGL = zeros(Float32, 1, n)
EqBGL = zeros(Float32, 1, n); HasEquity = zeros(Float32, 1, n)
for i in 1:n
    if batch[i].has_equity
        eq = batch[i].equity
        EqWin[1,i]=eq[1]; EqGW[1,i]=eq[2]; EqBGW[1,i]=eq[3]
        EqGL[1,i]=eq[4]; EqBGL[1,i]=eq[5]; HasEquity[1,i]=1.0f0
    end
end
batch_data = (; W, X, A, P, V, IsChance, EqWin, EqGW, EqBGW, EqGL, EqBGL, HasEquity)

learning_params = AlphaZero.LearningParams(
    samples_weighing_policy=AlphaZero.CONSTANT_WEIGHT,
    optimiser=AlphaZero.Adam(lr=0.001),
    l2_regularization=1f-4,
    nonvalidity_penalty=1f0,
    batch_size=256,
    loss_computation_batch_size=256,
    min_checkpoints_per_epoch=1,
    max_batches_per_checkpoint=100,
    num_checkpoints=1)

# Helper functions
bce(ŷ, y, w) = -sum((y .* log.(ŷ .+ eps(Float32)) .+
                      (1f0 .- y) .* log.(1f0 .- ŷ .+ eps(Float32))) .* w) / sum(w)

function analyze_checkpoint(ckpt_path, label)
    net = FluxLib.FCResNetMultiHead(
        gspec, FluxLib.FCResNetMultiHeadHP(width=128, num_blocks=3))
    FluxLib.load_weights(ckpt_path, net)

    L, Lp, Lv, Lreg, Linv = AlphaZero.losses(net, learning_params, mean(W), 0.0f0, batch_data)

    # Per-head breakdown
    P̂, V̂w, V̂gw, V̂bgw, V̂gl, V̂bgl, p_inv = FluxLib.forward_normalized_multihead(net, X, A)
    W_eq = W .* HasEquity
    Lvw = bce(V̂w, EqWin, W_eq)
    Lvgw = bce(V̂gw, EqGW, W_eq)
    Lvbgw = bce(V̂bgw, EqBGW, W_eq)
    Lvgl = bce(V̂gl, EqGL, W_eq)
    Lvbgl = bce(V̂bgl, EqBGL, W_eq)

    # Weight magnitude
    params = Network.params(net)
    weight_norm = sqrt(sum(sum(w .* w) for w in params))

    println("$label:")
    Printf.@printf("  Total=%.3f  Lp=%.3f  Lv=%.3f  Lreg=%.4f  Linv=%.5f\n",
        Float64(L), Float64(Lp), Float64(Lv), Float64(Lreg), Float64(Linv))
    Printf.@printf("  Value heads: win=%.3f  g|w=%.3f  bg|w=%.3f  g|l=%.3f  bg|l=%.3f\n",
        Float64(Lvw), Float64(Lvgw), Float64(Lvbgw), Float64(Lvgl), Float64(Lvbgl))
    Printf.@printf("  Weight L2 norm=%.1f  p_invalid_mean=%.5f\n", Float64(weight_norm), Float64(mean(p_inv)))

    # Per-head prediction stats
    eq_mask = vec(HasEquity) .> 0
    for (name, pred) in [("win", V̂w), ("g|w", V̂gw), ("bg|w", V̂bgw), ("g|l", V̂gl), ("bg|l", V̂bgl)]
        p = vec(pred)[eq_mask]
        Printf.@printf("    %s: mean=%.3f range=[%.3f, %.3f]\n", name, mean(p), minimum(p), maximum(p))
    end
    println()
end

using Printf

# Analyze all checkpoints
println("\n" * "=" ^ 80)
println("LOSS COMPONENT ANALYSIS ACROSS CHECKPOINTS")
println("(Evaluated on same $(n) random-game samples)")
println("=" ^ 80)

# Analyze all checkpoints
for iter in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    ckpt = joinpath(session_dir, "checkpoints", "iter_$iter.data")
    if isfile(ckpt)
        analyze_checkpoint(ckpt, "iter_$iter")
    end
end
