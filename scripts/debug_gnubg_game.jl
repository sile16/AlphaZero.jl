#!/usr/bin/env julia
# Debug script: play one game between AZ and GnuBG with full state logging
# Verifies board state agreement and reward calculation

using Random

ENV["BACKGAMMON_OBS_TYPE"] = "minimal"
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))

using AlphaZero
using AlphaZero: GI, FluxLib, MctsParams, MctsPlayer, ConstSchedule
using BackgammonNet
using Flux

include(joinpath(@__DIR__, "GnubgPlayer.jl"))

# Parse args
ply = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1
checkpoint = length(ARGS) >= 2 ? ARGS[2] : nothing

println("=" ^ 60)
println("Debug GnuBG Game (ply=$ply)")
println("=" ^ 60)

gspec = GameSpec()

# Load AZ player if checkpoint provided, otherwise use random
if checkpoint !== nothing
    network = FluxLib.FCResNetMultiHead(gspec, FluxLib.FCResNetMultiHeadHP(width=128, num_blocks=3))
    FluxLib.load_weights(checkpoint, network)
    println("Loaded AZ network: $(sum(length, Flux.params(network))) params")
    mcts_params = MctsParams(
        num_iters_per_turn=100,
        cpuct=2.0,
        temperature=ConstSchedule(0.0),
        dirichlet_noise_ϵ=0.0,
        dirichlet_noise_α=1.0,
        chance_mode=:passthrough
    )
    az_player = MctsPlayer(gspec, network, mcts_params)
    use_az = true
else
    use_az = false
    println("No checkpoint provided — using random player vs GnuBG")
end

gnubg_player = GnubgPlayer.GnubgBaseline(ply=ply)

# Initialize gnubg
println("Initializing GnuBG...")
let
    dummy = GI.init(gspec)
    AlphaZero.think(gnubg_player, dummy)
end
println("GnuBG initialized")

# Helper: print board state
function print_board(env, move_num)
    game = env.game
    println("\n--- Move $move_num ---")
    println("  Current player: $(game.current_player)")
    println("  Dice: $(game.dice)")
    println("  Remaining actions: $(game.remaining_actions)")
    println("  Terminated: $(game.terminated)")

    # Decode board from nibble bitboard
    p0_pts = Int[]
    p1_pts = Int[]
    for pt in 1:24
        p0_count = Int((game.p0 >> (pt << 2)) & 0xF)
        p1_count = Int((game.p1 >> (pt << 2)) & 0xF)
        if p0_count > 0
            push!(p0_pts, pt)
        end
        if p1_count > 0
            push!(p1_pts, pt)
        end
    end

    # Bar
    p0_bar = Int((game.p0 >> (BackgammonNet.IDX_P0_BAR << 2)) & 0xF)
    p1_bar = Int((game.p1 >> (BackgammonNet.IDX_P1_BAR << 2)) & 0xF)
    p0_off = Int((game.p0 >> (BackgammonNet.IDX_P0_OFF << 2)) & 0xF)
    p1_off = Int((game.p1 >> (BackgammonNet.IDX_P1_OFF << 2)) & 0xF)

    println("  P0: bar=$p0_bar, off=$p0_off, points=$([(pt, Int((game.p0 >> (pt << 2)) & 0xF)) for pt in 1:24 if Int((game.p0 >> (pt << 2)) & 0xF) > 0])")
    println("  P1: bar=$p1_bar, off=$p1_off, points=$([(pt, Int((game.p1 >> (pt << 2)) & 0xF)) for pt in 1:24 if Int((game.p1 >> (pt << 2)) & 0xF) > 0])")

    # Also get GnuBG evaluation of this position
    if !game.terminated && game.remaining_actions > 0
        try
            probs = GnubgPlayer.evaluate_position(game; ply=ply)
            equity = GnubgPlayer.GnubgInterface.evaluate(game; ply=ply)
            println("  GnuBG eval (ply=$ply): win=$(round(probs[1], digits=4)), wg=$(round(probs[2], digits=4)), wbg=$(round(probs[3], digits=4)), lg=$(round(probs[4], digits=4)), lbg=$(round(probs[5], digits=4))")
            println("  GnuBG equity: $(round(equity, digits=4))")
        catch e
            println("  GnuBG eval failed: $e")
        end
    end
end

# Sample chance outcome
function _sample_chance(rng, outcomes)
    r = rand(rng)
    acc = 0.0
    for i in eachindex(outcomes)
        acc += outcomes[i][2]
        if r <= acc
            return i
        end
    end
    return length(outcomes)
end

# Play one game: AZ as white vs GnuBG
rng = MersenneTwister(42)
env = GI.init(gspec)
env.rng = rng

move_num = 0
az_is_white = true

println("\nPlaying: $(az_is_white ? "AZ(white)" : "GnuBG(white)") vs $(az_is_white ? "GnuBG(black)" : "AZ(black)")")
println("GnuBG ply: $ply")

while !GI.game_terminated(env)
    # Chance node
    if GI.is_chance_node(env)
        outcomes = GI.chance_outcomes(env)
        idx = _sample_chance(rng, outcomes)
        dice_action = outcomes[idx][1]
        GI.apply_chance!(env, dice_action)
        println("  [Chance] Dice rolled")
        continue
    end

    global move_num += 1

    # Print state before move
    print_board(env, move_num)

    # Check for forced moves
    actions = GI.available_actions(env)
    if length(actions) == 1
        println("  -> Forced: action=$(actions[1])")
        GI.play!(env, actions[1])
        continue
    end

    is_white = GI.white_playing(env)
    use_az_here = (is_white && az_is_white) || (!is_white && !az_is_white)

    if use_az_here && use_az
        actions_az, π = AlphaZero.think(az_player, env)
        action = actions_az[argmax(π)]
        println("  -> AZ plays: action=$action ($(length(actions)) legal moves)")
        AlphaZero.reset_player!(az_player)
    else
        actions_gnubg, π = AlphaZero.think(gnubg_player, env)
        action = actions_gnubg[argmax(π)]
        println("  -> GnuBG($(ply)ply) plays: action=$action ($(length(actions)) legal moves)")
    end

    GI.play!(env, action)

    if move_num >= 200
        println("\n[TIMEOUT] Game exceeded 200 moves")
        break
    end
end

# Final state
println("\n" * "=" ^ 60)
println("GAME OVER")
println("=" ^ 60)
game = env.game
println("Terminated: $(game.terminated)")
println("Reward field: $(game.reward)")
white_reward = GI.white_reward(env)
println("White reward (GI.white_reward): $white_reward")

# Verify: count checkers
p0_total = 0
p1_total = 0
for pt in 1:24
    p0_total += Int((game.p0 >> (pt << 2)) & 0xF)
    p1_total += Int((game.p1 >> (pt << 2)) & 0xF)
end
p0_total += Int((game.p0 >> (BackgammonNet.IDX_P0_BAR << 2)) & 0xF)
p1_total += Int((game.p1 >> (BackgammonNet.IDX_P1_BAR << 2)) & 0xF)
p0_off = Int((game.p0 >> (BackgammonNet.IDX_P0_OFF << 2)) & 0xF)
p1_off = Int((game.p1 >> (BackgammonNet.IDX_P1_OFF << 2)) & 0xF)

println("P0: $(p0_total) on board + $p0_off off = $(p0_total + p0_off)")
println("P1: $(p1_total) on board + $p1_off off = $(p1_total + p1_off)")
println("Winner: $(white_reward > 0 ? "White (P0)" : white_reward < 0 ? "Black (P1)" : "Draw")")
println("Reward magnitude: $(abs(white_reward)) (1=single, 2=gammon, 3=backgammon)")
