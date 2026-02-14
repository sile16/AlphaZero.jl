#!/usr/bin/env julia
# Verify board state and reward agreement between our engine and GnuBG
# STRICT: Any mismatch crashes immediately with full diagnostic info.

ENV["BACKGAMMON_OBS_TYPE"] = "minimal"
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))
using BackgammonNet
include(joinpath(@__DIR__, "GnubgPlayer.jl"))
using .GnubgPlayer: GnubgInterface
using AlphaZero
using AlphaZero: GI
using Random

GnubgInterface._init()
gspec = GameSpec()

# Verify checker counts are always 15 per side
function verify_checker_counts(game::BackgammonNet.BackgammonGame, context::String)
    p0_total = 0
    p1_total = 0
    for pt in 1:24
        p0_total += Int((game.p0 >> (pt << 2)) & 0xF)
        p1_total += Int((game.p1 >> (pt << 2)) & 0xF)
    end
    p0_bar = Int((game.p0 >> (BackgammonNet.IDX_P0_BAR << 2)) & 0xF)
    p1_bar = Int((game.p1 >> (BackgammonNet.IDX_P1_BAR << 2)) & 0xF)
    p0_off = Int((game.p0 >> (BackgammonNet.IDX_P0_OFF << 2)) & 0xF)
    p1_off = Int((game.p1 >> (BackgammonNet.IDX_P1_OFF << 2)) & 0xF)
    p0_total += p0_bar + p0_off
    p1_total += p1_bar + p1_off
    @assert p0_total == 15 "[$context] P0 checker count is $p0_total, expected 15 (board=$p0_bar bar, $p0_off off)"
    @assert p1_total == 15 "[$context] P1 checker count is $p1_total, expected 15 (board=$p1_bar bar, $p1_off off)"
end

# Verify GnuBG can evaluate this position without error
function verify_gnubg_eval(game::BackgammonNet.BackgammonGame, context::String)
    board = GnubgInterface._to_gnubg_board(game)
    probs = GnubgInterface._gnubg.probabilities(board, 0)
    win = Float64(probs[1])
    wg = Float64(probs[2])
    wbg = Float64(probs[3])
    lg = Float64(probs[4])
    lbg = Float64(probs[5])

    # Basic sanity: probabilities in [0,1]
    @assert 0.0 <= win <= 1.0 "[$context] win=$win out of range"
    @assert 0.0 <= wg <= 1.0 "[$context] wg=$wg out of range"
    @assert 0.0 <= wbg <= 1.0 "[$context] wbg=$wbg out of range"
    @assert 0.0 <= lg <= 1.0 "[$context] lg=$lg out of range"
    @assert 0.0 <= lbg <= 1.0 "[$context] lbg=$lbg out of range"

    # Gammon/backgammon conditionals can't exceed win/loss
    @assert wg <= win "[$context] wg=$wg > win=$win"
    @assert wbg <= wg "[$context] wbg=$wbg > wg=$wg"
    loss = 1.0 - win
    if loss > 0.001
        @assert lg <= loss + 0.001 "[$context] lg=$lg > loss=$loss"
        @assert lbg <= lg + 0.001 "[$context] lbg=$lbg > lg=$lg"
    end

    return (win, wg, wbg, lg, lbg)
end

# Verify board encoding roundtrip: our nibble extraction matches _to_gnubg_board
function verify_board_encoding(game::BackgammonNet.BackgammonGame, context::String)
    board = GnubgInterface._to_gnubg_board(game)
    cp = Int(game.current_player)

    our_onroll = zeros(Int, 25)
    our_opp = zeros(Int, 25)

    if cp == 0
        for pt in 1:24
            idx = 25 - pt
            our_onroll[pt + 1] = Int((game.p0 >> (idx << 2)) & 0xF)
            our_opp[pt + 1] = Int((game.p1 >> (idx << 2)) & 0xF)
        end
        our_onroll[1] = Int((game.p0 >> (BackgammonNet.IDX_P0_BAR << 2)) & 0xF)
        our_opp[1] = Int((game.p1 >> (BackgammonNet.IDX_P1_BAR << 2)) & 0xF)
    else
        for pt in 1:24
            our_onroll[pt + 1] = Int((game.p1 >> (pt << 2)) & 0xF)
            our_opp[pt + 1] = Int((game.p0 >> (pt << 2)) & 0xF)
        end
        our_onroll[1] = Int((game.p1 >> (BackgammonNet.IDX_P1_BAR << 2)) & 0xF)
        our_opp[1] = Int((game.p0 >> (BackgammonNet.IDX_P0_BAR << 2)) & 0xF)
    end

    gnubg_opp = Int.(board[1])
    gnubg_onr = Int.(board[2])

    @assert our_opp == gnubg_opp "[$context] Opponent board mismatch!\n  Our:   $our_opp\n  GnuBG: $gnubg_opp"
    @assert our_onroll == gnubg_onr "[$context] On-roll board mismatch!\n  Our:   $our_onroll\n  GnuBG: $gnubg_onr"
end

# Compute expected reward independently from raw game state
function compute_expected_reward(game::BackgammonNet.BackgammonGame)
    p0_off = Int((game.p0 >> (BackgammonNet.IDX_P0_OFF << 2)) & 0xF)
    p1_off = Int((game.p1 >> (BackgammonNet.IDX_P1_OFF << 2)) & 0xF)
    p0_bar = Int((game.p0 >> (BackgammonNet.IDX_P0_BAR << 2)) & 0xF)
    p1_bar = Int((game.p1 >> (BackgammonNet.IDX_P1_BAR << 2)) & 0xF)

    if p0_off >= 15
        # P0 (white) won
        if p1_off == 0
            # Check for backgammon: any p1 checker in P0's home (pts 19-24) or on bar
            # P0's home board is physical points 19-24 (P0 moves toward 24→off)
            p1_in_p0_home = 0
            for pt in 19:24
                p1_in_p0_home += Int((game.p1 >> (pt << 2)) & 0xF)
            end
            if p1_bar > 0 || p1_in_p0_home > 0
                return 3.0  # backgammon
            else
                return 2.0  # gammon
            end
        else
            return 1.0  # single
        end
    elseif p1_off >= 15
        # P1 (black) won
        if p0_off == 0
            # Check for backgammon: any p0 checker in P1's home (pts 1-6) or on bar
            # P1's home board is physical points 1-6 (P1 moves toward 1→off)
            p0_in_p1_home = 0
            for pt in 1:6
                p0_in_p1_home += Int((game.p0 >> (pt << 2)) & 0xF)
            end
            if p0_bar > 0 || p0_in_p1_home > 0
                return -3.0  # backgammon
            else
                return -2.0  # gammon
            end
        else
            return -1.0  # single
        end
    else
        error("Game terminated but neither player has 15 off! p0_off=$p0_off, p1_off=$p1_off")
    end
end

# Play a full game, verifying state at every step
function play_verified_game(seed::Int)
    rng = MersenneTwister(seed)
    env = GI.init(gspec)
    env.rng = rng

    gnubg_1ply = GnubgPlayer.GnubgBaseline(ply=1)
    move_count = 0
    board_checks = 0

    while !GI.game_terminated(env)
        if GI.is_chance_node(env)
            outcomes = GI.chance_outcomes(env)
            r = rand(rng)
            acc = 0.0
            idx = length(outcomes)
            for i in eachindex(outcomes)
                acc += outcomes[i][2]
                if r <= acc; idx = i; break; end
            end
            GI.apply_chance!(env, outcomes[idx][1])
            continue
        end

        move_count += 1
        game = env.game
        ctx = "seed=$seed, move=$move_count"

        # Always verify checker counts
        verify_checker_counts(game, ctx)

        # Verify board encoding at decision points with remaining moves
        if game.remaining_actions > 0
            board_checks += 1
            verify_board_encoding(game, ctx)
            verify_gnubg_eval(game, ctx)
        end

        # Let GnuBG choose the move
        actions = GI.available_actions(env)
        @assert length(actions) >= 1 "[$ctx] No available actions!"
        if length(actions) == 1
            GI.play!(env, actions[1])
        else
            actions_g, π = AlphaZero.think(gnubg_1ply, env)
            action = actions_g[argmax(π)]
            GI.play!(env, action)
        end

        @assert move_count <= 500 "[$ctx] Game exceeded 500 moves — likely infinite loop"
    end

    # Verify final state
    game = env.game
    verify_checker_counts(game, "seed=$seed, final")

    our_reward = GI.white_reward(env)
    expected_reward = compute_expected_reward(game)
    if our_reward != expected_reward
        p0_off = Int((game.p0 >> (BackgammonNet.IDX_P0_OFF << 2)) & 0xF)
        p1_off = Int((game.p1 >> (BackgammonNet.IDX_P1_OFF << 2)) & 0xF)
        p0_bar = Int((game.p0 >> (BackgammonNet.IDX_P0_BAR << 2)) & 0xF)
        p1_bar = Int((game.p1 >> (BackgammonNet.IDX_P1_BAR << 2)) & 0xF)
        println("  REWARD MISMATCH DIAGNOSTICS:")
        println("  GI.white_reward=$our_reward, expected=$expected_reward, game.reward=$(game.reward)")
        println("  P0: off=$p0_off, bar=$p0_bar")
        println("  P1: off=$p1_off, bar=$p1_bar")
        for pt in 1:24
            p0c = Int((game.p0 >> (pt << 2)) & 0xF)
            p1c = Int((game.p1 >> (pt << 2)) & 0xF)
            if p0c > 0 || p1c > 0
                println("  Point $pt: P0=$p0c, P1=$p1c")
            end
        end
        error("seed=$seed: REWARD MISMATCH! GI.white_reward=$our_reward, expected=$expected_reward")
    end

    return move_count, board_checks
end

println("Verifying board state and reward agreement (strict mode)...")
println()

for seed in 1:20
    moves, checks = play_verified_game(seed)
    println("Game $seed: OK ($moves moves, $checks board checks, reward verified)")
end

println()
println("ALL 20 GAMES PASSED - board encoding, checker counts, gnubg eval, and rewards all verified.")
