# BgblitzPlayer.jl - BGBlitz evaluation via TCP server or subprocess
#
# Server mode (recommended for eval):
#   BgblitzPlayer.start_server(slots=4, ply=0)
#   action = BgblitzPlayer.best_move(conn, game_env)
#   BgblitzPlayer.stop_server()
#
# Subprocess mode (backward compat):
#   player = BgblitzFast(ply=0)
#   AlphaZero.think(player, env)

module BgblitzPlayer

using BackgammonNet
using Sockets

export BgblitzFast

# =============================================================================
# Constants
# =============================================================================

const BGBLITZ_JAVA = "/Applications/BGBlitz/Bgblitz.app/Contents/runtime/Contents/Home/bin/java"
const BGBLITZ_APP = "/Applications/BGBlitz/Bgblitz.app/Contents/app"
const BGBLITZ_SYSTEM = "/Applications/BGBlitz/system"

# =============================================================================
# TCP Server Mode
# =============================================================================

mutable struct ServerState
    proc::Base.Process
    port::Int
    connections::Channel{TCPSocket}
    num_slots::Int
end

const _server = Ref{Union{Nothing, ServerState}}(nothing)

"""
    start_server(; slots=4, ply=0) -> ServerState

Launch a single BGBlitz JVM with `slots` evaluator instances.
Returns when all evaluators are ready and TCP connections are established.
"""
function start_server(; slots::Int=4, ply::Int=0)
    if _server[] !== nothing
        stop_server()
    end

    bridge_dir = joinpath(@__DIR__, "bgblitz-bridge")
    cp = join([
        joinpath(BGBLITZ_APP, "bgblitz.jar"),
        joinpath(BGBLITZ_APP, "tools.jar"),
        joinpath(BGBLITZ_APP, "res.jar"),
        joinpath(BGBLITZ_APP, "tools_jni.jar"),
        bridge_dir
    ], ":")

    cmd = `$BGBLITZ_JAVA
        -Dapple.awt.UIElement=true
        -Xmx512m
        -DBGB_SYSTEM_DIR=$BGBLITZ_SYSTEM
        --enable-preview
        --add-modules jdk.incubator.vector
        -cp $cp
        BgblitzServer --slots $slots --ply $ply`

    proc = open(cmd, "r+")

    # Read READY port=N from server
    port = 0
    for _ in 1:120  # up to 120 seconds for evaluator init
        line = readline(proc)
        if startswith(line, "READY port=")
            port = parse(Int, split(split(line, "port=")[2])[1])
            break
        elseif startswith(line, "ERROR")
            error("BGBlitz server failed: $line")
        end
    end
    if port == 0
        error("BGBlitz server did not report port")
    end

    # Connect N TCP sockets and fill the pool
    pool = Channel{TCPSocket}(slots)
    for i in 1:slots
        conn = connect("127.0.0.1", port)
        # Read per-connection READY
        ready = readline(conn)
        if !startswith(ready, "READY")
            error("BGBlitz connection $i failed: $ready")
        end
        put!(pool, conn)
    end

    srv = ServerState(proc, port, pool, slots)
    _server[] = srv
    return srv
end

"""
    stop_server()

Shut down the BGBlitz server and close all connections.
"""
function stop_server()
    srv = _server[]
    if srv === nothing
        return
    end

    # Close any remaining connections still in the pool
    while isready(srv.connections)
        try
            conn = take!(srv.connections)
            println(conn, "QUIT")
            flush(conn)
            close(conn)
        catch
        end
    end

    # Send QUIT on stdin to shut down the JVM
    try
        println(srv.proc, "QUIT")
        flush(srv.proc)
    catch
    end

    _server[] = nothing
end

"""
    take_connection() -> TCPSocket

Borrow a connection from the pool. Call `return_connection(conn)` when done.
"""
take_connection() = take!(_server[].connections)

"""
    return_connection(conn::TCPSocket)

Return a borrowed connection to the pool.
"""
return_connection(conn::TCPSocket) = put!(_server[].connections, conn)

"""
    send_command(conn::TCPSocket, cmd::String) -> String

Send a command and read the response. No log-line filtering needed
since the TCP server sends clean protocol responses only.
"""
function send_command(conn::TCPSocket, cmd::String)::String
    println(conn, cmd)
    flush(conn)
    return readline(conn)
end

"""
    best_move(conn::TCPSocket, env) -> Int

Get BGBlitz's best move for the current position. Returns an action index.
Uses the given TCP connection (caller manages borrowing/returning).
"""
function best_move(conn::TCPSocket, env)::Int
    bg_game = env.game
    xgid = _to_xgid(bg_game)
    response = send_command(conn, "BESTMOVE $xgid")

    action = _parse_move_response(response, bg_game)

    # Verify action is legal
    legal = BackgammonNet.legal_actions(bg_game)
    if !(action in legal)
        @warn "BGBlitz returned illegal action $action, falling back to evaluation"
        action = _best_move_evaluate(conn, bg_game)
    end

    return action
end

# =============================================================================
# XGID Conversion: BackgammonNet -> XGID string
# =============================================================================

# BackgammonNet Player Mapping:
#   BG P0 (White) = XGID X (RED, lowercase) - moves 1->24, bar at XGID index 0
#   BG P1 (Black) = XGID O (GREEN, uppercase) - moves 24->1, bar at XGID index 25
#
# Point numbering is the SAME: BG absolute point N = XGID point N

function _to_xgid(g::BackgammonGame)::String
    board = zeros(Int, 26)

    for pt in 1:24
        p0_count = Int((g.p0 >> (pt << 2)) & 0xF)
        p1_count = Int((g.p1 >> (pt << 2)) & 0xF)
        if p0_count > 0
            board[pt + 1] = -p0_count
        elseif p1_count > 0
            board[pt + 1] = p1_count
        end
    end

    p0_bar = Int((g.p0 >> (BackgammonNet.IDX_P0_BAR << 2)) & 0xF)
    p1_bar = Int((g.p1 >> (BackgammonNet.IDX_P1_BAR << 2)) & 0xF)
    board[1] = -p0_bar
    board[26] = p1_bar

    chars = Vector{Char}(undef, 26)
    for i in 1:26
        v = board[i]
        if v == 0
            chars[i] = '-'
        elseif v > 0
            chars[i] = Char('A' + v - 1)
        else
            chars[i] = Char('a' + abs(v) - 1)
        end
    end
    pos_str = String(chars)

    turn = g.current_player == 0 ? -1 : 1
    d1, d2 = Int(g.dice[1]), Int(g.dice[2])
    dice_str = d1 > 0 && d2 > 0 ? "$(max(d1,d2))$(min(d1,d2))" : "00"

    return "XGID=$(pos_str):0:0:$(turn):$(dice_str):0:0:0:0:0"
end

# =============================================================================
# Move Conversion: BGBlitz -> BackgammonNet action
# =============================================================================

function _parse_move_response(line::String, g::BackgammonGame)::Int
    if !startswith(line, "MOVE ")
        @warn "Unexpected BGBlitz response: $line"
        return BackgammonNet.encode_action(BackgammonNet.PASS_LOC, BackgammonNet.PASS_LOC)
    end

    move_part = split(line[6:end], "|")[1]

    if strip(move_part) == "PASS"
        return BackgammonNet.encode_action(BackgammonNet.PASS_LOC, BackgammonNet.PASS_LOC)
    end

    parts = split(strip(move_part), ",")
    froms = Int[]
    tos = Int[]
    for p in parts
        ft = split(strip(p), "/")
        push!(froms, parse(Int, ft[1]))
        push!(tos, parse(Int, ft[2]))
    end

    return _convert_bgblitz_move(froms, tos, g)
end

function _convert_bgblitz_move(froms::Vector{Int}, tos::Vector{Int}, g::BackgammonGame)::Int
    d1 = Int(g.dice[1])
    d2 = Int(g.dice[2])
    n_moves = length(froms)

    if n_moves == 0
        return BackgammonNet.encode_action(BackgammonNet.PASS_LOC, BackgammonNet.PASS_LOC)
    end

    n_take = min(n_moves, 2)
    dies = [abs(froms[i] - tos[i]) for i in 1:n_take]

    function to_canonical(bgblitz_from::Int)
        if bgblitz_from == 25
            return 0  # bar
        else
            return 25 - bgblitz_from
        end
    end

    canonical_sources = [to_canonical(froms[i]) for i in 1:n_take]

    if n_take == 1
        if dies[1] == d1
            return BackgammonNet.encode_action(canonical_sources[1], BackgammonNet.PASS_LOC)
        else
            return BackgammonNet.encode_action(BackgammonNet.PASS_LOC, canonical_sources[1])
        end
    end

    if d1 == d2
        return BackgammonNet.encode_action(canonical_sources[1], canonical_sources[2])
    end

    if dies[1] == d1 && dies[2] == d2
        return BackgammonNet.encode_action(canonical_sources[1], canonical_sources[2])
    elseif dies[1] == d2 && dies[2] == d1
        return BackgammonNet.encode_action(canonical_sources[2], canonical_sources[1])
    else
        # Ambiguous (bear-off where die > distance) - can't use conn here,
        # return a sentinel that the caller should handle
        return -1
    end
end

# =============================================================================
# Fallback: Evaluate all legal actions via EVAL command
# =============================================================================

function _best_move_evaluate(conn::TCPSocket, g::BackgammonGame)::Int
    actions = BackgammonNet.legal_actions(g)

    if isempty(actions)
        return BackgammonNet.encode_action(BackgammonNet.PASS_LOC, BackgammonNet.PASS_LOC)
    end

    best_action = actions[1]
    best_equity = -Inf

    for action in actions
        g2 = BackgammonNet.clone(g)
        BackgammonNet.apply_action!(g2, action)

        xgid = _to_xgid_eval(g2)
        response = send_command(conn, "EVAL $xgid")

        if startswith(response, "EVAL ")
            equity = parse(Float64, split(response[6:end])[1])
            if g2.current_player != g.current_player
                equity = -equity
            end
            if equity > best_equity
                best_equity = equity
                best_action = action
            end
        end
    end

    return best_action
end

function _to_xgid_eval(g::BackgammonGame)::String
    board = zeros(Int, 26)
    for pt in 1:24
        p0_count = Int((g.p0 >> (pt << 2)) & 0xF)
        p1_count = Int((g.p1 >> (pt << 2)) & 0xF)
        if p0_count > 0
            board[pt + 1] = -p0_count
        elseif p1_count > 0
            board[pt + 1] = p1_count
        end
    end
    p0_bar = Int((g.p0 >> (BackgammonNet.IDX_P0_BAR << 2)) & 0xF)
    p1_bar = Int((g.p1 >> (BackgammonNet.IDX_P1_BAR << 2)) & 0xF)
    board[1] = -p0_bar
    board[26] = p1_bar

    chars = Vector{Char}(undef, 26)
    for i in 1:26
        v = board[i]
        if v == 0; chars[i] = '-'
        elseif v > 0; chars[i] = Char('A' + v - 1)
        else; chars[i] = Char('a' + abs(v) - 1)
        end
    end

    turn = g.current_player == 0 ? -1 : 1
    return "XGID=$(String(chars)):0:0:$(turn):00:0:0:0:0:0"
end

# =============================================================================
# AlphaZero Player Interface (subprocess mode - backward compat)
# =============================================================================

mutable struct BgblitzProcess
    proc::Base.Process
    input::IO
    output::IO
    ply::Int
end

const _process = Ref{Union{Nothing, BgblitzProcess}}(nothing)

function _get_process(ply::Int)
    p = _process[]
    if p !== nothing && process_running(p.proc) && p.ply == ply
        return p
    end
    _process[] = _launch(ply)
    return _process[]
end

function _launch(ply::Int)
    bridge_dir = joinpath(@__DIR__, "bgblitz-bridge")
    cp = join([
        joinpath(BGBLITZ_APP, "bgblitz.jar"),
        joinpath(BGBLITZ_APP, "tools.jar"),
        joinpath(BGBLITZ_APP, "res.jar"),
        joinpath(BGBLITZ_APP, "tools_jni.jar"),
        bridge_dir
    ], ":")

    cmd = `$BGBLITZ_JAVA
        -Dapple.awt.UIElement=true
        -Xmx512m
        -DBGB_SYSTEM_DIR=$BGBLITZ_SYSTEM
        --enable-preview
        --add-modules jdk.incubator.vector
        -cp $cp
        BgblitzBridge $ply`

    proc = open(cmd, "r+")
    ready = ""
    for _ in 1:50
        ready = readline(proc)
        if startswith(ready, "READY")
            break
        end
    end
    if !startswith(ready, "READY")
        error("BGBlitz bridge failed to start: $ready")
    end

    return BgblitzProcess(proc, proc, proc, ply)
end

function _send_command_subprocess(proc::BgblitzProcess, cmd::String)::String
    println(proc.input, cmd)
    flush(proc.input)
    for _ in 1:50
        line = readline(proc.output)
        if startswith(line, "MOVE ") || startswith(line, "EVAL ") ||
           startswith(line, "ERROR") || startswith(line, "READY")
            return line
        end
    end
    return "ERROR timeout reading response"
end

function cleanup()
    p = _process[]
    if p !== nothing && process_running(p.proc)
        try
            println(p.input, "QUIT")
            flush(p.input)
        catch
        end
    end
    _process[] = nothing

    # Also stop server if running
    stop_server()
end

import AlphaZero: AbstractPlayer, think, reset!

"""
BGBlitz player using TachiAI neural net evaluator.

    BgblitzFast()        # 0-ply (neural net only)
    BgblitzFast(ply=2)   # 2-ply lookahead
"""
struct BgblitzFast <: AbstractPlayer
    ply::Int
end

BgblitzFast(; ply::Int=0) = BgblitzFast(ply)

function think(p::BgblitzFast, game)
    bg_game = game.game
    proc = _get_process(p.ply)

    xgid = _to_xgid(bg_game)
    response = _send_command_subprocess(proc, "BESTMOVE $xgid")
    action = _parse_move_response(response, bg_game)

    legal = BackgammonNet.legal_actions(bg_game)
    if !(action in legal)
        @warn "BGBlitz returned illegal action $action, falling back to evaluation"
        # Use subprocess-based fallback
        best_action = legal[1]
        best_equity = -Inf
        for a in legal
            g2 = BackgammonNet.clone(bg_game)
            BackgammonNet.apply_action!(g2, a)
            xgid2 = _to_xgid_eval(g2)
            resp = _send_command_subprocess(proc, "EVAL $xgid2")
            if startswith(resp, "EVAL ")
                eq = parse(Float64, split(resp[6:end])[1])
                if g2.current_player != bg_game.current_player
                    eq = -eq
                end
                if eq > best_equity
                    best_equity = eq
                    best_action = a
                end
            end
        end
        action = best_action
    end

    num_actions = 676
    π = zeros(num_actions)
    if 1 <= action <= num_actions
        π[action] = 1.0
    end

    return collect(1:num_actions), π
end

reset!(::BgblitzFast) = nothing

# =============================================================================
# Benchmark Interface
# =============================================================================

import AlphaZero.Benchmark

struct BgblitzFastBenchmark <: Benchmark.Player
    ply::Int
end

BgblitzFastBenchmark(; ply::Int=0) = BgblitzFastBenchmark(ply)

Benchmark.name(p::BgblitzFastBenchmark) = "BGBlitz-$(p.ply)ply"

function Benchmark.instantiate(p::BgblitzFastBenchmark, ::Any, nn)
    return BgblitzFast(p.ply)
end

end # module
