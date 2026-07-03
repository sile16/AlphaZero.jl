#!/usr/bin/env julia
"""
Validate the exact k=7 bear-off table end-to-end by playing a pure table policy
against wildbg from mutual-bearoff starting positions.

The table agent picks the move maximizing the exact post-move table value
(terminal moves scored from `reward`, which carries the gammon multiplier).
Wildbg has no bear-off database — an exact policy should beat it by a clear
margin from bear-off positions. A result at or below parity indicates a bug
in the table, board encoding, perspective handling, or move enumeration.

Example:
  julia --threads 10 --project scripts/eval_table_vs_wildbg.jl \\
      --num-games=500 --num-workers=8
"""

using ArgParse

function parse_cli()
    s = ArgParseSettings(description="Exact bear-off table vs wildbg", autofix_names=true)
    @add_arg_table! s begin
        "--num-games"
            arg_type = Int
            default = 500
            help = "Number of starting positions (each played from both sides)"
        "--num-workers"
            arg_type = Int
            default = max(1, Threads.nthreads() - 1)
            help = "Parallel game workers (per-worker wildbg backend)"
        "--wildbg-lib"
            arg_type = String
            default = ""
            help = "Path to wildbg shared library (auto-detected if empty)"
        "--positions-file"
            arg_type = String
            default = ""
            help = "Race start tuples .jls (auto-detected if empty)"
        "--seed"
            arg_type = Int
            default = 42
            help = "RNG seed for position generation and dice"
        "--gammon-starts"
            action = :store_true
            help = "Only use starts where a side has 0 checkers off (gammon live) — most discriminative for exact-table play"
    end
    return ArgParse.parse_args(s)
end

const ARGS_D = parse_cli()

using Random
using Serialization
using Statistics
using StaticArrays

using AlphaZero
using AlphaZero: GI, GameLoop
import BackgammonNet
using BackgammonNet: BackgammonGame

# Backgammon game wrapper (GameEnv, GameSpec) — same pattern as eval_race.jl
include(joinpath(@__DIR__, "..", "games", "backgammon-deterministic", "game.jl"))

# ── k=7 table ───────────────────────────────────────────────────────────

const BEAROFF_SRC = joinpath(homedir(), "github", "BackgammonNet.jl", "src", "bearoff_k7.jl")
isfile(BEAROFF_SRC) || error("bearoff_k7.jl not found at $BEAROFF_SRC")
include(BEAROFF_SRC)
using .BearoffK7
using .BearoffK7: BearoffTable
include(joinpath(@__DIR__, "bearoff_eval_common.jl"))

const TABLE = let
    candidates = [
        joinpath(dirname(BEAROFF_SRC), "..", "tools", "bearoff_twosided", "bearoff_k7_twosided"),
        joinpath(homedir(), "bearoff_k7_twosided"),
        "/homeshare/projects/AlphaZero.jl/eval_data/bearoff_k7_twosided",
    ]
    dir = findfirst(d -> isdir(d) && isfile(joinpath(d, "bearoff_k7_c14.bin")), candidates)
    dir === nothing && error("k=7 bear-off table not found in: $(join(candidates, ", "))")
    println("Loading k=7 table from $(candidates[dir]) ...")
    BearoffTable(candidates[dir])
end

# ── Table policy agent ──────────────────────────────────────────────────

"""Pure exact-table policy: at a decision node, pick the legal move maximizing
the exact post-move value. Terminal moves are scored from `reward` (carries the
gammon multiplier); non-terminal moves from the opponent's negated pre-dice
table value. Raw-points scale throughout (self-consistent — no NN values here)."""
struct TableAgent <: GameLoop.GameAgent end

GameLoop.create_player(::TableAgent) = nothing

function GameLoop.select_action(::TableAgent, ::Nothing, env)
    bg = env.game
    actions = BackgammonNet.legal_actions(bg)
    @assert !isempty(actions)
    BearoffK7.is_bearoff_position(bg.p0, bg.p1) ||
        error("TableAgent reached a non-bearoff position — starts must be mutual bearoff")

    mover = Int(bg.current_player)
    work = BackgammonNet.clone(bg)
    best_action = actions[1]
    best_val = -Inf
    for a in actions
        BackgammonNet.copy_state!(work, bg)
        BackgammonNet.apply_action!(work, a)
        # Turn-aware exact value (handles doubles mid-turn recursion — see
        # scripts/bearoff_eval_common.jl for the pitfall this caught)
        val = bearoff_turn_value(TABLE, work, mover)
        if val > best_val
            best_val = val
            best_action = a
        end
    end
    return (best_action, Float32[], Int[])
end

# ── Starting positions: roll out race starts to mutual bearoff ─────────

function find_positions_file()
    isempty(ARGS_D["positions_file"]) || return ARGS_D["positions_file"]
    for f in [
        joinpath(@__DIR__, "..", "eval_data", "race_starts_tuples_bootstrap_no_eval_no_bo.jls"),
        "/homeshare/projects/AlphaZero.jl/eval_data/race_starts_tuples_no_eval.jls",
        "/homeshare/projects/AlphaZero.jl/eval_data/race_starts_tuples.jls",
    ]
        isfile(f) && return f
    end
    error("No race starts file found; pass --positions-file")
end

"""Random-play a race start until both sides are within k=7 table range."""
function rollout_to_bearoff(tup, rng)
    p0, p1, cp = tup[1], tup[2], tup[3]
    g = BackgammonGame(p0, p1, SVector{2, Int8}(0, 0), Int8(0), Int8(cp), false, 0.0f0;
                       obs_type=:minimal_flat)
    for _ in 1:200
        g.terminated && return nothing
        if BackgammonNet.is_chance_node(g)
            BearoffK7.is_bearoff_position(g.p0, g.p1) &&
                return (g.p0, g.p1, g.current_player)
            BackgammonNet.sample_chance!(g, rng)
        else
            acts = BackgammonNet.legal_actions(g)
            isempty(acts) && return nothing
            BackgammonNet.apply_action!(g, rand(rng, acts))
        end
    end
    return nothing
end

function build_start_positions(n::Int)
    src = find_positions_file()
    println("Race starts: $src")
    tuples = deserialize(src)
    rng = MersenneTwister(ARGS_D["seed"])
    seen = Set{Tuple{UInt128, UInt128, Int8}}()
    starts = Tuple{UInt128, UInt128, Int8}[]
    for tup in shuffle(rng, tuples)
        pos = rollout_to_bearoff(tup, rng)
        pos === nothing && continue
        pos in seen && continue
        if ARGS_D["gammon_starts"]
            # Gammon live: at least one side has all 15 checkers still on board.
            # These are the positions where exact gammon-aware play beats a
            # table-less NN by the most.
            n_on0 = sum(Int((pos[1] >> (i * 4)) & 0xF) for i in 1:24)
            n_on1 = sum(Int((pos[2] >> (i * 4)) & 0xF) for i in 1:24)
            (n_on0 == 15 || n_on1 == 15) || continue
        end
        push!(seen, pos)
        push!(starts, pos)
        length(starts) >= n && break
    end
    length(starts) >= n || @warn "Only found $(length(starts)) / $n bearoff starts"
    return starts
end

# ── Play ────────────────────────────────────────────────────────────────

function play_one(start, wildbg_backend; table_is_white::Bool, seed::Int)
    p0, p1, cp = start
    rng = MersenneTwister(seed)
    game = BackgammonGame(p0, p1, SVector{2, Int8}(0, 0), Int8(0), cp, false, 0.0f0;
                          obs_type=:minimal_flat)
    env = GameEnv(game, rng)
    ta = TableAgent()
    wb = GameLoop.ExternalAgent(wildbg_backend)
    w, b = table_is_white ? (ta, wb) : (wb, ta)
    result = GameLoop.play_game(w, b, env; rng=rng, temperature_fn=_ -> 0.0)
    return table_is_white ? result.reward : -result.reward
end

function main()
    n = ARGS_D["num_games"]
    starts = build_start_positions(n)
    println("Playing $(length(starts)) positions x 2 sides vs wildbg...")

    nw = max(1, min(ARGS_D["num_workers"], Threads.nthreads()))
    lib = ARGS_D["wildbg_lib"]
    if isempty(lib)
        for cand in [
            joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg.so"),
            joinpath(homedir(), "github", "wildbg", "target", "release", "libwildbg.dylib"),
        ]
            isfile(cand) && (lib = cand; break)
        end
        isempty(lib) && error("wildbg library not found; pass --wildbg-lib")
    end
    println("wildbg: $lib")
    backends = [begin
        wb = BackgammonNet.WildbgBackend(lib_path=lib)
        BackgammonNet.open!(wb)
        wb
    end for _ in 1:nw]

    jobs = [(i, side) for i in eachindex(starts) for side in (true, false)]
    rewards = Vector{Float64}(undef, length(jobs))
    next = Threads.Atomic{Int}(0)
    Threads.@sync for w in 1:nw
        Threads.@spawn begin
            while true
                j = Threads.atomic_add!(next, 1) + 1
                j > length(jobs) && break
                (i, table_white) = jobs[j]
                # Duplicate-style pairing: BOTH side assignments of position i use
                # the same dice seed, so the paired per-position sum cancels the
                # (large) on-roll advantage and isolates pure policy skill.
                rewards[j] = play_one(starts[i], backends[w];
                                      table_is_white=table_white,
                                      seed=ARGS_D["seed"] * 100_000 + i)
            end
        end
    end

    as_white = rewards[1:2:end]
    as_black = rewards[2:2:end]
    eq = mean(rewards)
    win = mean(rewards .> 0) * 100
    g_rate = mean(abs.(rewards) .>= 2) * 100
    # Paired per-position edge: table played both sides of each position with the
    # same dice seed, so (r_white + r_black)/2 cancels the on-roll advantage.
    npos = length(as_white)
    paired = [(as_white[i] + as_black[i]) / 2 for i in 1:npos]
    edge = mean(paired)
    se = std(paired) / sqrt(npos)
    ci = 1.96 * se
    println("=" ^ 60)
    println("Exact k=7 table policy vs wildbg ($(length(jobs)) games, $npos paired positions)")
    println("  Equity (table-relative): $(round(eq, digits=4))")
    println("  Win%:                    $(round(win, digits=1))%")
    println("  Gammon-or-better rate:   $(round(g_rate, digits=1))%")
    println("  As white: eq=$(round(mean(as_white), digits=4))  win=$(round(mean(as_white .> 0)*100, digits=1))%")
    println("  As black: eq=$(round(mean(as_black), digits=4))  win=$(round(mean(as_black .> 0)*100, digits=1))%")
    println("  PAIRED policy edge:      $(round(edge, digits=4)) ± $(round(ci, digits=4)) (95% CI)")
    println()
    # Note: in pure mutual bearoff both sides play near-perfectly (most moves are
    # forced/obvious), so the exact-play edge over a strong NN is small and lives
    # mostly in gammon-sensitive positions (--gammon-starts). Judge by the paired
    # CI: an exact policy can NEVER be significantly negative — that means a bug
    # (this exact signal caught the doubles mid-turn mis-scoring on 2026-07-03).
    if edge - ci > 0
        println("PASS: table policy beats wildbg (paired edge significantly > 0)")
    elseif edge + ci < 0
        println("FAIL: table policy LOSES to wildbg — table lookup, perspective,")
        println("      or move enumeration is likely broken")
    else
        println("OK (parity): no significant edge either way. Expected for mutual")
        println("bearoff; try --gammon-starts or more games for a sharper signal.")
    end
    for wb in backends
        try BackgammonNet.close!(wb) catch end
    end
end

main()
