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
        "--policy"
            arg_type = String
            default = "table"
            help = "Table-side policy: 'table' (pure exact-table argmax) or 'mcts' (BatchedMCTS with exact bearoff evaluator)"
        "--objective"
            arg_type = String
            default = "money"
            help = "Scoring/selection objective: money|dmp|gg|gs (weights over plain/gammon win/loss). Applied to table move selection AND game scoring; wildbg always plays money."
        "--mcts-iters"
            arg_type = Int
            default = 30
            help = "MCTS simulations per turn (--policy=mcts only)"
        "--cpuct"
            arg_type = Float64
            default = 2.0
            help = "MCTS exploration constant (--policy=mcts only)"
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

# ── Objective + policy configuration ────────────────────────────────────

const OBJECTIVE = Symbol(ARGS_D["objective"])
const POLICY = Symbol(ARGS_D["policy"])

# Weight vector (w_pw, w_gw, w_pl, w_gl) over (plain win, gammon win, plain
# loss, gammon loss). Applied to BOTH table move selection and game scoring.
# wildbg always plays money — for non-money objectives it optimizes the wrong
# thing, which is the point of these runs.
const OBJ_WEIGHTS_MAP = Dict(
    :money => (1.0, 2.0, -1.0, -2.0),  # current behavior (bearoff has no backgammons)
    :dmp   => (1.0, 1.0, -1.0, -1.0),  # double match point: win is win, gammon irrelevant
    :gg    => (1.0, 2.0, -1.0, -1.0),  # gammon-go: value winning gammons, ignore losing gammons
    :gs    => (1.0, 1.0, -1.0, -2.0),  # gammon-save: ignore winning gammons, fear losing gammons
)
haskey(OBJ_WEIGHTS_MAP, OBJECTIVE) || error("Unknown --objective=$(OBJECTIVE); use money|dmp|gg|gs")
POLICY in (:table, :mcts) || error("Unknown --policy=$(POLICY); use table|mcts")
const OBJ_WEIGHTS = OBJ_WEIGHTS_MAP[OBJECTIVE]
const MONEY_WEIGHTS = OBJ_WEIGHTS_MAP[:money]
const GSPEC = GameSpec()

"""Score a raw table-relative game reward (±1/±2) in objective units."""
function score_reward(raw_r::Float64, weights)::Float64
    weights == MONEY_WEIGHTS && return raw_r  # identity: raw reward IS money value
    ar = abs(raw_r)
    if raw_r > 0
        return ar >= 2 ? weights[2] : weights[1]
    elseif raw_r < 0
        return ar >= 2 ? weights[4] : weights[3]
    else
        return 0.0
    end
end

# ── Shared table move-selection primitives (used by both TableAgent and the
#    MCTS move-agreement instrumentation) ─────────────────────────────────

"""Exact objective value of playing `action` from decision node `bg` (raw
points, current-player perspective; handles doubles mid-turn recursion)."""
function table_action_value(bg, action, weights)::Float64
    work = BackgammonNet.clone(bg)
    BackgammonNet.apply_action!(work, action)
    return bearoff_turn_value(TABLE, work, Int(bg.current_player); weights=weights)
end

"""Argmax-by-exact-objective-value legal move at decision node `bg`.
Returns (best_action, best_value). Iterates `legal_actions` order; ties break to
the first-seen action (same convention the old TableAgent used)."""
function table_best_action(bg, weights)
    actions = BackgammonNet.legal_actions(bg)
    mover = Int(bg.current_player)
    work = BackgammonNet.clone(bg)
    best_a = actions[1]
    best_v = -Inf
    for a in actions
        BackgammonNet.copy_state!(work, bg)
        BackgammonNet.apply_action!(work, a)
        v = bearoff_turn_value(TABLE, work, mover; weights=weights)
        if v > best_v
            best_v = v
            best_a = a
        end
    end
    return (best_a, best_v)
end

# ── Table policy agent ──────────────────────────────────────────────────

"""Pure exact-table policy: at a decision node, pick the legal move maximizing
the exact post-move value under the configured objective weights. Terminal moves
are scored from `reward` (carries the gammon multiplier); non-terminal moves from
the opponent's negated pre-dice table value. Raw-points scale throughout."""
struct TableAgent <: GameLoop.GameAgent
    weights::NTuple{4, Float64}
end

GameLoop.create_player(::TableAgent) = nothing

function GameLoop.select_action(agent::TableAgent, ::Nothing, env)
    bg = env.game
    BearoffK7.is_bearoff_position(bg.p0, bg.p1) ||
        error("TableAgent reached a non-bearoff position — starts must be mutual bearoff")
    best_action, _ = table_best_action(bg, agent.weights)
    return (best_action, Float32[], Int[])
end

# ── MCTS policy agent (BatchedMCTS + exact bearoff evaluator) ────────────

"""Uniform-prior, V=0 oracle. In pure mutual bearoff EVERY tree node is a
bearoff position, so the bearoff evaluator supplies exact values and this
oracle's V is never used; the uniform prior only spreads initial exploration.
(Kept valid for safety in case a non-bearoff leaf is ever reached.)"""
function uniform_oracle(state)
    acts = GI.available_actions(GI.init(GSPEC, state))
    n = max(1, length(acts))
    return (fill(Float32(1.0 / n), n), 0.0f0)
end

"""Objective-aware BatchedMCTS bearoff evaluator, mirroring
selfplay_client.jl's `make_bearoff_evaluator`: chance node → pre-dice table
lookup; decision node → exact best-move value. Returns WHITE-relative equity
NORMALIZED /3 (documented, monotonic so argmax is unaffected; consistent with
the money-scaled reward path via reward_scale), or `nothing` if not bearoff."""
function make_objective_bearoff_evaluator(table, weights)
    return function(game_env)
        bg = game_env.game
        BearoffK7.is_bearoff_position(bg.p0, bg.p1) || return nothing
        if BackgammonNet.is_chance_node(bg)
            r = BearoffK7.lookup(table, bg)
            mover_eq = _bearoff_objective_value(r, weights) / 3.0
            return bg.current_player == 0 ? mover_eq : -mover_eq
        end
        acts = BackgammonNet.legal_actions(bg)
        isempty(acts) && return nothing
        best = bearoff_best_move_value(table, bg; weights=weights)
        best == -Inf && return nothing
        best /= 3.0
        return bg.current_player == 0 ? best : -best
    end
end

const MCTS_PARAMS = MctsParams(
    num_iters_per_turn = ARGS_D["mcts_iters"],
    cpuct = ARGS_D["cpuct"],
    gamma = 1.0,
    temperature = ConstSchedule(0.0),
    dirichlet_noise_ϵ = 0.0,
    dirichlet_noise_α = 0.3,   # unused (ϵ = 0)
    prior_temperature = 1.0,
    chance_mode = :passthrough,
)

# One immutable agent config shared across worker threads; play_game creates a
# fresh per-game BatchedMctsPlayer (own tree) so this is thread-safe. The oracle
# and evaluator are pure and read TABLE (read-only) only.
const MCTS_AGENT = GameLoop.MctsAgent(
    uniform_oracle, nothing, MCTS_PARAMS, 8, GSPEC;
    bearoff_eval = make_objective_bearoff_evaluator(TABLE, OBJ_WEIGHTS))

# The table-side agent used this run.
const TABLE_AGENT = POLICY == :mcts ? MCTS_AGENT : TableAgent(OBJ_WEIGHTS)

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

"""Return (starts, s15s): `starts` is a vector of (p0,p1,cp); `s15s[i]` is which
side (0/1) has all 15 checkers on board (gammon-live), or -1 if neither/both.
For gg/gs objectives with --gammon-starts, only EXACTLY-one-side-15 positions are
kept so each paired position contributes one gammon-favorite orientation (table
plays the side ahead → can win a gammon) and one gammon-underdog orientation
(table plays the 15-on-board side → must save the gammon)."""
function build_start_positions(n::Int)
    src = find_positions_file()
    println("Race starts: $src")
    tuples = deserialize(src)
    rng = MersenneTwister(ARGS_D["seed"])
    seen = Set{Tuple{UInt128, UInt128, Int8}}()
    starts = Tuple{UInt128, UInt128, Int8}[]
    s15s = Int[]
    need_exactly_one = OBJECTIVE in (:gg, :gs)
    for tup in shuffle(rng, tuples)
        pos = rollout_to_bearoff(tup, rng)
        pos === nothing && continue
        pos in seen && continue
        n_on0 = sum(Int((pos[1] >> (i * 4)) & 0xF) for i in 1:24)
        n_on1 = sum(Int((pos[2] >> (i * 4)) & 0xF) for i in 1:24)
        if ARGS_D["gammon_starts"]
            if need_exactly_one
                # Exactly one side gammon-live: needed to orient favorite/underdog.
                ((n_on0 == 15) ⊻ (n_on1 == 15)) || continue
            else
                # Gammon live: at least one side has all 15 checkers on board.
                (n_on0 == 15 || n_on1 == 15) || continue
            end
        end
        s15 = n_on0 == 15 ? (n_on1 == 15 ? -1 : 0) : (n_on1 == 15 ? 1 : -1)
        push!(seen, pos)
        push!(starts, pos)
        push!(s15s, s15)
        length(starts) >= n && break
    end
    length(starts) >= n || @warn "Only found $(length(starts)) / $n bearoff starts"
    return (starts, s15s)
end

# ── Play ────────────────────────────────────────────────────────────────

# Instrument table-side decisions when it can tell us something: MCTS mode
# (move-agreement vs pure table) or a non-money objective (how often the
# objective changes the optimal move vs money / vs wildbg).
const INSTRUMENT = (POLICY == :mcts) || (OBJECTIVE != :money)

"""Post-process a recorded game trace: for every non-trivial TABLE-side decision,
compare the played move to the exact table optimum under the objective.

Returns a NamedTuple:
- `decisions`   : # non-trivial (>1 legal) table decisions
- `agree`       : # where the MCTS-chosen move == table objective argmax (mcts only)
- `gaps`        : exact-value gap Δ = best_v − chosen_v for each disagreement (mcts only)
- `obj_changes` : # where the exact objective argmax != the exact MONEY argmax
                  (i.e. positions where optimizing this objective actually changes
                  the optimal move vs money — the honest "does the objective matter"
                  signal). We use the EXACT table money-optimum as the money-player
                  reference instead of wildbg: wildbg's move API is stateful (it
                  caches doubles sub-move plans and must be driven in-sequence, so
                  it cannot be safely queried out-of-band on arbitrary trace states),
                  and the exact money argmax is a noise-free, stronger reference.
"""
function analyze_table_decisions(trace, table_is_white::Bool)
    table_player = table_is_white ? 0 : 1
    decisions = 0
    agree = 0
    gaps = Float64[]
    obj_changes = 0
    for te in trace
        te.is_chance && continue
        te.player == table_player || continue
        length(te.legal_actions) > 1 || continue
        bg = te.state  # BackgammonGame at a rolled-dice decision node
        tb_a, tb_v = table_best_action(bg, OBJ_WEIGHTS)
        decisions += 1
        if POLICY == :mcts
            chosen = te.action
            if chosen == tb_a
                agree += 1
            else
                chosen_v = table_action_value(bg, chosen, OBJ_WEIGHTS)
                push!(gaps, tb_v - chosen_v)
            end
        end
        if OBJECTIVE != :money
            mb_a, _ = table_best_action(bg, MONEY_WEIGHTS)
            tb_a != mb_a && (obj_changes += 1)
        end
    end
    return (decisions=decisions, agree=agree, gaps=gaps, obj_changes=obj_changes)
end

function play_one(start, wildbg_backend; table_is_white::Bool, seed::Int)
    p0, p1, cp = start
    rng = MersenneTwister(seed)
    game = BackgammonGame(p0, p1, SVector{2, Int8}(0, 0), Int8(0), cp, false, 0.0f0;
                          obs_type=:minimal_flat)
    env = GameEnv(game, rng)
    wb = GameLoop.ExternalAgent(wildbg_backend)
    w, b = table_is_white ? (TABLE_AGENT, wb) : (wb, TABLE_AGENT)
    result = GameLoop.play_game(w, b, env; rng=rng, temperature_fn=_ -> 0.0,
                                record_trace=INSTRUMENT)
    raw = table_is_white ? result.reward : -result.reward
    stats = INSTRUMENT ? analyze_table_decisions(result.trace, table_is_white) : nothing
    return (raw, stats)
end

function main()
    n = ARGS_D["num_games"]
    starts, s15s = build_start_positions(n)
    println("Config: policy=$(POLICY)  objective=$(OBJECTIVE) weights=$(OBJ_WEIGHTS)  " *
            "instrument=$(INSTRUMENT)")
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
    rewards = Vector{Float64}(undef, length(jobs))   # raw table-relative reward (±1/±2)
    stats = Vector{Any}(undef, length(jobs))
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
                raw, st = play_one(starts[i], backends[w];
                                   table_is_white=table_white,
                                   seed=ARGS_D["seed"] * 100_000 + i)
                rewards[j] = raw
                stats[j] = st
            end
        end
    end

    # jobs are (i, true) then (i, false): as_white = table plays player 0.
    as_white = rewards[1:2:end]
    as_black = rewards[2:2:end]
    npos = length(as_white)

    # Objective-unit scoring (money = identity).
    obj = [score_reward(r, OBJ_WEIGHTS) for r in rewards]
    obj_white = obj[1:2:end]
    obj_black = obj[2:2:end]

    eq = mean(obj)
    win = mean(rewards .> 0) * 100                  # game win% (raw, objective-independent)
    g_rate = mean(abs.(rewards) .>= 2) * 100
    paired = [(obj_white[i] + obj_black[i]) / 2 for i in 1:npos]
    edge = mean(paired)
    se = std(paired) / sqrt(npos)
    ci = 1.96 * se

    println("=" ^ 64)
    println("Exact k=7 table [$(POLICY)] vs wildbg — objective=$(OBJECTIVE) " *
            "($(length(jobs)) games, $npos paired positions)")
    println("  Mean objective value:    $(round(eq, digits=4))")
    println("  Win%:                    $(round(win, digits=1))%")
    println("  Gammon-or-better rate:   $(round(g_rate, digits=1))%")
    println("  As white: obj=$(round(mean(obj_white), digits=4))  win=$(round(mean(as_white .> 0)*100, digits=1))%")
    println("  As black: obj=$(round(mean(obj_black), digits=4))  win=$(round(mean(as_black .> 0)*100, digits=1))%")
    println("  PAIRED policy edge:      $(round(edge, digits=4)) ± $(round(ci, digits=4)) (95% CI, objective units)")

    # ── gg/gs orientation buckets (favorite vs underdog) ──────────────────
    if OBJECTIVE in (:gg, :gs)
        fav = Float64[]     # table plays the side AHEAD (opponent has 15 on board) → can win a gammon
        und = Float64[]     # table plays the 15-on-board side → must save the gammon
        for j in eachindex(jobs)
            (i, table_white) = jobs[j]
            s = s15s[i]
            s == -1 && continue
            table_player = table_white ? 0 : 1
            if table_player == s
                push!(und, obj[j])   # table is the gammon-live underdog
            else
                push!(fav, obj[j])   # table is the gammon-favorite
            end
        end
        _m(v) = isempty(v) ? NaN : mean(v)
        _ci(v) = length(v) > 1 ? 1.96 * std(v) / sqrt(length(v)) : NaN
        println("  Orientation split (objective units):")
        println("    favorite (table ahead, can win gammon): obj=$(round(_m(fav), digits=4)) ± $(round(_ci(fav), digits=4)) (n=$(length(fav)))")
        println("    underdog (table has 15 on board, saves): obj=$(round(_m(und), digits=4)) ± $(round(_ci(und), digits=4)) (n=$(length(und)))")
    end

    # ── MCTS move-agreement instrumentation ───────────────────────────────
    if POLICY == :mcts
        tot_dec = 0; tot_agree = 0; all_gaps = Float64[]
        for st in stats
            st === nothing && continue
            tot_dec += st.decisions
            tot_agree += st.agree
            append!(all_gaps, st.gaps)
        end
        agree_pct = tot_dec == 0 ? NaN : 100 * tot_agree / tot_dec
        ndis = tot_dec - tot_agree
        real_dis = count(g -> g > 1e-9, all_gaps)   # disagreements that are NOT exact ties
        maxg = isempty(all_gaps) ? 0.0 : maximum(all_gaps)
        meang = isempty(all_gaps) ? 0.0 : mean(all_gaps)
        println("  MCTS vs pure-table move agreement:")
        println("    decisions=$(tot_dec)  agree=$(tot_agree) ($(round(agree_pct, digits=2))%)  disagree=$(ndis)")
        println("    disagreement Δ (best−chosen, raw pts): max=$(round(maxg, digits=6))  mean=$(round(meang, digits=6))")
        println("    disagreements with Δ>1e-9 (NON-tie, would indicate a wiring bug): $(real_dis)")
    end

    # ── Objective move-change instrumentation ─────────────────────────────
    # How often does optimizing this objective change the exact-optimal move vs
    # money? If ~0%, the objective edge is genuinely small (wildbg, a money
    # player, would play the same moves) — report that honestly.
    if OBJECTIVE != :money
        tot_dec = 0; tot_objchg = 0
        for st in stats
            st === nothing && continue
            tot_dec += st.decisions
            tot_objchg += st.obj_changes
        end
        chg_pct = tot_dec == 0 ? NaN : 100 * tot_objchg / tot_dec
        println("  Objective effect on move choice (table decisions=$(tot_dec)):")
        println("    obj-argmax != money-argmax (exact): $(tot_objchg) ($(round(chg_pct, digits=2))%)")
    end

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
