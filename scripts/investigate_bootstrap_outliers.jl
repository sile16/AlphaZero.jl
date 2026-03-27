#!/usr/bin/env julia
"""
Investigate the large-error positions from bootstrap vs bearoff table comparison.

Dumps detailed info for the worst positions: board state, dice, BGBlitz values,
exact table values, and possible explanations.

Run on Jarvis:
    julia --project scripts/investigate_bootstrap_outliers.jl
"""

using BackgammonNet
using Serialization
using Printf
using Statistics
using StaticArrays

# ── Load BearoffK6 ──────────────────────────────────────────────────────

const BEAROFF_SRC = joinpath(dirname(dirname(pathof(BackgammonNet))), "src", "bearoff_k6.jl")
include(BEAROFF_SRC)
using .BearoffK6

const TABLE_DIR = joinpath(dirname(BEAROFF_SRC), "..", "tools", "bearoff_twosided", "bearoff_k6_twosided")
const TABLE = BearoffTable(TABLE_DIR)
println("Bearoff table loaded")

# ── Helpers ──────────────────────────────────────────────────────────────

function equity_conditional(pw, gw, bgw, gl, bgl)
    pw * (1 + gw + bgw) - (1 - pw) * (1 + gl + bgl)
end

function board_summary(game::BackgammonGame)
    # Extract home board positions for mover and opponent
    cp = Int(game.current_player)
    mover_pos = BearoffK6.extract_home_position(cp == 0 ? game.p0 : game.p1, cp)
    opp_pos = BearoffK6.extract_home_position(cp == 0 ? game.p1 : game.p0, 1 - cp)
    mover_total = sum(mover_pos)
    opp_total = sum(opp_pos)
    mover_off = 15 - mover_total
    opp_off = 15 - opp_total
    return (; mover_pos, opp_pos, mover_total, opp_total, mover_off, opp_off, cp)
end

function print_position(game::BackgammonGame; prefix="")
    bs = board_summary(game)
    dice = (Int(game.dice[1]), Int(game.dice[2]))
    remaining = Int(game.remaining_actions)
    is_chance = BackgammonNet.is_chance_node(game)
    phase = is_chance ? "CHANCE (pre-dice)" : "DECISION (post-dice)"

    println("$(prefix)Phase: $phase | Player: P$(bs.cp) | Dice: $dice | Remaining: $remaining")
    println("$(prefix)Mover home: $(bs.mover_pos) ($(bs.mover_total) on board, $(bs.mover_off) off)")
    println("$(prefix)Opp   home: $(bs.opp_pos) ($(bs.opp_total) on board, $(bs.opp_off) off)")

    # c14 vs c15
    i_m = BearoffK6.board_to_index(bs.cp == 0 ? game.p0 : game.p1, bs.cp)
    i_o = BearoffK6.board_to_index(bs.cp == 0 ? game.p1 : game.p0, 1 - bs.cp)
    is_c15 = !(min(i_m, i_o) < BearoffK6.N_LE14 && max(i_m, i_o) < BearoffK6.N_LE14)
    println("$(prefix)Table region: $(is_c15 ? "c15 (gammons possible)" : "c14 (no gammons)")")
end

# ── Load bootstrap ──────────────────────────────────────────────────────

const BOOTSTRAP_PATH = length(ARGS) >= 1 ? ARGS[1] :
    joinpath(dirname(dirname(pathof(BackgammonNet))),
             "data", "bootstrap", "bootstrap_5000g_bgblitz1ply.jls")

println("Loading bootstrap from: $BOOTSTRAP_PATH")
flush(stdout)
raw_data = deserialize(BOOTSTRAP_PATH)

states = raw_data.states
equities_pre = raw_data.equity_preroll
equities_post = raw_data.equity_postroll
values = raw_data.values
n_total = length(states)
println("  $n_total samples loaded")

# ── Find bearoff positions and compute errors ───────────────────────────

println("\nScanning all bearoff positions...")
flush(stdout)

struct ErrorRecord
    idx::Int
    post_eq_diff::Float64
    post_pw_diff::Float64
    pre_eq_diff::Float64
    pre_pw_diff::Float64
    bg_post::NTuple{5,Float64}
    bg_pre::NTuple{5,Float64}
    tbl::NTuple{5,Float64}
    tbl_eq::Float64
    bg_post_eq::Float64
    bg_pre_eq::Float64
end

records = ErrorRecord[]

for i in 1:n_total
    s = states[i]
    s isa BackgammonGame || continue
    BearoffK6.is_bearoff_position(s.p0, s.p1) || continue

    r = BearoffK6.lookup(TABLE, s)
    tbl = (Float64(r.p_win), Float64(r.p_gammon_win), Float64(r.p_bg_win),
           Float64(r.p_gammon_loss), Float64(r.p_bg_loss))
    tbl_eq = equity_conditional(tbl...)

    bg_post = Float64.(equities_post[i])
    bg_pre = Float64.(equities_pre[i])
    bg_post_eq = equity_conditional(bg_post...)
    bg_pre_eq = equity_conditional(bg_pre...)

    push!(records, ErrorRecord(i,
        abs(bg_post_eq - tbl_eq), abs(bg_post[1] - tbl[1]),
        abs(bg_pre_eq - tbl_eq), abs(bg_pre[1] - tbl[1]),
        bg_post, bg_pre, tbl, tbl_eq, bg_post_eq, bg_pre_eq))
end

println("  $(length(records)) bearoff positions found")

# ════════════════════════════════════════════════════════════════════════
# POST-ROLL outliers (should be small — both evaluating same position)
# ════════════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("TOP 20 POST-ROLL OUTLIERS (BGBlitz post-roll vs bearoff table)")
println("These should be small — both evaluate the same specific position")
println("="^80)

sort!(records, by=r -> r.post_eq_diff, rev=true)

for (rank, rec) in enumerate(records[1:min(20, length(records))])
    s = states[rec.idx]
    println("\n--- #$rank | |Δ equity| = $(round(rec.post_eq_diff, digits=6)) ---")
    print_position(s, prefix="  ")

    @printf("  BGBlitz post:  pw=%.6f gw=%.6f bgw=%.6f gl=%.6f bgl=%.6f → eq=%.6f\n",
            rec.bg_post..., rec.bg_post_eq)
    @printf("  Bearoff table: pw=%.6f gw=%.6f bgw=%.6f gl=%.6f bgl=%.6f → eq=%.6f\n",
            rec.tbl..., rec.tbl_eq)
    @printf("  |Δ pw|=%.6f  |Δ gw|=%.6f  |Δ gl|=%.6f  |Δ eq|=%.6f\n",
            rec.post_pw_diff,
            abs(rec.bg_post[2] - rec.tbl[2]),
            abs(rec.bg_post[4] - rec.tbl[4]),
            rec.post_eq_diff)

    # Check: is this a near-terminal position?
    actions = BackgammonNet.legal_actions(s)
    n_terminal = 0
    for a in actions
        gc = BackgammonNet.clone(s)
        BackgammonNet.apply_action!(gc, a)
        gc.terminated && (n_terminal += 1)
    end
    println("  Legal actions: $(length(actions)) ($n_terminal lead to terminal)")

    # Game outcome value
    @printf("  Game outcome value (player-relative): %.4f\n", rec.bg_pre[1] != rec.bg_post[1] ? values[rec.idx] : values[rec.idx])
end

# ════════════════════════════════════════════════════════════════════════
# PRE-ROLL outliers (expected larger — different concepts at decision nodes)
# ════════════════════════════════════════════════════════════════════════

println("\n\n" * "="^80)
println("TOP 20 PRE-ROLL OUTLIERS (BGBlitz pre-roll vs bearoff table)")
println("At decision nodes: BGBlitz pre-roll = its own pre-dice estimate")
println("                    bearoff table = exact pre-dice average over all dice")
println("="^80)

sort!(records, by=r -> r.pre_eq_diff, rev=true)

for (rank, rec) in enumerate(records[1:min(20, length(records))])
    s = states[rec.idx]
    println("\n--- #$rank | |Δ equity| = $(round(rec.pre_eq_diff, digits=6)) ---")
    print_position(s, prefix="  ")

    @printf("  BGBlitz pre:   pw=%.6f gw=%.6f bgw=%.6f gl=%.6f bgl=%.6f → eq=%.6f\n",
            rec.bg_pre..., rec.bg_pre_eq)
    @printf("  BGBlitz post:  pw=%.6f gw=%.6f bgw=%.6f gl=%.6f bgl=%.6f → eq=%.6f\n",
            rec.bg_post..., rec.bg_post_eq)
    @printf("  Bearoff table: pw=%.6f gw=%.6f bgw=%.6f gl=%.6f bgl=%.6f → eq=%.6f\n",
            rec.tbl..., rec.tbl_eq)
    @printf("  |Δ pre-post pw|=%.6f  |Δ pre-table pw|=%.6f\n",
            abs(rec.bg_pre[1] - rec.bg_post[1]),
            rec.pre_pw_diff)

    # Check dice — is this an extreme dice roll that makes pre/post diverge?
    dice = (Int(s.dice[1]), Int(s.dice[2]))
    is_doubles = dice[1] == dice[2]
    println("  Doubles: $is_doubles | Dice sum: $(dice[1]+dice[2])")
end

# ════════════════════════════════════════════════════════════════════════
# Distribution analysis
# ════════════════════════════════════════════════════════════════════════

println("\n\n" * "="^80)
println("ERROR DISTRIBUTION ANALYSIS")
println("="^80)

# Post-roll error histogram
post_eq_errs = [r.post_eq_diff for r in records]
println("\nPost-roll |Δ equity| distribution:")
for threshold in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    n = count(e -> e > threshold, post_eq_errs)
    pct = 100 * n / length(post_eq_errs)
    @printf("  > %.3f: %5d / %d (%.2f%%)\n", threshold, n, length(post_eq_errs), pct)
end

# Pre-roll error histogram
pre_eq_errs = [r.pre_eq_diff for r in records]
println("\nPre-roll |Δ equity| distribution:")
for threshold in [0.01, 0.05, 0.1, 0.5, 1.0, 1.5]
    n = count(e -> e > threshold, pre_eq_errs)
    pct = 100 * n / length(pre_eq_errs)
    @printf("  > %.2f: %5d / %d (%.2f%%)\n", threshold, n, length(pre_eq_errs), pct)
end

# Check if pre-roll errors correlate with position asymmetry
println("\nPre-roll error vs position asymmetry:")
asymmetries = Float64[]
pre_errs = Float64[]
for rec in records
    s = states[rec.idx]
    bs = board_summary(s)
    push!(asymmetries, abs(bs.mover_total - bs.opp_total))
    push!(pre_errs, rec.pre_eq_diff)
end

# Bin by asymmetry
for (lo, hi) in [(0,2), (3,5), (6,8), (9,11), (12,15)]
    mask = lo .<= asymmetries .<= hi
    n = count(mask)
    n == 0 && continue
    errs = pre_errs[mask]
    @printf("  Asymmetry %d-%d: n=%d, mean |Δ eq|=%.4f, max=%.4f\n",
            lo, hi, n, mean(errs), maximum(errs))
end

# Check if pre-roll errors correlate with mover's off count
println("\nPre-roll error vs mover's checkers off:")
for off in 0:5:15
    mask = [board_summary(states[r.idx]).mover_off == off for r in records]
    n = count(mask)
    n == 0 && continue
    errs = pre_errs[mask]
    @printf("  Mover off=%d: n=%d, mean |Δ eq|=%.4f, max=%.4f\n",
            off, n, mean(errs), maximum(errs))
end

println("\n" * "="^80)
println("DONE")
println("="^80)
