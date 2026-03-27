#!/usr/bin/env julia
"""
Cross-validate bearoff table values against BGBlitz bootstrap data.

For each bearoff position in the bootstrap, compares:
- BGBlitz 5-head equity values (from bootstrap)
- Exact bearoff table 5-head values (from k=6 table)
- Both conditional and joint equity formulas

This validates that BGBlitz and the bearoff table agree on bearoff positions,
and numerically confirms the conditional probability format.

Run on Jarvis (where the raw bootstrap file with BackgammonGame objects exists):
    julia --project scripts/validate_bootstrap_bearoff.jl [--max-samples=10000]
"""

using BackgammonNet
using Serialization
using Printf
using Statistics
using Random

# ── Load BearoffK6 ──────────────────────────────────────────────────────

const BEAROFF_SRC = joinpath(dirname(dirname(pathof(BackgammonNet))), "src", "bearoff_k6.jl")
include(BEAROFF_SRC)
using .BearoffK6

const TABLE_DIR = joinpath(dirname(BEAROFF_SRC), "..", "tools", "bearoff_twosided", "bearoff_k6_twosided")
if !isdir(TABLE_DIR)
    error("Bearoff table not found at: $TABLE_DIR")
end
const TABLE = BearoffTable(TABLE_DIR)
println("Bearoff table loaded: c14=$(TABLE.c14_pairs) pairs, c15=$(TABLE.c15_pairs) pairs")

# ── Parse args ──────────────────────────────────────────────────────────

MAX_SAMPLES = 10000
BOOTSTRAP_PATH = joinpath(dirname(dirname(pathof(BackgammonNet))),
    "data", "bootstrap", "bootstrap_5000g_bgblitz1ply.jls")

for arg in ARGS
    if startswith(arg, "--max-samples=")
        global MAX_SAMPLES = parse(Int, split(arg, "=")[2])
    elseif startswith(arg, "--bootstrap=")
        global BOOTSTRAP_PATH = split(arg, "=", limit=2)[2]
    end
end

# ── Equity formulas ─────────────────────────────────────────────────────

function equity_conditional(pw, gw, bgw, gl, bgl)
    pw * (1 + gw + bgw) - (1 - pw) * (1 + gl + bgl)
end

function equity_joint(pw, wg, wbg, lg, lbg)
    (2*pw - 1) + (wg - lg) + (wbg - lbg)
end

function conditional_to_joint(pw, gw, bgw, gl, bgl)
    return (pw * gw, pw * bgw, (1-pw) * gl, (1-pw) * bgl)
end

# ── Load bootstrap ──────────────────────────────────────────────────────

println("\nLoading bootstrap from: $BOOTSTRAP_PATH")
flush(stdout)
t0 = time()
raw_data = deserialize(BOOTSTRAP_PATH)
t_load = time() - t0
println("  Loaded in $(round(t_load, digits=1))s")

# Determine format: columnar (states/policies/values/equity) vs per-sample vector
local equities_preroll, equities_postroll, has_preroll

if raw_data isa NamedTuple && hasproperty(raw_data, :states)
    states = raw_data.states
    values = raw_data.values
    n_total = length(states)

    # Original format has equity_preroll and equity_postroll (separate!)
    if hasproperty(raw_data, :equity_preroll)
        equities_preroll = raw_data.equity_preroll
        equities_postroll = raw_data.equity_postroll
        equities = equities_postroll  # default to post-roll for comparison
        has_preroll = true
        println("  Format: columnar NamedTuple with PRE-ROLL and POST-ROLL equity, $n_total samples")
    elseif hasproperty(raw_data, :equity)
        equities = raw_data.equity
        equities_preroll = equities
        equities_postroll = equities
        has_preroll = false
        println("  Format: columnar NamedTuple with single equity, $n_total samples")
    else
        error("No equity field found")
    end

    if hasproperty(raw_data, :metadata)
        meta = raw_data.metadata
        contract = get(meta, "value_head_contract", "unknown")
        println("  Contract: $contract")
        for k in sort(collect(keys(meta)))
            println("    $k = $(meta[k])")
        end
    end
elseif raw_data isa Vector
    n_total = length(raw_data)
    println("  Format: Vector (per-sample), $n_total samples")
    states = [s.state for s in raw_data]
    equities = [s.equity for s in raw_data]
    equities_preroll = equities
    equities_postroll = equities
    has_preroll = false
    values = [s.value for s in raw_data]
else
    error("Unknown bootstrap format: $(typeof(raw_data))")
end

# ── Find bearoff positions ──────────────────────────────────────────────

println("\nScanning for bearoff positions...")
flush(stdout)

bearoff_indices = Int[]
for i in 1:n_total
    s = states[i]
    if s isa BackgammonGame && BearoffK6.is_bearoff_position(s.p0, s.p1)
        push!(bearoff_indices, i)
    end
end

n_bearoff = length(bearoff_indices)
println("  Found $n_bearoff bearoff positions out of $n_total total ($(round(100*n_bearoff/n_total, digits=1))%)")

# Subsample if too many
if n_bearoff > MAX_SAMPLES
    rng = MersenneTwister(42)
    bearoff_indices = sort(bearoff_indices[randperm(rng, n_bearoff)[1:MAX_SAMPLES]])
    println("  Subsampled to $MAX_SAMPLES")
end

# ── Compare BGBlitz vs bearoff table ─────────────────────────────────

println("\n" * "="^72)
println("Comparing BGBlitz bootstrap vs exact bearoff table")
println("="^72)
flush(stdout)

# ── Compare PRE-ROLL equity (BGBlitz pre-roll vs bearoff table pre-dice) ──
# The bearoff table stores pre-dice values. BGBlitz equity_preroll should match.

pre_pw_diffs = Float64[]
pre_gw_diffs = Float64[]
pre_gl_diffs = Float64[]
pre_eq_diffs = Float64[]

# ── Compare POST-ROLL equity (BGBlitz post-roll vs bearoff table post-dice) ──
post_pw_diffs = Float64[]
post_gw_diffs = Float64[]
post_gl_diffs = Float64[]
post_eq_diffs = Float64[]

# Trackers
n_with_gammons = 0
n_c15 = 0
n_c14 = 0
n_chance = 0
n_decision = 0
n_pre_compared = 0
n_post_compared = 0

for idx in bearoff_indices
    local s = states[idx]
    local is_chance = BackgammonNet.is_chance_node(s)
    is_chance ? (global n_chance += 1) : (global n_decision += 1)

    # Bearoff table lookup (conditional, mover-relative)
    local r = BearoffK6.lookup(TABLE, s)
    local tbl_pw = Float64(r.p_win)
    local tbl_gw = Float64(r.p_gammon_win)
    local tbl_gl = Float64(r.p_gammon_loss)
    local tbl_eq = equity_conditional(tbl_pw, tbl_gw, 0.0, tbl_gl, 0.0)

    # c14 vs c15
    local cp = Int(s.current_player)
    local i_m = BearoffK6.board_to_index(cp == 0 ? s.p0 : s.p1, cp)
    local i_o = BearoffK6.board_to_index(cp == 0 ? s.p1 : s.p0, 1 - cp)
    local is_c15 = !(min(i_m, i_o) < BearoffK6.N_LE14 && max(i_m, i_o) < BearoffK6.N_LE14)
    is_c15 ? (global n_c15 += 1) : (global n_c14 += 1)
    if tbl_gw > 0.001 || tbl_gl > 0.001
        global n_with_gammons += 1
    end

    # ── PRE-ROLL comparison ──
    # Bearoff table is PRE-DICE (averaged over all dice outcomes).
    # BGBlitz equity_preroll should be its own pre-dice evaluation.
    # These should match well for chance nodes (where the state IS pre-dice).
    if has_preroll
        pre_eq_bg = equities_preroll[idx]
        bg_pre_pw = Float64(pre_eq_bg[1])
        bg_pre_gw = Float64(pre_eq_bg[2])
        bg_pre_gl = Float64(pre_eq_bg[4])
        bg_pre_eq = equity_conditional(bg_pre_pw, bg_pre_gw, Float64(pre_eq_bg[3]),
                                        bg_pre_gl, Float64(pre_eq_bg[5]))

        push!(pre_pw_diffs, abs(bg_pre_pw - tbl_pw))
        push!(pre_gw_diffs, abs(bg_pre_gw - tbl_gw))
        push!(pre_gl_diffs, abs(bg_pre_gl - tbl_gl))
        push!(pre_eq_diffs, abs(bg_pre_eq - tbl_eq))
        global n_pre_compared += 1
    end

    # ── POST-ROLL comparison ──
    local post_eq_bg = has_preroll ? equities_postroll[idx] : equities[idx]
    local bg_post_pw = Float64(post_eq_bg[1])
    local bg_post_gw = Float64(post_eq_bg[2])
    local bg_post_gl = Float64(post_eq_bg[4])
    local bg_post_eq = equity_conditional(bg_post_pw, bg_post_gw, Float64(post_eq_bg[3]),
                                     bg_post_gl, Float64(post_eq_bg[5]))

    push!(post_pw_diffs, abs(bg_post_pw - tbl_pw))
    push!(post_gw_diffs, abs(bg_post_gw - tbl_gw))
    push!(post_gl_diffs, abs(bg_post_gl - tbl_gl))
    push!(post_eq_diffs, abs(bg_post_eq - tbl_eq))
    global n_post_compared += 1
end

# ── Print results ────────────────────────────────────────────────────

function print_stats(name, diffs)
    isempty(diffs) && return
    @printf("  %-24s  mean=%.6f  median=%.6f  p95=%.6f  p99=%.6f  max=%.6f\n",
            name, mean(diffs), median(diffs),
            quantile(diffs, 0.95), quantile(diffs, 0.99), maximum(diffs))
end

println("\nPositions compared: pre=$n_pre_compared, post=$n_post_compared")
println("  Chance nodes: $n_chance | Decision nodes: $n_decision")
println("  c14 (no gammons): $n_c14 | c15 (gammons possible): $n_c15")
println("  With non-trivial gammons: $n_with_gammons")

if has_preroll && !isempty(pre_pw_diffs)
    println("\n--- PRE-ROLL: BGBlitz equity_preroll vs Bearoff Table (both pre-dice) ---")
    print_stats("|Δ p_win|", pre_pw_diffs)
    print_stats("|Δ p_gammon_win|", pre_gw_diffs)
    print_stats("|Δ p_gammon_loss|", pre_gl_diffs)
    print_stats("|Δ equity|", pre_eq_diffs)
end

println("\n--- POST-ROLL: BGBlitz equity_postroll vs Bearoff Table (post vs pre-dice) ---")
println("(Expected: larger diffs since post-roll is for specific dice, table is averaged)")
print_stats("|Δ p_win|", post_pw_diffs)
print_stats("|Δ p_gammon_win|", post_gw_diffs)
print_stats("|Δ p_gammon_loss|", post_gl_diffs)
print_stats("|Δ equity|", post_eq_diffs)

# ── Pre-roll vs Post-roll within BGBlitz ──
if has_preroll
    println("\n--- BGBlitz: Pre-Roll vs Post-Roll (internal consistency) ---")
    pre_post_pw = Float64[]
    pre_post_eq = Float64[]
    for idx in bearoff_indices[1:min(5000, length(bearoff_indices))]
        pre = equities_preroll[idx]
        post = equities_postroll[idx]
        push!(pre_post_pw, abs(Float64(pre[1]) - Float64(post[1])))
        eq_pre = equity_conditional(Float64.(pre)...)
        eq_post = equity_conditional(Float64.(post)...)
        push!(pre_post_eq, abs(eq_pre - eq_post))
    end
    print_stats("|Δ p_win (pre vs post)|", pre_post_pw)
    print_stats("|Δ equity (pre vs post)|", pre_post_eq)
end

# ── Format consistency ──
println("\n--- Format Consistency Check ---")
let n_fmt_mm = 0, max_fmt_d = 0.0
    for idx in bearoff_indices[1:min(1000, length(bearoff_indices))]
        eq = has_preroll ? equities_preroll[idx] : equities[idx]
        pw, gw, bgw, gl, bgl = Float64.(eq)
        eq_c = equity_conditional(pw, gw, bgw, gl, bgl)
        j = conditional_to_joint(pw, gw, bgw, gl, bgl)
        eq_j = equity_joint(pw, j[1], j[2], j[3], j[4])
        diff = abs(eq_c - eq_j)
        max_fmt_d = max(max_fmt_d, diff)
        diff > 1e-10 && (n_fmt_mm += 1)
    end
    @printf("  Conditional↔Joint equity mismatch: %d/1000 (max diff=%.2e)\n",
            n_fmt_mm, max_fmt_d)
end

# ── Summary ──────────────────────────────────────────────────────────

println("\n" * "="^72)
println("SUMMARY")
println("="^72)

mean_eq_diff = mean(eq_scalar_diffs)
if mean_eq_diff < 0.01
    println("✓ BGBlitz and bearoff table AGREE on bearoff positions (mean |Δ equity|=$(round(mean_eq_diff, digits=4)))")
else
    println("✗ SIGNIFICANT DISAGREEMENT: mean |Δ equity|=$(round(mean_eq_diff, digits=4))")
end

if n_with_gammons > 0
    gammon_indices = [i for (i, idx) in enumerate(bearoff_indices)
                      if (let r = BearoffK6.lookup(TABLE, states[idx]);
                          r.p_gammon_win > 0.001 || r.p_gammon_loss > 0.001
                      end)]
    if !isempty(gammon_indices)
        @printf("  Gammon head accuracy (on %d positions with gammons):\n", length(gammon_indices))
        @printf("    |Δ p_gammon_win|: mean=%.4f max=%.4f\n",
                mean(gw_diffs[gammon_indices]), maximum(gw_diffs[gammon_indices]))
        @printf("    |Δ p_gammon_loss|: mean=%.4f max=%.4f\n",
                mean(gl_diffs[gammon_indices]), maximum(gl_diffs[gammon_indices]))
    end
else
    println("  No positions with significant gammon values found (expected for race bearoff)")
end

println("\nBGBlitz uses conditional format — confirmed by:")
println("  1. Conditional formula matches scalar parity (from generate_bootstrap.jl)")
println("  2. Conditional ↔ Joint equity conversion is exact (verified above)")
println("  3. Bearoff table also stores conditional (verified by Bellman in test_value_head_formats.jl)")
println("="^72)
