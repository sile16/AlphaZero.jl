# Combined bear-off dispatch — prefer EXACT k=7, else the n=18 one-sided race table.
#
# The two tables are COMPLEMENTARY:
#   - k=7 two-sided : deep bear-off only (both sides ≤7-pt home) — EXACT (optimal play)
#   - n=18 one-sided: broad race band (both sides ≤18-pt frame) — approximate
#                     (efficiency personality; MAE ~0.001 vs exact on the overlap)
# The k=7 domain is a strict subset of the one-sided domain, so a single lookup
# gives the EXACT value in the deep endgame and the approximate value across the
# whole race band. Dispatch order: is_bearoff_position → k=7, else is_onesided_race
# → one-sided, else not covered.
#
# Requires BearoffK7 + BearoffOneSided loaded and bearoff_eval_common.jl included
# (for the generic bearoff_lookup / bearoff_covers / bearoff_equity). Everything
# downstream is duck-typed on `.pW/.pWG/.pLG`, so BearoffResult and OneSidedResult
# are interchangeable.

# ── one-sided methods (drop-in for the generics defined in bearoff_eval_common) ──
bearoff_lookup(table::BearoffOneSided.OneSidedTable, game) = BearoffOneSided.lookup(table, game)
bearoff_covers(table::BearoffOneSided.OneSidedTable, p0::UInt128, p1::UInt128) =
    BearoffOneSided.is_onesided_race(table, p0, p1)

# ── the combined table ──────────────────────────────────────────────────────────
"""
    CombinedBearoff(k7, onesided)

Holds both back-ends; either may be `nothing` (e.g. only the race table copied to a
machine). Use `bearoff_covers(t, p0, p1)` to gate and `bearoff_lookup(t, game)` to
evaluate — exact k=7 is preferred wherever it applies.
"""
struct CombinedBearoff
    k7::Union{Nothing, BearoffK7.BearoffTable}
    onesided::Union{Nothing, BearoffOneSided.OneSidedTable}
end

"""Covered iff either sub-table covers the position."""
function bearoff_covers(t::CombinedBearoff, p0::UInt128, p1::UInt128)
    (t.k7 !== nothing && BearoffK7.is_bearoff_position(p0, p1)) && return true
    (t.onesided !== nothing && BearoffOneSided.is_onesided_race(t.onesided, p0, p1)) && return true
    return false
end

"""Exact k=7 first (its domain ⊂ one-sided's), else the one-sided race table.
Assumes `bearoff_covers(t, p0, p1)` — errors if the position is in neither."""
function bearoff_lookup(t::CombinedBearoff, game)
    p0 = game.p0; p1 = game.p1
    if t.k7 !== nothing && BearoffK7.is_bearoff_position(p0, p1)
        return BearoffK7.lookup(t.k7, game)          # EXACT
    elseif t.onesided !== nothing && BearoffOneSided.is_onesided_race(t.onesided, p0, p1)
        return BearoffOneSided.lookup(t.onesided, game)  # approximate, broader
    else
        error("bearoff_lookup(CombinedBearoff): position covered by neither table — " *
              "gate on bearoff_covers(t, p0, p1) first")
    end
end

"""
    load_combined_bearoff(; k7_dir=nothing, onesided_dir=nothing) -> CombinedBearoff

Load whichever tables are present locally (each optional). Mirrors the existing
local-preferred resolver; a `nothing` dir or missing directory yields a `nothing`
sub-table rather than an error, so a machine with only the race table still works.
"""
function load_combined_bearoff(; k7_dir=nothing, onesided_dir=nothing)
    k7 = (k7_dir !== nothing && isdir(k7_dir) &&
          isfile(joinpath(k7_dir, "bearoff_k7_c14.bin"))) ? BearoffK7.BearoffTable(k7_dir) : nothing
    os = (onesided_dir !== nothing && isdir(onesided_dir) &&
          isfile(joinpath(onesided_dir, "onesided_all.bin"))) ? BearoffOneSided.OneSidedTable(onesided_dir) : nothing
    return CombinedBearoff(k7, os)
end
