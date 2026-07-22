module BearoffTables

import BackgammonNet

export RuntimeTables, parse_table_selection, validate_table_release,
       load_runtime_tables, exact_k7_covers, exact_k7_lookup,
       n15_covers, n15_lookup, n15_turn_value_equity, n15_best_move_value

const TABLE_SELECTIONS = ("none", "k7", "n15", "k7+n15")

"""Normalize the server-owned bearoff table selection."""
function parse_table_selection(value::AbstractString)::String
    selection = lowercase(strip(String(value)))
    selection in TABLE_SELECTIONS || throw(ArgumentError(
        "bearoff table selection must be one of $(join(TABLE_SELECTIONS, ", ")); got $(repr(value))"))
    return selection
end

_uses_k7(selection::AbstractString) = selection in ("k7", "k7+n15")
_uses_n15(selection::AbstractString) = selection in ("n15", "k7+n15")

function _validate_k7_release(dir::AbstractString)
    release = BackgammonNet.K7_TABLE_RELEASE
    files = (
        ("bearoff_k7_c14.bin", release.c14_size),
        ("bearoff_k7_c15.bin", release.c15_size),
    )
    for (name, expected_size) in files
        path = joinpath(dir, name)
        isfile(path) || error("missing configured k7 table file: $path")
        filesize(path) == expected_size || error(
            "$name size mismatch: got $(filesize(path)), expected $expected_size")
    end
    metadata = joinpath(dir, "bearoff_k7_meta.txt")
    isfile(metadata) || error("missing configured k7 metadata: $metadata")
    BackgammonNet.file_hash(metadata) == release.meta_hash || error(
        "bearoff_k7_meta.txt hash does not match the pinned k7 release")
    return Dict{String,Any}(
        "name" => "k7",
        "semantics" => "exact_money_optimal_training_teacher",
        "contract" => String(BackgammonNet.K7_TABLE_IDENTITY["contract"]),
        "c14_bytes" => release.c14_size,
        "c14_pinned_hash" => release.c14_hash,
        "c15_bytes" => release.c15_size,
        "c15_pinned_hash" => release.c15_hash,
        "metadata_hash" => release.meta_hash,
    )
end

function _validate_n15_release(dir::AbstractString)
    release = BackgammonNet.N15_BUNDLE_V3_RELEASE
    header = joinpath(dir, "header.txt")
    isfile(header) || error("missing configured n15 table header: $header")
    filesize(header) == release.header_size || error(
        "n15 header size mismatch: got $(filesize(header)), expected $(release.header_size)")
    BackgammonNet.file_hash(header) == release.header_hash || error(
        "n15 header hash does not match the pinned release")

    records = Dict{String,Any}[]
    for expected in release.files
        path = joinpath(dir, expected.name)
        isfile(path) || error("missing configured n15 table file: $path")
        filesize(path) == expected.size || error(
            "$(expected.name) size mismatch: got $(filesize(path)), expected $(expected.size)")
        digest = BackgammonNet.sampled_page_hash(
            path; pages=BackgammonNet.VALUE_TABLE_SAMPLE_PAGES,
            page_bytes=BackgammonNet.VALUE_TABLE_SAMPLE_PAGE_BYTES)
        digest.hash == expected.sampled_hash || error(
            "$(expected.name) sampled-page hash does not match the pinned n15 release")
        push!(records, Dict{String,Any}(
            "name" => expected.name,
            "bytes" => expected.size,
            "sampled_hash" => expected.sampled_hash,
        ))
    end
    return Dict{String,Any}(
        "name" => "n15",
        "semantics" => "coherent_e_rr_runtime_approximation_not_training_teacher",
        "contract" => release.contract,
        "header_hash" => release.header_hash,
        "files" => records,
    )
end

"""Validate selected local files and return a path-independent fleet contract."""
function validate_table_release(selection::AbstractString;
                                k7_dir::AbstractString=BackgammonNet.default_bearoff_k7_dir(),
                                n15_dir::AbstractString=BackgammonNet.default_bearoff_n15_dir())
    selected = parse_table_selection(selection)
    tables = Dict{String,Any}[]
    _uses_k7(selected) && push!(tables, _validate_k7_release(k7_dir))
    _uses_n15(selected) && push!(tables, _validate_n15_release(n15_dir))
    return Dict{String,Any}(
        "selection" => selected,
        "selection_contract" => "server_pinned_explicit_bearoff_tables_v1",
        "tables" => tables,
    )
end

struct RuntimeTables{K,N}
    selection::String
    identity::Dict{String,Any}
    k7::K
    n15::N
end

_plain_identity(value::AbstractDict) =
    Dict(String(key) => _plain_identity(child) for (key, child) in pairs(value))
_plain_identity(value::AbstractVector) = Any[_plain_identity(child) for child in value]
_plain_identity(value) = value

"""Load exactly the tables selected by the server; never inspect optional files."""
function load_runtime_tables(selection::AbstractString;
                             expected_identity=nothing,
                             k7_dir::AbstractString=BackgammonNet.default_bearoff_k7_dir(),
                             n15_dir::AbstractString=BackgammonNet.default_bearoff_n15_dir())
    selected = parse_table_selection(selection)
    identity = validate_table_release(selected; k7_dir, n15_dir)
    expected_identity === nothing || identity == _plain_identity(expected_identity) || error(
        "local bearoff release identity disagrees with the server-pinned identity")
    k7 = _uses_k7(selected) ? BackgammonNet.BearoffK7.BearoffTable(String(k7_dir)) : nothing
    n15 = _uses_n15(selected) ?
        BackgammonNet.BearoffOneSidedCompact.CompactBundle(String(n15_dir)) : nothing
    return RuntimeTables(selected, identity, k7, n15)
end

@inline exact_k7_covers(tables::RuntimeTables, game)::Bool =
    tables.k7 !== nothing &&
    BackgammonNet.BearoffK7.is_bearoff_position(game.p0, game.p1)

@inline function exact_k7_lookup(tables::RuntimeTables, game)
    exact_k7_covers(tables, game) || return nothing
    return BackgammonNet.BearoffK7.lookup(tables.k7, game)
end

@inline n15_covers(tables::RuntimeTables, game)::Bool =
    tables.n15 !== nothing &&
    BackgammonNet.BearoffOneSidedCompact.covers(tables.n15, game.p0, game.p1)

@inline function n15_lookup(tables::RuntimeTables, game)
    n15_covers(tables, game) || return nothing
    return BackgammonNet.BearoffOneSidedCompact.lookup(tables.n15, game)
end

@inline function _heads_for_mover(result, state, mover::Integer)
    pW, pWG, pLG = Float64(result.pW), Float64(result.pWG), Float64(result.pLG)
    if Int(state.current_player) != Int(mover)
        pW, pWG, pLG = 1.0 - pW, pLG, pWG
    end
    heads = BackgammonNet.race_heads_to_joint_cumulative(
        Float32(pW), Float32(pWG), Float32(pLG))
    return Float64(BackgammonNet.compute_equity_joint(heads)), heads
end

function n15_turn_value_equity(table::BackgammonNet.BearoffOneSidedCompact.CompactBundle,
                               game, mover::Integer)
    boundary_value = function(state)
        if state.terminated
            heads = BackgammonNet.terminal_heads_target(state, mover)
            tuple = (heads.p_win, heads.p_gammon_win, heads.p_bg_win,
                     heads.p_gammon_loss, heads.p_bg_loss)
            return Float64(BackgammonNet.compute_equity_joint(tuple)), tuple
        end
        BackgammonNet.BearoffOneSidedCompact.covers(table, state.p0, state.p1) || error(
            "n15 does not cover a turn boundary reached from an n15-covered state")
        result = BackgammonNet.BearoffOneSidedCompact.lookup(table, state)
        return _heads_for_mover(result, state, mover)
    end
    return BackgammonNet.turn_aware_best(game, mover, boundary_value; objective=first)
end

function n15_best_move_value(table::BackgammonNet.BearoffOneSidedCompact.CompactBundle,
                             game)::Float64
    (game.terminated || BackgammonNet.is_chance_node(game) ||
     game.phase != BackgammonNet.PHASE_CHECKER_PLAY) && error(
        "n15_best_move_value requires a checker-play state")
    mover = Int(game.current_player)
    best = -Inf
    work = BackgammonNet.clone(game)
    for action in BackgammonNet.legal_actions(game)
        BackgammonNet.copy_state!(work, game)
        BackgammonNet.apply_legal_action!(work, action)
        value, _ = n15_turn_value_equity(table, work, mover)
        best = max(best, value)
    end
    return best
end

end
