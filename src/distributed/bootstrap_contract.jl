module BootstrapContract

import BackgammonNet

export BOOTSTRAP_ROW_SELECTORS, select_bootstrap_rows

const BOOTSTRAP_ROW_SELECTORS = ("all", "contact", "race", "exact-race")
const EXACT_RACE_KINDS = ("race_exact_k7", "race_natural_exact_k7")
const EVAL_ROLES = ("eval", "spot_eval")

function _full_game_race_indices(data, n::Int)
    hasproperty(data, :race_candidate_indices) || error(
        "full-game bootstrap is missing race_candidate_indices")
    indices = Int[]
    seen = falses(n)
    for raw in data.race_candidate_indices
        idx = Int(raw)
        1 <= idx <= n || error("race_candidate_indices contains out-of-range row $idx")
        seen[idx] && error("race_candidate_indices contains duplicate row $idx")
        seen[idx] = true
        push!(indices, idx)
    end
    return indices, seen
end

"""Select checker rows by an explicit ML family decision.

Selectors are teacher-neutral:

- `all`: every checker row in the artifact;
- `contact`: contact rows only;
- `race`: race rows regardless of whether their approved artifact teacher is
  gnubg (full-game) or exact k7 (race artifacts);
- `exact-race`: exact-k7 race artifacts only.

The BackgammonNet loader remains authoritative for schema, provenance, and
teacher-policy validation. This layer chooses which already-approved rows the
ML run intends to consume and rejects incompatible artifact/selector pairs.
"""
function _select_bootstrap_rows(data, states, training_mode::AbstractString,
                                selector::AbstractString;
                                validate_positions::Bool=true)
    mode = String(training_mode)
    mode in ("dual", "race") || throw(ArgumentError(
        "unsupported training mode for bootstrap ingestion: $mode"))
    selected_family = lowercase(strip(String(selector)))
    selected_family in BOOTSTRAP_ROW_SELECTORS || throw(ArgumentError(
        "bootstrap row selector must be one of $(join(BOOTSTRAP_ROW_SELECTORS, ", ")); " *
        "got $(repr(selector))"))

    metadata = data.metadata
    kind = String(BackgammonNet.metadata_get(metadata, "artifact_kind", ""))
    role = String(BackgammonNet.metadata_get(metadata, "artifact_role", ""))
    role in EVAL_ROLES && throw(ArgumentError(
        "evaluation artifact role $role may not be used for training"))
    # `artifact` reached this API through BackgammonNet.load_training_artifact,
    # which validates the complete role vocabulary. This consumer adds only the
    # use-case rule: eval/spot_eval rows must never enter training. In particular,
    # the descriptive `bootstrap` role is a valid training input here.
    teacher_policy = String(BackgammonNet.metadata_get(
        metadata, "teacher_policy", "missing"))
    teacher_ply = BackgammonNet.metadata_get(metadata, "teacher_ply", nothing)
    engine_value = BackgammonNet.metadata_get(metadata, "engine_value", nothing)
    engine_policy = BackgammonNet.metadata_get(metadata, "engine_policy", nothing)
    engine_cube = BackgammonNet.metadata_get(metadata, "engine_cube", nothing)
    game_mode = BackgammonNet.metadata_get(metadata, "game_mode", nothing)
    source_mode = BackgammonNet.metadata_get(metadata, "source_mode", nothing)
    source_selector = BackgammonNet.metadata_get(metadata, "source_selector", nothing)
    exact_table_identity = BackgammonNet.metadata_get(
        metadata, "exact_table_identity", nothing)
    variant_id = BackgammonNet.metadata_get(metadata, "variant_id", nothing)
    block_id = BackgammonNet.metadata_get(metadata, "block_id", nothing)
    producer_repo_commit = BackgammonNet.metadata_get(
        metadata, "producer_repo_commit", nothing)

    n = length(states)
    all_indices = collect(1:n)
    race_indices, race_mask = if kind == "full_game_gnubg"
        _full_game_race_indices(data, n)
    elseif kind in EXACT_RACE_KINDS
        (copy(all_indices), trues(n))
    elseif kind == "contact_gnubg"
        (Int[], falses(n))
    else
        throw(ArgumentError("unsupported bootstrap artifact_kind=$(repr(kind))"))
    end
    contact_indices = findall(!, race_mask)

    indices = if selected_family == "all"
        all_indices
    elseif selected_family == "contact"
        kind in EXACT_RACE_KINDS && throw(ArgumentError(
            "selector=contact is incompatible with exact-race artifact_kind=$kind"))
        contact_indices
    elseif selected_family == "race"
        kind == "contact_gnubg" && throw(ArgumentError(
            "selector=race is incompatible with contact_gnubg"))
        race_indices
    else # exact-race
        kind in EXACT_RACE_KINDS || throw(ArgumentError(
            "selector=exact-race requires artifact_kind race_exact_k7 or " *
            "race_natural_exact_k7; got $kind"))
        all_indices
    end
    isempty(indices) && throw(ArgumentError(
        "selector=$selected_family selected zero checker rows from artifact_kind=$kind"))

    # A race-only network cannot consume contact rows. This is a model-family
    # compatibility check, not a restriction on which approved teacher may label
    # race rows.
    if mode == "race" && any(!race_mask[idx] for idx in indices)
        throw(ArgumentError(
            "training_mode=race with selector=$selected_family selected contact rows; " *
            "choose selector=race or exact-race for this artifact"))
    end

    if validate_positions
        for idx in indices
            game = states[idx]
            if kind in EXACT_RACE_KINDS
                BackgammonNet.BearoffK7.is_bearoff_position(game.p0, game.p1) || error(
                    "exact-k7 race artifact contains a row outside k7 coverage at index $idx")
            elseif race_mask[idx]
                BackgammonNet.is_race_position(game) || error(
                    "full-game race_candidate_indices contains a non-race row at index $idx")
            else
                BackgammonNet.is_contact_position(game) || error(
                    "$kind contains a selected non-contact row outside its race queue at index $idx")
            end
        end
    end

    selected_race = count(idx -> race_mask[idx], indices)
    return (
        indices=indices,
        selector=selected_family,
        artifact_kind=kind,
        artifact_role=role,
        teacher_policy=teacher_policy,
        teacher_ply=teacher_ply,
        engine_value=engine_value,
        engine_policy=engine_policy,
        engine_cube=engine_cube,
        game_mode=game_mode,
        source_mode=source_mode,
        source_selector=source_selector,
        exact_table_identity=exact_table_identity,
        variant_id=variant_id,
        block_id=block_id,
        producer_repo_commit=producer_repo_commit,
        selected_rows=length(indices),
        selected_contact=length(indices) - selected_race,
        selected_race=selected_race,
        artifact_rows=n,
    )
end

select_bootstrap_rows(artifact::BackgammonNet.TrainingArtifact,
                      training_mode::AbstractString,
                      selector::AbstractString) =
    _select_bootstrap_rows(artifact.data, artifact.states, training_mode, selector)

end
