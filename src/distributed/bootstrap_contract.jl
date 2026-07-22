module BootstrapContract

import BackgammonNet

export bootstrap_checker_indices

const EXACT_RACE_KINDS = ("race_exact_k7", "race_natural_exact_k7")

"""Return checker rows eligible for the configured training family.

Race-only training accepts only exact-k7 artifacts. Dual training accepts a
single canonical family, and removes every gnubg-labeled race candidate from a
full-game artifact so those rows cannot enter the race network.
"""
function _bootstrap_checker_indices(data, states, training_mode::AbstractString;
                                    validate_positions::Bool=true)
    mode = String(training_mode)
    mode in ("dual", "race") || throw(ArgumentError(
        "unsupported training mode for bootstrap ingestion: $mode"))
    metadata = data.metadata
    kind = String(BackgammonNet.metadata_get(metadata, "artifact_kind", ""))
    role = String(BackgammonNet.metadata_get(metadata, "artifact_role", ""))
    role in ("train", "spot_train") || throw(ArgumentError(
        "bootstrap ingestion requires a train artifact_role, got $(repr(role))"))

    n = length(states)
    indices = if kind in EXACT_RACE_KINDS
        collect(1:n)
    elseif kind == "contact_gnubg"
        mode == "race" && throw(ArgumentError(
            "race training accepts only exact-k7 race artifacts; got contact_gnubg"))
        collect(1:n)
    elseif kind == "full_game_gnubg"
        mode == "race" && throw(ArgumentError(
            "race training accepts only exact-k7 race artifacts; full-game race rows are gnubg labels"))
        hasproperty(data, :race_candidate_indices) || error(
            "full-game bootstrap is missing race_candidate_indices")
        excluded = falses(n)
        for raw in data.race_candidate_indices
            idx = Int(raw)
            1 <= idx <= n || error("race_candidate_indices contains out-of-range row $idx")
            excluded[idx] && error("race_candidate_indices contains duplicate row $idx")
            excluded[idx] = true
        end
        findall(!, excluded)
    else
        throw(ArgumentError("unsupported bootstrap artifact_kind=$(repr(kind))"))
    end

    if validate_positions
        for idx in indices
            game = states[idx]
            if kind in EXACT_RACE_KINDS
                BackgammonNet.BearoffK7.is_bearoff_position(game.p0, game.p1) || error(
                    "exact-k7 race artifact contains a row outside k7 coverage at index $idx")
            else
                BackgammonNet.is_contact_position(game) || error(
                    "$kind bootstrap contains a non-contact selected row at index $idx")
            end
        end
    end
    return indices
end

bootstrap_checker_indices(artifact::BackgammonNet.TrainingArtifact,
                          training_mode::AbstractString) =
    _bootstrap_checker_indices(artifact.data, artifact.states, training_mode)

end
