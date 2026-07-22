# AlphaZero ↔ BackgammonNet training-artifact integration test.
#
# PENDING v4 RE-ENABLE — tracked in docs/backgammon_status.md.
#
# BackgammonNet's training-artifact format moved to `backgammon_training_v4`
# (BackgammonNet.TRAINING_ARTIFACT_SCHEMA). v4 stores RAW checker action
# equities (`checker_action_ids` / `checker_action_equities`) instead of shaped
# policy probabilities, and its loader rejects any generation-time policy-shaping
# metadata (temperature / density contract). Hand-building a valid v4 artifact
# here would duplicate a large slice of BackgammonNet's teacher + action-equity
# contract inside AlphaZero — exactly what AGENTS.md says not to do.
#
# AlphaZero consumes bootstrap artifacts ONLY through BackgammonNet's fail-closed
# `load_training_artifact` + `fill_training_batch!`, and trusts BackgammonNet's
# rigorous release process and artifact manifest for contract conformance. This
# integration test should be re-enabled against a REAL BackgammonNet-produced v4
# artifact (or a public BackgammonNet v4 fixture builder) once the verified
# release lands. Until then it is skipped rather than reproducing the v4 contract.

using BackgammonNet
using Test

# Sanity: the schema constant we build against is exported and is the v4 line.
@test BackgammonNet.TRAINING_ARTIFACT_SCHEMA == "backgammon_training_v4"

# Consumption assertions, pending a real v4 artifact / builder (see header):
#   fill_training_batch! populates observations (OBS_FLAT_MIN_PLUS × n),
#   policies (CHECKER_ACTIONS × n, summing to 1), value scalars, and value heads
#   satisfying check_probability_contract.
@test_skip fill_training_batch_consumes_real_v4_artifact
