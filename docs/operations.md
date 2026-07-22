# AlphaZero distributed operations

## Before a run

Run the server with `--preflight`. It validates the BackgammonNet version,
AlphaZero ML contract, 21-outcome chance model, value-head contract, both network
forwards, distributed protocol, weight checksums, configured artifacts, and the
session directory. It writes `preflight_report.json` and exits without opening a
server or starting training.

Evaluation files are immutable inputs. Build their sidecar once with
`scripts/build_eval_manifest.jl --input POSITIONS.jls`; training verifies the
artifact hash, position-set hash, count, and ML contract before use.

Validate finalized training/evaluation artifacts with
`scripts/validate_training_data.jl`. BackgammonNet performs authoritative schema,
policy, and value-label validation; AlphaZero adds corpus composition, duplicate,
legality, and train/eval leakage reporting.

That consumer check is necessary but not sufficient for releasing training
data. Before configuring `--bootstrap-file`, run BackgammonNet's current strict
and live/independent audit lanes and verify the artifact is included in a
`backgammon_artifact_manifest_v2` release. Pre-reset artifacts and reports are
quarantined as documented in [`backgammon_status.md`](backgammon_status.md).

## Health and draining

- `GET /api/health` is a liveness check.
- `GET /api/ready` returns 200 only when weights and the contract are ready and
  the server accepts uploads.
- Authenticated `POST /api/drain` stops new uploads. The learner exits at a safe
  iteration boundary and writes a final transactional checkpoint.

Checkpoint bundles under `checkpoints/bundle_iter_*` are the only resumable
training state. Each has
a manifest containing SHA-256 hashes plus the exact AlphaZero commit,
BackgammonNet version/commit, Julia version, contract/config fingerprints, and
seed. Resume scans newest-first and automatically falls back past corrupt or
incomplete bundles. Published `contact_latest.data` and `race_latest.data` files
are evaluation conveniences and are never accepted as resume state.
