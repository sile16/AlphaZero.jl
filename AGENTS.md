# AlphaZero / Backgammon Integration Notes

This repo consumes `BackgammonNet.jl` as a package dependency. Do not copy
BackgammonNet source files into AlphaZero or include them by path.

## Current Contract

- Required BackgammonNet version: `0.7.0+` (Julia 1.12.6+).
- Backgammon value-head contract: use `BackgammonNet.VALUE_HEAD_CONTRACT`,
  `BackgammonNet.VALUE_HEAD_ORDER`, and `BackgammonNet.check_probability_contract`.
- Equity perspective is side-to-move unless a local API explicitly says
  white-relative.
- The AlphaZero ML wrapper always exposes the standard 21 dice outcomes,
  including the first playable roll. Initial-player selection is resolved
  uniformly during setup rather than represented as an MCTS chance node.
- `BACKGAMMON_TAVLA_ENABLED=true` enables Tavla scoring independently of that
  ML opening convention; it is disabled by default.
- Distributed clients must match the server's versioned ML contract fingerprint
  (observation/value/action/chance/rule settings) before producing samples.
- Sample uploads use idempotent batch IDs and must be retried without discarding
  local samples until the server acknowledges the batch.
- GNUBG, Wildbg, and BGBlitz integration/capability logic belongs in
  BackgammonNet. AlphaZero should call its public backend APIs rather than
  copying or reimplementing bridge code.
- Bearoff table generation and runtime lookup code belongs in BackgammonNet.
- Canonical local bearoff artifact root is:
  - `../BackgammonNet.jl/data/bearoff/`

Expected table subdirectories:

- `bearoff_k7_twosided`
- `bearoff_n15_bundle_v3`

Bearoff tables are selected explicitly by the training server with
`--bearoff-tables=none|k7|n15|k7+n15`. Every client validates and loads exactly
that set; local file presence never enables a table. The exact money-optimal k7
two-sided table is the only source permitted for hard/truncation targets. The
n15 one-sided race bundle is a coherent E(R,R) runtime approximation for MCTS
leaves and is never described or consumed as an exact training teacher.

Use the BackgammonNet helpers instead of hardcoded paths:

- `BackgammonNet.default_bearoff_root()`
- `BackgammonNet.default_bearoff_k7_dir()`
- `BackgammonNet.default_bearoff_n15_dir()`

Upstream code must check each concrete configured table's coverage and invoke
that table's lookup directly. Do not recreate an implicit combined/fallback
table. k7 takes precedence when both configured tables cover a position.

Environment overrides are reserved for unusual deployments:

- `BACKGAMMONNET_BEAROFF_DIR`
- `BACKGAMMONNET_BEAROFF_K7_DIR`
- `BACKGAMMONNET_BEAROFF_N15_DIR`

## Active Distributed Code

The active distributed training stack is:

- `scripts/training_server.jl`
- `scripts/selfplay_client.jl`
- `src/distributed/`

Legacy archive trees were removed. Use git history for old launchers and
experiments.

The authoritative project status, evidence policy, and revalidation queue are
in `docs/backgammon_status.md`. Dated files under `notes/` are historical and
must not be used as current evidence.

## Cube Scope

BackgammonNet supports cube, match equity tables, Janowski conversion, and
rule-aware search values. The current `games/backgammon-deterministic/`
AlphaZero wrapper uses BackgammonNet's checker-policy head-local space:
checker actions `1:676`. BackgammonNet's engine IDs remain unified at 680
actions (cube IDs `677:680`), but the current AlphaZero network has no split
cube-policy heads. Keep `BACKGAMMON_CUBE_ENABLED=false` for this wrapper.

# Goal 1
Train a NN that outputs value and policy for board positions for Race positions that uses MCTS for better tree search of best moves.

# Goal 2
Train a NN that outputs value and policy for board positions for Contact positions that uses MCTS for better tree search of best moves.

# Goal 3
Beat wildbg in play , by using alphazero style self play to make the models better.  We can use bear off table to train the model, and bootstrap games to make training faster.


Use both Jarvis (x86) and Neo (M3 Mac Studio) , Balance incremental testing with faster wall clock time to acheive our goals.
