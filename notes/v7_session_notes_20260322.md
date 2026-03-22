# v7 Session Notes (2026-03-22)

## Overview

Implemented and deployed v7: unified game loop + distributed eval architecture.
Two major changes from v6:
1. **Unified game loop** — single `play_game()` replacing 5 duplicate game loops
2. **Distributed eval** — eval moved from server to clients via chunked work queue

## v6 Results (baseline)

v6 tested 10x replay ratio (4000 gradient steps vs v5's ~390).
Race-only, 256w×5b, PER, cosine LR, bootstrap-seeded.

| Iter | Equity | Win% | Notes |
|------|--------|------|-------|
| 10 | +0.015 | 50.1% | Strong from start |
| 20 | +0.007 | 50.0% | Slight dip |
| 30 | +0.016 | 50.1% | Best, recovered |

Key finding: 10x replay ratio works — play strength even with wildbg despite rising training loss (1.49 → 2.12). Server blocked 17-34min during eval.

## v7 Implementation

### Architecture
- `src/game_loop.jl` — GameLoop module: MctsAgent, ExternalAgent, play_game()
- `src/distributed/eval_manager.jl` — EvalManager: chunked eval job tracking
- `src/distributed/server.jl` — 4 new HTTP endpoints (status/checkout/submit/heartbeat)
- `scripts/selfplay_client.jl` — --eval-capable flag, PAUSE_SELFPLAY atomics, check_and_do_eval!()

### Implementation Method
- 9 tasks executed via subagent-driven development
- 4 subagents ran in parallel (worktree isolation)
- Tasks 1-4: pure refactor (unified game loop)
- Tasks 6-9: distributed eval feature
- Plan review caught 3 critical issues (temperature handling, bear-off equity type, name collision)

### Bugs Found During Deployment (5 total)
All stemmed from JSON/MsgPack format mismatch between server and client:
1. `JSON.parse` not imported in client (silent catch swallowed error)
2. Server returns JSON, client used MsgPack.unpack for checkout
3. Same for heartbeat and submit responses
4. Server expected `az_is_white` in request body (client doesn't send it)
5. Missing `finalize_eval()` call + TB logging

### Training Run
- 3 server restarts needed for bug fixes (iter 30, 50, 55 checkpoints)
- SSH tunnel drops repeatedly (Neo→Jarvis) — major reliability issue
- Checkpoint interval reduced to 5 (from 10) to minimize restart data loss

## v7 Eval Results

| Iter | Equity | Win% | Chunks | Time | Notes |
|------|--------|------|--------|------|-------|
| 40 | ~even | ~50% | 40/40 | 120s | Client-side only (no TB) |
| 60 | +0.007 | 49.6% | 39/40 | 147s | Full end-to-end with TB |
| 70 | pending | | | | In progress |

Play strength consistent with v6 — even with wildbg large.

## Key Metrics Comparison: v6 vs v7 Eval

| Metric | v6 (server-side) | v7 (distributed) |
|--------|-------------------|-------------------|
| Eval time | 17-34 min | 2-2.5 min |
| Server blocked | Yes (entire duration) | Never |
| API responsive during eval | No | Yes |
| Eval at iter 60 | +0.007 | +0.007 |

## Lessons Learned

1. **Subagent-implemented code needs integration testing** — parallel subagents produced correct code individually but format mismatches at boundaries (JSON vs MsgPack)
2. **Silent catch blocks hide bugs** — the first 3 bugs were invisible because errors were caught and swallowed
3. **SSH tunnels are unreliable** — tunnel drops 5+ times during training, each time stalling Neo's sample uploads. Should switch to direct connection (jarvis:9090)
4. **Server restarts lose training progress** — even with checkpoints, we lost iters. The distributed eval architecture avoids this by never blocking the server
5. **TB step offset after restart** — when server resumes from checkpoint, TB step counter is offset by START_ITER. Eval points show as step 41/61 instead of iter 40/60

## Files Changed (v7)

### New
- `src/game_loop.jl` (429 lines) — unified game loop
- `src/distributed/eval_manager.jl` (216 lines) — eval job manager
- `test/test_game_loop.jl` (99 lines)
- `test/test_eval_manager.jl` (225 lines)
- `scripts/launch_v7.sh` (99 lines)
- `docs/superpowers/specs/` — design spec + refinements
- `docs/superpowers/plans/` — implementation plan

### Modified
- `scripts/selfplay_client.jl` — +446/-8 (eval mode + play_game wiring)
- `scripts/training_server.jl` — +166/-60 (distributed eval + play_game wiring)
- `scripts/eval_race.jl` — -50 lines (play_game wiring)
- `scripts/eval_vs_wildbg.jl` — -50 lines (play_game wiring)
- `src/distributed/server.jl` — +137 (eval endpoints + finalize)
- `src/AlphaZero.jl` — +5 (include game_loop)

## Next Steps

1. **Fix SSH tunnel reliability** — switch Neo to direct jarvis:9090 connection
2. **Fix TB step offset** — ensure eval steps match actual iteration numbers
3. **Continue training** — let v7 run to 200 iterations, compare with v6
4. **Observation experiments** — test different BACKGAMMON_OBS_TYPE (was the original goal before v7 detour)
