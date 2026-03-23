# v8 Session Notes (2026-03-23)

## Overview

v8: multihead masking fix + strict checksums + robustness improvements.
Fresh bootstrap start with corrected conditional equity head training.

## Key Changes from v7

1. **Multihead conditional masking** — P(gammon|win) and P(bg|win) only trained on won games; P(gammon|loss) and P(bg|loss) only on lost games. Previous versions trained all heads on all samples, learning joint probabilities instead of conditionals.
2. **Strict weight checksums** — SHA256 validation on all weight transfers (no checksum=0 tolerance)
3. **@inbounds audit** — systematic bounds check removal in MCTS hot paths (~2-5% speedup)
4. **Eval player reuse** — EvalAlphaZeroAgent was creating new BatchedMctsPlayer per MOVE (fixed: create once, reuse)
5. **Selfplay direct MCTS** — bypasses GameLoop.play_game() to avoid GC pressure under threading

## Critical Performance Bug Found and Fixed

**GameLoop.play_game() caused 20-30x selfplay throughput regression.**

v6 used inline BatchedMCTS calls directly. v7/v8 routed through GameLoop.play_game() which:
- Allocates TraceEntry structs per move
- Calls _is_contact_position() per move
- Creates PositionValueSample vectors
- Under 32 threads: 28% more GC pressure (91GB vs 71GB per 5 games)
- Julia's stop-the-world GC amplifies this to 20-30x throughput loss

**Profiling results (single-threaded, 256w×5b, 1600 MCTS):**

| Path | Time/5 games | Allocations | Memory |
|------|-------------|-------------|--------|
| GameLoop.play_game() | 118.7s | 13.7M | 91 GB |
| Direct BatchedMCTS (v6 style) | 92.1s | 10.7M | 71 GB |

Fix: restored v6's lean inline game loop in `_play_games_loop()`. GameLoop retained for eval scripts only.

**Additional perf issue:** `create_player()` checked `chance_mode == :passthrough` but MctsParams defaults to `:full`, routing to slow classic MctsPlayer. Fixed to use BatchedMCTS for both `:passthrough` and `:full`.

**Eval perf issue:** EvalAlphaZeroAgent created new BatchedMctsPlayer per MOVE (not per game). Fixed: player created once in constructor.

## SSH Tunnel / Tailscale Issue

**Root cause found:** Tailscale network extension on Neo intercepts outbound TCP from non-Apple-signed binaries (Julia, Python). System binaries (curl, nc) are Apple-signed and bypass the extension.

- Tailscale removed from Neo but requires reboot to fully unload
- SSH tunnel (autossh) used as workaround until reboot
- Recurring tunnel drops killed Neo client ~5 times during training

## Jarvis OOM Crashes

Confirmed via `journalctl`: Linux OOM killer terminates Jarvis selfplay client.
- Server: 28GB RSS (3M PER buffer + GPU model + bear-off tables)
- Client: 34GB RSS (12 workers × MCTS trees + FastWeights)
- Total: 62GB on 64GB machine → OOM

**Fix options for v9:** Reduce Jarvis workers (8 instead of 12), reduce buffer (2M), or run client only on Neo with direct IP (post-Tailscale removal + reboot).

## Training Results

**Config:** 256w×5b race, 4000 gradient steps/iter, PER, cosine LR, bootstrap (2.37M samples), 1600 MCTS iters, 44 workers (32 Neo + 12 Jarvis).

### Loss Trend
| Iter | Loss |
|------|------|
| 1 | 1.615 |
| 2 | 1.544 |
| 3 | 1.530 |
| 4 | 1.515 |
| 5 | 1.525 |
| 6 | 1.541 |
| 7 | 1.573 |
| 8 | 1.599 |
| 9 | 1.630 |
| 10 | 1.663 |
| 15 | 1.813 |
| 20 | 1.968 |
| 23 | 2.080 |

Loss rises steadily from iter 4. Same pattern as v6 (10x replay ratio). Theory: overfitting to bootstrap data before self-play dilutes it.

### Eval Results (vs wildbg large, 600 MCTS, 1000 positions × 2 sides)
| Iter | Equity | Win% | Time |
|------|--------|------|------|
| 10 | +0.01 | 49.7% | 132s |
| 20 | +0.02 | 50.0% | 117s |

**Key finding: play strength holds at ~even with wildbg despite loss doubling (1.66→1.97).** Consistent with v6/v7 observation that loss plateau != strength plateau.

## Throughput

| Phase | games/min | Notes |
|-------|-----------|-------|
| Iter 0 (bootstrap weights) | ~0.5 | Untrained network plays 200+ move games |
| Iter 1+ (trained) | ~400 | After selfplay perf fix |
| Iter 1+ (old code) | ~40 | Before selfplay perf fix (GameLoop overhead) |
| Expected (v6 baseline) | ~400 | Direct BatchedMCTS, 1600 MCTS iters |

## Code Cleanup

Removed 9,050 lines of unused example games:
- games/pig/, games/pig-deterministic/
- games/connect-four/, games/mancala/, games/grid-world/, games/ospiel_ttt/
- test/stochastic_mcts.jl
- Kept tictactoe (used by 8 test files)

## Position Analysis

Verified race starting positions (98,516) have balanced pip count distribution:
- Mean pip diff (white-black): 0.1 (symmetric)
- Std: 59.2, range [-201, 222]
- Bell curve centered at 0 as expected

**All 2000 eval positions are IN the training set.** Created excluded training set: `race_starts_tuples_no_eval.jls` (96,514 positions) for v9.

## Files Changed

### Performance
- `scripts/selfplay_client.jl` — direct BatchedMCTS in selfplay loop, player reuse in eval agent
- `src/batched_mcts.jl` — @inbounds on hot loops, EMPTY_INT_VEC for terminal nodes
- `src/mcts.jl` — Ntot() explicit @inbounds loop instead of generator
- `src/game_loop.jl` — create_player routes :full to BatchedMCTS, accepts pre-created players

### Correctness
- `src/learning.jl` — conditional head masking, split_equity_targets helper
- `src/memory.jl` — equity_vector, flip_equity_perspective utilities
- `src/distributed/protocol.jl` — SHA256 weight checksums, strict validation
- `src/distributed/server.jl` — better eval submit error handling (404/409)
- `src/distributed/eval_manager.jl` — reject duplicate chunk submissions
- `src/distributed/client.jl` — weight version validation on download

### Cleanup
- Removed 6 game directories (9,050 lines)
- `src/examples.jl` — only tictactoe remains

## Lessons Learned

1. **GameLoop abstractions kill threading performance** — per-move allocations compound under Julia's stop-the-world GC. Hot paths must minimize allocation.
2. **Profile under realistic threading, not single-thread** — the GameLoop overhead was only 29% single-threaded but 20-30x under 32 threads.
3. **Tailscale network extensions break non-Apple-signed binaries** — Julia (Homebrew, ad-hoc signed) can't make outbound LAN TCP connections when Tailscale is active.
4. **Weight version enforcement breaks eval** — eval job created at iter N, but by download time server is at iter N+1. Don't enforce strict version matching on eval downloads.
5. **Jarvis can't run server + 12-worker client** — 62GB combined exceeds 64GB. Need fewer workers or separate machines.
6. **Rising loss != play quality degradation** — v8 loss doubled from 1.5 to 2.1 but eval equity stayed at ~0 vs wildbg. Bootstrap→self-play transition artifact.
7. **Player-per-move in eval** was even worse than player-per-game in selfplay. Always check if expensive objects are being created in inner loops.
