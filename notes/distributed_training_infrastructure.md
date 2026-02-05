# Distributed Training Infrastructure Plan

## Current Status (2026-02-01)

### Completed Today
- [x] `train_distributed.jl` - Julia Distributed training script
- [x] Worker-based parallel evaluation (moved from master to workers)
- [x] Final evaluation on workers (--final-eval-games, default 1000)
- [x] Validated: 12 iterations + parallel eval + final eval working
- [x] Resource profiling: ~11 GB RAM, ~400% CPU for 6 workers
- [x] Documentation in CLAUDE.md and this file
- [x] TavlaTalk architecture analysis for browser worker integration

### Next TODOs (Priority Order)

#### Immediate (Before Mac Studio Arrives)
1. [ ] Create `scripts/validate_distributed.jl` - Test harness for correctness
   - Deterministic seed comparison
   - Sample buffer verification
   - Model checkpoint validation
2. [ ] Add WandB integration to `train_distributed.jl`
3. [ ] Run extended training (200+ iterations) to verify stability
4. [ ] Compare training curves: thread-based vs distributed

#### When Mac Studio Arrives (2 days)
5. [ ] Test multi-machine Julia Distributed (Jarvis ↔ Mac Studio)
6. [ ] Benchmark: throughput with remote workers
7. [ ] Configure optimal worker count for 512GB system

#### Browser Workers (After Julia Distributed Stable)
8. [ ] Add WebSocket server to Julia training loop (HTTP.jl)
9. [ ] Implement model serving endpoint (ONNX distribution)
10. [ ] Implement sample collection endpoint
11. [ ] Modify tavlatalk worker.js for training integration
12. [ ] Add resource control UI (CPU/GPU sliders)
13. [ ] Build leaderboard and stats dashboard

#### Production Scaling
14. [ ] End-to-end validation: Julia + browser workers
15. [ ] Security: authentication, rate limiting
16. [ ] Monitoring and alerting
17. [ ] Public launch of browser worker participation

---

## Overview

This document outlines the plan for distributed AlphaZero training using Julia Distributed for server workers and WebAssembly/WebGPU for browser-based workers.

## Infrastructure

### Dedicated Systems

| System | RAM | CPU/GPU | Role |
|--------|-----|---------|------|
| Current (Jarvis) | 62 GB | 24 cores | Primary training server |
| Mac Studio Max | 512 GB | TBD (arriving in 2 days) | Secondary training server |

### Resource Utilization (Validated 2026-02-01)

Per `train_distributed.jl` with 6 workers + 1 master:

| Component | Memory | CPU |
|-----------|--------|-----|
| Per Worker | 1.5-1.8 GB | 55-68% per core |
| Master | 1.4 GB | 16-35% |
| **Total (6+1)** | **~11.3 GB** | **~400%** (4 cores equiv) |

**Bottleneck**: CPU, not memory. On 62GB system, can run 1-2 instances comfortably.

### Performance Baselines

| Approach | Throughput | Notes |
|----------|------------|-------|
| Thread-based (train_cluster.jl) | ~59 games/min | Single machine only |
| Distributed (train_distributed.jl) | ~47 games/min | Multi-machine capable |
| Browser worker (estimated) | 2-5 games/sec per tab | Via tavlatalk |

## Architecture

### Julia Distributed (Server Workers)

```
Master Process                    Worker Processes (N)
┌─────────────────────┐          ┌──────────────────────┐
│ - Training loop     │  pmap()  │ - Self-play games    │
│ - Replay buffer     │ <------> │ - MCTS evaluation    │
│ - Weight updates    │  samples │ - Network inference  │
│ - Checkpointing     │          │ - Eval games         │
│ - WandB logging     │          │                      │
└─────────────────────┘          └──────────────────────┘
```

### Browser Workers (WASM/WebGPU)

```
Julia Training Server              Browser Workers
┌─────────────────────┐           ┌─────────────────────┐
│ WebSocket Server    │ <-------> │ Worker Thread       │
│ - Model distribution│   ONNX    │ - GameBridge (WASM) │
│ - Sample collection │  samples  │ - InferenceEngine   │
│ - Worker registry   │   stats   │   (ONNX/WebGPU)     │
│ - Leaderboard       │           │ - MCTS self-play    │
└─────────────────────┘           └─────────────────────┘
```

## Browser Worker Features (User-Facing)

### Engagement Features
1. **Leaderboard** - Track contributions by user/device
2. **Progress Dashboard** - Show current training stats
   - Games completed
   - Samples contributed
   - Model version/iteration
   - Games/second throughput
3. **Resource Controls** - Let users adjust:
   - CPU/GPU utilization percentage
   - Number of concurrent workers
   - MCTS iterations (quality vs speed)
   - Batch size for inference

### Stats to Display
- Global: Total games, total samples, model iteration, eval vs GnuBG
- Per-user: Games contributed, samples contributed, uptime
- Performance: Games/sec, simulations/sec, GPU utilization

## TavlaTalk Integration

Existing repo: `/home/sile/github/tavlatalk/`

### Current Capabilities
- ONNX Runtime Web with WebGPU backend
- WASM-based game logic (AssemblyScript)
- Worker thread for background MCTS
- Self-play loop implemented in `selfplay-test.html`

### Required Additions
1. **WebSocket client** - Connect to Julia training server
2. **Model update handler** - Download new ONNX on version change
3. **Sample collector** - Capture observations, policies, outcomes
4. **Stats reporter** - Send performance metrics
5. **UI for worker control** - Resource sliders, start/stop

### Communication Protocol

**Server → Browser:**
```javascript
{type: 'MODEL_UPDATE', version, onnxUrl, checksum}
{type: 'PLAY_GAMES', gameCount, mcts: {iterations, batchSize, cpuct}}
```

**Browser → Server:**
```javascript
{type: 'WORKER_READY', workerId, backend, features}
{type: 'GAMES_COMPLETE', workerId, games: [...], aggregateStats}
{type: 'EVAL_RESULT', checkpointId, result}
```

## Testing & Validation Requirements

### CRITICAL: Pre-Production Validation

Before scaling to expensive compute (Mac Studio, many browser workers), we MUST validate:

1. **Sample Quality**
   - [ ] Verify observations match between Julia and WASM
   - [ ] Verify policy format is consistent
   - [ ] Verify outcome/reward calculation matches
   - [ ] Compare sample distributions from Julia vs browser workers

2. **Training Stability**
   - [ ] Run 500+ iteration training, verify no NaN/Inf in losses
   - [ ] Verify checkpoint save/load works correctly
   - [ ] Verify weight synchronization doesn't introduce drift
   - [ ] Compare training curves: Julia-only vs hybrid

3. **Performance Validation**
   - [ ] Benchmark Julia workers: games/min, samples/min
   - [ ] Benchmark browser workers: games/sec, latency
   - [ ] Measure WebSocket overhead for sample transmission
   - [ ] Verify no memory leaks in long-running workers

4. **Correctness Tests**
   - [ ] Unit tests for game logic (Julia vs WASM)
   - [ ] MCTS tree verification (same position → similar distribution)
   - [ ] End-to-end: Train with browser workers, verify model improves

5. **Integration Tests**
   - [ ] Multi-machine Julia Distributed (when Mac Studio arrives)
   - [ ] Mixed Julia + browser workers training
   - [ ] Model update propagation timing
   - [ ] Worker crash recovery / reconnection

### Test Harness

Create `scripts/validate_distributed.jl`:
- Run N iterations with deterministic seed
- Compare final model hash against known-good baseline
- Verify sample counts and buffer statistics
- Run eval against fixed opponent, compare results

## Implementation Phases

### Phase 1: Julia Distributed (COMPLETED 2026-02-01)
- [x] `train_distributed.jl` with pmap workers
- [x] Parallel evaluation on workers
- [x] Final evaluation on workers
- [x] Checkpoint saving with eval results

### Phase 2: Testing Infrastructure
- [ ] Create `validate_distributed.jl` test harness
- [ ] Add deterministic replay for debugging
- [ ] Implement sample comparison tools
- [ ] Set up CI for distributed training tests

### Phase 3: WebSocket Server (Julia Side)
- [ ] Add HTTP.jl WebSocket support
- [ ] Implement model serving endpoint
- [ ] Implement sample collection endpoint
- [ ] Add worker registry and health checks
- [ ] Integrate with training loop

### Phase 4: Browser Worker Client (TavlaTalk)
- [ ] Add WebSocket connection to worker.js
- [ ] Implement model update handler
- [ ] Implement sample collection and upload
- [ ] Add resource usage controls
- [ ] Build leaderboard UI

### Phase 5: Production Scaling
- [ ] Deploy on Mac Studio Max
- [ ] Enable public browser worker participation
- [ ] Add monitoring and alerting
- [ ] Performance optimization

## Risk Mitigation

1. **Wasted Compute**: Rigorous validation BEFORE scaling
2. **Sample Corruption**: Checksums on model updates, sample verification
3. **Security**: Rate limiting, authentication for browser workers
4. **Reliability**: Worker heartbeats, automatic reconnection

## Notes

- Browser workers add latency but massive parallelism
- Prioritize Julia workers for training, browser for evaluation
- Consider adaptive contribution weighting based on worker reliability
