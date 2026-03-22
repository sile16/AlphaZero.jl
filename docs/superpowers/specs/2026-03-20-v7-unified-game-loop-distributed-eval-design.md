# v7: Unified Game Loop + Distributed Eval

## Problem

1. **Server blocks during eval** — the training server runs eval synchronously, blocking HTTP for 20-34 minutes. Clients get upload errors, samples are lost, and the server becomes unmonitorable.

2. **Memory pressure crashes** — running eval + training + self-play on Jarvis (62GB) caused an OOM crash during v6, losing 8 iterations of training.

3. **Code duplication** — the same game loop is implemented 5 times across 4 files (selfplay_client.jl CPU/GPU, training_server.jl, eval_race.jl, eval_vs_wildbg.jl), each with slight variations and independent bugs.

## Solution

Two changes:

**A. Unified game loop** — one `play_game()` function that handles both self-play and eval, parameterized by agent type and recording mode.

**B. Distributed eval** — move eval from the training server into self-play clients via a work-queue. Server stays responsive at all times.

---

## Part A: Unified Game Loop

### New file: `src/game_loop.jl`

#### Agent Interface

```julia
abstract type GameAgent end

struct MctsAgent <: GameAgent
    oracle           # single state → (policy, value)
    batch_oracle     # batch states → [(policy, value)...]
    mcts_params::MctsParams
    batch_size::Int
    gspec            # game spec for BatchedMCTS
    bearoff_eval     # optional bearoff evaluator (nothing if disabled)
end

struct ExternalAgent <: GameAgent
    backend          # BackgammonNet.WildbgBackend or similar
end
```

`MctsAgent` wraps BatchedMCTS. `ExternalAgent` wraps any BackgammonNet backend (wildbg, gnubg, etc.).

#### Game Result

```julia
struct GameResult
    reward::Float64                             # white-relative outcome
    trace::Vector{TraceEntry}                   # position trace (for training samples)
    value_samples::Vector{PositionValueSample}  # NN vs opponent value comparison
    num_moves::Int
    bearoff_truncated::Bool                     # game ended early via bear-off table
    first_bearoff_equity::Union{Nothing, Float64}    # exact equity at first bear-off position
    first_bearoff_white_playing::Union{Nothing, Bool} # who was moving at first bear-off
end

struct TraceEntry
    state           # game state
    player::Int     # current player (0=white, 1=black)
    action::Int     # chosen action
    legal_actions::Vector{Int}  # all legal actions (needed for sparse policy reconstruction)
    policy::Vector{Float32}     # MCTS visit distribution over legal_actions
    is_chance::Bool
    is_bearoff::Bool
    is_contact::Bool
end

struct PositionValueSample
    nn_val::Float64
    opponent_val::Float64
    is_contact::Bool
end
```

#### Core Function

```julia
function play_game(
    white::GameAgent, black::GameAgent, position;
    record_trace::Bool = false,          # for training samples
    record_value_comparison::Bool = false, # for eval stats
    value_oracle = nothing,              # separate oracle for value comparison (not the agent's oracle)
    opponent_value_fn = nothing,         # fn(state) → Float64, e.g., wildbg evaluate
    bearoff_truncation::Bool = false,
    bearoff_lookup = nothing,            # fn(state) → equity, for truncation/first-bearoff tracking
    rng = Random.default_rng(),
    seed::Int = 0
) :: GameResult
```

**Logic (full, including bear-off handling):**
```julia
function play_game(white, black, position; kwargs...)
    seed > 0 && Random.seed!(rng, seed)
    env = initialize_game(position)
    trace = TraceEntry[]
    value_samples = PositionValueSample[]
    num_moves = 0
    bearoff_truncated = false
    first_bearoff_equity = nothing
    first_bearoff_white_playing = nothing

    # Create MCTS players once per game (not per move)
    white_player = create_player(white)  # returns BatchedMctsPlayer or nothing
    black_player = create_player(black)

    while !game_terminated(env)
        if is_chance_node(env)
            # Check bear-off truncation at chance nodes (pre-dice)
            if bearoff_truncation && bearoff_lookup !== nothing
                bo_eq = bearoff_lookup(current_state(env))
                if bo_eq !== nothing
                    # Record first bear-off if not yet seen
                    if first_bearoff_equity === nothing
                        first_bearoff_equity = bo_eq
                        first_bearoff_white_playing = white_to_move(env)
                    end
                    bearoff_truncated = true
                    reward = bo_eq  # exact equity as game result
                    break
                end
            end

            # Track first bear-off even without truncation
            if bearoff_lookup !== nothing && first_bearoff_equity === nothing
                bo_eq = bearoff_lookup(current_state(env))
                if bo_eq !== nothing
                    first_bearoff_equity = bo_eq
                    first_bearoff_white_playing = white_to_move(env)
                end
            end

            sample_chance!(env, rng)
            continue
        end

        avail = legal_actions(env)

        # Single-action bypass — skip MCTS for forced moves
        if length(avail) == 1
            if record_trace
                push!(trace, TraceEntry(current_state(env), current_player(env),
                    avail[1], avail, Float32[1.0], false,
                    is_bearoff_position(env), is_contact_position(env)))
            end
            play!(env, avail[1])
            num_moves += 1
            continue
        end

        is_white_turn = white_to_move(env)
        current_agent = is_white_turn ? white : black
        player = is_white_turn ? white_player : black_player

        action, policy, legal = select_action(current_agent, env, player)
        num_moves += 1

        if record_trace && current_agent isa MctsAgent
            push!(trace, TraceEntry(current_state(env), current_player(env),
                action, legal, policy, false,
                is_bearoff_position(env), is_contact_position(env)))
        end

        if record_value_comparison && current_agent isa MctsAgent && value_oracle !== nothing && opponent_value_fn !== nothing
            nn_v = value_oracle(current_state(env))
            opp_v = opponent_value_fn(current_state(env))
            push!(value_samples, PositionValueSample(nn_v, opp_v, is_contact_position(env)))
        end

        play!(env, action)
    end

    if !bearoff_truncated
        reward = white_reward(env)
    end

    # Reset players for reuse
    white_player !== nothing && reset_player!(white_player)
    black_player !== nothing && reset_player!(black_player)

    return GameResult(reward, trace, value_samples, num_moves,
                      bearoff_truncated, first_bearoff_equity, first_bearoff_white_playing)
end
```

#### Player lifecycle

MCTS players are created once per game, not per move:

```julia
function create_player(agent::MctsAgent)
    BatchedMctsPlayer(agent.gspec, agent.oracle, agent.mcts_params,
                      agent.batch_size; batch_oracle=agent.batch_oracle,
                      bearoff_evaluator=agent.bearoff_eval)
end

create_player(agent::ExternalAgent) = nothing
```

#### select_action dispatch

```julia
function select_action(agent::MctsAgent, env, player)
    actions, policy = BatchedMCTS.think(player, env)
    action = sample_from_policy(policy, agent.mcts_params.temperature)
    return action, policy, actions
end

function select_action(agent::ExternalAgent, env, player)
    action = BackgammonNet.agent_move(BackgammonNet.BackendAgent(agent.backend), env.game)
    return action, Float32[], Int[]
end
```

#### How existing code paths map to play_game()

**Self-play** (selfplay_client.jl):
```julia
az = MctsAgent(oracle, batch_oracle, train_mcts_params, 50, gspec, bearoff_eval)
result = play_game(az, az, start_position;
    record_trace=true,
    bearoff_truncation=BEAROFF_TRUNCATION,
    bearoff_lookup=bearoff_chance_lookup,
    seed=seed)
samples = convert_trace_to_samples(result)  # uses result.trace, .reward, .first_bearoff_*, .bearoff_truncated
```

**Eval** (training_server.jl / eval_race.jl / eval_vs_wildbg.jl):
```julia
az = MctsAgent(oracle, batch_oracle, eval_mcts_params, 50, gspec, nothing)
wb = ExternalAgent(wildbg_backend)
result = play_game(az, wb, position;
    record_value_comparison=true,
    value_oracle=nn_value_fn,
    opponent_value_fn=wildbg_value_fn,
    seed=seed)
```

**GPU self-play** (selfplay_client.jl GPU path):
```julia
az_gpu = MctsAgent(gpu_oracle, gpu_batch_oracle, train_mcts_params, 50, gspec, bearoff_eval)
# Same play_game() call — GPU vs CPU is an oracle concern, not a game loop concern
result = play_game(az_gpu, az_gpu, start_position; record_trace=true, seed=seed)
```

### What gets deleted

- `training_server.jl`: `eval_race_game_server()`, `EvalAlphaZeroAgent` struct, inline eval loop (~150 lines)
- `selfplay_client.jl`: inline game loop in `_play_games_loop()` (~80 lines)
- `eval_race.jl`: `eval_race_game()` and its inline loop (~40 lines)
- `eval_vs_wildbg.jl`: `eval_game()` and its inline loop (~40 lines)

**Total removed: ~310 lines. Replaced by ~200 lines in game_loop.jl.**

Parallelism stays in each caller — `play_game()` handles a single game. The `Threads.@threads` work-stealing loops in eval scripts and `continuous_worker` loops in selfplay remain unchanged.

### Temperature and noise

Controlled entirely by `MctsParams` — no special handling in `play_game()`:

- Self-play: `temperature=schedule, dirichlet_noise_ϵ=0.25, dirichlet_noise_α=0.3`
- Eval: `temperature=ConstSchedule(0.0), dirichlet_noise_ϵ=0.0`

### Bear-off table

Two separate concerns:

1. **MCTS bear-off evaluator** — inside `MctsAgent.bearoff_eval`, used during tree search. Self-play enables it, eval disables it. Invisible to `play_game()`.

2. **Bear-off game-level events** — `bearoff_lookup` param for truncation and first-bearoff tracking. Only used by self-play for training sample generation. `play_game()` handles this explicitly.

`convert_trace_to_samples` stays in selfplay_client.jl. It uses `GameResult.trace`, `.reward`, `.bearoff_truncated`, `.first_bearoff_equity`, and `.first_bearoff_white_playing` to produce the three-way branching (truncated / first-bearoff-seen / normal).

---

## Part B: Distributed Eval

### Server Side

#### Eval Job State

New file: `src/distributed/eval_manager.jl`

All eval state protected by a single `ReentrantLock` (following existing `ServerState` pattern in `server.jl`):

```julia
const EVAL_LOCK = ReentrantLock()

mutable struct EvalChunk
    chunk_id::Int
    position_range::UnitRange{Int}    # indices into EVAL_POSITIONS
    az_is_white::Bool                 # true for first pass, false for second
    checked_out_by::Union{Nothing, String}
    checkout_time::Float64
    completed::Bool
end

mutable struct EvalJob
    iter::Int
    weights_version::Int              # matches weight version counter
    chunks::Vector{EvalChunk}
    results::Dict{Int, EvalChunkResult}  # chunk_id → result
    created_at::Float64
end

struct EvalChunkResult
    chunk_id::Int
    az_is_white::Bool                 # needed for white/black aggregation
    rewards::Vector{Float64}
    value_samples::Vector{PositionValueSample}
end

const EVAL_JOB = Ref{Union{Nothing, EvalJob}}(nothing)
const CHUNK_SIZE = 50  # games per chunk
const CHECKOUT_LEASE_SECONDS = 300  # 5 minutes
const EVAL_JOB_TIMEOUT_SECONDS = 1800  # 30 minutes — abandon if no progress
```

#### Chunk Layout

2000 positions × 2 sides = 4000 games. Chunk size = 50:
- Chunks 1-40: positions 1-2000, AZ as white (50 positions per chunk)
- Chunks 41-80: positions 1-2000, AZ as black (50 positions per chunk)
- Total: 80 chunks

#### Creating an Eval Job

Called from the training loop after checkpoint save (replaces `run_eval!()`):

```julia
function create_eval_job!(iter::Int)
    lock(EVAL_LOCK) do
        n_pos = length(EVAL_POSITIONS)
        chunks = EvalChunk[]

        # White games
        for i in 1:CHUNK_SIZE:n_pos
            push!(chunks, EvalChunk(length(chunks)+1, i:min(i+CHUNK_SIZE-1, n_pos), true, nothing, 0.0, false))
        end
        # Black games
        for i in 1:CHUNK_SIZE:n_pos
            push!(chunks, EvalChunk(length(chunks)+1, i:min(i+CHUNK_SIZE-1, n_pos), false, nothing, 0.0, false))
        end

        EVAL_JOB[] = EvalJob(iter, current_weights_version(), chunks, Dict(), time())
        println("Eval job created for iter $iter: $(length(chunks)) chunks of $CHUNK_SIZE games")
    end
end
```

#### New API Endpoints

All endpoints acquire `EVAL_LOCK` for thread safety.

**`GET /api/eval/status`** — check if eval work is available
```json
Response: {"eval_iter": 20, "total_chunks": 80, "completed": 35, "available": 12}
// or {"eval_iter": 0} if no eval in progress
```

**`POST /api/eval/checkout`** — claim a chunk
```json
Request: {"client_name": "jarvis-cpu"}
Response: {
    "chunk_id": 42,
    "position_indices": [1, 50],
    "az_is_white": false,
    "weights_version": 20
}
// or {"chunk_id": 0} if no chunks available
```

Logic:
- Acquire EVAL_LOCK
- Find first unclaimed, non-expired chunk
- Mark as checked_out_by client, set checkout_time
- Return position range (client loads positions from NFS or requests via separate endpoint)

**Positions**: Clients with NFS access load positions directly from the shared file. For future non-NFS clients (cloud, WASM), add a `GET /api/eval/positions?range=1:50` endpoint that serves serialized positions via MsgPack. Position data is small (~25 bytes × 50 = 1.25 KB per chunk).

**Eval weights**: Client downloads weights via existing `/api/weights` endpoint, using the pinned `weights_version` from the checkout response. Eval weights are stored as separate `FastWeights` instances on the client, distinct from self-play weights.

**`POST /api/eval/submit`** — submit completed chunk
```json
Request: {
    "chunk_id": 42,
    "rewards": [0.5, -1.0, ...],
    "value_samples": [{"nn_val": 0.3, "opponent_val": 0.4, "is_contact": false}, ...]
}
Response: {"accepted": true, "eval_complete": false}
```

Logic:
- Acquire EVAL_LOCK
- Validate chunk_id and that client holds the checkout
- Store results with `az_is_white` from chunk metadata
- If all chunks complete → call `finalize_eval!()`, set `EVAL_JOB[] = nothing`
- Return whether the full eval is now complete

**`POST /api/eval/heartbeat`** — extend checkout lease
```json
Request: {"client_name": "jarvis-cpu", "chunk_id": 42}
Response: {"lease_extended": true}
// or {"lease_extended": false, "reason": "chunk expired or reclaimed"} if lease already expired
```

Client should stop working on a chunk if heartbeat returns `lease_extended: false`.

#### Checkout Expiry + Job Timeout

Background task (runs every 60s):
```julia
function expire_stale_eval_state!()
    lock(EVAL_LOCK) do
        job = EVAL_JOB[]
        job === nothing && return

        now = time()

        # Job-level timeout: abandon eval if no progress for 30 min
        if now - job.created_at > EVAL_JOB_TIMEOUT_SECONDS
            n_completed = count(c -> c.completed, job.chunks)
            if n_completed < length(job.chunks)
                println("WARNING: Eval job iter $(job.iter) timed out ($n_completed/$(length(job.chunks)) chunks). Abandoning.")
                EVAL_JOB[] = nothing
                return
            end
        end

        # Chunk-level expiry
        for chunk in job.chunks
            if chunk.checked_out_by !== nothing && !chunk.completed
                if now - chunk.checkout_time > CHECKOUT_LEASE_SECONDS
                    println("Eval chunk $(chunk.chunk_id) expired (was: $(chunk.checked_out_by)), returning to pool")
                    chunk.checked_out_by = nothing
                end
            end
        end
    end
end
```

#### Result Aggregation

When all chunks submitted, split by `az_is_white` for per-side stats:

```julia
function finalize_eval!(job::EvalJob)
    white_rewards = Float64[]
    black_rewards = Float64[]
    all_vsamples = PositionValueSample[]

    for result in values(job.results)
        if result.az_is_white
            append!(white_rewards, result.rewards)
        else
            append!(black_rewards, result.rewards)
        end
        append!(all_vsamples, result.value_samples)
    end

    all_rewards = vcat(white_rewards, black_rewards)
    equity = mean(all_rewards)
    win_pct = 100 * count(r -> r > 0, all_rewards) / length(all_rewards)
    white_equity = mean(white_rewards)
    black_equity = mean(black_rewards)

    # Value accuracy stats
    nn_vals = [s.nn_val for s in all_vsamples]
    opp_vals = [s.opponent_val for s in all_vsamples]
    value_mse = mean((nn_vals .- opp_vals).^2)
    value_corr = cor(nn_vals, opp_vals)

    # Log to TensorBoard
    log_value(TB_LOGGER, "eval/equity/value", equity, step=job.iter)
    log_value(TB_LOGGER, "eval/win_pct/value", win_pct, step=job.iter)
    log_value(TB_LOGGER, "eval/white_equity/value", white_equity, step=job.iter)
    log_value(TB_LOGGER, "eval/black_equity/value", black_equity, step=job.iter)
    log_value(TB_LOGGER, "eval/value_mse/value", value_mse, step=job.iter)
    log_value(TB_LOGGER, "eval/value_corr/value", value_corr, step=job.iter)
    log_value(TB_LOGGER, "eval/games/value", length(all_rewards), step=job.iter)

    EVAL_JOB[] = nothing
    println("Eval iter $(job.iter) complete: equity=$(round(equity, digits=4)), win%=$(round(win_pct, digits=1)), $(length(all_rewards)) games")
end
```

### Client Side

#### New flag

```
--eval-capable    Enable eval mode (client will do eval when available)
--wildbg-lib      Path to wildbg shared library (required if eval-capable)
```

#### Eval Weight Management

Eval weights are stored separately from self-play weights to avoid corruption:

```julia
mutable struct EvalWeightState
    iter::Int           # eval iteration these weights correspond to
    fast_weights::Any   # separate FastWeights instance (or nothing)
end

const EVAL_WEIGHT_STATE = EvalWeightState(0, nothing)
```

When a new eval job is detected, the client downloads weights for that specific version and builds a fresh `FastWeights` from them. Self-play continues using the existing `CONTACT_FAST_WEIGHTS` / `RACE_FAST_WEIGHTS`.

#### Worker Pause/Resume

The client's continuous self-play workers run in infinite loops on spawned threads. To switch to eval mode, use a shared atomic flag:

```julia
const PAUSE_SELFPLAY = Threads.Atomic{Bool}(false)
const ACTIVE_SELFPLAY_GAMES = Threads.Atomic{Int}(0)

# In continuous_worker():
while true
    if PAUSE_SELFPLAY[]
        sleep(0.1)
        continue
    end
    Threads.atomic_add!(ACTIVE_SELFPLAY_GAMES, 1)
    try
        # ... normal self-play ...
    finally
        Threads.atomic_sub!(ACTIVE_SELFPLAY_GAMES, 1)
    end
end
```

When entering eval mode:
1. Set `PAUSE_SELFPLAY[] = true`
2. Wait for in-flight games to finish (~1-2 seconds)
3. Run eval using the same worker threads
4. Set `PAUSE_SELFPLAY[] = false`

This is simple, avoids thread lifecycle issues, and reuses the existing thread pool.

#### Client Eval Loop

```julia
function check_and_do_eval!()
    # Check if eval work available
    resp = HTTP.get("$SERVER/api/eval/status")
    status = JSON.parse(resp.body)
    status["eval_iter"] == 0 && return false

    eval_iter = status["eval_iter"]

    # Download eval weights if needed (once per eval iter)
    if EVAL_WEIGHT_STATE.iter != eval_iter
        load_eval_weights!(EVAL_WEIGHT_STATE, eval_iter)
    end

    # Pause self-play workers and wait for in-flight games to finish
    PAUSE_SELFPLAY[] = true
    while ACTIVE_SELFPLAY_GAMES[] > 0
        sleep(0.05)
    end

    # Open wildbg backends (keep alive across chunks within one eval)
    wildbg_backends = [begin
        wb = BackgammonNet.WildbgBackend(nets=WILDBG_NETS_VARIANT)
        BackgammonNet.open!(wb)
        wb
    end for _ in 1:NUM_WORKERS]

    # Build eval agents
    eval_oracle = make_fast_oracle(EVAL_WEIGHT_STATE.fast_weights)
    eval_params = MctsParams(num_iters_per_turn=EVAL_MCTS_ITERS, cpuct=1.5,
                             temperature=ConstSchedule(0.0), dirichlet_noise_ϵ=0.0)
    az = MctsAgent(eval_oracle, eval_batch_oracle, eval_params, 50, gspec, nothing)

    # Process chunks until none available
    while true
        resp = HTTP.post("$SERVER/api/eval/checkout",
                         body=MsgPack.pack(Dict("client_name" => CLIENT_NAME)))
        chunk = MsgPack.unpack(resp.body)
        chunk["chunk_id"] == 0 && break

        positions = load_eval_positions(chunk["position_indices"])
        az_is_white = chunk["az_is_white"]

        # Play games in parallel using worker threads
        rewards, vsamples = parallel_eval_chunk(az, wildbg_backends, positions, az_is_white, chunk["chunk_id"])

        # Submit results
        HTTP.post("$SERVER/api/eval/submit",
                  body=MsgPack.pack(Dict("chunk_id" => chunk["chunk_id"],
                                          "rewards" => rewards,
                                          "value_samples" => vsamples)))
    end

    # Cleanup
    for wb in wildbg_backends
        BackgammonNet.close(wb)
    end

    # Resume self-play with latest training weights
    PAUSE_SELFPLAY[] = false
    sync_weights!()

    return true
end
```

#### Heartbeat

During `parallel_eval_chunk`, a background task sends heartbeats:

```julia
function parallel_eval_chunk(az, wildbg_backends, positions, az_is_white, chunk_id)
    chunk_done = Threads.Atomic{Bool}(false)

    # Heartbeat in background
    heartbeat_task = @async begin
        while !chunk_done[]
            try
                resp = HTTP.post("$SERVER/api/eval/heartbeat",
                    body=MsgPack.pack(Dict("client_name" => CLIENT_NAME, "chunk_id" => chunk_id)))
                result = MsgPack.unpack(resp.body)
                if !result["lease_extended"]
                    println("WARNING: Chunk $chunk_id lease lost, abandoning")
                    break
                end
            catch e
                # Server busy, retry next cycle
            end
            sleep(60)
        end
    end

    # Play games with Threads.@threads work-stealing (same pattern as current eval)
    rewards = Vector{Float64}(undef, length(positions))
    vsamples = Vector{Vector{PositionValueSample}}(undef, length(positions))
    claimed = Threads.Atomic{Int}(0)

    Threads.@threads for tid in 1:length(wildbg_backends)
        wb = ExternalAgent(wildbg_backends[tid])
        while true
            job = Threads.atomic_add!(claimed, 1) + 1
            job > length(positions) && break
            pos = positions[job]
            w, b = az_is_white ? (az, wb) : (wb, az)
            result = play_game(w, b, pos; record_value_comparison=true,
                               value_oracle=eval_value_fn, opponent_value_fn=wildbg_value_fn,
                               seed=job + chunk_id * 10000)
            rewards[job] = result.reward
            vsamples[job] = result.value_samples
        end
    end

    chunk_done[] = true
    return rewards, vcat(vsamples...)
end
```

---

## v7 Deployment

### Jarvis (i7-10700K, 16 threads)
- Training server: no server-side eval (create_eval_job! only)
- Self-play client: `--num-workers 12 --eval-capable --wildbg-lib /path/to/libwildbg.so`
- During eval: self-play paused, 12 workers on eval
- Eval throughput: 12 workers × 17.3 games/min = ~208 games/min
- 4000 games / 208 = **~19 min per eval** (uncontested)
- During self-play: 12 workers producing samples

### Neo (M3 Max, 32 threads)
- Self-play client: `--num-workers 32` (no eval)
- Pure self-play: 32 workers × 8.8 games/min = ~282 games/min
- Neo is 3.6x less efficient at eval per core — keep it on self-play
- Never pauses — continuous sample production even during Jarvis eval

### Alternative: Both eval-capable
- If faster eval needed, both clients take chunks
- Neo's 32 workers + Jarvis's 12 workers = 44 workers on eval
- 4000 games / (44 × ~10 avg games/min) = **~9 min per eval**
- Trade-off: ALL self-play pauses during eval

---

## Migration Path

### Training loop change (training_server.jl)

Replace:
```julia
if iter % EVAL_INTERVAL == 0
    run_eval!(network, iter)  # blocks for 20-34 min
end
```

With:
```julia
if iter % EVAL_INTERVAL == 0
    create_eval_job!(iter)  # instant, non-blocking
end
```

### Standalone eval scripts

`eval_race.jl` and `eval_vs_wildbg.jl` continue to work for one-off benchmarks. They switch to using `play_game()` from `game_loop.jl` but keep their own parallel worker orchestration (`Threads.@threads` work-stealing). These are not affected by the distributed eval feature.

---

## Implementation Order

1. **`src/game_loop.jl`** — unified play_game(), GameAgent types, GameResult, TraceEntry with full bear-off handling
2. **Wire into selfplay_client.jl** — replace inline game loop with play_game() calls; verify convert_trace_to_samples works with new GameResult
3. **Wire into training_server.jl** — replace eval_race_game_server() with play_game()
4. **Wire into eval_race.jl / eval_vs_wildbg.jl** — replace inline loops
5. **Test** — run self-play + eval on same checkpoint, verify equity within ±0.02 and win% within ±1%
6. **`src/distributed/eval_manager.jl`** — EvalJob, EvalChunk, create/checkout/submit/expire with EVAL_LOCK
7. **Add API endpoints** — 4 new routes in server.jl (status, checkout, submit, heartbeat)
8. **Client eval mode** — --eval-capable flag, PAUSE_SELFPLAY atomic, eval weight management, check_and_do_eval!(), heartbeat
9. **Remove run_eval!()** from training loop, replace with create_eval_job!()
10. **Update launch_v7.sh** — Jarvis eval-capable, Neo self-play only
11. **Test end-to-end** — verify distributed eval results match standalone eval_race.jl within noise (±0.02 equity, ±1% win%)

Steps 1-5 can be tested independently (pure refactor, no behavior change).
Steps 6-11 add the distributed eval feature.

---

## Risks and Mitigations

**Risk: Eval results differ after refactor**
- Mitigation: Run both old and new paths on same checkpoint, compare equity/win% within noise margin (±0.02 equity at 2000 games)

**Risk: Chunk checkout overhead slows eval**
- Mitigation: 50-game chunks = ~15s of work on Jarvis. HTTP round-trip is <100ms. Overhead < 1%.

**Risk: Client dies mid-eval, chunks stuck**
- Mitigation: 5-minute lease with heartbeat. Expired chunks return to pool. Worst case: 5 min delay per lost chunk. Job-level 30-minute timeout abandons stale eval jobs entirely.

**Risk: No eval-capable clients connected**
- Mitigation: Job-level timeout (30 min). Server logs warning and abandons eval. Training continues unblocked.

**Risk: Wildbg backend thread safety**
- Mitigation: Create per-worker backend instances (already done in current eval code). Backends kept alive across chunks within one eval job to avoid repeated disk loads, closed after eval completes.

**Risk: Eval weights corrupt self-play weights**
- Mitigation: Eval uses separate `FastWeights` instances stored in `EVAL_WEIGHT_STATE`. Self-play weights in `CONTACT_FAST_WEIGHTS` / `RACE_FAST_WEIGHTS` are never touched during eval.

**Risk: Eval weights lag behind training**
- Mitigation: Eval job is pinned to a specific weights version. Clients download those exact weights. Training continues with newer weights — this is fine, eval measures a snapshot.

**Risk: Self-play workers don't pause cleanly**
- Mitigation: `PAUSE_SELFPLAY` atomic flag + `ACTIVE_SELFPLAY_GAMES` atomic counter. Workers increment before starting a game, decrement after. Eval waits until counter hits zero — deterministic, no sleep-based race conditions.

---

## Implementation Notes

### Deterministic Worker Synchronization
Use `ACTIVE_SELFPLAY_GAMES::Threads.Atomic{Int}(0)` instead of a 2-second sleep:
- Workers: `atomic_add!(ACTIVE_SELFPLAY_GAMES, 1)` before game, `atomic_sub!(ACTIVE_SELFPLAY_GAMES, 1)` after
- Eval entry: set `PAUSE_SELFPLAY[] = true`, then `while ACTIVE_SELFPLAY_GAMES[] > 0; sleep(0.05); end`
- Guarantees complete resource isolation during eval

### Network Fault Tolerance
- **Submit retries**: Exponential backoff (1s, 2s, 4s, 8s, max 30s) for `/api/eval/submit`. Cap at 30s to stay within 5-minute lease.
- **Heartbeat resilience**: 3 consecutive heartbeat failures before chunk abandonment (single transient failure is ignored).
- **Checkout retry**: If checkout fails, sleep 5s and retry (server may be temporarily busy).

### Payload Management
- MsgPack for all eval wire protocol (consistent with existing sample upload protocol)
- Server `/api/eval/submit` body size limit: 50MB (sufficient for 50 games × ~20 value samples each)
- Position data for non-NFS clients: ~1.25 KB per chunk (25 bytes × 50 positions) — negligible

### Result Integrity
- Weight version from checkout response is a hard requirement — client must verify weights match before playing
- Side-specific aggregation uses `EvalChunkResult.az_is_white` (not positional ordering) to prevent side-swap errors
