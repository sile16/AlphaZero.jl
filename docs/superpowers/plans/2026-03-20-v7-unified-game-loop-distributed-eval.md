# v7: Unified Game Loop + Distributed Eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify 5 duplicate game loops into one `play_game()` function, then move eval from the training server into self-play clients via a chunked work queue so the server never blocks.

**Architecture:** Part A creates `src/game_loop.jl` with agent abstractions (`MctsAgent`, `ExternalAgent`) and a single `play_game()` function parameterized for self-play vs eval. Part B adds `src/distributed/eval_manager.jl` for server-side eval job management and client-side eval mode with worker pause/resume. The server creates eval jobs (non-blocking), clients claim chunks, play games, and submit results.

**Tech Stack:** Julia 1.12, BatchedMCTS, FastWeights inference, BackgammonNet.jl, HTTP.jl, MsgPack.jl, TensorBoardLogger.jl

**Spec:** `docs/superpowers/specs/2026-03-20-v7-unified-game-loop-distributed-eval-design.md`

---

## File Structure

### New Files
- `src/game_loop.jl` — Unified game loop: `GameAgent`, `MctsAgent`, `ExternalAgent`, `GameResult`, `TraceEntry`, `PositionValueSample`, `play_game()`, `select_action()`, `create_player()`
- `src/distributed/eval_manager.jl` — Server eval job manager: `EvalChunk`, `EvalJob`, `EvalChunkResult`, `create_eval_job!()`, `checkout_chunk!()`, `submit_chunk!()`, `expire_stale_eval_state!()`, `finalize_eval!()`
- `test/test_game_loop.jl` — Tests for unified game loop
- `test/test_eval_manager.jl` — Tests for eval manager
- `scripts/launch_v7.sh` — v7 launch script

### Modified Files
- `src/AlphaZero.jl` — Include `game_loop.jl`, export new types
- `scripts/selfplay_client.jl` — Replace inline game loop with `play_game()` calls; add `--eval-capable` flag, `PAUSE_SELFPLAY`/`ACTIVE_SELFPLAY_GAMES` atomics, `check_and_do_eval!()`
- `scripts/training_server.jl` — Remove `run_eval!()`/`eval_race_game_server()`; add eval manager routes; replace eval call with `create_eval_job!()`
- `src/distributed/server.jl` — Add 4 new HTTP routes for eval API
- `scripts/eval_race.jl` — Replace inline game loop with `play_game()` calls
- `scripts/eval_vs_wildbg.jl` — Replace inline game loop with `play_game()` calls

---

## PART A: Unified Game Loop (Pure Refactor)

### Task 1: Create game_loop.jl with types and play_game()

**Files:**
- Create: `src/game_loop.jl`
- Modify: `src/AlphaZero.jl`
- Test: `test/test_game_loop.jl`

This is the core of the refactor. All types and the main `play_game()` function.

- [ ] **Step 1: Write test file for game loop types**

Create `test/test_game_loop.jl` with basic type construction tests:

```julia
using Test
using AlphaZero
using AlphaZero: GameLoop

@testset "GameLoop types" begin
    @testset "TraceEntry construction" begin
        te = GameLoop.TraceEntry(
            nothing,     # state
            0,           # player
            1,           # action
            [1, 2, 3],   # legal_actions
            Float32[0.5, 0.3, 0.2],  # policy
            false,       # is_chance
            false,       # is_bearoff
            true          # is_contact
        )
        @test te.player == 0
        @test te.action == 1
        @test length(te.legal_actions) == 3
    end

    @testset "PositionValueSample construction" begin
        pvs = GameLoop.PositionValueSample(0.5, 0.3, true)
        @test pvs.nn_val == 0.5
        @test pvs.opponent_val == 0.3
        @test pvs.is_contact == true
    end

    @testset "GameResult construction" begin
        gr = GameLoop.GameResult(
            1.0,                              # reward
            GameLoop.TraceEntry[],            # trace
            GameLoop.PositionValueSample[],   # value_samples
            10,                                # num_moves
            false,                             # bearoff_truncated
            nothing,                           # first_bearoff_equity
            nothing                            # first_bearoff_white_playing
        )
        @test gr.reward == 1.0
        @test gr.num_moves == 10
        @test gr.bearoff_truncated == false
    end

    @testset "MctsAgent construction" begin
        oracle = s -> (ones(Float32, 5), 0.5f0)
        params = AlphaZero.MctsParams(
            num_iters_per_turn=10,
            dirichlet_noise_ϵ=0.0,
            dirichlet_noise_α=1.0)
        # Just test that the struct can be constructed
        agent = GameLoop.MctsAgent(oracle, nothing, params, 10, nothing, nothing)
        @test agent.batch_size == 10
    end

    @testset "ExternalAgent construction" begin
        agent = GameLoop.ExternalAgent(:mock_backend)
        @test agent.backend == :mock_backend
    end
end
```

- [ ] **Step 2: Create src/game_loop.jl with type definitions**

Create `src/game_loop.jl`:

```julia
module GameLoop

using ..AlphaZero: GI, BatchedMCTS, MctsParams
import BackgammonNet
import Random

export GameAgent, MctsAgent, ExternalAgent, GameResult, TraceEntry, PositionValueSample
export play_game, create_player, select_action

# ── Agent Types ──────────────────────────────────────────────────────

abstract type GameAgent end

struct MctsAgent <: GameAgent
    oracle           # single state → (policy, value)
    batch_oracle     # batch states → [(policy, value)...] or nothing
    mcts_params::MctsParams
    batch_size::Int
    gspec            # game spec for BatchedMCTS
    bearoff_eval     # optional bearoff evaluator closure, or nothing
end

struct ExternalAgent <: GameAgent
    backend          # BackgammonNet.WildbgBackend or similar
end

# ── Result Types ─────────────────────────────────────────────────────

struct TraceEntry
    state           # game state (opaque)
    player::Int     # current player (0=white, 1=black)
    action::Int     # chosen action
    legal_actions::Vector{Int}  # all legal actions (for sparse policy reconstruction)
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

struct GameResult
    reward::Float64
    trace::Vector{TraceEntry}
    value_samples::Vector{PositionValueSample}
    num_moves::Int
    bearoff_truncated::Bool
    first_bearoff_equity::Union{Nothing, Float64}
    first_bearoff_white_playing::Union{Nothing, Bool}
end

# ── Player Lifecycle ─────────────────────────────────────────────────

function create_player(agent::MctsAgent)
    BatchedMCTS.BatchedMctsPlayer(
        agent.gspec, agent.oracle, agent.mcts_params;
        batch_size=agent.batch_size, batch_oracle=agent.batch_oracle,
        bearoff_evaluator=agent.bearoff_eval)
end

create_player(::ExternalAgent) = nothing

# ── Action Selection ─────────────────────────────────────────────────

function select_action(agent::MctsAgent, env, player)
    actions, policy = BatchedMCTS.think(player, env)
    # Temperature-based sampling is handled inside BatchedMCTS via MctsParams
    # For eval (temp=0), think() returns deterministic argmax policy
    # For training (temp>0), we sample from the policy
    if maximum(policy) ≈ 1.0
        # Deterministic (eval mode or single dominant action)
        idx = argmax(policy)
    else
        idx = _sample_policy(policy)
    end
    return actions[idx], policy, actions
end

function select_action(agent::ExternalAgent, env, ::Nothing)
    action = BackgammonNet.agent_move(BackgammonNet.BackendAgent(agent.backend), env.game)
    return action, Float32[], Int[]
end

function _sample_policy(policy::Vector{Float32})
    r = rand(Float32)
    cumsum = 0.0f0
    for i in eachindex(policy)
        cumsum += policy[i]
        if r <= cumsum
            return i
        end
    end
    return length(policy)
end

# ── Core Game Loop ───────────────────────────────────────────────────

"""
    play_game(white, black, position; kwargs...) -> GameResult

Unified game loop for both self-play and evaluation.

# Arguments
- `white::GameAgent` — agent playing white
- `black::GameAgent` — agent playing black
- `position` — starting position (game state or tuple for initialization)

# Keyword Arguments
- `record_trace::Bool=false` — record position/policy trace (for training samples)
- `record_value_comparison::Bool=false` — record NN vs opponent value (for eval stats)
- `value_oracle=nothing` — fn(state) → Float64, for value comparison (separate from agent oracle)
- `opponent_value_fn=nothing` — fn(state) → Float64, e.g., wildbg evaluate
- `bearoff_truncation::Bool=false` — truncate game at first bear-off position
- `bearoff_lookup=nothing` — fn(state) → Union{Float64, Nothing}, for bear-off detection/values
- `rng=Random.default_rng()` — random number generator
- `seed::Int=0` — seed for rng (0 = don't seed)
"""
function play_game(
    white::GameAgent, black::GameAgent, position;
    record_trace::Bool = false,
    record_value_comparison::Bool = false,
    value_oracle = nothing,
    opponent_value_fn = nothing,
    bearoff_truncation::Bool = false,
    bearoff_lookup = nothing,
    rng = Random.default_rng(),
    seed::Int = 0
)
    seed > 0 && Random.seed!(rng, seed)

    # Initialize game environment from position
    env = _init_env(position)

    trace = TraceEntry[]
    value_samples = PositionValueSample[]
    num_moves = 0
    bearoff_truncated = false
    first_bearoff_equity = nothing
    first_bearoff_white_playing = nothing
    reward = 0.0

    # Create MCTS players once per game (reused across moves)
    white_player = create_player(white)
    black_player = create_player(black)

    while !GI.game_terminated(env)
        if GI.is_chance_node(env)
            # Bear-off detection at chance nodes (pre-dice)
            if bearoff_lookup !== nothing
                bo_eq = bearoff_lookup(GI.current_state(env))
                if bo_eq !== nothing
                    # Record first bear-off if not yet seen
                    if first_bearoff_equity === nothing
                        first_bearoff_equity = bo_eq
                        first_bearoff_white_playing = GI.white_playing(env)
                    end
                    # Truncate if requested
                    if bearoff_truncation
                        bearoff_truncated = true
                        reward = bo_eq
                        break
                    end
                end
            end

            # Sample chance outcome (passthrough)
            outcomes = GI.chance_outcomes(env)
            idx = _sample_chance(rng, outcomes)
            GI.apply_chance!(env, outcomes[idx][1])
            continue
        end

        avail = GI.available_actions(env)

        # Single-action bypass — skip MCTS for forced moves
        if length(avail) == 1
            if record_trace
                state = GI.current_state(env)
                push!(trace, TraceEntry(
                    state, GI.white_playing(env) ? 0 : 1,
                    avail[1], avail, Float32[1.0], false,
                    _is_bearoff_position(state), _is_contact_position(state)))
            end
            GI.play!(env, avail[1])
            num_moves += 1
            continue
        end

        is_white_turn = GI.white_playing(env)
        current_agent = is_white_turn ? white : black
        player = is_white_turn ? white_player : black_player

        state = GI.current_state(env)
        action, policy, legal = select_action(current_agent, env, player)
        num_moves += 1

        if record_trace && current_agent isa MctsAgent
            push!(trace, TraceEntry(
                state, is_white_turn ? 0 : 1,
                action, legal, policy, false,
                _is_bearoff_position(state), _is_contact_position(state)))
        end

        if record_value_comparison && current_agent isa MctsAgent &&
           value_oracle !== nothing && opponent_value_fn !== nothing
            nn_v = value_oracle(state)
            opp_v = opponent_value_fn(state)
            push!(value_samples, PositionValueSample(nn_v, opp_v, _is_contact_position(state)))
        end

        GI.play!(env, action)
    end

    if !bearoff_truncated
        reward = GI.white_reward(env)
    end

    # Reset players for potential reuse by caller
    white_player !== nothing && BatchedMCTS.reset_player!(white_player)
    black_player !== nothing && BatchedMCTS.reset_player!(black_player)

    return GameResult(reward, trace, value_samples, num_moves,
                      bearoff_truncated, first_bearoff_equity, first_bearoff_white_playing)
end

# ── Helpers ──────────────────────────────────────────────────────────

function _sample_chance(rng, outcomes)
    r = rand(rng)
    cumsum = 0.0
    for (i, (_, prob)) in enumerate(outcomes)
        cumsum += prob
        if r <= cumsum
            return i
        end
    end
    return length(outcomes)
end

function _init_env(position)
    # Position can be a game environment directly, or a tuple for BackgammonNet
    if position isa Tuple
        # BackgammonNet position tuple (p0, p1, bar, off, current_player, ...)
        return BackgammonNet.make_game(position)
    else
        # Already a game environment
        return position
    end
end

function _is_bearoff_position(state)
    state isa BackgammonNet.BackgammonGame ? !BackgammonNet.is_contact_position(state) && BackgammonNet.is_bearoff_k6(state) : false
end

function _is_contact_position(state)
    state isa BackgammonNet.BackgammonGame ? BackgammonNet.is_contact_position(state) : true
end

end # module GameLoop
```

- [ ] **Step 3: Add game_loop.jl to AlphaZero.jl module**

In `src/AlphaZero.jl`, add after the BatchedMCTS include:

```julia
include("game_loop.jl")
```

- [ ] **Step 4: Run type tests to verify compilation**

Run: `julia --project -e 'using AlphaZero; using AlphaZero: GameLoop; println("GameLoop loaded: ", names(GameLoop))'`

Expected: Module loads without error, lists exported names.

- [ ] **Step 5: Run test_game_loop.jl**

Run: `julia --project -e 'include("test/test_game_loop.jl")'`

Expected: All type construction tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/game_loop.jl src/AlphaZero.jl test/test_game_loop.jl
git commit -m "feat: add unified game loop module with types and play_game()"
```

---

### Task 2: Wire play_game() into selfplay_client.jl

**Files:**
- Modify: `scripts/selfplay_client.jl` (lines 664-769: `_play_games_loop()`)

Replace the inline game loop in `_play_games_loop()` with a call to `GameLoop.play_game()`. The `convert_trace_to_samples()` function stays — it receives the `GameResult` and produces upload-ready samples.

**Key mapping:**
- Current `_play_games_loop()` creates `BatchedMctsPlayer` at line 674, loops from line 695 to 741
- Replace with: construct `MctsAgent`, call `play_game()`, extract trace from `GameResult`
- The three-way `convert_trace_to_samples` branching uses `result.bearoff_truncated`, `result.first_bearoff_equity`, etc.

- [ ] **Step 1: Study the current _play_games_loop() carefully**

Read `scripts/selfplay_client.jl` lines 660-770 to understand all edge cases.

- [ ] **Step 2: Replace the game loop in _play_games_loop()**

In `_play_games_loop()`, replace the BatchedMctsPlayer creation + while loop (lines ~674-741) with:

```julia
# Create agent (replaces BatchedMctsPlayer creation)
using AlphaZero: GameLoop
az_agent = GameLoop.MctsAgent(
    CPU_SINGLE_ORACLE, CPU_BATCH_ORACLE,
    mcts_params, INFERENCE_BATCH_SIZE, gspec, BEAROFF_EVALUATOR)

# Play game (replaces inline while loop)
result = GameLoop.play_game(az_agent, az_agent, start_pos;
    record_trace=true,
    bearoff_truncation=BEAROFF_TRUNCATION,
    bearoff_lookup=BEAROFF_CHANCE_LOOKUP,
    rng=rng, seed=game_seed)
```

- [ ] **Step 3: Adapt convert_trace_to_samples() call**

Update the post-game sample conversion to use `GameResult` fields instead of inline trace variables. Map:
- `result.trace` → extract states, policies, actions, is_chance flags
- `result.reward` → final_reward
- `result.bearoff_truncated` → use bear-off equity path
- `result.first_bearoff_equity` → first_bearoff_bo
- `result.first_bearoff_white_playing` → first_bearoff_wp

- [ ] **Step 4: Test self-play produces samples**

Run a quick 10-game self-play test:
```bash
julia --threads 4 --project scripts/selfplay_client.jl --server http://127.0.0.1:9090 --api-key test --num-workers 2 --client-name test
```

Verify: Client starts, plays games, produces samples (check log output for "Batch N: M games, K samples").

- [ ] **Step 5: Commit**

```bash
git add scripts/selfplay_client.jl
git commit -m "refactor: replace selfplay inline game loop with play_game()"
```

---

### Task 3: Wire play_game() into training_server.jl eval

**Files:**
- Modify: `scripts/training_server.jl` (lines 405-541: `EvalAlphaZeroAgent`, `eval_race_game_server()`, `run_eval!()`)

Replace the eval game loop with `GameLoop.play_game()`. The `run_eval!()` orchestration (parallel workers, result aggregation) stays — only the per-game function changes.

- [ ] **Step 1: Replace eval_race_game_server() with play_game()**

Replace the `eval_race_game_server()` function body. Keep the same function signature for `run_eval!()` but internally use:

```julia
az = GameLoop.MctsAgent(single_oracle, batch_oracle, eval_mcts_params, 50, gspec, nothing)
wb = GameLoop.ExternalAgent(wildbg_backend)

w, b = az_is_white ? (az, wb) : (wb, az)
result = GameLoop.play_game(w, b, position;
    record_value_comparison=true,
    value_oracle=nn_value_fn,
    opponent_value_fn=wildbg_value_fn,
    seed=seed)
```

- [ ] **Step 2: Delete EvalAlphaZeroAgent struct**

Remove the `EvalAlphaZeroAgent` struct and its `agent_move` method (lines 405-422). No longer needed.

- [ ] **Step 3: Test eval still works**

Run: `julia --threads 4 --project scripts/eval_race.jl <checkpoint> --width=128 --blocks=3 --num-workers=2 --mcts-iters=100 --num-games=10`

Verify: Eval completes, reports equity and win%.

- [ ] **Step 4: Commit**

```bash
git add scripts/training_server.jl
git commit -m "refactor: replace training server eval loop with play_game()"
```

---

### Task 4: Wire play_game() into standalone eval scripts

**Files:**
- Modify: `scripts/eval_race.jl` (lines 109-186: `AlphaZeroAgent`, `eval_race_game()`)
- Modify: `scripts/eval_vs_wildbg.jl` (lines 183-269: `AlphaZeroAgent`, `eval_game()`)

- [ ] **Step 1: Replace eval_race.jl game loop**

Replace `eval_race_game()` with `GameLoop.play_game()` call. Keep the parallel orchestration in `main()`.

- [ ] **Step 2: Replace eval_vs_wildbg.jl game loop**

Replace `eval_game()` with `GameLoop.play_game()` call. Keep the parallel orchestration and dual-model routing.

- [ ] **Step 3: Test eval_race.jl**

Run a quick eval with small game count to verify results are reasonable.

- [ ] **Step 4: Commit**

```bash
git add scripts/eval_race.jl scripts/eval_vs_wildbg.jl
git commit -m "refactor: replace standalone eval scripts with play_game()"
```

---

### Task 5: Verify Part A refactor — compare old vs new eval results

- [ ] **Step 1: Run eval_race.jl on a known checkpoint with play_game()**

Run on the v6 iter 10 checkpoint (known result: +0.015 equity, 50.1% wins):
```bash
julia --threads 16 --project scripts/eval_race.jl \
  /home/sile/alphazero-server-race-v6/checkpoints/race_iter_10.data \
  --width=256 --blocks=5 --num-workers=12 --mcts-iters=600 --num-games=100
```

Expected: Equity within ±0.05 of +0.015 (100 games has wider noise than 2000).

- [ ] **Step 2: Commit Part A complete**

```bash
git commit --allow-empty -m "milestone: Part A complete — unified game loop verified"
```

---

## PART B: Distributed Eval

### Task 6: Create eval_manager.jl

**Files:**
- Create: `src/distributed/eval_manager.jl`
- Test: `test/test_eval_manager.jl`

- [ ] **Step 1: Write eval manager tests**

Create `test/test_eval_manager.jl`:

```julia
using Test

# Test the eval manager logic in isolation
@testset "EvalManager" begin
    include("../src/distributed/eval_manager.jl")

    @testset "create_eval_job!" begin
        # Mock EVAL_POSITIONS as 100 positions
        positions = collect(1:100)
        job = EvalManager.create_eval_job(100, positions, 1, 50)
        @test job.iter == 100
        # 100 positions × 2 sides / 50 per chunk = 4 chunks
        @test length(job.chunks) == 4
        @test job.chunks[1].az_is_white == true
        @test job.chunks[3].az_is_white == false
    end

    @testset "checkout_chunk!" begin
        positions = collect(1:100)
        job = EvalManager.create_eval_job(1, positions, 1, 50)

        chunk = EvalManager.checkout_chunk!(job, "client-a")
        @test chunk !== nothing
        @test chunk.checked_out_by == "client-a"
        @test chunk.checkout_time > 0

        # Second checkout gets different chunk
        chunk2 = EvalManager.checkout_chunk!(job, "client-b")
        @test chunk2 !== nothing
        @test chunk2.chunk_id != chunk.chunk_id
    end

    @testset "submit_chunk!" begin
        positions = collect(1:50)
        job = EvalManager.create_eval_job(1, positions, 1, 50)

        chunk = EvalManager.checkout_chunk!(job, "client-a")
        rewards = rand(50)
        ok = EvalManager.submit_chunk!(job, chunk.chunk_id, "client-a", rewards, [])
        @test ok == true
        @test chunk.completed == true
        @test haskey(job.results, chunk.chunk_id)
    end

    @testset "expire_stale_checkouts!" begin
        positions = collect(1:50)
        job = EvalManager.create_eval_job(1, positions, 1, 50)

        chunk = EvalManager.checkout_chunk!(job, "client-a")
        # Manually set old checkout time
        chunk.checkout_time = time() - 600  # 10 min ago

        EvalManager.expire_stale_checkouts!(job, 300)
        @test chunk.checked_out_by === nothing  # expired
    end

    @testset "all chunks complete triggers finalize" begin
        positions = collect(1:50)
        job = EvalManager.create_eval_job(1, positions, 1, 50)

        # Checkout and submit all chunks
        for _ in 1:length(job.chunks)
            chunk = EvalManager.checkout_chunk!(job, "client-a")
            n = length(chunk.position_range)
            EvalManager.submit_chunk!(job, chunk.chunk_id, "client-a", rand(n), [])
        end

        @test EvalManager.is_complete(job)
    end
end
```

- [ ] **Step 2: Create src/distributed/eval_manager.jl**

```julia
module EvalManager

using Statistics: mean, cor

export EvalChunk, EvalJob, EvalChunkResult
export create_eval_job, checkout_chunk!, submit_chunk!, expire_stale_checkouts!
export is_complete, finalize_eval

# ── Types ────────────────────────────────────────────────────────────

mutable struct EvalChunk
    chunk_id::Int
    position_range::UnitRange{Int}
    az_is_white::Bool
    checked_out_by::Union{Nothing, String}
    checkout_time::Float64
    completed::Bool
end

struct EvalChunkResult
    chunk_id::Int
    az_is_white::Bool
    rewards::Vector{Float64}
    value_nn::Vector{Float64}
    value_opp::Vector{Float64}
    value_is_contact::Vector{Bool}
end

mutable struct EvalJob
    iter::Int
    weights_version::Int
    chunks::Vector{EvalChunk}
    results::Dict{Int, EvalChunkResult}
    created_at::Float64
end

# ── Job Creation ─────────────────────────────────────────────────────

function create_eval_job(iter::Int, positions, weights_version::Int, chunk_size::Int=50)
    n_pos = length(positions)
    chunks = EvalChunk[]

    # White games
    for i in 1:chunk_size:n_pos
        r = i:min(i + chunk_size - 1, n_pos)
        push!(chunks, EvalChunk(length(chunks) + 1, r, true, nothing, 0.0, false))
    end
    # Black games
    for i in 1:chunk_size:n_pos
        r = i:min(i + chunk_size - 1, n_pos)
        push!(chunks, EvalChunk(length(chunks) + 1, r, false, nothing, 0.0, false))
    end

    return EvalJob(iter, weights_version, chunks, Dict{Int, EvalChunkResult}(), time())
end

# ── Chunk Operations ─────────────────────────────────────────────────

function checkout_chunk!(job::EvalJob, client_name::String)
    now = time()
    for chunk in job.chunks
        if !chunk.completed && chunk.checked_out_by === nothing
            chunk.checked_out_by = client_name
            chunk.checkout_time = now
            return chunk
        end
    end
    return nothing
end

function submit_chunk!(job::EvalJob, chunk_id::Int, client_name::String,
                       rewards::Vector{Float64}, value_data::Vector)
    idx = findfirst(c -> c.chunk_id == chunk_id, job.chunks)
    idx === nothing && return false

    chunk = job.chunks[idx]
    chunk.checked_out_by != client_name && return false

    # Parse value data
    nn_vals = Float64[]
    opp_vals = Float64[]
    is_contact = Bool[]
    for v in value_data
        if v isa Dict || v isa AbstractDict
            push!(nn_vals, Float64(v["nn_val"]))
            push!(opp_vals, Float64(v["opponent_val"]))
            push!(is_contact, Bool(v["is_contact"]))
        end
    end

    chunk.completed = true
    job.results[chunk_id] = EvalChunkResult(
        chunk_id, chunk.az_is_white, rewards,
        nn_vals, opp_vals, is_contact)
    return true
end

function extend_lease!(job::EvalJob, chunk_id::Int, client_name::String)
    idx = findfirst(c -> c.chunk_id == chunk_id, job.chunks)
    idx === nothing && return false
    chunk = job.chunks[idx]
    chunk.checked_out_by != client_name && return false
    chunk.completed && return false
    chunk.checkout_time = time()
    return true
end

function expire_stale_checkouts!(job::EvalJob, lease_seconds::Float64=300.0)
    now = time()
    for chunk in job.chunks
        if chunk.checked_out_by !== nothing && !chunk.completed
            if now - chunk.checkout_time > lease_seconds
                println("  Eval chunk $(chunk.chunk_id) expired (was: $(chunk.checked_out_by))")
                chunk.checked_out_by = nothing
            end
        end
    end
end

function is_complete(job::EvalJob)
    all(c -> c.completed, job.chunks)
end

function status(job::EvalJob)
    completed = count(c -> c.completed, job.chunks)
    available = count(c -> !c.completed && c.checked_out_by === nothing, job.chunks)
    return (eval_iter=job.iter, total_chunks=length(job.chunks),
            completed=completed, available=available)
end

# ── Result Aggregation ───────────────────────────────────────────────

function finalize_eval(job::EvalJob)
    white_rewards = Float64[]
    black_rewards = Float64[]
    all_nn = Float64[]
    all_opp = Float64[]

    for result in values(job.results)
        if result.az_is_white
            append!(white_rewards, result.rewards)
        else
            append!(black_rewards, result.rewards)
        end
        append!(all_nn, result.value_nn)
        append!(all_opp, result.value_opp)
    end

    all_rewards = vcat(white_rewards, black_rewards)
    n = length(all_rewards)
    equity = n > 0 ? mean(all_rewards) : 0.0
    win_pct = n > 0 ? 100.0 * count(r -> r > 0, all_rewards) / n : 0.0
    white_equity = length(white_rewards) > 0 ? mean(white_rewards) : 0.0
    black_equity = length(black_rewards) > 0 ? mean(black_rewards) : 0.0

    value_mse = length(all_nn) > 0 ? mean((all_nn .- all_opp).^2) : 0.0
    value_corr = length(all_nn) > 1 ? cor(all_nn, all_opp) : 0.0

    return (equity=equity, win_pct=win_pct,
            white_equity=white_equity, black_equity=black_equity,
            value_mse=value_mse, value_corr=value_corr,
            n_games=n)
end

end # module EvalManager
```

- [ ] **Step 3: Run eval manager tests**

Run: `julia --project -e 'include("test/test_eval_manager.jl")'`

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/distributed/eval_manager.jl test/test_eval_manager.jl
git commit -m "feat: add eval manager for distributed eval job tracking"
```

---

### Task 7: Add eval API endpoints to training server

**Files:**
- Modify: `scripts/training_server.jl` — add eval routes, replace `run_eval!()` with `create_eval_job!()`
- Modify: `src/distributed/server.jl` — register new routes

- [ ] **Step 1: Include eval_manager.jl in training_server.jl**

Add near the top of training_server.jl:
```julia
include(joinpath(@__DIR__, "..", "src", "distributed", "eval_manager.jl"))
using .EvalManager
```

Add server-level state:
```julia
const EVAL_JOB = Ref{Union{Nothing, EvalManager.EvalJob}}(nothing)
const EVAL_LOCK = ReentrantLock()
const EVAL_CHUNK_SIZE = 50
const EVAL_CHECKOUT_LEASE = 300.0
const EVAL_JOB_TIMEOUT = 1800.0
```

- [ ] **Step 2: Add 4 eval HTTP handlers**

Add handler functions to training_server.jl:

```julia
function handle_eval_status(req, state)
    lock(EVAL_LOCK) do
        job = EVAL_JOB[]
        if job === nothing
            return HTTP.Response(200, json_headers(), JSON3.write(Dict("eval_iter" => 0)))
        end
        s = EvalManager.status(job)
        return HTTP.Response(200, json_headers(), JSON3.write(Dict(
            "eval_iter" => s.eval_iter,
            "total_chunks" => s.total_chunks,
            "completed" => s.completed,
            "available" => s.available,
            "weights_version" => job.weights_version)))
    end
end

function handle_eval_checkout(req, state)
    body = MsgPack.unpack(req.body)
    client_name = body["client_name"]
    lock(EVAL_LOCK) do
        job = EVAL_JOB[]
        if job === nothing
            return HTTP.Response(200, [], MsgPack.pack(Dict("chunk_id" => 0)))
        end
        chunk = EvalManager.checkout_chunk!(job, client_name)
        if chunk === nothing
            return HTTP.Response(200, [], MsgPack.pack(Dict("chunk_id" => 0)))
        end
        return HTTP.Response(200, [], MsgPack.pack(Dict(
            "chunk_id" => chunk.chunk_id,
            "position_range_start" => first(chunk.position_range),
            "position_range_end" => last(chunk.position_range),
            "az_is_white" => chunk.az_is_white,
            "weights_version" => job.weights_version)))
    end
end

function handle_eval_submit(req, state)
    body = MsgPack.unpack(req.body)
    chunk_id = body["chunk_id"]
    client_name = body["client_name"]
    rewards = Float64.(body["rewards"])
    value_data = get(body, "value_samples", [])

    lock(EVAL_LOCK) do
        job = EVAL_JOB[]
        job === nothing && return HTTP.Response(404, [], "No eval job")

        ok = EvalManager.submit_chunk!(job, chunk_id, client_name, rewards, value_data)
        !ok && return HTTP.Response(400, [], "Invalid chunk or client")

        eval_complete = EvalManager.is_complete(job)
        if eval_complete
            result = EvalManager.finalize_eval(job)
            # Log to TensorBoard
            with_logger(TB_LOGGER) do
                @info "eval/equity" value=result.equity log_step_increment=0
                @info "eval/win_pct" value=result.win_pct log_step_increment=0
                @info "eval/white_equity" value=result.white_equity log_step_increment=0
                @info "eval/black_equity" value=result.black_equity log_step_increment=0
                @info "eval/value_mse" value=result.value_mse log_step_increment=0
                @info "eval/value_corr" value=result.value_corr log_step_increment=0
                @info "eval/games" value=result.n_games log_step_increment=0
            end
            println("Eval iter $(job.iter) complete: equity=$(round(result.equity, digits=4)), win%=$(round(result.win_pct, digits=1))")
            EVAL_JOB[] = nothing
        end

        return HTTP.Response(200, [], MsgPack.pack(Dict(
            "accepted" => true, "eval_complete" => eval_complete)))
    end
end

function handle_eval_heartbeat(req, state)
    body = MsgPack.unpack(req.body)
    chunk_id = body["chunk_id"]
    client_name = body["client_name"]
    lock(EVAL_LOCK) do
        job = EVAL_JOB[]
        job === nothing && return HTTP.Response(200, [], MsgPack.pack(Dict("lease_extended" => false)))
        ok = EvalManager.extend_lease!(job, chunk_id, client_name)
        return HTTP.Response(200, [], MsgPack.pack(Dict("lease_extended" => ok)))
    end
end
```

- [ ] **Step 3: Register routes in create_router()**

Add to `src/distributed/server.jl` `create_router()`:
```julia
HTTP.register!(router, "GET", "/api/eval/status", req -> handle_eval_status(req, state))
HTTP.register!(router, "POST", "/api/eval/checkout", req -> handle_eval_checkout(req, state))
HTTP.register!(router, "POST", "/api/eval/submit", req -> handle_eval_submit(req, state))
HTTP.register!(router, "POST", "/api/eval/heartbeat", req -> handle_eval_heartbeat(req, state))
```

- [ ] **Step 4: Replace run_eval!() call with create_eval_job!()**

In the training loop (around line 1203), replace:
```julia
if EVAL_ENABLED && iter % EVAL_INTERVAL == 0
    eval_result = run_eval!(eval_net, iter)
    # ... TB logging ...
end
```

With:
```julia
if EVAL_ENABLED && iter % EVAL_INTERVAL == 0
    lock(EVAL_LOCK) do
        wv = ARGS["training_mode"] == "race" ? server_state.race_version[] : server_state.contact_version[]
        EVAL_JOB[] = EvalManager.create_eval_job(iter, EVAL_POSITIONS, wv, EVAL_CHUNK_SIZE)
        println("Eval job created for iter $iter")
    end
end
```

- [ ] **Step 5: Add background expiry task**

Add near the server startup:
```julia
@async begin
    while true
        sleep(60)
        lock(EVAL_LOCK) do
            job = EVAL_JOB[]
            job === nothing && return
            EvalManager.expire_stale_checkouts!(job, EVAL_CHECKOUT_LEASE)
            # Job-level timeout
            if time() - job.created_at > EVAL_JOB_TIMEOUT
                n = count(c -> c.completed, job.chunks)
                if n < length(job.chunks)
                    println("WARNING: Eval job iter $(job.iter) timed out ($n/$(length(job.chunks))). Abandoning.")
                    EVAL_JOB[] = nothing
                end
            end
        end
    end
end
```

- [ ] **Step 6: Commit**

```bash
git add scripts/training_server.jl src/distributed/server.jl src/distributed/eval_manager.jl
git commit -m "feat: add distributed eval API endpoints and non-blocking eval job creation"
```

---

### Task 8: Add client eval mode to selfplay_client.jl

**Files:**
- Modify: `scripts/selfplay_client.jl` — add `--eval-capable`, pause/resume atomics, `check_and_do_eval!()`

- [ ] **Step 1: Add --eval-capable CLI flag**

Add to the ArgParse settings:
```julia
"--eval-capable"
    help = "Enable eval mode (client does eval when server has eval jobs)"
    action = :store_true
"--eval-mcts-iters"
    help = "MCTS iterations for eval games (default: 600)"
    arg_type = Int
    default = 600
"--wildbg-lib"
    help = "Path to wildbg shared library (required if eval-capable)"
    arg_type = String
    default = ""
```

- [ ] **Step 2: Add pause/resume atomics**

Add near the top-level constants:
```julia
const PAUSE_SELFPLAY = Threads.Atomic{Bool}(false)
const ACTIVE_SELFPLAY_GAMES = Threads.Atomic{Int}(0)
const EVAL_CAPABLE = parsed_args["eval_capable"]
```

- [ ] **Step 3: Wrap continuous_worker game loop with atomics**

In `continuous_worker()`, wrap the game-playing code:
```julia
while true
    if PAUSE_SELFPLAY[]
        sleep(0.1)
        continue
    end
    Threads.atomic_add!(ACTIVE_SELFPLAY_GAMES, 1)
    try
        # ... existing self-play game code ...
    finally
        Threads.atomic_sub!(ACTIVE_SELFPLAY_GAMES, 1)
    end
end
```

- [ ] **Step 4: Implement check_and_do_eval!()**

Add the eval loop function that checks for eval work, pauses self-play, plays eval games, and submits results. Follow the spec's `check_and_do_eval!()` pattern with:
- Separate `EvalWeightState` for eval weights
- `parallel_eval_chunk()` using `Threads.@threads` work-stealing
- Heartbeat via `@async` task
- Exponential backoff on submit failures

- [ ] **Step 5: Integrate eval check into main loop**

In the main sample upload loop, add periodic eval check:
```julia
# Check for eval work every 30 seconds
if EVAL_CAPABLE && time() - last_eval_check > 30.0
    last_eval_check = time()
    try
        check_and_do_eval!()
    catch e
        println("Eval check error: $e")
    end
end
```

- [ ] **Step 6: Commit**

```bash
git add scripts/selfplay_client.jl
git commit -m "feat: add client-side eval mode with pause/resume and chunked eval"
```

---

### Task 9: Create launch_v7.sh

**Files:**
- Create: `scripts/launch_v7.sh`

- [ ] **Step 1: Create v7 launch script**

Based on launch_v6.sh but with:
- Jarvis client: `--num-workers 12 --eval-capable --wildbg-lib /home/sile/github/wildbg/target/release/libwildbg.so`
- Neo client: `--num-workers 32` (no eval)
- Server: no `--eval-workers` needed (eval is distributed)
- All logs use `>>` (append)
- `--checkpoint-interval 5` (more frequent saves to reduce data loss)

- [ ] **Step 2: Commit**

```bash
git add scripts/launch_v7.sh
git commit -m "feat: add v7 launch script with distributed eval"
```

---

### Task 10: End-to-end test

- [ ] **Step 1: Start training server locally**

Start server with a small config for testing:
```bash
julia --threads 4 --project scripts/training_server.jl \
  --port 9090 --data-dir /tmp/v7-test --training-mode race \
  --total-iterations 5 --games-per-iteration 20 --training-steps 10 \
  --eval-games 20 --eval-workers 0 --checkpoint-interval 2
```

- [ ] **Step 2: Start eval-capable client**

```bash
julia --threads 4 --project scripts/selfplay_client.jl \
  --server http://127.0.0.1:9090 --api-key alphazero-dev-key \
  --num-workers 2 --eval-capable --client-name test-eval
```

- [ ] **Step 3: Verify eval job creation and completion**

Monitor:
- Server creates eval job at iter 2 (checkpoint interval)
- Client detects eval job, pauses self-play
- Client checks out chunks, plays eval games, submits results
- Server aggregates results, logs to TensorBoard
- Client resumes self-play

- [ ] **Step 4: Verify TensorBoard shows eval metrics**

Check that eval/equity, eval/win_pct etc. appear in TB.

- [ ] **Step 5: Final commit**

```bash
git commit --allow-empty -m "milestone: v7 distributed eval end-to-end verified"
```

---

## Task Dependencies

```
Task 1 (game_loop.jl) ──┬── Task 2 (selfplay_client.jl)
                         ├── Task 3 (training_server.jl eval)
                         └── Task 4 (eval scripts)
                                    │
                         Task 5 (verify Part A) ── Task 6 (eval_manager.jl)
                                                          │
                                                   Task 7 (server API)
                                                          │
                                                   Task 8 (client eval)
                                                          │
                                                   Task 9 (launch script)
                                                          │
                                                   Task 10 (e2e test)
```

Tasks 2, 3, 4 can run in parallel after Task 1.
Tasks 6-10 are sequential (each builds on the previous).
