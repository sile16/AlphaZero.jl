# Regression tests for the 2026-07-03 external-review bug classes:
#
# 1. MCTS value scale: terminal rewards (±1/±2/±3) must be normalized by
#    GI.reward_scale before entering Q totals, so they mix with NN values
#    ([-1,1]) on the same scale. A terminal gammon must back up Q ≈ 2/3, NOT 2.0.
# 2. Replay buffer torn reads: once the circular buffer wraps, per_add_batch!
#    overwrites entries in place; extract_batch must never observe a sample
#    whose columns come from two different writes.
# 3. Partition correctness: contact samples must never land in the race
#    partition (race-only training must not see contact states).

using Test
using Random
using StaticArrays

using AlphaZero
using AlphaZero: GI, MCTS, BatchedMCTS

import BackgammonNet

# buffer.jl is not part of the AlphaZero package — training_server.jl includes it
# directly. Do the same here (guarded so repeated test runs don't re-include).
if !isdefined(Main, :PERBuffer)
    include(joinpath(@__DIR__, "..", "src", "distributed", "buffer.jl"))
end

const GAMES_DIR_SB = joinpath(@__DIR__, "..", "games")
if !isdefined(Main, :BackgammonDeterministic)
    include(joinpath(GAMES_DIR_SB, "backgammon-deterministic", "main.jl"))
end
const BGD_SB = Main.BackgammonDeterministic

@testset "MCTS Q-scale: terminal gammon backs up 2/3, not 2.0" begin
    gspec = BGD_SB.GameSpec()

    # White (P0): 14 off, 1 checker on point 24 — every roll bears off the last
    # checker. Black (P1): 15 checkers in home, 0 off — so the win is a gammon
    # (raw reward 2.0). Normalized Q at the root must be 2/3.
    p0 = (UInt128(14) << (25 * 4)) | (UInt128(1) << (24 * 4))
    p1 = (UInt128(5) << (1 * 4)) | (UInt128(5) << (2 * 4)) | (UInt128(5) << (3 * 4))
    bg = BackgammonNet.BackgammonGame(
        p0, p1, SVector{2, Int8}(0, 0), Int8(0), Int8(0), false, 0.0f0;
        obs_type=:minimal_flat)
    # Resolve the chance node deterministically (any dice bear off from pt 24)
    BackgammonNet.apply_chance!(bg, BackgammonNet.chance_outcomes(bg)[1][1])

    env = GI.init(gspec, bg)
    @test !GI.game_terminated(env)
    @test !GI.is_chance_node(env)

    # Uniform oracle with V = 0: root Q comes purely from the terminal reward.
    uniform_oracle(state) = begin
        n = length(GI.available_actions(gspec, state))
        (ones(Float64, n) ./ n, 0.0)
    end

    mcts = MCTS.Env(gspec, uniform_oracle;
                    cpuct=2.0, noise_ϵ=0.0, chance_mode=:passthrough)
    benv = BatchedMCTS.BatchedEnv(mcts, 8)
    BatchedMCTS.batched_explore!(benv, env, 64)

    root_state = GI.current_state(env)
    @test haskey(mcts.tree, root_state)
    info = mcts.tree[root_state]

    visited = [(s.W / s.N) for s in info.stats if s.N > 0]
    @test !isempty(visited)
    for q in visited
        # Every action bears off the last checker -> terminal gammon.
        # Raw reward is 2.0; the reward_scale fix must normalize it to 2/3.
        @test isapprox(q, 2 / 3; atol=1e-9)
        # The bug class this guards: any Q outside [-1,1] means some value
        # source entered the tree unnormalized.
        @test abs(q) <= 1.0 + 1e-9
    end
end

@testset "Buffer: no torn reads across circular overwrite" begin
    state_dim, num_actions, capacity = 4, 6, 64
    buf = PERBuffer(capacity, state_dim, num_actions)

    # Sentinel encoding: every column of a sample carries the same id, so any
    # extracted sample whose fields disagree was torn mid-overwrite.
    function sentinel_batch(id::Int, n::Int)
        ids = Float32.(id .+ (0:n-1))
        states = repeat(ids', state_dim, 1)
        policies = repeat(ids', num_actions, 1)
        values = copy(ids)
        equities = repeat(ids', 5, 1)
        has_eq = fill(true, n)
        is_chance = fill(false, n)
        is_contact = fill(false, n)
        is_bearoff = fill(false, n)
        (states, policies, values, equities, has_eq, is_chance, is_contact, is_bearoff)
    end

    # Fill past capacity so the circular pointer wraps before the stress phase
    let (s, p, v, e, h, ch, c, b) = sentinel_batch(0, capacity + 8)
        per_add_batch!(buf, s, p, v, e, h, ch, c, b)
    end

    stop = Threads.Atomic{Bool}(false)
    torn = Threads.Atomic{Int}(0)

    reader = Threads.@spawn begin
        rng = MersenneTwister(1)
        while !stop[]
            idx = rand(rng, 1:capacity, 16)
            batch = extract_batch(buf, idx)
            for j in 1:length(idx)
                ref = batch.values[j]
                same = all(batch.states[:, j] .== ref) &&
                       all(batch.policies[:, j] .== ref) &&
                       all(batch.equities[:, j] .== ref)
                same || Threads.atomic_add!(torn, 1)
            end
        end
    end

    # Writer keeps overwriting wrapped entries with fresh sentinel ids
    for round in 1:300
        (s, p, v, e, h, ch, c, b) = sentinel_batch(round * 1000, 32)
        per_add_batch!(buf, s, p, v, e, h, ch, c, b)
    end
    stop[] = true
    wait(reader)

    @test torn[] == 0
end

@testset "Buffer: partition never mixes contact into race" begin
    state_dim, num_actions, capacity = 4, 6, 32
    buf = PERBuffer(capacity, state_dim, num_actions)

    n = 20
    states = zeros(Float32, state_dim, n)
    policies = zeros(Float32, num_actions, n)
    # Encode contact flag in the value so we can cross-check after extraction
    is_contact = [isodd(i) for i in 1:n]
    values = Float32[c ? 1.0 : -1.0 for c in is_contact]
    equities = zeros(Float32, 5, n)
    has_eq = fill(true, n)
    is_chance = fill(false, n)
    is_bearoff = fill(false, n)
    per_add_batch!(buf, states, policies, values, equities, has_eq,
                   is_chance, is_contact, is_bearoff)

    parts = partition_indices(buf)
    @test length(parts.contact) == count(is_contact)
    @test length(parts.race) == n - count(is_contact)
    @test isempty(intersect(Set(parts.contact), Set(parts.race)))

    # Race partition must contain only race samples (value -1 sentinel)
    race_batch = extract_batch(buf, parts.race)
    @test all(race_batch.values .== -1.0f0)
    @test all(.!race_batch.is_contact)
    contact_batch = extract_batch(buf, parts.contact)
    @test all(contact_batch.values .== 1.0f0)
    @test all(contact_batch.is_contact)
end

@testset "Buffer: preserves chance-node flags" begin
    state_dim, num_actions, capacity = 4, 6, 16
    buf = PERBuffer(capacity, state_dim, num_actions)

    n = 6
    states = zeros(Float32, state_dim, n)
    policies = zeros(Float32, num_actions, n)
    values = Float32.(1:n)
    equities = zeros(Float32, 5, n)
    has_eq = fill(true, n)
    is_chance = Bool[true, false, true, false, false, true]
    is_contact = fill(false, n)
    is_bearoff = fill(false, n)

    per_add_batch!(buf, states, policies, values, equities, has_eq,
                   is_chance, is_contact, is_bearoff)

    batch = extract_batch(buf, collect(1:n))
    @test batch.is_chance == is_chance

end

@testset "Buffer: reset clears bootstrap data and invalidates stale generations" begin
    state_dim, num_actions, capacity = 4, 6, 8
    buf = PERBuffer(capacity, state_dim, num_actions; beta_init=0.4f0, annealing_iters=10)

    n = capacity
    states = fill(7.0f0, state_dim, n)
    policies = fill(3.0f0, num_actions, n)
    values = fill(2.0f0, n)
    equities = fill(0.5f0, 5, n)
    has_eq = fill(true, n)
    is_chance = fill(true, n)
    is_contact = [isodd(i) for i in 1:n]
    is_bearoff = fill(true, n)
    per_add_batch!(buf, states, policies, values, equities, has_eq,
                   is_chance, is_contact, is_bearoff; initial_priority=4.0f0)
    per_anneal_beta!(buf)

    old_generations = copy(buf.generation)
    reset!(buf)

    @test buf.size == 0
    @test buf.write_pos == 1
    @test buf.beta == buf.beta_init
    @test buf.current_iter == 0
    @test all(buf.states .== 0.0f0)
    @test all(buf.policies .== 0.0f0)
    @test all(buf.values .== 0.0f0)
    @test all(buf.equities .== 0.0f0)
    @test all(.!buf.has_equity)
    @test all(.!buf.is_chance)
    @test all(.!buf.is_contact)
    @test all(.!buf.is_bearoff)
    @test all(buf.priorities .== 1.0f0)
    @test all(buf.generation .== old_generations .+ UInt32(1))
    @test partition_indices(buf) == (contact=Int[], race=Int[], all=Int[])
    @test_throws ErrorException per_sample(buf, 1, 0.6f0, 1.0f-6)

    # Refill starts from slot 1 with clean flags and priorities.
    let
        vals = Float32[11, 22]
        s = repeat(vals', state_dim, 1)
        p = repeat(vals', num_actions, 1)
        e = zeros(Float32, 5, 2)
        h = Bool[false, true]
        ch = Bool[false, false]
        c = Bool[true, false]
        b = Bool[false, true]
        per_add_batch!(buf, s, p, vals, e, h, ch, c, b; initial_priority=2.5f0)
    end
    @test buf.size == 2
    @test buf.write_pos == 3
    batch = extract_batch(buf, [1, 2])
    @test batch.values == Float32[11, 22]
    @test batch.is_contact == Bool[true, false]
    @test batch.is_bearoff == Bool[false, true]
    @test buf.priorities[1:2] == Float32[2.5, 2.5]
end

@testset "Buffer: clear_for_selfplay! establishes the reset invariant" begin
    state_dim, num_actions, capacity = 4, 6, 8
    buf = PERBuffer(capacity, state_dim, num_actions)
    n = 3
    s = rand(Float32, state_dim, n); p = rand(Float32, num_actions, n)
    vals = rand(Float32, n); e = rand(Float32, 5, n)
    h = fill(true, n); ch = fill(true, n); c = fill(true, n); b = fill(true, n)
    per_add_batch!(buf, s, p, vals, e, h, ch, c, b; initial_priority=3.0f0)
    gens = copy(buf.generation)

    clear_for_selfplay!(buf)   # semantic alias for reset! — same invariant

    @test buf.size == 0
    @test buf.write_pos == 1
    @test buf.beta == buf.beta_init
    @test all(buf.priorities .== 1.0f0)
    @test all(.!buf.is_chance) && all(.!buf.is_contact) && all(.!buf.is_bearoff)
    @test all(.!buf.has_equity)
    @test all(buf.generation .== gens .+ UInt32(1))
    @test partition_indices(buf) == (contact=Int[], race=Int[], all=Int[])
end

@testset "Buffer: reanalyze skips slots overwritten since extraction" begin
    # reanalyze extracts a batch, runs slow NN inference, then blends predictions
    # back by INDEX. A slot overwritten in between now holds a DIFFERENT sample;
    # its blend value is stale. The generation check must skip those slots.
    state_dim, num_actions, capacity = 4, 6, 8
    buf = PERBuffer(capacity, state_dim, num_actions)

    function mkbatch(vals::Vector{Float32})
        n = length(vals)
        states = repeat(vals', state_dim, 1)
        policies = repeat(vals', num_actions, 1)
        equities = zeros(Float32, 5, n)
        has_eq = fill(false, n)          # no equity blend — isolate the value path
        is_chance = fill(false, n)
        is_contact = fill(false, n)
        is_bearoff = fill(false, n)
        (states, policies, copy(vals), equities, has_eq, is_chance, is_contact, is_bearoff)
    end

    # Write 4 samples into slots 1..4 (generation → 1 each).
    let (s, p, v, e, h, ch, c, b) = mkbatch(Float32[10, 20, 30, 40])
        per_add_batch!(buf, s, p, v, e, h, ch, c, b)
    end
    idx = [1, 2, 3, 4]
    col = extract_batch(buf, idx)
    gens = copy(col.generations)         # [1,1,1,1]
    @test all(gens .== UInt32(1))

    # Overwrite slots 5,6,7,8 then 1,2 (write_pos wraps): slots 1 and 2 get gen 2.
    let (s, p, v, e, h, ch, c, b) = mkbatch(Float32[55, 66, 77, 88, 111, 222])
        per_add_batch!(buf, s, p, v, e, h, ch, c, b)
    end
    @test buf.generation[1] == UInt32(2)
    @test buf.generation[2] == UInt32(2)
    @test buf.values[1] == 111.0f0       # slot 1 now holds the overwrite sample
    @test buf.values[2] == 222.0f0

    # Reanalyze with the STALE generation snapshot: slots 1,2 mismatch (skip),
    # slots 3,4 still match (blend). new_values length matches idx.
    new_vals = Float32[1, 2, 3, 4]
    z = zeros(Float32, 4)
    skipped = reanalyze_update!(buf, idx, gens, new_vals, z, z, z, z, z; α_blend=1.0f0)
    @test skipped == 2                   # slots 1 and 2 skipped

    # Overwritten slots left untouched by the stale blend.
    @test buf.values[1] == 111.0f0
    @test buf.values[2] == 222.0f0
    # Matching slots blended (α=1.0 → replaced by new value).
    @test buf.values[3] == 3.0f0
    @test buf.values[4] == 4.0f0
end

@testset "Buffer: per_add_batch! rejects dim/length-skewed batches (no @inbounds OOB)" begin
    # A worker on a stale git commit / different BACKGAMMON_OBS_TYPE can POST a
    # wrong state_dim; a malformed payload can carry length-skewed columns. The
    # @inbounds copy would OOB-read (corruption/segfault) — validation rejects it.
    sd, na, cap = 4, 6, 8
    buf = PERBuffer(cap, sd, na)
    g = (rand(Float32, sd, 2), rand(Float32, na, 2), rand(Float32, 2),
         rand(Float32, 5, 2), fill(true, 2), fill(false, 2), fill(true, 2), fill(false, 2))
    per_add_batch!(buf, g...)                 # sanity: valid batch accepted
    @test buf.size == 2

    @test_throws ErrorException per_add_batch!(buf, rand(Float32, sd + 1, 2), g[2:8]...)   # wrong state_dim
    @test_throws ErrorException per_add_batch!(buf, g[1], rand(Float32, na + 1, 2), g[3:8]...) # wrong num_actions
    @test_throws ErrorException per_add_batch!(buf, g[1], g[2], rand(Float32, 3), g[4:8]...)   # values col skew
    @test_throws ErrorException per_add_batch!(buf, g[1:4]..., fill(true, 3), g[6:8]...)       # flag col skew
    @test buf.size == 2                       # rejected batches left the buffer untouched
end

@testset "Buffer: reanalyze writes NORMALIZED PER priority (matches training scale)" begin
    # Reanalyze priority must be on the same /3-normalized scale as
    # compute_td_errors, or reanalyzed slots are oversampled ~1.9x.
    sd, na, cap = 4, 6, 2
    buf = PERBuffer(cap, sd, na)
    vals = Float32[3.0, -3.0]                  # raw [-3,3] equity
    per_add_batch!(buf, rand(Float32, sd, 2), rand(Float32, na, 2), vals,
                   rand(Float32, 5, 2), fill(true, 2), fill(false, 2),
                   fill(true, 2), fill(false, 2))
    gens = copy(buf.generation[1:2])
    z = zeros(Float32, 2)
    reanalyze_update!(buf, [1, 2], gens, Float32[0.0, 0.0], z, z, z, z, z; α_blend=1.0f0)
    # slot1: |0 - 3|/3 = 1.0 ; slot2: |0 - (-3)|/3 = 1.0 (would be 3.0 raw without the fix)
    @test buf.priorities[1] ≈ 1.0f0
    @test buf.priorities[2] ≈ 1.0f0
end
