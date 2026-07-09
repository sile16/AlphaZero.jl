using Test
using Random

using AlphaZero
using AlphaZero: GI, Network, FluxLib, MctsParams, ConstSchedule, BatchedMCTS

import BackgammonNet

const GAMES_DIR = joinpath(@__DIR__, "..", "games")
if !isdefined(Main, :BackgammonDeterministic)
    include(joinpath(GAMES_DIR, "backgammon-deterministic", "main.jl"))
end
const BGD = Main.BackgammonDeterministic

function random_decision_states(gspec, n; seed=0)
    rng = MersenneTwister(seed)
    states = BackgammonNet.BackgammonGame[]
    while length(states) < n
        env = GI.init(gspec)
        for _ in 1:40
            GI.game_terminated(env) && break
            if GI.is_chance_node(env)
                outcomes = GI.chance_outcomes(env)
                GI.apply_chance!(env, outcomes[rand(rng, eachindex(outcomes))][1])
                continue
            end
            push!(states, GI.current_state(env))
            length(states) >= n && break
            actions = GI.available_actions(env)
            isempty(actions) && break
            GI.play!(env, actions[rand(rng, eachindex(actions))])
        end
    end
    return states
end

_sigmoid32(x) = 1.0f0 / (1.0f0 + exp(-Float32(x)))
_logit32(p) = Float32(log(Float32(p) / (1.0f0 - Float32(p))))

function force_constant_heads!(nn::FluxLib.FCResNetMultiHead, heads)
    for (head, p) in zip((nn.vhead_win, nn.vhead_gw, nn.vhead_bgw, nn.vhead_gl, nn.vhead_bgl), heads)
        dense = head.layers[end]
        fill!(dense.weight, 0.0f0)
        fill!(dense.bias, _logit32(p))
    end
    return nn
end

function expected_backgammon_search_value(nn::FluxLib.FCResNetMultiHead, gspec, state)
    X = reshape(Float32.(vec(GI.vectorize_state(gspec, state))), :, 1)
    A = reshape(Float32.(GI.actions_mask(gspec, state)), :, 1)
    _, Lw, Lgw, Lbgw, Lgl, Lbgl, _ = FluxLib.forward_normalized_multihead(nn, X, A)
    heads = (_sigmoid32(Lw[1, 1]), _sigmoid32(Lgw[1, 1]), _sigmoid32(Lbgw[1, 1]),
             _sigmoid32(Lgl[1, 1]), _sigmoid32(Lbgl[1, 1]))
    return BackgammonNet.search_value(state, heads; mode=:auto)
end

@testset "Backgammon Inference Regressions" begin
    gspec = BGD.GameSpec()
    state_dim = let env = GI.init(gspec)
        length(vec(GI.vectorize_state(gspec, GI.current_state(env))))
    end
    num_actions = GI.num_actions(gspec)
    cfg = AlphaZero.BackgammonInference.OracleConfig(
        state_dim, num_actions, gspec;
        vectorize_state! = BGD.vectorize_state_into!)

    @testset "deterministic wrapper delegates stable BackgammonNet APIs" begin
        @test num_actions == BackgammonNet.CHECKER_ACTIONS
        @test BGD.NUM_ACTIONS == BackgammonNet.CHECKER_ACTIONS

        @test GI.parse_action(gspec, "Bar | 5") == BackgammonNet.encode_action(BackgammonNet.BAR_LOC, 5)
        @test GI.parse_action(gspec, "Pass | Pass") ==
              BackgammonNet.encode_action(BackgammonNet.PASS_LOC, BackgammonNet.PASS_LOC)
        @test GI.parse_action(gspec, "No Double") == BackgammonNet.ACTION_CUBE_NO_DOUBLE
        @test GI.parse_action(gspec, "Double") == BackgammonNet.ACTION_CUBE_DOUBLE
        @test GI.parse_action(gspec, "Take") == BackgammonNet.ACTION_CUBE_TAKE
        @test GI.parse_action(gspec, "Drop") == BackgammonNet.ACTION_CUBE_PASS

        env = GI.init(gspec)
        @test env.game.cube_enabled == BGD.CUBE_ENABLED
        @test env.game.jacoby_enabled == BGD.JACOBY_ENABLED
        @test GI.heuristic_value(env) ≈ Float64(BackgammonNet.heuristic_value(env.game)) atol=1e-12

        disabled_game = BackgammonNet.initial_state(cube_enabled=false, jacoby_enabled=false, obs_type=:minimal_flat)
        BackgammonNet.sample_chance!(disabled_game, MersenneTwister(501))
        ctx = BackgammonNet.context_observation(disabled_game)
        @test ctx[5:7] == Float32[0, 0, 0]  # cube ownership all-zero means cube disabled
        @test ctx[14] == 0.0f0              # may_double is also off

        cube_env = GI.init(gspec)
        cube_env.game.cube_enabled = true
        cube_env.game.cube_owner = Int8(-1)
        cube_env.game.phase = BackgammonNet.PHASE_CUBE_DECISION
        @test BackgammonNet.may_double(cube_env.game)
        @test GI.available_actions(cube_env) == [BackgammonNet.ACTION_CUBE_NO_DOUBLE, BackgammonNet.ACTION_CUBE_DOUBLE]
        cube_mask = GI.actions_mask(cube_env)
        @test length(cube_mask) == BackgammonNet.MAX_ACTIONS
        @test findall(cube_mask) == [BackgammonNet.ACTION_CUBE_NO_DOUBLE, BackgammonNet.ACTION_CUBE_DOUBLE]

        response_env = GI.init(gspec)
        response_env.game.cube_enabled = true
        response_env.game.phase = BackgammonNet.PHASE_CUBE_RESPONSE
        @test GI.available_actions(response_env) == [BackgammonNet.ACTION_CUBE_TAKE, BackgammonNet.ACTION_CUBE_PASS]

        env.game.terminated = true
        env.game.reward = 2.0f0
        env.game.current_player = Int8(0)
        @test GI.heuristic_value(env) == 2.0
        env.game.current_player = Int8(1)
        @test GI.heuristic_value(env) == -2.0
    end

    @testset "current_state returns owning clone" begin
        env = GI.init(gspec)
        while GI.is_chance_node(env)
            GI.apply_chance!(env, GI.chance_outcomes(env)[1][1])
        end

        state = GI.current_state(env)
        actions_before = GI.available_actions(GI.init(gspec, state))
        vec_before = copy(GI.vectorize_state(gspec, state))

        for _ in 1:5
            GI.game_terminated(env) && break
            if GI.is_chance_node(env)
                GI.apply_chance!(env, GI.chance_outcomes(env)[1][1])
            else
                actions = GI.available_actions(env)
                isempty(actions) && break
                GI.play!(env, actions[1])
            end
        end

        actions_after = GI.available_actions(GI.init(gspec, state))
        vec_after = GI.vectorize_state(gspec, state)

        @test actions_before == actions_after
        @test vec_before == vec_after
        @test state !== env.game
        @test state._actions_buffer !== env.game._actions_buffer
    end

    @testset "state-level action APIs match env path" begin
        states = random_decision_states(gspec, 24; seed=43)

        chance_env = GI.init(gspec)
        push!(states, GI.current_state(chance_env))

        terminal_env = GI.init(gspec)
        rng = MersenneTwister(44)
        guard = 0
        while !GI.game_terminated(terminal_env) && guard < 2000
            if GI.is_chance_node(terminal_env)
                outcomes = GI.chance_outcomes(terminal_env)
                GI.apply_chance!(terminal_env, outcomes[rand(rng, eachindex(outcomes))][1])
            else
                actions = GI.available_actions(terminal_env)
                isempty(actions) && break
                GI.play!(terminal_env, actions[rand(rng, eachindex(actions))])
            end
            guard += 1
        end
        @test GI.game_terminated(terminal_env)
        push!(states, GI.current_state(terminal_env))

        for state in states
            state_before = BackgammonNet.clone(state)
            hash_before = hash(state)
            vec_before = copy(GI.vectorize_state(gspec, state))
            actions_cached_before = state._actions_cached
            actions_buffer_before = copy(state._actions_buffer)
            actions_buffer_id = state._actions_buffer

            direct_actions = GI.available_actions(gspec, state)
            env_actions = GI.available_actions(GI.init(gspec, state))
            direct_mask = GI.actions_mask(gspec, state)
            env_mask = GI.actions_mask(GI.init(gspec, state))

            @test direct_actions == env_actions
            @test direct_mask == env_mask
            @test direct_actions == findall(direct_mask)
            @test hash(state) == hash_before
            @test state == state_before
            @test GI.vectorize_state(gspec, state) == vec_before
            @test state._actions_cached == actions_cached_before
            @test state._actions_buffer === actions_buffer_id
            @test state._actions_buffer == actions_buffer_before
        end
    end

    @testset "state-level action APIs are shared-state thread safe" begin
        state = random_decision_states(gspec, 1; seed=45)[1]
        expected_actions = GI.available_actions(GI.init(gspec, state))
        expected_mask = GI.actions_mask(GI.init(gspec, state))
        actions_cached_before = state._actions_cached
        actions_buffer_before = copy(state._actions_buffer)
        actions_buffer_id = state._actions_buffer

        failures = Threads.Atomic{Int}(0)
        ntasks = max(2, Threads.nthreads())
        tasks = [Threads.@spawn begin
            for _ in 1:100
                actions = GI.available_actions(gspec, state)
                mask = GI.actions_mask(gspec, state)
                if actions != expected_actions || mask != expected_mask
                    Threads.atomic_add!(failures, 1)
                end
                yield()
            end
        end for _ in 1:ntasks]
        foreach(fetch, tasks)

        @test failures[] == 0
        @test state._actions_cached == actions_cached_before
        @test state._actions_buffer === actions_buffer_id
        @test state._actions_buffer == actions_buffer_before
    end

    @testset "flux oracle uses active batch slice only" begin
        struct BatchSensitiveNet <: Network.AbstractNetwork
            gspec::Any
            num_actions::Int
        end

        Network.game_spec(nn::BatchSensitiveNet) = nn.gspec
        Network.forward(nn::BatchSensitiveNet, state) = (
            ones(Float32, nn.num_actions, size(state, 2)),
            fill(Float32(size(state, 2)), 1, size(state, 2))
        )
        Network.convert_input(::BatchSensitiveNet, x) = x
        Network.convert_output(::BatchSensitiveNet, x) = x

        states = random_decision_states(gspec, 4; seed=1)
        _, batch_oracle = AlphaZero.BackgammonInference.make_cpu_oracles(
            :flux, BatchSensitiveNet(gspec, num_actions), cfg;
            batch_size=4, nslots=max(Threads.nthreads(), 1))

        warm = batch_oracle(states)
        @test length(warm) == 4
        @test all(r -> r[2] == 4.0f0, warm)

        single = batch_oracle(states[1:1])
        @test length(single) == 1
        @test single[1][2] == 1.0f0
    end

    @testset "shared backgammon oracles match state-aware evaluation" begin
        contact_net = FluxLib.FCResNetMultiHead(
            gspec, FluxLib.FCResNetMultiHeadHP(width=32, num_blocks=1))
        states = random_decision_states(gspec, 6; seed=2)

        for backend in (:fast, :flux)
            _, batch_oracle = AlphaZero.BackgammonInference.make_cpu_oracles(
                backend, contact_net, cfg;
                batch_size=4, nslots=max(Threads.nthreads(), 1))

            shared = batch_oracle(states[1:4])
            canonical = Network.evaluate_batch(contact_net, states[1:4])

            @test length(shared) == length(canonical)
            for i in eachindex(shared)
                p_shared, v_shared = shared[i]
                p_canon, _ = canonical[i]
                v_canon = expected_backgammon_search_value(contact_net, gspec, states[i])
                @test length(p_shared) == length(p_canon)
                @test isapprox(sum(p_shared), 1.0f0; atol=1f-4)
                @test isapprox(sum(p_canon), 1.0f0; atol=1f-4)
                @test maximum(abs.(p_shared .- p_canon)) ≤ 5f-4
                @test isapprox(v_shared, v_canon; atol=5f-4)
            end
        end
    end

    @testset "shared backgammon oracles use cubeful search value" begin
        heads = (0.62f0, 0.18f0, 0.04f0, 0.11f0, 0.03f0)
        contact_net = force_constant_heads!(
            FluxLib.FCResNetMultiHead(gspec, FluxLib.FCResNetMultiHeadHP(width=32, num_blocks=1)),
            heads)

        state = GI.current_state(GI.init(gspec))
        state.cube_enabled = true
        state.jacoby_enabled = false
        state.cube_value = Int16(4)
        state.cube_owner = state.current_player

        raw_value = BackgammonNet.compute_equity_joint(heads) / 3.0f0
        expected = BackgammonNet.search_value(state, heads; mode=:auto)
        @test !isapprox(expected, raw_value; atol=1f-3)

        disabled_state = BackgammonNet.clone(state)
        disabled_state.cube_enabled = false
        expected_disabled = BackgammonNet.search_value(disabled_state, heads; mode=:auto)
        @test isapprox(expected_disabled, raw_value; atol=1f-6)

        for backend in (:fast, :flux)
            _, batch_oracle = AlphaZero.BackgammonInference.make_cpu_oracles(
                backend, contact_net, cfg;
                batch_size=2, nslots=max(Threads.nthreads(), 1))

            _, v_cube = batch_oracle([state])[1]
            _, v_disabled = batch_oracle([disabled_state])[1]
            @test isapprox(v_cube, expected; atol=5f-5)
            @test isapprox(v_disabled, expected_disabled; atol=5f-5)
        end
    end

    @testset "action-aware batch oracle matches state-only oracle" begin
        contact_net = FluxLib.FCResNetMultiHead(
            gspec, FluxLib.FCResNetMultiHeadHP(width=32, num_blocks=1))
        states = random_decision_states(gspec, 8; seed=41)
        actions_by_state = [GI.available_actions(gspec, s) for s in states]

        for backend in (:fast, :flux)
            _, batch_oracle = AlphaZero.BackgammonInference.make_cpu_oracles(
                backend, contact_net, cfg;
                batch_size=8, nslots=max(Threads.nthreads(), 1))

            state_only = batch_oracle(states)
            action_aware = batch_oracle(states, actions_by_state)
            @test length(action_aware) == length(state_only)
            for i in eachindex(state_only)
                p_state, v_state = state_only[i]
                p_action, v_action = action_aware[i]
                @test length(p_action) == length(actions_by_state[i])
                @test length(p_action) == length(p_state)
                @test maximum(abs.(p_action .- p_state)) ≤ 5f-4
                @test isapprox(v_action, v_state; atol=5f-4)
            end
        end
    end

    @testset "shared gpu oracle builder matches canonical routing" begin
        primary_net = FluxLib.FCResNetMultiHead(
            gspec, FluxLib.FCResNetMultiHeadHP(width=32, num_blocks=1))
        secondary_net = FluxLib.FCResNetMultiHead(
            gspec, FluxLib.FCResNetMultiHeadHP(width=16, num_blocks=1))
        routed_cfg = AlphaZero.BackgammonInference.OracleConfig(
            state_dim, num_actions, gspec;
            vectorize_state! = BGD.vectorize_state_into!,
            route_state = s -> iseven(length(GI.available_actions(gspec, s))) ? 2 : 1)

        states = random_decision_states(gspec, 6; seed=29)
        _, batch_oracle = AlphaZero.BackgammonInference.make_gpu_oracles(
            primary_net, routed_cfg;
            secondary_net_gpu=secondary_net,
            batch_size=4,
            gpu_array_fn=identity,
            sync_fn=() -> nothing)

        shared = batch_oracle(states[1:4])
        @test length(shared) == 4
        for i in 1:4
            expected_net = routed_cfg.route_state(states[i]) == 2 ? secondary_net : primary_net
            p_canon, _ = Network.evaluate_batch(expected_net, [states[i]])[1]
            v_canon = expected_backgammon_search_value(expected_net, gspec, states[i])
            p_shared, v_shared = shared[i]
            @test length(p_shared) == length(p_canon)
            @test maximum(abs.(p_shared .- p_canon)) ≤ 5f-4
            @test isapprox(v_shared, v_canon; atol=5f-4)
        end
    end

    @testset "fast oracle safely reuses policy buffers across changing action counts" begin
        contact_net = FluxLib.FCResNetMultiHead(
            gspec, FluxLib.FCResNetMultiHeadHP(width=32, num_blocks=1))
        _, batch_oracle = AlphaZero.BackgammonInference.make_cpu_oracles(
            :fast, contact_net, cfg;
            batch_size=4, nslots=max(Threads.nthreads(), 1))

        states = random_decision_states(gspec, 24; seed=11)
        counts = [length(GI.available_actions(gspec, s)) for s in states]
        order = sortperm(counts)
        probe_states = [states[order[1]], states[order[end]], states[order[2]], states[order[end-1]]]

        for _ in 1:20
            for s in probe_states
                result = batch_oracle([s])[1]
                @test length(result[1]) == length(GI.available_actions(gspec, s))
                @test isapprox(sum(result[1]), 1.0f0; atol=5f-4)
            end
        end
    end

    @testset "fast oracle returns owning policy vectors across calls" begin
        contact_net = FluxLib.FCResNetMultiHead(
            gspec, FluxLib.FCResNetMultiHeadHP(width=32, num_blocks=1))
        _, batch_oracle = AlphaZero.BackgammonInference.make_cpu_oracles(
            :fast, contact_net, cfg;
            batch_size=4, nslots=max(Threads.nthreads(), 1))

        states = random_decision_states(gspec, 16; seed=19)
        counts = [length(GI.available_actions(gspec, s)) for s in states]
        order = sortperm(counts)
        low_state = states[order[1]]
        high_state = states[order[end]]

        first = batch_oracle([low_state])[1]
        first_policy = copy(first[1])
        first_len = length(first[1])

        for _ in 1:20
            batch_oracle([high_state])
            @test length(first[1]) == first_len
            @test first[1] == first_policy
        end
    end

    @testset "shared gpu oracle builder stays consistent under threaded load" begin
        contact_net = FluxLib.FCResNetMultiHead(
            gspec, FluxLib.FCResNetMultiHeadHP(width=32, num_blocks=1))
        _, batch_oracle = AlphaZero.BackgammonInference.make_gpu_oracles(
            contact_net, cfg;
            batch_size=4,
            gpu_array_fn=identity,
            sync_fn=() -> nothing)
        states = random_decision_states(gspec, 24; seed=31)

        failures = Channel{Any}(1)
        ntasks = max(Threads.nthreads() * 4, 8)
        Threads.@sync for task_id in 1:ntasks
            Threads.@spawn begin
                try
                    for iter in 1:10
                        state = states[mod1(task_id + iter, length(states))]
                        result = batch_oracle([state])[1]
                        actions = GI.available_actions(gspec, state)
                        @test length(result[1]) == length(actions)
                        @test isapprox(sum(result[1]), 1.0f0; atol=5f-4)
                        yield()
                    end
                catch err
                    put!(failures, err)
                end
            end
        end
        @test !isready(failures)
    end

    @testset "shared gpu server oracle batches concurrent clients correctly" begin
        contact_net = FluxLib.FCResNetMultiHead(
            gspec, FluxLib.FCResNetMultiHeadHP(width=32, num_blocks=1))
        _, batch_oracle, server = AlphaZero.BackgammonInference.make_gpu_server_oracles(
            contact_net, cfg;
            batch_size=4,
            num_workers=max(Threads.nthreads(), 2),
            gpu_array_fn=identity,
            sync_fn=() -> nothing,
            max_wait_ns=100_000)
        states = random_decision_states(gspec, 24; seed=37)

        failures = Channel{Any}(1)
        ntasks = max(Threads.nthreads() * 4, 8)
        Threads.@sync for task_id in 1:ntasks
            Threads.@spawn begin
                try
                    for iter in 1:5
                        s1 = states[mod1(task_id + iter, length(states))]
                        s2 = states[mod1(task_id + iter + 7, length(states))]
                        results = batch_oracle([s1, s2])
                        @test length(results) == 2
                        for (state, result) in zip((s1, s2), results)
                            actions = GI.available_actions(gspec, state)
                            @test length(result[1]) == length(actions)
                            @test isapprox(sum(result[1]), 1.0f0; atol=5f-4)
                        end
                        yield()
                    end
                catch err
                    put!(failures, err)
                end
            end
        end
        @test !isready(failures)
        close(server)
    end

    @testset "shared oracles stay consistent under threaded load" begin
        contact_net = FluxLib.FCResNetMultiHead(
            gspec, FluxLib.FCResNetMultiHeadHP(width=32, num_blocks=1))
        states = random_decision_states(gspec, 24; seed=3)

        for backend in (:fast, :flux)
            _, batch_oracle = AlphaZero.BackgammonInference.make_cpu_oracles(
                backend, contact_net, cfg;
                batch_size=8, nslots=max(Threads.nthreads(), 1))

            failures = Channel{Any}(Threads.nthreads())
            Threads.@sync for worker in 1:min(Threads.nthreads(), 4)
                Threads.@spawn begin
                    try
                        for offset in worker:4:length(states)-3
                            batch = states[offset:offset+3]
                            results = batch_oracle(batch)
                            @test length(results) == length(batch)
                            for i in eachindex(batch)
                                actions = GI.available_actions(gspec, batch[i])
                                policy, value = results[i]
                                @test length(policy) == length(actions)
                                @test isapprox(sum(policy), 1.0f0; atol=5f-4)
                                @test value isa Float32
                            end
                        end
                    catch err
                        put!(failures, err)
                    end
                end
            end
            worker_error = isready(failures) ? take!(failures) : nothing
            worker_error === nothing || @error "shared oracle worker failed" backend exception=worker_error
            @test worker_error === nothing
        end
    end

    @testset "fast oracle stays consistent with many tasks sharing threads" begin
        contact_net = FluxLib.FCResNetMultiHead(
            gspec, FluxLib.FCResNetMultiHeadHP(width=32, num_blocks=1))
        _, batch_oracle = AlphaZero.BackgammonInference.make_cpu_oracles(
            :fast, contact_net, cfg;
            batch_size=4, nslots=max(Threads.nthreads(), 1))
        states = random_decision_states(gspec, 32; seed=23)

        failures = Channel{Any}(1)
        ntasks = max(Threads.nthreads() * 4, 8)
        Threads.@sync for task_id in 1:ntasks
            Threads.@spawn begin
                try
                    for iter in 1:10
                        state = states[mod1(task_id + iter, length(states))]
                        result = batch_oracle([state])[1]
                        actions = GI.available_actions(gspec, state)
                        @test length(result[1]) == length(actions)
                        @test isapprox(sum(result[1]), 1.0f0; atol=5f-4)
                        yield()
                    end
                catch err
                    put!(failures, err)
                end
            end
        end
        @test !isready(failures)
    end

    @testset "BatchedMCTS stays consistent for both backends" begin
        contact_net = FluxLib.FCResNetMultiHead(
            gspec, FluxLib.FCResNetMultiHeadHP(width=32, num_blocks=1))
        params = MctsParams(
            num_iters_per_turn=20,
            cpuct=1.5,
            temperature=ConstSchedule(1.0),
            dirichlet_noise_ϵ=0.0,
            dirichlet_noise_α=0.3
        )

        for backend in (:fast, :flux)
            single_oracle, batch_oracle = AlphaZero.BackgammonInference.make_cpu_oracles(
                backend, contact_net, cfg;
                batch_size=8, nslots=max(Threads.nthreads(), 1))
            player = BatchedMCTS.BatchedMctsPlayer(
                gspec, single_oracle, params;
                batch_size=8, batch_oracle=batch_oracle,
                batch_oracle_with_actions=batch_oracle)

            env = GI.init(gspec)
            moves = 0
            while !GI.game_terminated(env) && moves < 20
                if GI.is_chance_node(env)
                    GI.apply_chance!(env, GI.chance_outcomes(env)[1][1])
                    continue
                end

                actions, policy = BatchedMCTS.think(player, env)
                @test !isempty(actions)
                @test length(actions) == length(policy)
                @test isapprox(sum(policy), 1.0; atol=1e-4)

                GI.play!(env, actions[argmax(policy)])
                BatchedMCTS.reset_player!(player)
                moves += 1
            end

            @test moves > 0
        end
    end

    @testset "make_cpu_oracles rejects nothing sources (never build broken oracle)" begin
        # :flux requires the network for every head it will evaluate.
        @test_throws ErrorException AlphaZero.BackgammonInference.make_cpu_oracles(
            :flux, nothing, cfg; batch_size=4, nslots=1)
        @test_throws ErrorException AlphaZero.BackgammonInference.make_cpu_oracles(
            :flux, nothing, cfg; secondary_net=nothing, secondary_fw=nothing,
            batch_size=4, nslots=1)

        # :fast needs a network OR FastWeights on each head; nothing+nothing is
        # the eval-oracle crash the guard now catches at the boundary.
        @test_throws ErrorException AlphaZero.BackgammonInference.make_cpu_oracles(
            :fast, nothing, cfg; batch_size=4, nslots=1)

        # Dual routing requested (secondary_net set) but the primary head has no
        # source — still errors, at the primary guard.
        contact_net = FluxLib.FCResNetMultiHead(
            gspec, FluxLib.FCResNetMultiHeadHP(width=16, num_blocks=1))
        @test_throws ErrorException AlphaZero.BackgammonInference.make_cpu_oracles(
            :fast, nothing, cfg; secondary_net=contact_net, batch_size=4, nslots=1)

        # Positive control: a valid single-head fast oracle still builds.
        single, batch = AlphaZero.BackgammonInference.make_cpu_oracles(
            :fast, contact_net, cfg; batch_size=4, nslots=1)
        @test single !== nothing && batch !== nothing
    end
end
