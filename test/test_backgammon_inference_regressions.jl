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

@testset "Backgammon Inference Regressions" begin
    gspec = BGD.GameSpec()
    state_dim = let env = GI.init(gspec)
        length(vec(GI.vectorize_state(gspec, GI.current_state(env))))
    end
    num_actions = GI.num_actions(gspec)
    cfg = AlphaZero.BackgammonInference.OracleConfig(
        state_dim, num_actions, gspec;
        vectorize_state! = BGD.vectorize_state_into!)

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

    @testset "shared backgammon oracles match canonical evaluation" begin
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
                p_canon, v_canon = canonical[i]
                @test length(p_shared) == length(p_canon)
                @test isapprox(sum(p_shared), 1.0f0; atol=1f-4)
                @test isapprox(sum(p_canon), 1.0f0; atol=1f-4)
                @test maximum(abs.(p_shared .- p_canon)) ≤ 5f-4
                @test isapprox(v_shared, v_canon; atol=5f-4)
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
            route_state = s -> iseven(length(GI.available_actions(GI.init(gspec, s)))) ? 2 : 1)

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
            p_canon, v_canon = Network.evaluate_batch(expected_net, [states[i]])[1]
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
        counts = [length(GI.available_actions(GI.init(gspec, s))) for s in states]
        order = sortperm(counts)
        probe_states = [states[order[1]], states[order[end]], states[order[2]], states[order[end-1]]]

        for _ in 1:20
            for s in probe_states
                result = batch_oracle([s])[1]
                @test length(result[1]) == length(GI.available_actions(GI.init(gspec, s)))
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
        counts = [length(GI.available_actions(GI.init(gspec, s))) for s in states]
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
                        actions = GI.available_actions(GI.init(gspec, state))
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
                                actions = GI.available_actions(GI.init(gspec, batch[i]))
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
            @test !isready(failures)
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
                        actions = GI.available_actions(GI.init(gspec, state))
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
                batch_size=8, batch_oracle=batch_oracle)

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
end
