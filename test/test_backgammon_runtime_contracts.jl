using Test
using BackgammonNet
import JSON3

include(joinpath(@__DIR__, "..", "src", "distributed", "bearoff_tables.jl"))
include(joinpath(@__DIR__, "..", "src", "distributed", "bootstrap_contract.jl"))

@testset "server-owned bearoff table selection" begin
    @test BearoffTables.parse_table_selection(" k7+n15 ") == "k7+n15"
    @test BearoffTables.parse_table_selection("NONE") == "none"
    @test_throws ArgumentError BearoffTables.parse_table_selection("auto")
    @test_throws ArgumentError BearoffTables.parse_table_selection("combined")
    parsed = JSON3.read("""{"selection":"k7","tables":[{"name":"k7"}]}""")
    @test BearoffTables._plain_identity(parsed) == Dict{String,Any}(
        "selection" => "k7", "tables" => Any[Dict("name" => "k7")])

    k7_dir = BackgammonNet.default_bearoff_k7_dir()
    n15_dir = BackgammonNet.default_bearoff_n15_dir()
    if isfile(joinpath(k7_dir, "bearoff_k7_c14.bin")) &&
       isfile(joinpath(n15_dir, "header.txt"))
        identity = BearoffTables.validate_table_release("k7+n15"; k7_dir, n15_dir)
        wire_identity = JSON3.read(JSON3.write(identity))
        tables = BearoffTables.load_runtime_tables(
            "k7+n15"; expected_identity=wire_identity, k7_dir, n15_dir)

        n15_game = BackgammonNet.initial_state(first_player=0)
        n15_game.p0 = UInt128(15) << (15 << 2) # distance 10
        n15_game.p1 = UInt128(15) << (8 << 2)  # distance 8
        n15_game.current_player = Int8(0)
        BackgammonNet.set_dice!(n15_game, 0, 0)
        @test !BearoffTables.exact_k7_covers(tables, n15_game)
        @test BearoffTables.n15_covers(tables, n15_game)
        @test BearoffTables.n15_lookup(tables, n15_game) !== nothing
        BackgammonNet.set_dice!(n15_game, 3, 1)
        @test isfinite(BearoffTables.n15_root_best_move_value(
            tables.k7, tables.n15, n15_game))

        # Root is n15-only, but both completed-turn successors enter k7. The
        # upstream recursion must re-check concrete coverage at each boundary
        # and use exact k7 there instead of retaining the root's n15 tier.
        transition = BackgammonNet.initial_state(first_player=0)
        transition.p0 = (UInt128(1) << (17 << 2)) | # one checker at distance 8
                        (UInt128(14) << (24 << 2))  # fourteen at distance 1
        transition.p1 = UInt128(15) << (6 << 2)
        transition.current_player = Int8(0)
        BackgammonNet.set_dice!(transition, 2, 1)
        @test !BearoffTables.exact_k7_covers(tables, transition)
        @test BearoffTables.n15_covers(tables, transition)
        exact_values = Float64[]
        for action in BackgammonNet.legal_actions(transition)
            successor = BackgammonNet.clone(transition)
            BackgammonNet.apply_legal_action!(successor, action)
            @test BackgammonNet.is_chance_node(successor)
            @test BearoffTables.exact_k7_covers(tables, successor)
            value, _ = BackgammonNet.bearoff_turn_value_equity(
                tables.k7, successor, Int(transition.current_player))
            push!(exact_values, value)
        end
        @test BearoffTables.n15_root_best_move_value(
            tables.k7, tables.n15, transition) == maximum(exact_values)

        k7_game = BackgammonNet.initial_state(first_player=0)
        k7_game.p0 = UInt128(15) << (18 << 2) # distance 7
        k7_game.p1 = UInt128(15) << (7 << 2)  # distance 7
        k7_game.current_player = Int8(0)
        BackgammonNet.set_dice!(k7_game, 0, 0)
        @test BearoffTables.exact_k7_covers(tables, k7_game)
        @test BearoffTables.exact_k7_lookup(tables, k7_game) !== nothing
    else
        @test_skip "local pinned k7+n15 tables unavailable"
    end
end

function fixture_data(kind; role="train", race_indices=Int32[],
                      teacher_policy="fixture_teacher",
                      producer_repo_commit=repeat("a", 40),
                      engine_value="gnubg-2ply", engine_policy="gnubg-2ply",
                      engine_cube="gnubg-2ply", game_mode="money",
                      source_mode="canonical_relabel",
                      source_selector="contact_close_calls")
    return (
        metadata=Dict{String,Any}(
            "artifact_kind" => kind,
            "artifact_role" => role,
            "teacher_policy" => teacher_policy,
            "producer_repo_commit" => producer_repo_commit,
            "engine_value" => engine_value,
            "engine_policy" => engine_policy,
            "engine_cube" => engine_cube,
            "game_mode" => game_mode,
            "source_mode" => source_mode,
            "source_selector" => source_selector,
        ),
        race_candidate_indices=race_indices,
    )
end

@testset "bootstrap family routing" begin
    states = fill(nothing, 5)
    select(data, mode, rows) = BootstrapContract._select_bootstrap_rows(
        data, states, mode, rows; validate_positions=false)

    exact = select(fixture_data("race_exact_k7"), "race", "exact-race")
    @test exact.indices == collect(1:5)
    @test exact.selected_race == 5
    @test exact.teacher_policy == "fixture_teacher"
    @test exact.producer_repo_commit == repeat("a", 40)
    @test length(exact.producer_repo_commit) == 40
    @test exact.engine_value == "gnubg-2ply"
    @test exact.engine_policy == "gnubg-2ply"
    @test exact.engine_cube == "gnubg-2ply"
    @test exact.game_mode == "money"
    @test exact.source_mode == "canonical_relabel"
    @test exact.source_selector == "contact_close_calls"
    @test select(fixture_data("race_natural_exact_k7"), "dual", "race").indices ==
          collect(1:5)

    full = fixture_data("full_game_gnubg"; race_indices=Int32[2, 5])
    @test select(full, "dual", "all").indices == collect(1:5)
    @test select(full, "dual", "contact").indices == [1, 3, 4]
    @test select(full, "dual", "race").indices == [2, 5]
    @test select(full, "race", "race").indices == [2, 5]
    @test_throws ArgumentError select(full, "race", "all")
    @test_throws ArgumentError select(full, "dual", "exact-race")

    contact = fixture_data("contact_gnubg")
    @test select(contact, "dual", "all").indices == collect(1:5)
    @test select(contact, "dual", "contact").indices == collect(1:5)
    @test_throws ArgumentError select(contact, "dual", "race")
    @test_throws ArgumentError select(contact, "race", "all")

    @test_throws ErrorException select(fixture_data("full_game_gnubg";
                                                    race_indices=Int32[2, 2]),
                                             "dual", "all")
    @test_throws ErrorException select(fixture_data("full_game_gnubg";
                                                    race_indices=Int32[6]),
                                             "dual", "all")
    for role in ("eval", "spot_eval")
        @test_throws ArgumentError select(
            fixture_data("race_exact_k7"; role), "race", "exact-race")
    end
    @test select(fixture_data("race_exact_k7"; role="bootstrap"),
                 "race", "exact-race").artifact_role == "bootstrap"
    @test_throws ArgumentError select(fixture_data("unknown"), "dual", "all")
    @test_throws ArgumentError select(full, "dual", "automatic")

    contact_game = BackgammonNet.initial_state(first_player=0)
    race_game = BackgammonNet.initial_state(first_player=0)
    race_game.p0 = UInt128(15) << (18 << 2)
    race_game.p1 = UInt128(15) << (7 << 2)
    BackgammonNet.set_dice!(race_game, 0, 0)
    validated = BootstrapContract._select_bootstrap_rows(
        fixture_data("full_game_gnubg"; race_indices=Int32[2]),
        [contact_game, race_game], "dual", "all")
    @test validated.selected_contact == 1
    @test validated.selected_race == 1
end
