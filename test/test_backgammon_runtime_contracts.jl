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
        @test isfinite(BearoffTables.n15_best_move_value(tables.n15, n15_game))

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

function fixture_data(kind; role="train", race_indices=Int32[])
    return (
        metadata=Dict{String,Any}(
            "artifact_kind" => kind,
            "artifact_role" => role,
        ),
        race_candidate_indices=race_indices,
    )
end

@testset "bootstrap family routing" begin
    states = fill(nothing, 5)
    select(data, mode) = BootstrapContract._bootstrap_checker_indices(
        data, states, mode; validate_positions=false)

    @test select(fixture_data("race_exact_k7"), "race") == collect(1:5)
    @test select(fixture_data("race_natural_exact_k7"), "dual") == collect(1:5)
    @test_throws ArgumentError select(fixture_data("full_game_gnubg"), "race")
    @test_throws ArgumentError select(fixture_data("contact_gnubg"), "race")
    @test select(fixture_data("full_game_gnubg";
                              race_indices=Int32[2, 5]), "dual") == [1, 3, 4]
    @test select(fixture_data("contact_gnubg"), "dual") == collect(1:5)
    @test_throws ErrorException select(fixture_data("full_game_gnubg";
                                                    race_indices=Int32[2, 2]), "dual")
    @test_throws ErrorException select(fixture_data("full_game_gnubg";
                                                    race_indices=Int32[6]), "dual")
    @test_throws ArgumentError select(fixture_data("race_exact_k7"; role="eval"), "race")
    @test_throws ArgumentError select(fixture_data("unknown"), "dual")
end
