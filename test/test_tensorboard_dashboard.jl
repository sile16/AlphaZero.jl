using Test
using TensorBoardLogger

include(joinpath(@__DIR__, "..", "src", "distributed", "tensorboard_dashboard.jl"))

@testset "TensorBoard dashboard namespace" begin
    allowed_groups = Set([
        "00_run",
        "01_ml_loss",
        "02_ml_perf",
        "03_selfplay_perf",
        "04_data",
        "05_eval_strength",
        "06_eval_bearoff",
        "07_system",
        "08_reliability",
        "09_promotion",
    ])

    sources = [
        read(joinpath(@__DIR__, "..", "scripts", "training_server.jl"), String),
        read(joinpath(@__DIR__, "..", "src", "distributed", "server.jl"), String),
    ]
    tags = String[]
    for source in sources
        append!(tags, (m.captures[1] for m in eachmatch(r"@info \"([^\"]+/[^\"]+)\"", source)))
    end
    groups = Set(first(split(tag, '/')) for tag in tags)

    @test !isempty(tags)
    @test groups <= allowed_groups
    @test allowed_groups <= groups
    @test length(unique(tags)) <= 72
    @test "01_ml_loss/contact_value" in tags
    @test "02_ml_perf/samples_per_sec" in tags
    @test "05_eval_strength/win_pct" in tags
    @test "05_eval_strength/contact_value_mse" in tags
    @test "04_data/train_sample_age_mean" in tags
    @test "06_eval_bearoff/fixed_mcts_top1" in tags
    @test "08_reliability/rejected_batches" in tags

    layout = tensorboard_dashboard_layout()
    @test length(layout) == 8
    @test collect(keys(layout)) == [
        "01 ML Loss", "02 ML Performance", "03 Self-play and Search",
        "04 Data and Labels", "05 Playing Strength", "06 Bearoff Evaluation",
        "07 Operations", "08 Promotion",
    ]
    @test collect(keys(layout["04 Data and Labels"])) ==
        ["Buffer composition", "Label coverage", "Training sample age (iterations)", "Outcomes"]
    @test sum(length, values(layout)) == 27
    @test sum(length, values(layout)) <= 27
    referenced = String[]
    for category in values(layout), (_, (_, chart_tags)) in category
        append!(referenced, chart_tags)
    end
    @test all(tag -> tag in tags, referenced)

    mktempdir() do dir
        logger = TBLogger(dir, tb_overwrite)
        @test install_tensorboard_dashboard!(logger) == layout
        close(logger)
        @test !isempty(readdir(dir))
    end
end
