import TensorBoardLogger

"""Minimal insertion-ordered `AbstractDict` used only for dashboard metadata."""
struct DashboardMap <: AbstractDict{String,Any}
    entries::Vector{Pair{String,Any}}
end

dashboard_map(entries::Pair...) = DashboardMap(
    Pair{String,Any}[String(first(entry)) => last(entry) for entry in entries])

Base.length(map::DashboardMap) = length(map.entries)
Base.iterate(map::DashboardMap, state::Int=1) =
    state > length(map.entries) ? nothing : (map.entries[state], state + 1)
Base.keys(map::DashboardMap) = String[first(entry) for entry in map.entries]
Base.values(map::DashboardMap) = Any[last(entry) for entry in map.entries]
function Base.getindex(map::DashboardMap, key::AbstractString)
    index = findfirst(entry -> first(entry) == key, map.entries)
    index === nothing && throw(KeyError(key))
    return last(map.entries[index])
end
Base.haskey(map::DashboardMap, key::AbstractString) =
    any(entry -> first(entry) == key, map.entries)
Base.get(map::DashboardMap, key::AbstractString, default) =
    haskey(map, key) ? map[key] : default

"""Return the bounded, combined-chart layout for TensorBoard Custom Scalars."""
function tensorboard_dashboard_layout()
    multiline(tags...) = (TensorBoardLogger.tb_multiline, String[tags...])
    return dashboard_map(
        "01 ML Loss" => dashboard_map(
            "Overall" => multiline(
                "01_ml_loss/overall", "01_ml_loss/contact_total", "01_ml_loss/race_total"),
            "Contact components" => multiline(
                "01_ml_loss/contact_policy", "01_ml_loss/contact_value",
                "01_ml_loss/contact_invalid"),
            "Race components" => multiline(
                "01_ml_loss/race_policy", "01_ml_loss/race_value",
                "01_ml_loss/race_invalid"),
        ),
        "02 ML Performance" => dashboard_map(
            "Iteration timing" => multiline(
                "02_ml_perf/train_seconds", "02_ml_perf/reanalyze_seconds",
                "02_ml_perf/iteration_seconds"),
            "Training throughput" => multiline("02_ml_perf/samples_per_sec"),
            "Training health" => multiline(
                "02_ml_perf/skipped_batches", "02_ml_perf/nonfinite_batches"),
        ),
        "03 Self-play and Search" => dashboard_map(
            "Game production" => multiline("03_selfplay_perf/games_per_sec"),
            "Sample production" => multiline("03_selfplay_perf/samples_per_sec"),
            "NN evaluations per simulation" => multiline(
                "03_selfplay_perf/nn_evals_per_sim"),
            "Oracle batch size" => multiline("03_selfplay_perf/oracle_batch_size"),
            "Lookup hit rates" => multiline(
                "03_selfplay_perf/tree_hit_rate", "03_selfplay_perf/bearoff_hit_rate"),
        ),
        "04 Data and Labels" => dashboard_map(
            "Buffer composition" => multiline(
                "04_data/contact_samples", "04_data/race_samples"),
            "Label coverage" => multiline(
                "04_data/equity_label_rate", "04_data/chance_sample_rate",
                "04_data/bearoff_sample_rate"),
            "Training sample age (iterations)" => multiline(
                "04_data/train_sample_age_mean", "04_data/train_sample_age_max"),
            "Outcomes" => multiline(
                "04_data/win_rate", "04_data/gammon_rate", "04_data/backgammon_rate"),
        ),
        "05 Playing Strength" => dashboard_map(
            "Equity and side balance" => multiline(
                "05_eval_strength/equity", "05_eval_strength/white_equity",
                "05_eval_strength/black_equity"),
            "Win rate" => multiline("05_eval_strength/win_pct"),
            "Value prediction MSE" => multiline(
                "05_eval_strength/contact_value_mse",
                "05_eval_strength/race_value_mse"),
            "Value prediction correlation" => multiline(
                "05_eval_strength/contact_value_corr",
                "05_eval_strength/race_value_corr"),
        ),
        "06 Bearoff Evaluation" => dashboard_map(
            "Value error" => multiline(
                "06_eval_bearoff/learned_value_mae",
                "06_eval_bearoff/fixed_value_mae"),
            "Policy top-k" => multiline(
                "06_eval_bearoff/fixed_policy_top1",
                "06_eval_bearoff/fixed_policy_top3"),
            "Raw NN versus MCTS" => multiline(
                "06_eval_bearoff/fixed_nn_top1",
                "06_eval_bearoff/fixed_mcts_top1"),
            "Decision regret" => multiline(
                "06_eval_bearoff/fixed_policy_expected_regret",
                "06_eval_bearoff/fixed_nn_regret",
                "06_eval_bearoff/fixed_mcts_regret"),
        ),
        "07 Operations" => dashboard_map(
            "Compute utilization" => multiline(
                "07_system/gpu_percent", "07_system/cpu_percent"),
            "Upload latency" => multiline("08_reliability/upload_latency_ms"),
            "Upload batch errors" => multiline(
                "08_reliability/duplicate_batches",
                "08_reliability/rejected_batches"),
        ),
        "08 Promotion" => dashboard_map(
            "Quality threshold" => multiline(
                "09_promotion/metric", "09_promotion/best_metric",
                "09_promotion/threshold"),
        ),
    )
end

# TensorBoardLogger 0.1.25's Custom Scalars helper uses keyword constructors
# that are incompatible with its generated ProtoBuf structs under current
# ProtoBuf/Julia. Build the same metadata event with positional constructors.
function _protobuf_with_defaults(T; kwargs...)
    defaults = TensorBoardLogger.PB.default_values(T)
    values = ntuple(fieldcount(T)) do i
        name = fieldname(T, i)
        haskey(kwargs, name) ? kwargs[name] : getproperty(defaults, name)
    end
    return T(values...)
end

function _write_custom_scalar_layout_compat!(logger, layout)
    TBL = TensorBoardLogger
    CS = TBL.tensorboard_plugin_custom_scalar

    categories = CS.Category[]
    for (category_name, chart_specs) in layout
        charts = CS.Chart[]
        for (chart_name, (chart_type, tags)) in chart_specs
            chart_type == TBL.tb_multiline || error("Only multiline charts are supported")
            content = CS.MultilineChartContent(tags)
            push!(charts, CS.Chart(chart_name, TBL.OneOf(:multiline, content)))
        end
        push!(categories, CS.Category(category_name, charts, false))
    end
    proto_layout = CS.Layout(Int32(1), categories)
    plugin_data = _protobuf_with_defaults(TBL.SummaryMetadata_PluginData;
                                           plugin_name="custom_scalars")
    metadata = _protobuf_with_defaults(TBL.SummaryMetadata; plugin_data=plugin_data)
    shape = _protobuf_with_defaults(TBL.TensorShapeProto)
    tensor = _protobuf_with_defaults(TBL.TensorProto;
        dtype=TBL._DataType.DT_STRING,
        tensor_shape=shape,
        string_val=[TBL.serialize_proto(proto_layout)])
    value = TBL.Summary_Value(
        "", "custom_scalars__config__", metadata, TBL.OneOf(:tensor, tensor))
    summary = TBL.SummaryCollection(value)
    TBL.write_event(logger, TBL.make_event(logger, summary; step=0))
    return layout
end

"""Install the Custom Scalars layout once at server startup."""
function install_tensorboard_dashboard!(logger)
    layout = tensorboard_dashboard_layout()
    try
        TensorBoardLogger.log_custom_scalar(logger, layout; step=0)
    catch e
        e isa MethodError || rethrow()
        _write_custom_scalar_layout_compat!(logger, layout)
    end
    return layout
end
