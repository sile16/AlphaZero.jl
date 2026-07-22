"""Small no-training preflight runner with machine-readable output."""
module Preflight

using Dates
using JSON

export run_preflight!, preflight_ok

preflight_ok(report) = all(check -> Bool(check["ok"]), report["checks"])

"""Run named zero-argument checks, write JSON, and throw if any check fails."""
function run_preflight!(checks::AbstractVector, report_path::AbstractString;
                        metadata::AbstractDict=Dict{String,Any}())
    results = Dict{String,Any}[]
    for (name, check) in checks
        started = time_ns()
        try
            detail = check()
            push!(results, Dict(
                "name" => String(name), "ok" => true,
                "detail" => detail === nothing ? "ok" : detail,
                "elapsed_ms" => (time_ns() - started) / 1e6))
        catch error
            push!(results, Dict(
                "name" => String(name), "ok" => false,
                "detail" => sprint(showerror, error),
                "elapsed_ms" => (time_ns() - started) / 1e6))
        end
    end
    report = Dict{String,Any}(
        "schema" => "alphazero_preflight_v1",
        "created_at_utc" => string(Dates.now(Dates.UTC)),
        "metadata" => Dict{String,Any}(String(k) => v for (k, v) in metadata),
        "checks" => results,
    )
    mkpath(dirname(report_path))
    open(report_path, "w") do io
        JSON.print(io, report, 2)
    end
    preflight_ok(report) || error("preflight failed; see $report_path")
    return report
end

end # module
