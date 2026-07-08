#!/usr/bin/env julia
"""
sharpen_bootstrap.jl — produce a SHARPENED copy of a soft bootstrap policy target.

Loads a raw columnar bootstrap NamedTuple (states, policies, values, equity),
raises each policy vector to a power `p` over its ORIGINALLY-NONZERO (legal)
entries and renormalizes to sum 1. Power-sharpening is a strictly monotone map on
positive entries, so the argmax is provably UNCHANGED — only the mass distribution
is concentrated toward the top move. states/values/equity are copied verbatim.

Usage:
    julia --project scripts/sharpen_bootstrap.jl <in.jls> <out.jls> [power]
"""

using BackgammonNet
using Serialization
using Statistics: mean

function main()
    length(ARGS) >= 2 || error("usage: sharpen_bootstrap.jl <in.jls> <out.jls> [power]")
    in_path  = ARGS[1]
    out_path = ARGS[2]
    p = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 4.0

    println("Loading: $in_path")
    d = Serialization.deserialize(in_path)
    d isa NamedTuple && hasproperty(d, :policies) ||
        error("expected raw columnar NamedTuple with :policies, got $(typeof(d))")

    pol = d.policies
    n = length(pol)
    println("Samples: $n   power p=$p")

    sharp = Vector{Vector{Float32}}(undef, n)
    argmax_moved = 0
    old_maxprobs = Float64[]
    new_maxprobs = Float64[]
    old_nnz = Int[]
    new_nnz = Int[]

    for i in 1:n
        v = pol[i]
        am_old = argmax(v)
        # sharpen only the strictly-positive (legal) entries; keep zeros zero
        w = similar(v)
        s = 0.0
        @inbounds for j in eachindex(v)
            if v[j] > 0
                x = Float64(v[j])^p
                w[j] = Float32(x)
                s += x
            else
                w[j] = 0.0f0
            end
        end
        # renormalize over legal mass (s > 0 always: a valid dist has ≥1 positive entry)
        s > 0 || error("sample $i has no positive policy mass")
        @inbounds for j in eachindex(w)
            w[j] = Float32(Float64(w[j]) / s)
        end
        am_new = argmax(w)
        am_new == am_old || (argmax_moved += 1)
        sharp[i] = w
        if i <= 3000
            push!(old_maxprobs, maximum(v)); push!(new_maxprobs, maximum(w))
            push!(old_nnz, count(!iszero, v)); push!(new_nnz, count(!iszero, w))
        end
    end

    argmax_moved == 0 || error("ARGMAX MOVED on $argmax_moved samples — sharpening is broken")
    println("Argmax unchanged on ALL $n samples ✓")
    println("mean max-prob:  soft=$(round(mean(old_maxprobs),digits=4))  sharp=$(round(mean(new_maxprobs),digits=4))")
    println("mean nnz:       soft=$(round(mean(old_nnz),digits=2))  sharp=$(round(mean(new_nnz),digits=2)) (nnz preserved by construction)")

    out = merge(d, (policies = sharp,))
    println("Saving: $out_path")
    Serialization.serialize(out_path, out)

    # verify readable + argmax match on a reload
    r = Serialization.deserialize(out_path)
    @assert length(r.policies) == n
    mism = 0
    for i in 1:n
        argmax(r.policies[i]) == argmax(pol[i]) || (mism += 1)
        abs(sum(r.policies[i]) - 1.0) < 1e-4 || error("sample $i does not sum to 1: $(sum(r.policies[i]))")
    end
    mism == 0 || error("reload argmax mismatch on $mism samples")
    # fields preserved
    @assert r.values == d.values
    @assert r.equity == d.equity
    @assert length(r.states) == length(d.states)
    println("Reload OK: readable, sums=1, argmax matches soft on all $n, values/equity/states preserved ✓")
end

main()
