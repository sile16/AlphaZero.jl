using Test
using Flux

include(joinpath(@__DIR__, "..", "src", "distributed", "numerical_safety.jl"))

@testset "Non-finite gradient guard" begin
    finite = (layers=((weight=ones(Float32, 2, 2), bias=zeros(Float32, 2), σ=nothing),),)
    @test _all_finite_gradient(finite)

    nan_gradient = (layers=((weight=Float32[1 NaN; 2 3], bias=zeros(Float32, 2), σ=nothing),),)
    inf_gradient = Dict(:weight => Float32[1, Inf])
    @test !_all_finite_gradient(nan_gradient)
    @test !_all_finite_gradient(inf_gradient)
    @test _all_finite_gradient(nothing)

    model = Chain(Dense(2 => 3, tanh), Dense(3 => 1))
    _, gradients = Flux.withgradient(model) do current
        sum(abs2, current(rand(Float32, 2, 4)))
    end
    @test _all_finite_gradient(gradients[1])
end
