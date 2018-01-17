using Ensemble
using Base.Test: @testset

include("TestStats.jl")
include("TestEnsembleSampler.jl")
include("TestEnsembleGibbs.jl")
include("TestEnsembleNest.jl")
include("TestEnsembleKombine.jl")

using .TestStats
using .TestEnsembleSampler
using .TestEnsembleGibbs
using .TestEnsembleNest
using .TestEnsembleKombine

@testset "Ensemble.jl package tests" begin
    TestStats.testall()
    TestEnsembleSampler.testall()
    TestEnsembleGibbs.testall()
    TestEnsembleNest.testall()
    TestEnsembleKombine.testall()
end
