using Ensemble
using Test: @testset

include("TestStats.jl")
include("TestEnsembleSampler.jl")
include("TestEnsembleGibbs.jl")
include("TestEnsembleNest.jl")
include("TestEnsembleKombine.jl")
include("TestEnsemblePTSampler.jl")

using .TestStats
using .TestEnsembleSampler
using .TestEnsembleGibbs
using .TestEnsembleNest
using .TestEnsembleKombine
using .TestEnsemblePTSampler

@testset "Ensemble.jl package tests" begin
    TestStats.testall()
    TestEnsembleSampler.testall()
    TestEnsembleGibbs.testall()
    TestEnsembleNest.testall()
    TestEnsembleKombine.testall()
    TestEnsemblePTSampler.testall()
end
