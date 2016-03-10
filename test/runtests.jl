using Ensemble
using Base.Test

include("TestStats.jl")
include("TestEnsembleSampler.jl")
include("TestEnsembleGibbs.jl")
include("TestEnsembleNest.jl")

using .TestStats
using .TestEnsembleSampler
using .TestEnsembleGibbs
using .TestEnsembleNest

TestStats.testall()
TestEnsembleSampler.testall()
TestEnsembleGibbs.testall()
TestEnsembleNest.testall()
