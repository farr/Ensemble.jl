using Ensemble
using Base.Test

using TestStats
using TestEnsembleSampler
using TestEnsembleGibbs
using TestEnsembleNest

TestStats.testall()
TestEnsembleSampler.testall()
TestEnsembleGibbs.testall()
TestEnsembleNest.testall()
