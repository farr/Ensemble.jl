using Ensemble
using Base.Test

using TestStats
using TestEnsembleSampler
using TestEnsembleGibbs

TestStats.testall()
TestEnsembleSampler.testall()
TestEnsembleGibbs.testall()
