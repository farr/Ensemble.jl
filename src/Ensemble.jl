module Ensemble

include("Stats.jl")

include("Acor.jl")
include("EnsembleSampler.jl")
include("EnsembleNest.jl")
include("EnsembleGibbs.jl")
include("Optimize.jl")
include("Parameterizations.jl")
include("Plots.jl")

using .Acor
using .EnsembleSampler
using .EnsembleGibbs
using .EnsembleNest
using .Optimize
using .Parameterizations
using .Plots
using .Stats

export Acor, EnsembleSampler, EnsembleGibbs, EnsembleNest, Optimize, Parameterizations, Plots, Stats

end
