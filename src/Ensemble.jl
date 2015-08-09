module Ensemble

include("Acor.jl")
include("EnsembleSampler.jl")
include("EnsembleGibbs.jl")
include("Optimize.jl")
include("Parameterizations.jl")
include("Plots.jl")
include("Stats.jl")

using .Acor
using .EnsembleSampler
using .EnsembleGibbs
using .Optimize
using .Parameterizations
using .Plots
using .Stats

export Acor, EnsembleSampler, EnsembleGibbs, Optimize, Parameterizations, Plots, Stats

end
