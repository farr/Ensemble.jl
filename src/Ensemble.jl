module Ensemble

include("Acor.jl")
include("EnsembleSampler.jl")
include("EnsembleGibbs.jl")
include("Stats.jl")

using .Acor
using .EnsembleSampler
using .EnsembleGibbs
using .Stats

export Acor, EnsembleSampler, EnsembleGibbs, Stats

end
