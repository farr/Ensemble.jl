"""
    EnsembleGibbs

A module mixing the ensemble walker stretch move on a subset of
parameter space with a Gibbs sampler on the complimentary subset.  Can
be useful for foward modelling in the presence of selection effects:
the stretch move adjusts model parameters given observed data and a
draw from the latent (un-observed) data, then the Gibbs step draws new
latent data."""

module EnsembleGibbs

"""
    propose(ps, qs)

Return `(ps_new, zs)`, the proposed updates to `ps` and the associated
one-dimensional parameters, `zs`.
"""
function propose(ps::Array{Float64, 2}, qs::Array{Float64, 2})
    n = size(ps,2)
    nd = size(ps,1)

    zs = exp(log(0.5) + (log(2.0) - log(0.5))*rand(n))
    inds = rand(1:n, (n,))

    ps_new = zeros(size(ps))
    for i in 1:n
        for j in 1:nd
            ps_new[j,i] = qs[j,inds[i]] + zs[i]*(ps[j,i] - qs[j,inds[i]])
        end
    end

    return ps_new, zs
end

"""
    lnprobs(ps, gs, lnprobfn)

Return an array of log-probabilities given parameters `ps` and
corresponding Gibbs samples `gs`.

The log-probability function, `lnprobfn`, should accept these
arguments as `lnprobfn(p, g)`.
"""
function lnprobs(ps, gs, lnprobfn)
    ps_any = Any[ps[:,i] for i in 1:size(ps,2)]
    Array{Float64, 1}(pmap(lnprobfn, ps_any, gs))
end

"""
    gibbses(ps, gs, gupdate)

Return new Gibbs samples to replace `gs` using `gupdate(p, g)`.
"""
function gibbses(ps, gs, gupdate)
    ps_any = Any[ps[:,i] for i in 1:size(ps,2)]
    Array{Any, 1}(pmap(gupdate, ps_any, gs))
end

"""
    update_half(ps, gs, lnps, qs, lnprobfn)

Return `(ps_new, gs, lnps_new)` from an MCMC step for half the system,
treating `qs` as fixed.
"""
function update_half(ps, gs, lnps, qs, lnprobfn)
    nd = size(ps, 1)
    n = size(ps, 2)

    ps_new, zs = propose(ps, qs)

    lnps_new = lnprobs(ps_new, gs, lnprobfn)
    
    lnpacc = lnps_new - lnps + nd*log(zs)

    acc = log(rand(n)) .< lnpacc

    ps_out = zeros(size(ps))
    lnps_out = zeros(size(lnps))

    ps_out[:,acc] = ps_new[:,acc]
    lnps_out[acc] = lnps_new[acc]

    ps_out[:,~acc] = ps[:,~acc]
    lnps_out[~acc] = lnps[~acc]

    ps_out, gs, lnps_out
end

"""
    update(ensemble, gibbs, lnprobs, lnprobfn, gibbsupdate)

Return `(new_ensemble, new_gibbs, new_lnprobs)` from a complete MCMC
then Gibbs cycle.
"""
function update(ensemble, gibbs, lnprob, lnprobfn, gibbsupdate)
    @assert(size(ensemble, 2) % 2 == 0)

    n = size(ensemble, 2)
    nd = size(ensemble, 1)

    ensemble = copy(ensemble)
    gibbs = copy(gibbs)
    lnprob = copy(lnprob)
    
    nh = div(n, 2)

    ps, gs, lnps = update_half(ensemble[:,1:nh], gibbs[1:nh], lnprob[1:nh], ensemble[:,nh+1:end], lnprobfn)

    ensemble[:,1:nh] = ps
    gibbs[1:nh] = gs
    lnprob[1:nh] = lnps

    ps, gs, lnps = update_half(ensemble[:,nh+1:end], gibbs[nh+1:end], lnprob[nh+1:end], ensemble[:,1:nh], lnprobfn)

    ensemble[:,nh+1:end] = ps
    gibbs[nh+1:end] = gs
    lnprob[nh+1:end] = lnps

    gibbs = gibbses(ensemble, gibbs, gibbsupdate)
    lnprob = lnprobs(ensemble, gibbs, lnprobfn)

    ensemble, gibbs, lnprob
end

"""
    run_mcmc(ensemble, gibbs, lnprobs, lnprobfn, gibbsupdate, steps; thin=1)

Run `steps` worth of `update` and return an array of the result,
optionally thinned.
"""
function run_mcmc(ensemble, gibbs, lnprobs, lnprobfn, gibbsupdate, steps; thin=1)
    nsave = div(steps, thin)

    chain = zeros((size(ensemble, 1), size(ensemble, 2), nsave))
    gibbschain = Array{Any,2}(size(ensemble, 2), nsave)
    chainlnprob = zeros((size(ensemble, 2), nsave))
    isave = 1
    for i in 1:steps
        ensemble, gibbs, lnprobs = update(ensemble, gibbs, lnprobs, lnprobfn, gibbsupdate)

        if i % thin == 0
            chain[:,:,isave] = ensemble
            gibbschain[:, isave] = gibbs
            chainlnprob[:,isave] = lnprobs

            isave = isave + 1
        end
    end

    chain, gibbschain, chainlnprob
end

end
