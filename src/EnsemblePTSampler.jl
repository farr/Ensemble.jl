module EnsemblePTSampler

function propose(ps, qs)
    nd, nw, nt = size(ps)

    ps_out = zeros(nd, nw, nt)
    zs = zeros(nw, nt)

    for i in 1:nw
        for j in 1:nt
            zs[i,j] = exp(log(0.5) + (log(2.0)-log(0.5))*rand())
            ii = rand(1:nw)
            ps_out[:,i,j] = qs[:,ii,j] + zs[i,j]*(ps[:,i,j] - qs[:,ii,j])
        end
    end

    ps_out, zs
end

function lnlike_lnprior(x, ll, lp)
    p = lp(x)
    if p == -Inf
        (-Inf, -Inf)
    else
        (ll(x), p)
    end
end

function lnlikes_lnpriors(ps, ll, lp)
    nd, nw, nt = size(ps)
    if length(workers()) > 1
        parr = Array{Float64,1}[]
        for j in 1:nt
            for i in 1:nw
                push!(parr, ps[:,i,j])
            end
        end
        lls_lps = pmap(x -> lnlike_lnprior(x, ll, lp), parr, batch_size=div(nw*nt,(4*length(workers()))))
        lls = reshape(Float64[l for (l,p) in lls_lps], (nw, nt))
        lps = reshape(Float64[p for (l,p) in lls_lps], (nw, nt))
        (lls, lps)
    else
        lls = zeros(nw, nt)
        lps = zeros(nw, nt)
        for i in 1:nw
            for j in 1:nt
                lps[i,j] = lp(ps[:,i,j])
                if lps[i,j] == -Inf
                    lls[i,j] = -Inf
                else
                    lls[i,j] = ll(ps[:,i,j])
                end
            end
        end
        (lls, lps)
    end
end

function update_half(ps, llps, lpps, qs, ll, lp, betas)
    nd, nw, nt = size(ps)

    prop_ps, zs = propose(ps, qs)
    prop_llps, prop_lpps = lnlikes_lnpriors(prop_ps, ll, lp)

    new_ps = zeros(size(ps)...)
    new_llps = zeros(size(llps)...)
    new_lpps = zeros(size(lpps)...)

    for i in 1:nw
        for j in 1:nt
            lpacc = betas[j]*(prop_llps[i,j] - llps[i,j]) + prop_lpps[i,j] - lpps[i,j] + nd*log(zs[i,j])
            if log(rand()) < lpacc
                new_ps[:,i,j] = prop_ps[:,i,j]
                new_llps[i,j] = prop_llps[i,j]
                new_lpps[i,j] = prop_lpps[i,j]
            else
                new_ps[:,i,j] = ps[:,i,j]
                new_llps[i,j] = llps[i,j]
                new_lpps[i,j] = lpps[i,j]
            end
        end
    end

    (new_ps, new_llps, new_lpps)
end

"""
    mcmc_step(pts, lnlikes, lnpriors, loglike, logprior, betas)

Returns `(new_pts, new_lnlikes, new_lnpriors)` from a single MCMC step operating
on the given ensemble.

# Arguments
* `pts` a `(ndim, nwalkers, ntemps)` array of the current state of the chain.
* `lnlikes` a `(nwalkers, ntemps)` array giving the current log-likelihoods.
* `lnpriors` a `(nwalkers, ntemps)` array giving the current log-priors.
* `loglike` a function that gives the log-likelihood for parameter values.
* `logprior` a function that gives the log-prior for parameter values.
* `betas` a `(ntemps,)` array with the current inverse temperatures.

"""
function mcmc_step(pts, lnlikes, lnpriors, loglike, logprior, betas)
    nd, nw, nt = size(pts)

    @assert nw%2==0 "must have even number of walkers"

    ihalf = div(nw, 2)

    ps = pts[:,1:ihalf,:]
    llps = lnlikes[1:ihalf,:]
    lpps = lnpriors[1:ihalf,:]

    qs = pts[:,ihalf+1:end,:]
    llqs = lnlikes[ihalf+1:end,:]
    lpqs = lnpriors[ihalf+1:end,:]

    new_ps, new_llps, new_lpps = update_half(ps, llps, lpps, qs, loglike, logprior, betas)
    new_qs, new_llqs, new_lpqs = update_half(qs, llqs, lpqs, new_ps, loglike, logprior, betas)

    new_pts = cat(2, new_ps, new_qs)
    new_lnlikes = cat(1, new_llps, new_llqs)
    new_lnpriors = cat(1, new_lpps, new_lpqs)
    (new_pts, new_lnlikes, new_lnpriors)
end

"""
    tswap_step(pts, lnlikes, lnpriors, betas)

Implements the parallel-tempering swap step between ensembles of different
temperatures.  Arguments are as for `mcmc_step`.

"""
function tswap_step(pts, lnlikes, lnpriors, betas)
    nd, nw, nt = size(pts)

    new_pts = copy(pts)
    new_lnlikes = copy(lnlikes)
    new_lnpriors = copy(lnpriors)
    tswaps = zeros(Int, nt-1)

    for k in nt:-1:2
        for j in 1:nw
            jj = rand(1:nw)

            llhigh = new_lnlikes[j,k]
            lllow = new_lnlikes[jj, k-1]

            bhigh = betas[k]
            blow = betas[k-1]
            db = bhigh-blow

            lpacc = db*(lllow - llhigh)

            if log(rand()) < lpacc
                phigh = new_pts[:,j,k]
                lphigh = new_lnpriors[j,k]

                new_pts[:,j,k] = new_pts[:,jj,k-1]
                new_lnlikes[j,k] = new_lnlikes[jj,k-1]
                new_lnpriors[j,k] = new_lnpriors[jj,k-1]

                new_pts[:,jj,k-1] = phigh
                new_lnlikes[jj,k-1] = llhigh
                new_lnpriors[jj,k-1] = lphigh

                tswaps[k-1] += 1
            end # If not accepted, do nothing
        end
    end

    new_pts, new_lnlikes, new_lnpriors, tswaps
end


"""
    tevolve(swapfraction, betas, tc)

Returns `new_betas`, which are evolved according to a rule to equalise
temperature swap rates over a timescale `tc`.

The rule is formulated in terms of a quantity `x[i]`, which is the logit
transform of `beta[i]` relative to its neighbours (recall that `beta` is the
*inverse* temperature, so decreases as `i` increases):

    x[i] = log(beta[i] - beta[i+1]) - log(beta[i-1] - beta[i])

`x` maps the range between `beta[i+1]` and `beta[i-1]` to `-Inf` to `Inf`.  We
evolve `x` via

    x_new[i] = x[i] + 2.0*(swapfraction[i] - swapfraction[i-1])/(tc*(swapfraction[i] + swapfraction[i-1]))

where `swapfraction[i]` measures the fraction of accepted swaps between
temperature `i` and temperature `i+1` (i.e. between chain `i` and the
next-highest temperature chain) and `tc` is a "time constant" that controls the
(exponential) rate of convergence of `x`.

This is similar to the evolution rule in [Vousden, Farr, & Mandel
(2016)](https://ui.adsabs.harvard.edu/#abs/2016MNRAS.455.1919V/abstract).

"""
function tevolve(swapfraction, betas, tc)
    new_betas = copy(betas)

    for i in 2:size(betas, 1)-1
        if swapfraction[i] == 0 && swapfraction[i-1] == 0
            # Do nothing---we don't know which way to move!
        else
            x = log(betas[i] - betas[i+1]) - log(betas[i-1] - betas[i])
            xnew = x + 2.0*(swapfraction[i] - swapfraction[i-1])/(tc*(swapfraction[i] + swapfraction[i-1]))
            exnew = exp(xnew)
            new_betas[i] = (new_betas[i+1] + exnew*new_betas[i-1])/(1.0 + exnew)
        end
    end

    new_betas
end

"""
    run_mcmc(pts, loglike, logprior, betas, nstep; [thin=1])

Run a MCMC simulation of length `nstep`.

The simulation will consist of a "burnin" phase and a "sampling" phase, each
involving half `nstep` iterations.  During "burnin", the temperatures of the
chain will adapt with a time constant of 1/4 of the burnin length.  During
sampling the temperatures are fixed to preserve detailed balance.

The function will return `(chain, chainloglike, chainlogprior, new_betas)` where
each returned chain value has an extra dimension appended counting steps of the
chain (so `chain` is of shape `(ndim, nwalkers, ntemp, nstep)`, for example).

# Arguments
* `pts` a `(ndim, nwalkers, ntemp)` array with the current state.
* `loglike` a function giving the log-likelihood for parameters.
* `logprior` a function giving the log-prior for parameters.
* `betas` a `(ntemp,)` array of inverse temperatures.
* `thin=1` save the output every `thin` steps.

"""
function run_mcmc(pts, loglike, logprior, betas, nstep; thin=1)
    nd, nw, nt = size(pts)
    lnlikes, lnpriors = lnlikes_lnpriors(pts, loglike, logprior)

    nhalf = div(nstep, 2)

    for i in 1:nhalf
        pts, lnlikes, lnpriors = mcmc_step(pts, lnlikes, lnpriors, loglike, logprior, betas)
        pts, lnlikes, lnpriors, ntswap = tswap_step(pts, lnlikes, lnpriors, betas)
        swapfraction = ntswap / nw
        tc = 10.0 + i/10.0 # tc > 10 always, but asymptotes to "current length / 10"
        betas = tevolve(swapfraction, betas, tc)
    end

    nsave = div(nhalf, thin)
    chain = zeros(nd, nw, nt, nsave)
    chainlnlike = zeros(nw, nt, nsave)
    chainlnprior = zeros(nw, nt, nsave)
    isave = 1

    for i in 1:nhalf
        pts, lnlikes, lnpriors = mcmc_step(pts, lnlikes, lnpriors, loglike, logprior, betas)
        pts, lnlikes, lnpriors, _ = tswap_step(pts, lnlikes, lnpriors, betas)

        if i % thin == 0
            chain[:,:,:,isave] = pts
            chainlnlike[:,:,isave] = lnlikes
            chainlnprior[:,:,isave] = lnpriors
            isave += 1
        end
    end

    chain, chainlnlike, chainlnprior, betas
end

function trapz(ys, xs)
    sum(0.5*diff(xs).*(ys[1:end-1] + ys[2:end]))
end

"""
    lnZ(lnlikes, betas)

Returns `(ln_Z, delta_ln_Z)` estimating the log-evidence and uncertainty using
thermodynamic integration.

# Arguments
* `lnlikes` an `(nwalkers, ntemps, nsteps)` array of log-likelihoods.
* `betas` an `(ntemps,)` array of the corresponding beta values.

"""
function lnZ(lnlikes, betas)
    mean_lnlikes = vec(mean(lnlikes, (1, 3)))

    bs_thin = [betas[1]]
    lnls_thin = [mean_lnlikes[1]]
    for i in 2:2:size(betas,1)-1
        push!(bs_thin, betas[i])
        push!(lnls_thin, mean_lnlikes[i])
    end
    push!(bs_thin, betas[end])
    push!(lnls_thin, mean_lnlikes[end])

    lnZ = -trapz(mean_lnlikes, betas)
    lnZ2 = -trapz(lnls_thin, bs_thin)

    lnZ, abs(lnZ-lnZ2)
end

end
