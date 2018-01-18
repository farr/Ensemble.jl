module EnsemblePTSampler

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

    new_pts = zeros(size(pts)...)
    new_lnlikes = zeros(size(lnlikes)...)
    new_lnpriors = zeros(size(lnpriors)...)

    for k in 1:nt
        for j in 1:nw
            p = pts[:,j,k]

            jj = j
            while jj == j
                jj = rand(1:nw)
            end

            q = pts[:,jj,k]
            z = exp(log(0.5) + (log(2.0)-log(0.5))*rand())

            pnew = q + z*(p-q)
            lpnew = logprior(pnew)

            if lpnew == -Inf
                new_pts[:,j,k] = pts[:,j,k]
                new_lnlikes[j,k] = lnlikes[j,k]
                new_lnpriors[j,k] = lnpriors[j,k]
            else
                llnew = loglike(pnew)

                lpacc = betas[k]*(llnew-lnlikes[j,k]) + lpnew - lnpriors[j,k] + nd*log(z)

                if log(rand()) < lpacc
                    new_pts[:,j,k] = pnew
                    new_lnlikes[j,k] = llnew
                    new_lnpriors[j,k] = lpnew
                else
                    new_pts[:,j,k] = pts[:,j,k]
                    new_lnlikes[j,k] = lnlikes[j,k]
                    new_lnpriors[j,k] = lnpriors[j,k]
                end
            end
        end
    end

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
    lnlikes = [loglike(pts[:,j,k]) for j in 1:nw, k in 1:nt]
    lnpriors = [logprior(pts[:,j,k]) for j in 1:nw, k in 1:nt]

    nhalf = div(nstep, 2)
    tc = div(nhalf, 4)

    for i in 1:nhalf
        pts, lnlikes, lnpriors = mcmc_step(pts, lnlikes, lnpriors, loglike, logprior, betas)
        pts, lnlikes, lnpriors, ntswap = tswap_step(pts, lnlikes, lnpriors, betas)
        swapfraction = ntswap / nw
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
