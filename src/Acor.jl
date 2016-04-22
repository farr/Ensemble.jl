module Acor

""" Returns an estimate of the autocorrelation function of the given
series.  """
function acf(xs::Array{Float64, 1})
    n = size(xs, 1)
    N = 1
    while N < 2*n
        N = N << 1
    end

    mu = mean(xs)

    ys = zeros(N)
    for i in 1:n
        ys[i] = xs[i] - mu
    end

    ys_tilde = rfft(ys)
    ac = irfft(abs2(ys_tilde), N)

    ac[1:n]/ac[1]
end

""" Returns an estimate of the autocorrelation length of the given
series, or `Inf` if this is not possible to accurately estimate.  """
function acl(xs::Array{Float64, 1})
    ac = acf(xs)

    n = size(ac, 1)

    cumac = 2.0*cumsum(ac) - 1.0
    for i in 1:div(n,2)
        if cumac[i] < i/5
            return cumac[i]
        end
    end

    return Inf
end

""" Returns an array giving the estimated ACL for each parameter,
treating `xs` as the output of an ensemble MCMC (shape `(ndim,
nwalkers, nsteps)`).  """
function acl(xs::Array{Float64, 3})
    ndim = size(xs, 1)
    nwalk = size(xs, 2)
    nstep = size(xs, 3)

    acls = zeros(ndim)
    means = zeros(nstep)

    for i in 1:ndim
        for k in 1:nstep
            means[k] = 0.0
            for j in 1:nwalk
                means[k] += xs[i,j,k]
            end
            means[k] /= nwalk
        end
        acls[i] = acl(means)
    end

    acls
end

""" Returns an array of shape `(ndim, nwalkers*nsteps)` where suitable
for thinning to generate independent samples.

This involves both a flattening and a dimension-permutation, so that
successive samples from the same ensemble don't sit adjacent to each
other in memory."""
function flatten(pts)
    p = permutedims(pts, (1, 3, 2))
    reshape(p, (size(p,1), size(p,2)*size(p,3)))
end

""" Thins an ensemble by the maximum autocorrelation length of the
    parameters in the ensemble. """
function thin(pts)
    l = maximum(acl(pts))

    if l == Inf
        warn("Cannot thin because ACL is infinite.")
        zeros(size(pts, 1), size(pts, 2), 0)
    else
        n = ceil(l)
        pts[:,:,1:n:end]
    end
end

""" Returns an estimate of the Gelman-Rubin R statistic for each
parameter, treating each walker as an independent simulation.  The R
statistic estimates the factor by which the variance of the estimated
distribution can be reduced by running the simulation longer."""
function gelman_rubin_rs(ps)
    ndim, nwalkers, nsteps = size(ps)

    rs = zeros(ndim)
    means = zeros(ndim, nwalkers)
    vars = zeros(ndim, nwalkers)

    for i in 1:ndim
        for j in 1:nwalkers
            for k in 1:nsteps
                means[i,j] += ps[i,j,k]
            end
            means[i,j] /= nsteps

            for k in 1:nsteps
                x = ps[i,j,k] - means[i,j]
                vars[i,j] += x*x
            end
            vars[i,j] /= nsteps
        end

        Bon = var(means[i,:])
        W = mean(vars[i,:])

        s2 = (nsteps-1.0)/nsteps*W + Bon

        rs[i] = s2/W
    end

    rs
end

""" An estimate of the AIC (Akaike Information Criterion) from the
MCMC samples of the log-likelihood.  The AIC is `-2.0*max(lnps) +
nparams`.  The estimate assumes Gaussianity of the likelihood
function; under this assumption the mean of the log-likelihood is
`nparams/2.0` below the maximum and the variance of the log-likelihood
is `nparams/2.0`.  So the estimate of the AIC is 

    waic = -2.0*mean(lnps) + 2.0*var(lnps)

This estimate is related to (or equal to?) the WAIC described by XXXX
(Gelman ref.).
"""
waic(lnps) = -2.0*mean(lnps) + 2.0*var(lnps)

"""Returns an array of acceptance rates for each walker from the given
chain. """
function acceptance_rate(ps)
    nw = size(ps, 2)
    nt = size(ps, 3)

    ar = zeros(nw)

    for j in 1:nw
        acc = 0
        for k in 1:nt-1
            if ps[:,j,k] == ps[:,j,k+1]
                # Pass
            else
                acc += 1
            end
        end
        ar[j] = float(acc)/float(nt)
    end

    ar
end

""" Returns the acceptance rate given that the chain `ps` has been
thinned by `thin`."""
function acceptance_rate(ps, thin)
    ar = acceptance_rate(ps)

    1 - exp(log1p(-ar)/thin)
end

end
