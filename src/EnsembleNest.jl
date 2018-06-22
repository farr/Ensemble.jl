module EnsembleNest

using ..EnsembleSampler
using ..Stats

using HDF5

import Base:
    write

export NestState, ndim, nlive, ndead, retire!, logZ, logdZ, postsample, run!

"""
Nested sampling state object.
"""
type NestState
    likelihood
    prior
    nmcmc::Int
    nmcmc_exact::Float64
    livepts::Array{Float64, 2}
    livelogps::Array{Float64, 1}
    livelogls::Array{Float64, 1}
    deadpts::Array{Float64, 2}
    deadlogls::Array{Float64,1}
    deadlogwts::Array{Float64,1}
    logx::Float64
    loglthresh::Float64
end

"""
    NestState(loglike, logprior, init, nmcmc)

Return a `NestState` object initialised from the like points in the 2D
array `init` using the given log-likelihood and log-prior functions.

`nmcmc` will be used initially to control the number of steps taken
before generating a live point to replace the low-likelihood point
that is being retired.  This parameter will be adjusted in a control
loop that tries to ensure that successive live points are
uncorrelated, as described below in the docs for `retire!`.  A setting
of `32` is a reasonable default.
"""
function NestState(logl, logp, pts::Array{Float64, 2}, nmcmc)
    npts = size(pts, 2)
    ndim = size(pts, 1)

    livelogps = zeros(npts)
    livelogls = zeros(npts)
    for i in 1:npts
        livelogps[i] = logp(pts[:,i])
        livelogls[i] = logl(pts[:,i])
    end

    NestState(logl, logp, nmcmc, nmcmc, copy(pts), livelogps, livelogls, zeros((ndim,0)), zeros(0), zeros(0), 0.0, -Inf)
end

"""Reading and writing NestState objects from HDF5"""
function NestState(f::Union{HDF5File,HDF5Group}; logl=nothing, logp=nothing)
    NestState(logl, logp, read(f, "nmcmc"), read(f, "nmcmc_exact"),
              read(f, "livepts"), read(f, "livelogps"), read(f, "livelogls"),
              read(f, "deadpts"), read(f, "deadlogls"), read(f, "deadlogwts"),
              read(f, "logx"), read(f, "loglthresh"))
end

function write(f::Union{HDF5File, HDF5Group}, ns::NestState)
    f["nmcmc"] = ns.nmcmc
    f["nmcmc_exact"] = ns.nmcmc_exact
    f["livepts", "compress", 3, "shuffle", ()] = ns.livepts
    f["livelogps", "compress", 3, "shuffle", ()] = ns.livelogps
    f["livelogls", "compress", 3, "shuffle", ()] = ns.livelogls
    f["deadpts", "compress", 3, "shuffle", ()] = ns.deadpts
    f["deadlogls", "compress", 3, "shuffle", ()] = ns.deadlogls
    f["deadlogwts", "compress", 3, "shuffle", ()] = ns.deadlogwts
    f["logx"] = ns.logx
    f["loglthresh"] = ns.loglthresh
end

"""Return the dimension of the problem in `n`."""
function ndim(n::NestState)
    size(n.livepts, 1)
end

"""Return the number of live points."""
function nlive(n::NestState)
    size(n.livepts, 2)
end

"""Return the number of dead (retired) points."""
function ndead(n::NestState)
    size(n.deadpts, 2)
end

function retire!(n::NestState)
    retire!(n::NestState, true)
end

"""
    retire!(nstate, verbose=true)

Retire the lowest-likelihood live point, using the stretch move in an MCMC to
produce its replacement.  If `verbose == true` then print information on the
retired point and the number of MCMC steps used.

The number of MCMC steps is adjusted so that it approaches `2*(2/accept_rate -
1)` exponentially with a rate that is `1/nlive(nstate)`.  If each accepted
stretch move generated a truly independent point, this would correspond to
running for 2 autocorrelation lengths of the resulting series to produce the
replacement live point.  The factor 2 is a "safety factor", and the exponential
approach with rate constant ensures that the internal MCMC adapts to the "local"
conditions of the likelihood vs. prior curve.  """
function retire!(n::NestState, verbose)
    nd = ndim(n)
    nl = nlive(n)

    # Find the point we are killing
    imin = indmin(n.livelogls)

    # Update the dead points, logl, logwt
    n.deadpts = cat(2, n.deadpts, reshape(n.livepts[:,imin], (nd, 1)))
    push!(n.deadlogls, n.livelogls[imin])
    push!(n.deadlogwts, n.logx - log(nl))

    # Update prior fraction
    n.logx += log1p(-1.0/nl)

    # Update threshold
    n.loglthresh = n.livelogls[imin]

    # Now update the retired point
    i = rand(1:nl)
    pt = n.livepts[:,i]
    ll = n.livelogls[i]
    lp = n.livelogps[i]

    nacc = 0

    for i in 1:n.nmcmc
        q = n.livepts[:,rand(1:nl)]
        z = exp(log(0.5) + rand()*(log(2.0)-log(0.5)))
        newpt = q + z*(pt - q)

        newlp = n.prior(newpt)

        logpacc = newlp - lp + nd*log(z)

        if log(rand()) < logpacc
            newll = n.likelihood(newpt)
            if newll > n.loglthresh
                nacc += 1
                pt = newpt
                ll = newll
                lp = newlp
            end
        end
    end

    n.livepts[:,imin] = pt
    n.livelogls[imin] = ll
    n.livelogps[imin] = lp

    facc = float(nacc)/float(n.nmcmc)

    # Based on past acceptance rate, estimate the best-possible bpACL
    # = 2/p - 1), and then plan to run for 2*bpACL.  Average the plan
    # over the past nlive retirings to compute the next mcmc length.
    # If there were no acceptances, plan to run for twice as long
    # (still averaging over the last nlive retirings).
    if nacc == 0
        n.nmcmc_exact = (1.0 + 1.0/nl)*n.nmcmc_exact
    else
        n.nmcmc_exact = (1.0 - 1.0/nl)*n.nmcmc_exact + 2.0/nl*(2.0/facc - 1.0)
    end
    n.nmcmc = round(Int,n.nmcmc_exact)

    if verbose
        println(@sprintf("Retired point with ll = %.4f; accept = %.4f; next nmcmc = %d", n.deadlogls[end], facc, n.nmcmc))
    end

    n
end

""" Return the (natural) log of the evidence and an estimate of its
uncertainty.  """
function logZ(n::NestState)
    nl = nlive(n)

    loglivewt = n.logx - log(nl)

    logZdead = logsumexp(n.deadlogwts + n.deadlogls)
    logZlive = logsumexp(n.livelogls + loglivewt)
    logZlive_big = maximum(n.livelogls) + n.logx
    logZlive_small = minimum(n.livelogls) + n.logx

    logZ = logsumexp(logZdead, logZlive)
    logZ_big = logsumexp(logZdead, logZlive_big)
    logZ_small = logsumexp(logZdead, logZlive_small)

    logZ, (logZ_big - logZ_small)
end

""" Return the pair `(samples, lnlikes)` resulting from a posterior
sampling of the given nested state."""
function postsample(n::NestState)
    nd = ndim(n)
    nl = nlive(n)

    loglivewt = n.logx - log(nl)

    pts = cat(2, n.deadpts, n.livepts)
    lls = cat(1, n.deadlogls, n.livelogls)
    logwts = cat(1, n.deadlogls + n.deadlogwts, n.livelogls + loglivewt)

    logwtmax = maximum(logwts)
    post = zeros((nd, 0))
    logls = Float64[]

    for i in 1:size(pts, 2)
        if logwtmax + log(rand()) < logwts[i]
            post = cat(2, post, reshape(pts[:,i], (nd, 1)))
            logls = push!(logls, lls[i])
        end
    end

    post, logls
end

"""
    run!(nstate, dZStop, verbose=true, ckpt_file=nothing)

Run `retire!` on the live points in `nstate` until the uncertainty in
the log evidence calculation is smaller than `dZStop`.

`verbose` is as in `retire!`.

`ckpt_file` is a filename in which to store intermediate, serialised
states of the computation.  Deserialising one of these states and
calling `run!` on it again will continue the computation"""
function run!(n::NestState, dZStop; verbose=true, ckpt_file=nothing)
    while true
        for i in 1:nlive(n)
            retire!(n, verbose)
        end

        lZ, dlZ = logZ(n)

        if verbose
            println(@sprintf("Now evolved for %d steps, dlog(Z) = %.4f", ndead(n), dlZ))
        end

        if ckpt_file != nothing
            tmpfile = "$(ckpt_file).temp"
            h5open(f -> write(f, n), tmpfile, "w")
            mv(tmpfile, ckpt_file, remove_destination=true)
        end

        if dlZ < dZStop
            break
        end
    end
end

function dic(lls::Array{Float64, 1})
    -2.0*(mean(lls) - var(lls))
end

"""
    dic(nstate)

Return the DIC for the given nested state.

The DIC is defined as ``\mathrm{DIC} \equiv -2\left( \langle \log L
\rangle - \var \log L \right)``.
"""
function dic(ns::NestState)
    _, lls = postsample(ns)

    dic(lls)
end

end
