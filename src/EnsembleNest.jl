module EnsembleNest

using ..EnsembleSampler
using ..Stats

export NestState, ndim, nlive, ndead, retire!, logZ, logdZ, postsample, run!

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

function ndim(n::NestState)
    size(n.livepts, 1)
end

function nlive(n::NestState)
    size(n.livepts, 2)
end

function ndead(n::NestState)
    size(n.deadpts, 2)
end

function retire!(n::NestState)
    retire!(n::NestState, true)
end

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
    # = 2/p - 1), and then plan to run for 10*bpACL.  Average the plan
    # over the past nlive retirings to compute the next mcmc length.
    # If there were no acceptances, plan to run for twice as long
    # (still averaging over the last nlive retirings).
    if nacc == 0
        n.nmcmc_exact = (1.0 + 1.0/nl)*n.nmcmc_exact 
    else
        n.nmcmc_exact = (1.0 - 1.0/nl)*n.nmcmc_exact + 10.0/nl*(2.0/facc - 1.0)
    end
    n.nmcmc = round(Int,n.nmcmc_exact)

    if verbose
        println("Retired point with ll = $(n.deadlogls[end]); accept = $(facc); next nmcmc = $(n.nmcmc)")
    end
    
    n
end

function logZ(n::NestState)
    nl = nlive(n)
    
    loglivewt = n.logx - log(nl)

    logZdead = logsumexp(n.deadlogwts + n.deadlogls)
    logZlive = logsumexp(n.livelogls + loglivewt)

    logsumexp(logZdead, logZlive)
end

function logdZ(n::NestState)
    logZlow = minimum(n.livelogls) + n.logx
    logZhigh = maximum(n.livelogls) + n.logx

    logZhigh + log1p(-exp(logZlow - logZhigh))
end

function postsample(n::NestState)
    nd = ndim(n)
    nl = nlive(n)

    loglivewt = n.logx - log(nl)
    
    pts = cat(2, n.deadpts, n.livepts)
    logwts = cat(1, n.deadlogls + n.deadlogwts, n.livelogls + loglivewt)

    logwtmax = maximum(logwts)
    post = zeros((nd, 0))

    for i in 1:size(pts, 2)
        if logwtmax + log(rand()) < logwts[i]
            post = cat(2, post, reshape(pts[:,i], (nd, 1)))
        end
    end

    post
end

function run!(n::NestState, dZStop)
    run!(n, dZStop, true)
end

function run!(n::NestState, dZStop, verbose)
    while true
        for i in 1:nlive(n)
            retire!(n, verbose)
        end

        lZ = logZ(n)
        ldZ = logdZ(n)

        if verbose
            println("Now evolved for $(ndead(n)) steps, dZ/Z = $(exp(ldZ-lZ))")
        end
        
        if ldZ - lZ < log(dZStop)
            break
        end
    end
end

end
