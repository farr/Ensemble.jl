module EnsembleKombine

using ..Stats

import Base:
    rand

"""
    rescale(pts)

Transform ``pts`` (of size ``(ndim, npts)``) into zero-mean, unit covaraince.
"""
function rescale(pts)
    nd, np = size(pts)
    
    mu = vec(mean(pts, 2))
    cv = cov(pts, 2)

    repts = zeros(nd, np)
    for j in 1:np
        for i in 1:nd
            repts[i,j] = pts[i,j] - mu[i]
        end
    end

    F = cholfact(cv)

    # cv = L*L^T
    # so if x is N(0,1) then
    # y = Lx has covariance cv: y y^T = L x x^T L^T = cv

    F[:L] \ repts
end

function update_means(pts, mus, assigns)
    ncluster = maximum(assigns)

    new_mus = zeros(size(pts, 1), ncluster)
    for i in 1:ncluster
        ps = pts[:, assigns.==i]
        new_mus[:,i] = mean(ps, 2)
    end

    new_mus
end

function update_assigns(pts, mus, assigns)
    nd, np = size(pts)

    new_assigns = zeros(Int, size(pts,2))
    for j in 1:np
        p = pts[:,j]

        dmin = Inf
        imin = 0
        for k in 1:size(mus,2)
            d2 = sum((p.-mus[:,k]).*(p.-mus[:,k]))
            if d2 < dmin
                dmin = d2
                imin = k
            end

            new_assigns[j] = imin
        end
    end

    new_assigns
end

function kmeans_init(pts, n)
    nd, np = size(pts)

    mus = zeros(nd, n)
    assigns = ones(Int, np)

    inds = randperm(np)
    for j in 1:n
        mus[:,j] = pts[:,inds[j]]
    end

    mus, update_assigns(pts, mus, assigns)
end

function kmeans(pts, n, maxiter=10000)
    mus, assigns = kmeans_init(pts, n)

    new_mus = nothing
    new_assigns = nothing
    i = 0
    while true
        new_mus = update_means(pts, mus, assigns)
        new_assigns = update_assigns(pts, new_mus, assigns)

        if new_mus != mus
            mus = new_mus
            assigns = new_assigns
        elseif i > maxiter
            break
        else
            break
        end

        i += 1
    end

    new_mus, new_assigns
end

type ClusteredKDE
    pts::Array{Float64, 2}
    assigns::Array{Int, 1}
    cholfacts::Array{LinAlg.Cholesky{Float64,Array{Float64,2}}, 1}
end

function ClusteredKDE(pts, n)
    nd, np = size(pts)
    
    _, assigns = kmeans(pts, n)

    cfacts = LinAlg.Cholesky{Float64,Array{Float64,2}}[]
    for j in 1:n
        sel = assigns.==j
        ps = pts[:,sel]

        @assert size(ps,2)>n "one cluster has too few points"
        
        cv = cov(ps, 2)
        cv *= (1.0/np).^(2.0/(4.0 + nd)) # Scott's rule

        push!(cfacts, cholfact(cv))
    end

    ClusteredKDE(copy(pts), assigns, cfacts)
end

function ndim(ck::ClusteredKDE)
    size(ck.pts, 1)
end

function npts(ck::ClusteredKDE)
    size(ck.pts, 2)
end

function ncl(ck::ClusteredKDE)
    size(ck.cholfacts, 1)
end

function rand(ck::ClusteredKDE)
    j = rand(1:npts(ck))
    p = ck.pts[:, j]
    F = ck.cholfacts[ck.assigns[j]]

    p + F[:L]*randn(ndim(ck))
end

function rand(ck::ClusteredKDE, dims...)
    d = push!([ndim(ck)], dims...)

    pts = zeros(d...)
    cr = CartesianRange(size(pts)[2:end])
    for j in cr
        pts[:,j] = rand(ck)
    end
    pts
end

function logpdf(ck, x)
    logdets = [sum(log(diag(ck.cholfacts[i][:L]))) for i in 1:ncl(ck)]

    lpdf = -Inf
    for j in 1:npts(ck)
        p = ck.pts[:,j]
        a = ck.assigns[j]
        F = ck.cholfacts[a]

        dx = x - p
        r2 = sum(dx .* (F \ dx))

        l = -0.5*ndim(ck)*log(2.0*pi) -logdets[a] - 0.5*r2

        lpdf = logsumexp(lpdf, l)
    end
    lpdf - log(npts(ck))
end

function build_proposal_kde(pts, ncmax=nothing)
    # We need to have at least ndim+1 points in each cluster, so that
    # limits how many clusters we can have.  IF all points divided
    # equally, then this is the maximum number of clusters

    nd, np = size(pts)

    @assert np%2==0 "need even number of points!"

    if ncmax == nothing
        ncmax = floor(Int, np / (2*(nd+1)))
    end

    perm = randperm(np)
    ihalf = round(Int, np/2)
    kde_pts = pts[:, perm[1:ihalf]]
    test_pts = pts[:, perm[ihalf+1:end]]

    best = -Inf
    bestck = nothing
    for nc in 1:ncmax
        for j in 1:5 # Five trials at each nc
            try
                ck = ClusteredKDE(kde_pts, nc)
                logprob = sum([logpdf(ck, test_pts[:,k]) for k in 1:size(test_pts,2)])

                if logprob > best
                    best = logprob
                    bestck = ck
                end
            catch x
                if isa(x, AssertionError)
                    # Do nothing
                else
                    rethrow()
                end                
            end
        end
    end

    bestck
end

function mcmc_step(pts, lnprobs, lnprop, logpost, proposal)
    new_pts = rand(proposal, size(pts, 2))

    new_lnprobs = [logpost(new_pts[:,j]) for j in 1:size(new_pts,2)]

    new_lnprop = [logpdf(proposal, new_pts[:,j]) for j in 1:size(new_pts,2)]

    log_pacc = new_lnprobs .+ lnprop .- lnprobs .- new_lnprop

    out_pts = zeros(size(pts)...)
    out_lnprobs = zeros(size(lnprobs,1))
    out_lnprop = zeros(size(lnprop, 1))

    for j in 1:size(pts, 2)
        if log(rand()) < log_pacc[j]
            out_pts[:,j] = new_pts[:,j]
            out_lnprobs[j] = new_lnprobs[j]
            out_lnprop[j] = new_lnprop[j]
        else
            out_pts[:,j] = pts[:,j]
            out_lnprobs[j] = lnprobs[j]
            out_lnprop[j] = lnprop[j]
        end
    end

    (out_pts, out_lnprobs, out_lnprop)
end

function run_mcmc(pts, lnprobs, lnprop, logpost, proposal, steps; thin=1)
    nsave = div(steps, thin)

    chain = zeros(size(pts,1), size(pts,2), nsave)
    chainlnprob = zeros(size(pts,2), nsave)
    chainlnprop = zeros(size(pts,2), nsave)
    for i in 1:steps
        pts, lnprobs, lnprop = mcmc_step(pts, lnprobs, lnprop, logpost, proposal)

        if i % thin == 0
            chain[:,:,div(i,thin)] = pts
            chainlnprob[:,div(i,thin)] = lnprobs
            chainlnprop[:,div(i,thin)] = lnprop
        end
    end

    (chain, chainlnprob, chainlnprop)
end

end
