module EnsembleSampler

using ..Acor

## Store ensembles as (ndim, nwalkers) arrays.

function propose(ps::Array{Float64, 2}, qs::Array{Float64, 2})
    n = size(ps, 2)

    zs = exp(log(0.5) + rand((n,))*(log(2.0) - log(0.5)))
    inds = rand(1:n, (n,))

    ps_out = zeros(size(ps))
    for i in 1:n
        ps_out[:,i] = qs[:,inds[i]] + zs[i]*(ps[:,i] - qs[:,inds[i]])
    end
    ps_out, zs
end

function lnprobs(xs::Array{Float64, 2}, lnprobfn)
    xseq = Any[xs[:,i] for i=1:size(xs,2)]
    Array{Float64,1}(pmap(lnprobfn, xseq))
end

function update_half(ensemble::Array{Float64, 2}, lnprob::Array{Float64, 1}, lnprobfn, half)
    n = size(ensemble, 2)
    nd = size(ensemble, 1)

    if n % 2 != 0
        error("update_half requires even number of walkers")
    end

    n_half = div(n, 2)

    if half % 2 == 0
        ps = ensemble[:, 1:n_half]
        qs = ensemble[:, n_half+1:end]
        lnps = lnprob[1:n_half]
    else
        ps = ensemble[:, n_half+1:end]
        qs = ensemble[:, 1:n_half]
        lnps = lnprob[n_half+1:end]
    end

    ps_new, zs = propose(ps, qs)
    lnp_new = lnprobs(ps_new, lnprobfn)

    lnpacc = lnp_new - lnps + nd*log(zs)

    outhalf = zeros((nd, n_half))
    lnpouthalf = zeros(n_half)
    for i in 1:n_half
        if lnpacc[i] > 0 || log(rand()) < lnpacc[i]
            outhalf[:,i] = ps_new[:,i]
            lnpouthalf[i] = lnp_new[i]
        else
            outhalf[:,i] = ps[:,i]
            lnpouthalf[i] = lnps[i]
        end
    end

    out = zeros((nd, n))
    lnpout = zeros(n)
    if half % 2 == 0
        out[:,1:n_half] = outhalf
        out[:,n_half+1:end] = ensemble[:,n_half+1:end]

        lnpout[1:n_half] = lnpouthalf
        lnpout[n_half+1:end] = lnprob[n_half+1:end]
    else
        out[:,1:n_half] = ensemble[:,1:n_half]
        out[:,n_half+1:end] = outhalf

        lnpout[1:n_half] = lnprob[1:n_half]
        lnpout[n_half+1:end] = lnpouthalf
    end

    out, lnpout
end

function update(ensemble::Array{Float64, 2}, lnprob::Array{Float64,1}, lnprobfn)
    e, lp = update_half(ensemble, lnprob, lnprobfn, 0)
    update_half(e, lp, lnprobfn, 1)
end

function run_mcmc(ensemble, lnprob, lnprobfn, steps; thin=1)
    nsave = div(steps, thin)

    chain = zeros((size(ensemble, 1), size(ensemble, 2), nsave))
    chainlnprob = zeros((size(ensemble, 2), nsave))
    for i in 1:steps
        ensemble, lnprob = update(ensemble, lnprob, lnprobfn)

        if i % thin == 0
            chain[:,:,div(i, thin)] = ensemble
            chainlnprob[:,div(i,thin)] = lnprob
        end
    end

    chain, chainlnprob
end

function run_to_neff(ensemble, lnprob, lnprobfn, neff; callback=nothing)
    n = 8*neff
    n0 = n
    nd = size(ensemble, 1)
    nw = size(ensemble, 2)
    thin = 1

    ps = reshape(ensemble, (nd, nw, 1))
    lnps = reshape(lnprob, (nw, 1))
    
    lnpmax = maximum(lnps)

    while true
        ps, lnps = run_mcmc(ps[:,:,end], lnps[:,end], lnprobfn, n, thin=thin)

        if !(callback == nothing)
            callback(ps, lnps, n, thin)
        end
        
        acls = Acor.acl(ps)
        amax = maximum(acls)
        nee = size(ps, 3)/amax

        if nee > neff
            break
        end

        # Now check whether we need to re-centre around the maximum...
        lpm = maximum(lnps)

        if lpm > lnpmax + 2.0*sqrt(nd/2.0) # If we have increased max log(L) by 2-sigma
            ne = size(ps, 3)

            # Count number of samples we have above mean - 3*sigma
            lnpthresh = lpm - nd/2.0 - 3.0*sqrt(nd/2.0)  

            pslnps_above = Set()
            for i in 1:nw
                for j in 1:ne
                    p = ps[:,i,j]
                    l = lnps[i,j]

                    if l > lnpthresh
                        push!(pslnps_above, (p,l))
                    end
                end
            end
            na = length(pslnps_above)

            if na > 2*nd
                # Then we have at least 2*nd samples above threshold,
                # and we can re-start the sampling from general
                # position
                lnpmax = lpm
                n = n0
                thin = 1

                ps_above = zeros(nd, na)
                lnps_above = zeros(na)
                for (i, (p, lp)) in enumerate(pslnps_above)
                    ps_above[:,i] = p
                    lnps_above[i] = lp
                end
            
                new_ps = zeros(nd, nw, 1)
                new_lnps = zeros(nw, 1)

                for i in 1:nw
                    p = nothing
                    l = -Inf
                    while l < lnpthresh
                        j = rand(1:na)
                        p = ps_above[:,j]
                        l = lnps_above[j]
                    end
                    new_ps[:,i,1] = p
                    new_lnps[i,1] = l
                end
                
                ps = new_ps
                lnps = new_lnps
            end
        else            
            n *= 2
            thin *= 2
        end
    end

    ps, lnps
end

end
