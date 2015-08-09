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
end
