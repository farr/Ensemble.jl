module TestEnsembleSampler

using Ensemble

using Base.Test: @test_approx_eq_eps

function testgaussian()
    ndim = 10

    mu = randn(ndim)
    
    sigma = randn(ndim, ndim)
    sigma = sigma*transpose(sigma)

    sigmafact = factorize(sigma)

    function lnprob(x)
        y = x - mu
        -0.5*dot(y, sigmafact\y)
    end

    ps = randn(ndim, 100)
    lnps = EnsembleSampler.lnprobs(ps, lnprob)
    # Burnin 1000 steps
    for i in 1:1000
        ps, lnps = EnsembleSampler.update(ps, lnps, lnprob)
    end

    # Run 1000 steps
    chain, chainlnp = EnsembleSampler.run_mcmc(ps, lnps, lnprob, 1000, thin=10)

    chain_mean = zeros(ndim)
    chain_sigma = zeros(ndim,ndim)

    for i in 1:100
        for j in 1:100
            for k in 1:ndim
                chain_mean[k] += chain[k,j,i]
            end
        end
    end
    chain_mean /= 100*100

    for i in 1:100
        for j in 1:100
            for k in 1:ndim
                x = chain[k,j,i] - chain_mean[k]
                for l in 1:ndim
                    y = chain[l,j,i] - chain_mean[l]
                    chain_sigma[k,l] += x*y
                end
            end
        end
    end
    chain_sigma /= 100*100 - 1

    for i in 1:ndim
        @test_approx_eq_eps chain_mean[i] mu[i] 0.1*sigma[i,i]
    end

    chain_evals = eigvals(chain_sigma)
    evals = eigvals(sigma)
    
    for i in 1:ndim
        @test_approx_eq_eps chain_evals[i] evals[i] 0.1*evals[i]
    end

    chain_evecs = eigvecs(chain_sigma)
    evecs = eigvecs(sigma)

    for i in 1:ndim
        @test_approx_eq_eps abs(dot(chain_evecs[:,i], evecs[:,i])) 1.0 0.1
    end
end

sqr(x) = x*x
sqr{T}(x::AbstractArray{T,1}) = x.*x

# SNR = A/2*sqrt(n)/sigma

logit(x, a=0.0, b=1.0) = log(x-a) - log(b-x)

function invlogit(y::Float64, a=0.0, b=1.0)
    if y > 0
        ey = exp(-y)
        (a*ey + b)/(ey + 1.0)
    else
        ey = exp(y)
        (a + b*ey)/(1.0 + ey)
    end
end

function logitlj(y::Float64, a=0.0, b=1.0)
    if y > 0
        log(b-a) - y - 2.0*log1p(exp(-y))
    else
        log(b-a) + y - 2.0*log1p(exp(y))
    end
end

function make_sinusoidlnprob(ts::Array{Float64, 1}, data::Array{Float64, 1})
    function lnprob(x)
        lna::Float64 = x[1]
        lnP::Float64 = x[2]
        logitphi::Float64 = x[3]
        
        n = size(ts,1)

        phi = invlogit(logitphi, 0.0, 2*pi)

        a = exp(lna)
        P = exp(lnP)

        sig = zeros(n)
        for i in 1:n
            sig[i] = a*cos(2.0*pi*ts[i]/P + phi)
        end

        resid = data - sig

        ll = -0.5*dot(resid,resid)
        # flat prior in a, P, phi
        lp = lna + lnP + logitlj(logitphi, 0.0, 2*pi)

        ll+lp
    end
end

function testsinusoid()
    snr = 5.0
    
    n = 100
    sigma = 1.0
    
    a = 2.0*sigma*snr/sqrt(n)
    P = 1.0

    phi0 = 2*pi*rand()

    ts = sort(100.0*rand(n))

    noise = sigma*randn(n)
    signal = a*cos(2.0*pi*ts/P + phi0)

    data::Array{Float64, 1} = noise+signal

    lnprob = make_sinusoidlnprob(ts, data)

    ptrue = [log(a), log(P), logit(phi0, 0, 2*pi)]
    p0 = zeros(3, 100)
    for i in 1:100
        for j in 1:3
            p0[j,i] = ptrue[j] + 1e-4*randn()
        end
    end
    lnp0 = EnsembleSampler.lnprobs(p0, lnprob)

    for i in 1:1000
        p0, lnp0 = EnsembleSampler.update(p0, lnp0, lnprob)
    end

    chain, chainlnp = EnsembleSampler.run_mcmc(p0, lnp0, lnprob, 1000, thin=10)

    as = exp(chain[1,:,:])
    Ps = exp(chain[2,:,:])
    phis = [invlogit(c, 0, 2*pi) for c in chain[3,:,:]]

    @test_approx_eq_eps mean(as) a 3*std(as)
    @test_approx_eq_eps mean(Ps) P 3*std(Ps)
    @test_approx_eq_eps mean(phis) phi0 3*std(phis)
end

function testall()
    testgaussian()
    testsinusoid()
end

end
