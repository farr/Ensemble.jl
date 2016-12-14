module TestEnsembleNest

using Base.Test: @test_approx_eq_eps

using Ensemble
using .EnsembleNest

function test5DGaussian()
    nd = 5

    function logp(x)
        if any(x .> 1.0) || any(x .< 0.0)
            -Inf
        else
            0.0
        end
    end

    mu = 0.5*ones(nd) + 0.1*rand((nd,))
    sigma = 0.01*ones(nd)
    
    function logl(x)
        -0.5*nd*log(2.0*pi) - sum(log(sigma)) - 0.5*sum((x-mu).*(x-mu)./(sigma.*sigma))
    end

    nmcmc = 100
    nl = 1024
    xs = rand((nd, nl))
    ns = NestState(logl, logp, xs, nmcmc)

    run!(ns, 0.1)

    lZ, dlZ = logZ(ns)
    
    @test_approx_eq_eps lZ 0.0 1.0

    post, lnprobs = postsample(ns)
    npost = size(post, 2)

    ses = sigma / sqrt(npost)
    for i in 1:nd
        smu = mean(post[i,:])
        @test_approx_eq_eps smu mu[i] 3*ses[i]
    end
end

function testall()
    test5DGaussian()
end

end
