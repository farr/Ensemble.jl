module TestEnsemblePTSampler

using Base.Test: @test_approx_eq_eps, @testset

using Ensemble

function test_brewer()
    nd = 5
    nw = 128
    nt = 16

    function loglike(x)
        v = 0.1
        u = 0.01
        m = 0.031

        logl1 = -nd*log(v) - nd/2*log(2*pi) - 0.5*sum(x.*x/(v*v))
        logl2 = -nd*log(u) - nd/2*log(2*pi) - 0.5*sum((x-m).*(x-m)/(u*u))

        return Stats.logsumexp(logl1, log(99.0) + logl2)
    end

    function logprior(x)
        if any(x .> 0.5) || any(x .< -0.5)
            return -Inf
        else
            return 0.0
        end
    end

    pts = rand(nd,nw,nt) - 0.5

    betas = collect(linspace(1, 0, nt))

    chain, lnlikes, lnpriors, betas = EnsemblePTSampler.run_mcmc(pts, loglike, logprior, betas, 8192, thin=4)

    @testset "Brewer phase transition distribution" begin
        for i in 1:nd
            @test_approx_eq_eps mean(chain[i,:,1,:]) 0.99*0.031 0.1*sqrt(0.99*0.01^2 + 0.01*0.1^2)
        end

        lnZ, dlnZ = EnsemblePTSampler.lnZ(lnlikes, betas)
        @test_approx_eq_eps lnZ log(100.0) 3*dlnZ
    end

    chain, lnlikes, lnpriors, betas
end

function testall()
    @testset "EnsemblePTSampler test suite" begin
        test_brewer()
    end
end

end
