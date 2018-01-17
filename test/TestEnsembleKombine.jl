module TestEnsembleKombine

using Ensemble

using Base.Test: @test_approx_eq_eps, @testset

function testgaussian()
    ndim = 10
    mu = randn(ndim)
    sigma = randn(ndim, ndim)
    sigma = sigma'*sigma

    sigmafact = cholfact(sigma)

    function logpost(x)
        dx = x .- mu
        -0.5*sum(dx .* (sigmafact \ dx))
    end

    pts = randn(10, 128)

    prop = EnsembleKombine.build_proposal_kde(pts)

    lnprobs = [lnprob(pts[:,j,1]) for j in 1:size(pts,2)]
    lnprops = [logpdf(prop, pts[:,j]) for j in 1:size(pts,2)]

    # Burnin
    for j in 1:10
        pts, lnprobs, lnprops = EnsembleKombine.run_mcmc(pts, lnprobs, lnprops, logpost, prop, 128)

        pts = pts[:,:,end]
        lnprobs = lnprobs[:,end]
        lnprops = lnprops[:,end]
        prop = EnsembleKombine.build_proposal_kde(pts)
    end

    pts, lnprobs, lnprops = EnsembleKombine.run_mcmc(pts, lnprobs, lnprops, logpost, prop, 1024)

    @testset "Gaussian distribution tests" begin
        for i in 1:ndim
            @test_approx_eq_eps mean(pts[i,:,:]) mu[i] sqrt(sigma[i,i])/10.0
        end
    end
end

function testall()
    @testset "EnsembleKombine tests" begin
        testgaussian()
    end
end

end
