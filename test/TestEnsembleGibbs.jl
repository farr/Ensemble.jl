module TestEnsembleGibbs

using Base.Test: @test
using EnsembleGibbs
using Stats

const snr_half = 3.0
const mu = log(3.0)
const sigma = 1.0/3.0
const lambda = 100.0

const ptrue = [log(lambda), mu, log(sigma)]

log_odds(x) = 0.5*(x*x - snr_half*snr_half)

function logpselect(x)
    lo = log_odds(x)

    return lo - logsumexp(0.0, lo)
end
@vectorize_1arg Number logpselect

function logpnselect(x)
    lo = log_odds(x)

    return -logsumexp(0.0, lo)
end
@vectorize_1arg Number logpnselect

function draw(lambda=lambda, mu=mu, sigma=sigma)
    n = randpoi(lambda)
    xs = exp(mu + sigma*randn(n))
    psel = exp(logpselect(xs))

    sel = rand(n) .< psel

    xs[sel], xs[~sel]
end

function lognorm_logpdf(xs, mu, sigma)
    out = zeros(xs)

    log_inv_sqrt_2pi = -0.5*log(2.0*pi)
    log_sigma = log(sigma)
    sigma2 = sigma*sigma

    for i in eachindex(xs)
        log_x = log(xs[i])
        y = log_x - mu
        out[i] = log_inv_sqrt_2pi - log_sigma - log_x - 0.5*y*y/sigma2
    end

    out
end

function make_lnprob(xsdet)
    ndet = length(xsdet)
    function lnprob(params, xsndet)
        nndet = length(xsndet)
        log_lambda, mu, log_sigma = params
        lambda = exp(log_lambda)
        sigma = exp(log_sigma)

        sum(logpselect(xsdet)) + sum(logpnselect(xsndet)) + (ndet + nndet)*log_lambda - lambda + sum(lognorm_logpdf(xsdet, mu, sigma)) + sum(lognorm_logpdf(xsndet, mu, sigma))
    end
end

function gibbsupdate(params, xsndet)
    log_lambda, mu, log_sigma = params
    lambda = exp(log_lambda)
    sigma = exp(log_sigma)

    xsdet, xsndet = draw(lambda, mu, sigma)
    xsndet
end

function testall()
    xsdet, xsndet = draw()

    lnprob = make_lnprob(xsdet)
    ps = zeros(3, 100)
    for i in 1:100
        for j in 1:3
            ps[j,i] = ptrue[j] + 1e-3*randn()
        end
    end
    gs = EnsembleGibbs.gibbses(ps, [nothing for i in 1:100], gibbsupdate)
    lnprobs = EnsembleGibbs.lnprobs(ps, gs, lnprob)

    for i in 1:1000
        ps, gs, lnprobs = EnsembleGibbs.update(ps, gs, lnprobs, lnprob, gibbsupdate)
    end

    pts = zeros(3, 100, 100)
    for i in 1:100
        for j in 1:10
            ps, gs, lnprobs = EnsembleGibbs.update(ps, gs, lnprobs, lnprob, gibbsupdate)
        end
        pts[:,:,i] = ps
    end

    @test((mean(pts[1,:,:] - ptrue[1])/std(pts[1,:,:]) < 3))
    @test((mean(pts[2,:,:] - ptrue[2])/std(pts[2,:,:]) < 3))
    @test((mean(pts[3,:,:] - ptrue[3])/std(pts[3,:,:]) < 3))          
end

end
