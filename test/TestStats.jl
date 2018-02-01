module TestStats

using Base.Test: @test, @test, @testset
using Ensemble
using Ensemble.Stats: logsumexp

function testlogsumexp_random()
    for i in 1:100
        x = randn()
        y = randn()

        @test isapprox(logsumexp(x,y), log(exp(x)+exp(y)))
    end
end

function testlogsumexp_inf()
    @test logsumexp(-Inf, 0.0)==0.0
    @test logsumexp(0.0, -Inf)==0.0
    @test logsumexp(-Inf,-Inf)==-Inf
end

function testall()
    testlogsumexp_random()
    testlogsumexp_inf()
end

end
