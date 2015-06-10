module TestStats

using Base.Test: @test_approx_eq_eps, @test
using Stats

function testlogsumexp_random()
    for i in 1:100
        x = randn()
        y = randn()

        @test_approx_eq_eps logsumexp(x,y) log(exp(x)+exp(y)) 1e-12
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
