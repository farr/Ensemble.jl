module Acor

function acf(xs::Array{Float64, 1})
    n = size(xs, 1)
    N = 1
    while N < 2*n
        N = N << 1
    end

    ys = zeros(N)
    ys[1:n] = xs

    ys_tilde = rfft(ys)
    ac = irfft(abs2(ys_tilde), N)

    ac[1:n]/ac[1]
end

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

end
