module Parameterizations

bounded_param(x, low, high) = log(x-low) - log(high-x)

function bounded_value(p, low, high)
    if p > 0
        ep = exp(-p)
        (high + ep*low)/(one(p) + ep)
    else
        ep = exp(p)
        (high*ep + low)/(one(p) + ep)
    end
end

bounded_value(x::Vector, low, high) = [bounded_value(x[i], low, high) for i in eachindex(x)]

bounded_logjac(x, p, low, high) = log((high-x)*(x-low)/(high-low))
function bounded_logjac(x::Vector, p::Vector, low, high)
    lj = 0.0
    for i in eachindex(x)
        lj += bounded_logjac(x[i], p[i], low, high)
    end
    lj
end

function increasing_params(x)
    n = length(x)
    for i in 2:n
        @assert(x[i] > x[i-1], "values must be increasing")
    end

    p = zeros(x)
    p[1] = x[1]
    for i in 2:n
        p[i] = log(x[i]-x[i-1])
    end

    p
end

function increasing_values(p)
    n = length(p)

    x = zeros(p)
    x[1] = p[1]
    for i in 2:n
        x[i] = x[i-1] + exp(p[i])
    end

    x
end

function increasing_logjac(x, p)
    n = length(p)

    lj = 0.0
    for i in 2:n
        lj += p[i]
    end

    lj
end

function simplex_params(x)
    @assert(abs(sum(x) - 1) < 1e-8, "values must sum to one")

    n = length(x)

    p = zeros(n-1)
    remaining = one(p[1])
    for i in 1:n-1
        p[i] = bounded_param(x[i], zero(x[i]), remaining)
        remaining = remaining - x[i]
    end

    p
end

function simplex_values(p)
    n = length(p)

    x = zeros(n+1)
    remaining = one(p[1])
    for i in 1:n
        x[i] = bounded_value(p[i], zero(p[i]), remaining)
        remaining = remaining - x[i]
    end

    x[n+1] = remaining

    x
end

function simplex_logjac(x, p)
    lj = 0.0
    remaining = 1.0

    for i in 1:length(p)
        lj += bounded_logjac(x[i], p[i], zero(x[i]), remaining)
        remaining = remaining - x[i]
    end

    lj
end

function unit_disk_param(z)
    x = z[1]
    y = z[2]

    r2 = x*x + y*y

    a = 1.0 - r2

    Float64[x/a, y/a]
end

function unit_disk_value(p)
    u = p[1]
    v = p[2]

    R2 = u*u + v*v
    R = sqrt(R2)

    r = 0.5*(sqrt(4.0*R2 + 1.0) - 1.0)/R

    x = u*r/R
    y = v*r/R

    Float64[x, y]
end

function unit_disk_logjac(z, p)
    x = z[1]
    y = z[2]

    r2 = x*x + y*y

    3.0*log1p(-r2) - log1p(r2)
end

end
