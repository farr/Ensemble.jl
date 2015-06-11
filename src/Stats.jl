module Stats

export randpoi, logsumexp

@doc doc"""Return a Poisson-distributed integer with mean `lam`

""" ->
function randpoi(lam)
    if lam > 100.0
        sigma = sqrt(lam)
        mu = lam
        round(Int, lam + sigma*randn())
    else
        x = 0
        p = exp(-lam)
        s = p
        u = rand()
        while u > s
            x += 1
            p *= lam / x
            s += p
        end
        x
    end
end

@doc doc"""Compute `log(exp(x) + exp(y))` but accurately, and without
risk of overflow.

""" ->
function logsumexp(x, y)
    if x == -Inf
        y
    elseif y == -Inf
        x
    elseif x > y
        x + log1p(exp(y-x))
    else
        y + log1p(exp(x-y))
    end
end

@vectorize_2arg Number logsumexp

@doc doc"""If applied to a single array, returns `log(exp(x1) + exp(x2) + ...)`.

""" ->
function logsumexp{T <: Number}(x::AbstractArray{T})
    sum = -Inf
    for i in eachindex(x)
        sum = logsumexp(x[i], sum)
    end
    sum
end
        
end
