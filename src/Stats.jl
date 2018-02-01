module Stats

export logsumexp

"""
    logsumexp(x,y)
    logsumexp(x)

Return `log(exp(x) + exp(y))` but stably without overflow.

Single-argument version returns `log(exp(x[1]) + exp(x[2]) + ...)`.
"""
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
