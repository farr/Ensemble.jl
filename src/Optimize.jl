"""
    Optimize

Basic Powell's method optimizer.

"""
module Optimize

"""
    minbracket(f, a, b, c, epsabs, epsrel, epsdx)

Find a minimum of `f(x)` for `a <= x <= c`.  Uses basic bracketing.  

On entry, `f(b) < f(a)` and `f(b) < f(c)`.  Termination when either
`c-a <= epsdx` or the range of `f` on `a`, `b`, `c` satisfies `range
<= epsabs + epsrel*abs(f(b))`.

"""
function minbracket(f, a, b, c, epsabs, epsrel, epsdx)
    minbracket(f, a, f(x), b, f(b), c, f(c), epsabs, epsrel, epsdx)
end

function minbracket(f, a, fa, b, fb, c, fc, epsabs, epsrel, epsdx)
    @assert(fb < fa && fb < fc, "minimize: values must bracket a minimum")

    while c - a > epsdx && ((fc - fb) + (fa - fb))/2.0 > epsabs + epsrel/3.0*(abs(fa) + abs(fb) + abs(fc))
        if fa > fc
            x = a + 0.5*(b-a)
            fx = f(x)

            if fx > fb
                fa = fx
                a = x
            else
                fc = fb
                c = b
                fb = fx
                b = x
            end
        else
            x = b + 0.5*(c-b)
            fx = f(x)

            if fx > fb
                fc = fx
                c = x
            else
                fa = fb
                a = b
                fb = fx
                b = x
            end
        end
    end
    b, fb
end

function find_bounds(f, fx0)
    x0 = -1.0
    x1 = 0.0
    x2 = 1.0
    
    f0 = f(x0)
    f1 = fx0
    f2 = f(x2)

    while f1 > f2 || f1 > f0
        grad = (f1-f0)/(x1-x0) + (f2-f1)/(x2-x1) - (f2-f0)/(x2-x0)

        if grad > 0.0  # Going to expand to the left
            x0 = x1 - 2.0*(x1-x0)
            f0 = f(x0)
        else # Expand to the right
            x2 = x1 + 2.0*(x2-x1)
            f2 = f(x2)
        end
    end

    x0, f0, x1, f1, x2, f2
end

"""
    minpowell(f, x0, epsabs, epsrel, epsdx)
    minpowell(f, x0, direc, epsabs, epsrel, epsdx)

Returns the `x` that minimises `f` starting from `x0` using Powell's
method.

Termination when change in `f` on iteration is `df <= epsabs +
epsrel*abs(f)` or when the change in `x` satisfies `norm(dx) <=
epsdx`.

The optional `direc` argument gives the initial direction set.
`direc[:,i]` is the `i`th direction for the optimization.
"""
function minpowell(f, x0, epsabs, epsrel, epsdx)
    minpowell(f, x0, eye(size(x0,1)), epsabs, epsrel, epsdx)
end

function minpowell(f, x0, direc, epsabs, epsrel, epsdx)
    n = size(x0, 1)

    df_max = -Inf
    i_max = -1
    fx0 = Inf
    fbest = Inf
    new_direc = copy(direc)
    x = copy(x0)
    for i in 1:n
        dx = direc[:,i]

        function fone(y)
            xx = x + y*dx
            f(xx)
        end

        fone0 = fone(0.0)
        if i == 1
            fx0 = fone0
        end
        a, fa, b, fb, c, fc = find_bounds(fone, fone0)

        y, fy = minbracket(fone, a, fa, b, fb, c, fc, epsabs, epsrel, 0.0)

        fbest = fy
        
        # Update the scale of new_direc
        x = x + y*dx

        df = fone0 - fy
        if df > df_max
            df_max = df
            i_max = i
        end
    end

    dx = x - x0
    df = fx0 - fbest
    
    new_direc[:,i_max] = dx / norm(dx)

    if df < epsabs + abs(fbest)*epsrel || norm(dx) < epsdx
        x, fbest
    else
        minpowell(f, x, new_direc, epsabs, epsrel, epsdx)
    end
end

end
