module Plots

using Colors
using Gadfly

function chain_plot_layers(ps)
    niter = size(ps, 3)
    ndim = size(ps, 1)

    cs = distinguishable_colors(ndim)

    mus = mean(ps, 2)
    mu = mean(mus, 3)
    sigma = std(mus, 3)

    [layer(x=1:niter, y=(mus[i,1,:]-mu[i,1,1])/sigma[i,1,1], Geom.line, Theme(default_color=cs[i])) for i in 1:ndim]
end

function chain_plot(ps)
    plot(chain_plot_layers(ps)...)
end

function lnprob_plot(lnps; all=false)
    nw = size(lnps, 1)
    nt = size(lnps, 2)
    if all
        cs = Colors.distinguishable_colors(nw)
        layers = [layer(x=1:nt, y=lnps[i,:], Geom.line, Theme(default_color=cs[i])) for i in 1:nw]
        plot(layers...)
    else
        plot(x=1:nt, y=mean(lnps, 1), Geom.line)
    end
end

function parameter_plot(ps, i)
    nwalk = size(ps, 2)
    ntime = size(ps, 3)

    colors = Colors.distinguishable_colors(nwalk)

    layers = [layer(x=1:ntime, y=ps[i,j,:], Geom.line, Theme(default_color=colors[j])) for j in 1:nwalk]
    plot(layers...)
end

end
