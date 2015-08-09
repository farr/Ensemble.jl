module Plots

using Color
using Gadfly

""" Returns an array of layers containing  """

function chain_plot_layers(ps)
    niter = size(ps, 3)
    ndim = size(ps, 1)

    cs = distinguishable_colors(ndim)

    mus = mean(ps, 2)
    mu = mean(mus, 3)
    sigma = std(mus, 3)

    [layer(x=1:niter, y=(mus[i,1,:]-mu[i,1,1])/sigma[i,1,1], Geom.line, Theme(default_color=color(AlphaColorValue(cs[i], 1.0)))) for i in 1:ndim]
end

function chain_plot(ps)
    plot(chain_plot_layers(ps)...)
end

function lnprob_plot(lnps)
    plot(x=1:size(lnps, 2), y=mean(lnps, 1), Geom.line)
end

end
