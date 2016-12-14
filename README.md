The Ensemble Julia Package
==========================

[![Build Status](https://travis-ci.org/farr/Ensemble.jl.svg?branch=master)](https://travis-ci.org/farr/Ensemble.jl)

The Ensemble package implements in Julia the ensemble MCMC sampler of
[Goodman & Weare (2010)](https://goo.gl/EFNVCz), following in spirit
the implementation in the [emcee](http://dan.iel.fm/emcee/current/)
package from
[Foreman-Mackey, et al (2013)](http://adsabs.harvard.edu/abs/2013PASP..125..306F).

In addition to a basic implementation of the Goodman & Weare MCMC
algorithm in the `EnsembleSampler` module, this algorithm forms the
basis of a number of other stochastic sampling algorithms:

* A nested sampling algorithm based around the stretch move from
  Goodman & Weare in `EnsembleNest`.

* A combination MCMC/Gibbs sampling method in `EnsembleGibbs`.

* Various support libraries for stochastic sampling with these
  packages:

  - Computation of autocorrelation lengths following the spirit, if
    not the algorithm, of
    [Goodman's implementation](http://www.math.nyu.edu/faculty/goodman/software/acor/)
    of [Sokal's](http://www.stat.unc.edu/faculty/cji/Sokal.pdf)
    definition of the autocorrelation length in the module `Acor`.

  - Various useful re-parameterisations of constrained parameters to
    remove the constraints in the `Parameterizations` module.  Most of
    these are taken from the
    [Stan Users Manual](http://mc-stan.org/documentation/).

  - A basic
    [Powell's method](https://en.wikipedia.org/wiki/Powell's_method)
    optimiser in the `Optimize` module.

  - A stable implementation of the `logsumexp` function in the `Stats`
	module.

The `Ensemble` module exports these modules as top-level identifiers.



