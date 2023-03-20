# Auxiliary Kalman and particle Gibbs samplers for generalised Feynman-Kac models

This is the companion code for the paper [Auxiliary MCMC and particle Gibbs samplers for parallelisable
inference in latent dynamical systems](redacted) by REDACTED.

The models considered here are of the form

```math
    \pi(x_{0:T}) \propto p_0(x_0) \prod_{t=1}^T p_t(x_t | x_{t-1}) g(x_{0:T})
```

for which the auxiliary Kalman sampler can be implemented as long as $g$ is differentiable and Gaussian approximations of
$p_t(x_t | x_{t-1})$ can be computed.

For separable potentials, for example, $g(x_{0:T}) = \prod_{t=0}^T g_t(x_t)$, the second order auxiliary Kalman sampler can be implemented. This
is the case for the following models:

```math
    g(x_{0:T}) = \prod_{t=0}^T g_t(x_t) = \prod_{t=0}^T p(y_t | x_t)
```

Particle Gibbs samplers can be implemented for models of the form

```math
    \pi(x_{0:T}) \propto p_0(x_0)g_0(x_0) \prod_{t=1}^T p_t(x_t | x_{t-1}) g(x_t, x_{t-1}).
```

When $p_t(x_t | x_{t-1})$ is approximated by a Gaussian, and/or when $g$ is differentiable, improvements can be achieved as explained in the paper.

To install this, be sure to follow the official JAX installation instructions, and then run
```bash
    pip install -e .
```
or any other way of installing a Python package from source of your preference.

Examples can be found in the `examples` folder, together with scripts to reproduce the results described in the paper.
