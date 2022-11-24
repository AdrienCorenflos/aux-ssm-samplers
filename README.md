# Auxiliary Kalman and particle Gibbs samplers for generalised Feynman-Kac models
This is the companion code for the paper [stuff](stuff) by Adrien Corenflos and Simo Särkkä.

The models considered here are of the form
```math
    \pi(x_{0:T}) \propto p_0(x_0) \prod_{t=1}^T p_t(x_t | x_{t-1}) g(x_{0:T})
```
for which the auxiliary Kalman sampler can be implemented as long as $g$ is differentiable, or when 
```math
    g(x_{0:T}) = \prod_{t=0}^T g_t(x_t) = \prod_{t=0}^T p(y_t | x_t)
```
for some model $p(y_t | x_t)$ with known mean and covariance.

Particle Gibbs samplers can be implemented for models of the form
```math
    \pi(x_{0:T}) \propto p_0(x_0)g_0(x_0) \prod_{t=1}^T p_t(x_t | x_{t-1}) g(x_t, x_{t-1}).
```
When $p_t(x_t | x_{t-1})$ is approximated by a Gaussian, improvements can be achieved as explained in the paper.

