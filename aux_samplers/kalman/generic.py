"""
Implements the auxiliary Kalman sampling algorithm with custom proposal design.
"""
import jax
from chex import dataclass, Numeric, Array
from jax import numpy as jnp

from .._primitives.kalman import filtering, LGSSM, sampling, posterior_logpdf


@dataclass
class KalmanSampler:
    x: Array
    updated: Numeric


def get_kernel(dynamics_factory,
               observations_factory,
               log_likelihood_fn,
               parallel):
    """
    Returns a kernel and an init for the auxiliary Kalman sampler.

    Parameters:
    -----------
    dynamics_factory: Callable
        Function that takes as input the current trajectory of the sampler, and returns a tuple
        m0, P0, Fs, Qs, bs of the LGSSM.
    observations_factory: Callable
        Function that takes as input the current trajectory of the sampler, the auxiliary observations us, and delta, and returns a tuple
        ys, Hs, Rs, cs of the LGSSM.
    log_likelihood_fn: Callable
        Function that takes as input the current trajectory of the sampler, and returns the (unnormalised) log likelihood of the trajectory.
    parallel: bool
        Whether to use the parallel or sequential Kalman sampler.

    Returns
    -------
    kernel: Callable
        Auxiliary Kalman kernel.
    init: Callable
        Function to initialize the state of the sampler given a trajectory.
    """

    def kernel(key, state, delta):
        # Housekeeping
        x = state.x
        sqrt_delta = jnp.sqrt(delta)
        sqrt_half_delta = jnp.sqrt(0.5 * delta)
        auxiliary_key, sampling_key, accept_key = jax.random.split(key, 3)

        # Auxiliary observations
        u = x + sqrt_half_delta * jax.random.normal(auxiliary_key, x.shape)

        # Form the proposal LGSSM
        m0, P0, Fs, Qs, bs = dynamics_factory(x)
        ys, Hs, Rs, cs = observations_factory(x, u, delta)

        # Sample from the proposal
        lgssm = LGSSM(m0, P0, Fs, Qs, bs, Hs, Rs, cs)
        ms, Ps, ell = filtering(ys, lgssm, parallel)
        x_prop = sampling(key, ms, Ps, lgssm, parallel)

        # Proposal logpdf
        log_lgssm_prop = posterior_logpdf(ys, x_prop, ell, lgssm)
        log_target_prop = log_likelihood_fn(x_prop)

        # Form the reverse proposal LGSSM
        m0, P0, Fs, Qs, bs = dynamics_factory(x_prop)
        ys, Hs, Rs, cs = observations_factory(x_prop, u, delta)

        # Reverse proposal logpdf
        lgssm = LGSSM(m0, P0, Fs, Qs, bs, Hs, Rs, cs)
        _, _, ell = filtering(ys, lgssm, parallel)
        log_lgssm_rev = posterior_logpdf(ys, x, ell, lgssm)
        log_target_rev = log_likelihood_fn(x)

        # Acceptance ratio for pi(x, u)
        log_alpha = log_target_prop - log_target_rev
        log_alpha += log_lgssm_rev - log_lgssm_prop

        # Correct the lgssm proposal distribution to get pi(x | u)
        diff_prop, diff = (x_prop - u) / sqrt_delta, (x - u) / sqrt_delta

        correction = jnp.sum(diff_prop ** 2 - diff ** 2)
        log_alpha -= correction
        alpha = jnp.exp(jnp.minimum(0, log_alpha))

        # Accept or reject and update the state
        accept = jax.random.bernoulli(accept_key, alpha)
        x = jax.lax.select(accept, x_prop, x)

        return KalmanSampler(x=x, updated=accept)

    def init(x_star):
        return KalmanSampler(x=x_star, updated=False)

    return init, kernel


def delta_adaptation(delta, target_rate, acceptance_rate, adaptation_rate, min_delta=1e-20):
    """
    A simple adaptation rule for the delta parameter of the auxiliary Kalman sampler.

    Parameters
    ----------
    delta: float
        Current value of delta.
    target_rate: float
        Target acceptance rate.
    acceptance_rate: float
        Current average acceptance rate.
    adaptation_rate: float
        Adaptation rate.
    min_delta: float
        Minimum value of delta.

    Returns
    -------
    delta: float
        Adapted value of delta.

    """

    out = delta * jnp.exp(adaptation_rate * (acceptance_rate - target_rate))
    return jnp.maximum(out, min_delta)