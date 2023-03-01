"""
Implements the auxiliary Kalman sampling algorithm with custom proposal design.
"""

import jax
from chex import dataclass, Array
from jax import numpy as jnp

from .._primitives.base import SamplerState
from .._primitives.kalman import filtering, LGSSM, sampling, posterior_logpdf


@dataclass
class KalmanSampler(SamplerState):
    x: Array
    updated: bool


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
    return _get_kernel(dynamics_factory, observations_factory, log_likelihood_fn, parallel)


def _get_kernel(dynamics_factory,
                observations_factory,
                log_likelihood_fn,
                parallel):
    def kernel(key, state, delta):
        # Housekeeping
        x = state.x
        sqrt_delta = jnp.sqrt(delta)
        sqrt_half_delta = jnp.sqrt(0.5 * delta)
        auxiliary_key, sampling_key, accept_key = jax.random.split(key, 3)

        # Auxiliary observations
        u = x + sqrt_half_delta * jax.random.normal(auxiliary_key, x.shape)

        # Propose new state
        log_lgssm_prop, log_target_prop, x_prop = do_one(delta, sampling_key, u, x)

        # Form the reverse proposal LGSSM
        log_lgssm_rev, log_target_rev, _ = do_one(delta, sampling_key, u, x_prop, x)

        # Acceptance ratio for pi(x | u)
        alpha = _get_alpha(log_lgssm_prop, log_lgssm_rev, log_target_prop, log_target_rev, sqrt_delta, u, x, x_prop)

        # Accept or reject and update the state
        accept = jax.random.bernoulli(accept_key, alpha)
        x = jax.lax.select(accept, x_prop, x)

        return KalmanSampler(x=x, updated=accept)

    def do_one(delta, sampling_key, u, x, x_prop=None):
        # Form the proposal LGSSM
        m0, P0, Fs, Qs, bs, *_ = dynamics_factory(x)
        ys, Hs, Rs, cs, *_ = observations_factory(x, u, delta)
        # Sample from the proposal
        lgssm = LGSSM(m0, P0, Fs, Qs, bs, Hs, Rs, cs)
        ms, Ps, ell = filtering(ys, lgssm, parallel)
        if x_prop is None:
            x_prop = sampling(sampling_key, ms, Ps, lgssm, parallel)
        # Proposal logpdf
        log_lgssm_prop = posterior_logpdf(ys, x_prop, ell, lgssm)
        log_target_prop = log_likelihood_fn(x_prop)
        return log_lgssm_prop, log_target_prop, x_prop

    def init(x):
        return KalmanSampler(x=x, updated=True)

    return init, kernel


def _get_alpha(log_lgssm_prop, log_lgssm_rev, log_target_prop, log_target_rev, sqrt_delta, u, x, x_prop):
    log_alpha = log_target_prop - log_target_rev
    log_alpha += log_lgssm_rev - log_lgssm_prop
    # Correct the lgssm proposal distribution to get pi(x | u)
    diff_prop, diff = (x_prop - u) / sqrt_delta, (x - u) / sqrt_delta
    correction = jnp.sum(diff_prop ** 2 - diff ** 2)
    log_alpha -= correction
    alpha = jnp.exp(jnp.minimum(0, log_alpha))
    return alpha
