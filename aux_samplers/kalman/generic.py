"""
Implements the auxiliary Kalman sampling algorithm with custom proposal design.
"""

import jax
from chex import dataclass, Array
from jax import numpy as jnp

from .._primitives.base import SamplerState
from .._primitives.kalman import filtering, LGSSM, sampling, posterior_logpdf, coupled_sampling
from .._primitives.math.mvn import lindvall_roger


@dataclass
class KalmanSampler(SamplerState):
    x: Array
    updated: bool


@dataclass
class CoupledKalmanSampler:
    state_1: KalmanSampler
    state_2: KalmanSampler
    flags: Array

    @property
    def is_coupled(self):
        return jnp.all(self.flags)


def get_kernel(dynamics_factory,
               observations_factory,
               log_likelihood_fn,
               parallel,
               coupled=False,
               **coupled_kwargs):
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
    coupled: bool, optional
        Whether to use the coupled or uncoupled Kalman sampler.
    coupled_kwargs: dict, optional
        Additional keyword arguments for the coupled sampler. These are passed to the `coupled_sampling` function.
        `method` in particular is important, as it determines the coupling method for the marginals,
        see implemented methods in mvn.couplings. Other arguments are passed down to the coupling method.

    Returns
    -------
    kernel: Callable
        Auxiliary Kalman kernel.
    init: Callable
        Function to initialize the state of the sampler given a trajectory.
    """
    if not coupled:
        return _get_kernel(dynamics_factory, observations_factory, log_likelihood_fn, parallel)
    else:
        return _get_coupled_kernel(dynamics_factory, observations_factory, log_likelihood_fn, parallel, **coupled_kwargs)


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


def _get_coupled_kernel(dynamics_factory,
                        observations_factory,
                        log_likelihood_fn,
                        parallel,
                        **coupled_kwargs):
    def kernel(key, state: CoupledKalmanSampler, delta):
        # Housekeeping
        sqrt_delta = jnp.sqrt(delta)
        sqrt_half_delta = jnp.sqrt(0.5 * delta)
        auxiliary_key, sampling_key, accept_key = jax.random.split(key, 3)

        x_1, x_2 = state.state_1.x, state.state_2.x
        T = x_1.shape[0]

        # Auxiliary observations
        mvn_coupling = lambda k, a, b: lindvall_roger(k, a, b, sqrt_half_delta, sqrt_half_delta)
        aux_keys = jax.random.split(auxiliary_key, T)
        u_1, u_2, _ = jax.vmap(mvn_coupling)(aux_keys, x_1, x_2)

        # Propose new states
        log_lgssm_prop_1, log_lgssm_prop_2, log_target_prop_1, log_target_prop_2, x_prop_1, x_prop_2, coupled_ts = do_one(
            delta, sampling_key, u_1, u_2, x_1, x_2)

        # Form the reverse proposal LGSSMs
        log_lgssm_rev_1, log_lgssm_rev_2, log_target_rev_1, log_target_rev_2, *_ = do_one(
            delta, sampling_key, u_1, u_2, x_prop_1, x_prop_2, x_1, x_2)

        # Acceptance ratios for pi(x | u)
        alpha_1 = _get_alpha(log_lgssm_prop_1, log_lgssm_rev_1, log_target_prop_1, log_target_rev_1, sqrt_delta, u_1,
                             x_1, x_prop_1)
        alpha_2 = _get_alpha(log_lgssm_prop_2, log_lgssm_rev_2, log_target_prop_2, log_target_rev_2, sqrt_delta, u_2,
                             x_2, x_prop_2)

        # Accept or reject and update the state
        v = jax.random.uniform(accept_key, ())
        accept_1, accept_2 = v < alpha_1, v < alpha_2

        x_1 = jax.lax.select(accept_1, x_prop_1, x_1)
        x_2 = jax.lax.select(accept_2, x_prop_2, x_2)

        state_1 = KalmanSampler(x=x_1, updated=accept_1)
        state_2 = KalmanSampler(x=x_2, updated=accept_2)

        # Update the coupling flags:
        # if both proposals are rejected, the flags are unchanged
        # if only one proposal is accepted, the flags are set to false
        # if both proposals are accepted, the flags are set to coupled_ts

        index = 1 * (accept_1 != accept_2) + 2 * (accept_1 & accept_2)  # + 0 * (both rejected)
        cases = lambda: state.flags, lambda: jnp.zeros_like(state.flags), lambda: coupled_ts
        flags = jax.lax.switch(index, cases)
        state = CoupledKalmanSampler(state_1=state_1, state_2=state_2, flags=flags)

        return state

    def init(x_1, x_2):
        state_1 = KalmanSampler(x=x_1, updated=True)
        state_2 = KalmanSampler(x=x_2, updated=True)
        flags = jnp.zeros((x_1.shape[0],), dtype=jnp.bool_)
        return CoupledKalmanSampler(state_1=state_1, state_2=state_2, flags=flags)

    def do_one(delta, sampling_key, u_1, u_2, x_1, x_2, x_prop_1=None, x_prop_2=None):
        # Form the proposal LGSSM
        m0_1, P0_1, Fs_1, Qs_1, bs_1, *_ = dynamics_factory(x_1)
        m0_2, P0_2, Fs_2, Qs_2, bs_2, *_ = dynamics_factory(x_2)

        ys_1, Hs_1, Rs_1, cs_1, *_ = observations_factory(x_1, u_1, delta)
        ys_2, Hs_2, Rs_2, cs_2, *_ = observations_factory(x_2, u_2, delta)

        # Sample from the proposal
        lgssm_1 = LGSSM(m0_1, P0_1, Fs_1, Qs_1, bs_1, Hs_1, Rs_1, cs_1)
        lgssm_2 = LGSSM(m0_2, P0_2, Fs_2, Qs_2, bs_2, Hs_2, Rs_2, cs_2)

        ms_1, Ps_1, ell_1 = filtering(ys_1, lgssm_1, parallel)
        ms_2, Ps_2, ell_2 = filtering(ys_2, lgssm_2, parallel)

        if x_prop_1 is None or x_prop_2 is None:
            x_prop_1, x_prop_2, coupled_ts = coupled_sampling(sampling_key,
                                                              lgssm_1, lgssm_2,
                                                              ms_1, Ps_1, ms_2, Ps_2,
                                                              parallel,
                                                              **coupled_kwargs)
        else:
            coupled_ts = None
        # Proposal logpdf
        log_lgssm_prop_1 = posterior_logpdf(ys_1, x_prop_1, ell_1, lgssm_1)
        log_lgssm_prop_2 = posterior_logpdf(ys_2, x_prop_2, ell_2, lgssm_2)

        log_target_prop_1 = log_likelihood_fn(x_prop_1)
        log_target_prop_2 = log_likelihood_fn(x_prop_2)

        return log_lgssm_prop_1, log_lgssm_prop_2, log_target_prop_1, log_target_prop_2, x_prop_1, x_prop_2, coupled_ts

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
