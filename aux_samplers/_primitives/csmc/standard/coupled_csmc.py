"""
Implements the coupled cSMC kernel of Jacob et al. as well as the coupled backward sampling version of Lee et al.
"""

from typing import Optional

import jax
from jax import numpy as jnp, tree_map

from aux_samplers._primitives.csmc.base import Distribution, UnivariatePotential, Dynamics, Potential, CSMCState, \
    CoupledCSMCState
from aux_samplers._primitives.csmc.resamplings import coupled_multinomial
from aux_samplers._primitives.math.utils import normalize
from aux_samplers._primitives.math.generic_couplings import index_max_coupling


def get_coupled_kernel(M0: Distribution, G0: UnivariatePotential, Mt: Dynamics, Gt: Potential, N: int,
                       backward: bool = False, Pt: Optional[Dynamics] = None):
    """
    Get a coupled cSMC kernel.

    Parameters:
    -----------
    M0:
        Initial distribution.
    G0:
        Initial potential.
    Mt:
        Dynamics of the model.
    Gt:
        Potential of the model.
    N: int
        Total number of particles to use in the cSMC sampler.
    backward: bool
        Whether to perform backward sampling or not. If True, the dynamics must implement a valid logpdf method.
    Pt:
        Dynamics of the true model. If None, it is assumed to be the same as Mt.
    Returns:
    --------
    kernel: Callable
        cSMC kernel.
    init: Callable
        Function to initialize the state of the sampler given a trajectory.
    """

    if backward and Pt is None:
        Pt = Mt
    elif backward and not hasattr(Pt, "logpdf"):
        raise ValueError("When `backward` is True, `Pt` must implement a valid logpdf method.")

    def kernel(key, coupled_state):
        key_fwd, key_bwd = jax.random.split(key)
        state_1, state_2, coupled_flags = coupled_state
        w_1_T, xs_1, log_ws_1, As_1, w_2_T, xs_2, log_ws_2, As_2, coupled_flags = _ccsmc(key_fwd,
                                                                                         state_1,
                                                                                         state_2,
                                                                                         coupled_flags,
                                                                                         M0, G0, Mt, Gt, N)
        if not backward:
            x_1, ancestors_1, x_2, ancestors_1, coupled_flags = _coupled_backward_scanning_pass(key_bwd, w_1_T, w_2_T,
                                                                                                xs_1, xs_2, As_1, As_2,
                                                                                                coupled_flags)
        else:
            x_1, ancestors_1, x_2, ancestors_2, coupled_flags = _coupled_backward_sampling_pass(key_bwd, Pt, w_1_T,
                                                                                                w_2_T, xs_1, xs_2,
                                                                                                log_ws_1, log_ws_2,
                                                                                                coupled_flags)

        state_1 = CSMCState(x=x_1, ancestors=ancestors_1)
        state_2 = CSMCState(x=x_2, ancestors=ancestors_2)
        coupled_state = CoupledCSMCState(state_1=state_1, state_2=state_2, coupled_flags=coupled_flags)
        return coupled_state

    def init(x_star):
        T, *_ = x_star.shape
        ancestors = jnp.zeros((T,), dtype=jnp.int_)
        return CSMCState(x=x_star, ancestors=ancestors)

    return init, kernel


def _ccsmc(key, x_star_1, x_star_2, coupling_flag, M0, G0, Mt, Gt, N):
    """JAX scan implementation of the coupled version of the cSMC algorithm
     using common random numbers for sampling and index coupled resampling."""
    T, *_ = x_star_1.shape
    keys = jax.random.split(key, T)

    # Sample initial statea
    x0_1 = x0_2 = M0.sample(keys[0], N)
    are_coupled_0 = jnp.ones((N,), dtype=jnp.bool_)
    # Replace the first particle with the star trajectory
    x0_1 = x0_1.at[0].set(x_star_1[0])
    x0_2 = x0_2.at[0].set(x_star_2[0])
    are_coupled_0 = are_coupled_0.at[0].set(coupling_flag[0])

    # Compute initial weights and normalize
    log_w0_1, log_w0_2 = G0(x0_1), G0(x0_2)
    w0_1, w0_2 = normalize(log_w0_1), normalize(log_w0_2)

    def body(carry, inp):
        w_1_t_m_1, x_1_t_m_1, w_2_t_m_1, x_2_t_m_1, coupled_t_m_1 = carry
        Mt_params, Gt_params, x_1_star_t, x_2_star_t, key_t, flag_t = inp
        resampling_key, sampling_key = jax.random.split(key_t)
        # Conditional resampling
        A_1_t, A_2_t, coupled_index = coupled_multinomial(resampling_key, w_1_t_m_1, w_2_t_m_1)

        x_1_t_m_1 = jnp.take(x_1_t_m_1, A_1_t, axis=0)
        x_2_t_m_1 = jnp.take(x_2_t_m_1, A_2_t, axis=0)

        coupled_t = coupled_t_m_1 & coupled_index
        coupled_t = coupled_t.at[0].set(flag_t)

        # Sample new particles
        x_1_t = Mt.sample(sampling_key, x_1_t_m_1, Mt_params)
        x_2_t = Mt.sample(sampling_key, x_2_t_m_1, Mt_params)

        x_1_t = x_1_t.at[0].set(x_1_star_t)
        x_2_t = x_2_t.at[0].set(x_2_star_t)

        # Compute weights
        log_w_1_t = Gt(x_1_t, x_1_t_m_1, Gt_params)
        log_w_2_t = Gt(x_2_t, x_2_t_m_1, Gt_params)

        # Normalize weights
        w_1_t, w_2_t = normalize(log_w_1_t), normalize(log_w_2_t)

        # Return next step
        next_carry = (w_1_t, x_1_t, w_2_t, x_2_t, coupled_t)
        save = (x_1_t, log_w_1_t, A_1_t, x_2_t, log_w_2_t, A_2_t, coupled_t)

        return next_carry, save

    init = (w0_1, x0_1, w0_2, x0_2, are_coupled_0)
    inputs = (Mt.params, Gt.params, x_star_1[1:], x_star_2[1:], keys[1:], coupling_flag[1:])
    (w_1_T, _, w_2_T, *_), (xs_1, log_ws_1, As_1, xs_2, log_ws_2, As_2, coupled_flags) = jax.lax.scan(body,
                                                                                                      init,
                                                                                                      inputs)

    log_ws_1 = jnp.insert(log_ws_1, 0, log_w0_1, axis=0)
    log_ws_2 = jnp.insert(log_ws_2, 0, log_w0_2, axis=0)
    xs_1 = jnp.insert(xs_1, 0, x0_1, axis=0)
    xs_2 = jnp.insert(xs_2, 0, x0_2, axis=0)
    return w_1_T, xs_1, log_ws_1, As_1, w_2_T, xs_2, log_ws_2, As_2, coupled_flags


def _coupled_backward_scanning_pass(key, w_1_T, w_2_T, xs_1, xs_2, As_1, As_2, coupled_flags):
    # Sample initial state

    B_1_T, B_2_T, coupled_index_T = index_max_coupling(key, w_1_T, w_2_T, 1)
    x_1_T, x_2_T = xs_1[-1, B_1_T], xs_2[-1, B_2_T]
    coupled_T = coupled_flags[-1, B_1_T] & coupled_index_T

    def body(carry, inp):
        B_1_t, B_2_t = carry
        xs_1_t_m_1, A_1_t, xs_2_t_m_1, A_2_t, flags_t_m_1 = inp
        B_1_t_m_1, B_2_t_m_1 = A_1_t[B_1_t], A_2_t[B_2_t]
        coupled_t_m_1 = flags_t_m_1[B_1_t_m_1] & (B_1_t_m_1 == B_2_t_m_1)
        x_1_t_m_1 = xs_1_t_m_1[B_1_t_m_1]
        x_2_t_m_1 = xs_2_t_m_1[B_2_t_m_1]
        next_carry = (B_1_t_m_1, B_2_t_m_1)
        save = (x_1_t_m_1, B_1_t_m_1, x_2_t_m_1, B_2_t_m_1, coupled_t_m_1)
        return next_carry, save

    # xs[-2::-1] is the reversed list of xs[:-1], I know, not readable...
    _, scanned_out = jax.lax.scan(body, (B_1_T, B_2_T),
                                  (xs_1[-2::-1], As_1[::-1], xs_2[-2::-1], As_2[::-1], coupled_flags[-2::-1]))
    xs_1, Bs_1, xs_2, Bs_2, coupled_flags = scanned_out

    xs_1 = jnp.insert(xs_1, 0, x_1_T, axis=0)
    xs_2 = jnp.insert(xs_2, 0, x_2_T, axis=0)

    Bs_1 = jnp.insert(Bs_1, 0, B_1_T, axis=0)
    Bs_2 = jnp.insert(Bs_2, 0, B_2_T, axis=0)

    coupled_flags = jnp.insert(coupled_flags, 0, coupled_T, axis=0)
    return xs_1[::-1], Bs_1[::-1], xs_2[::-1], Bs_2[::-1], coupled_flags[::-1]


def _coupled_backward_sampling_pass(key, Pt: Dynamics, w_1_T, w_2_T, xs_1, xs_2, log_ws_1, log_ws_2, coupled_flags):
    """JAX implementation of the coupled backward sampling pass."""
    T = xs_1.shape[0]
    keys = jax.random.split(key, T)

    B_1_T, B_2_T, coupled_index_T = index_max_coupling(keys[0], w_1_T, w_2_T, 1)
    x_1_T, x_2_T = xs_1[-1, B_1_T], xs_2[-1, B_2_T]
    coupled_T = coupled_flags[-1, B_1_T] & coupled_index_T

    def body(carry, inp):
        x_1_t, x_2_t = carry
        op_key, xs_1_t_m_1, log_w_1_t_m_1, xs_2_t_m_1, log_w_2_t_m_1, Pt_m_1_params = inp
        log_w_1 = Pt.logpdf(x_1_t, xs_1_t_m_1, Pt_m_1_params) + log_w_1_t_m_1
        log_w_2 = Pt.logpdf(x_2_t, xs_2_t_m_1, Pt_m_1_params) + log_w_2_t_m_1
        w_1, w_2 = normalize(log_w_1), normalize(log_w_2)
        B_1_t_m_1, B_2_t_m_1, coupled_index_t_m_1 = index_max_coupling(op_key, w_1, w_2, 1)
        x_1_t_m_1, x_2_t_m_1 = xs_1_t_m_1[B_1_t_m_1], xs_2_t_m_1[B_2_t_m_1]
        coupled_t_m_1 = coupled_flags[-1, B_1_t_m_1] & coupled_index_t_m_1

        next_carry = (x_1_t_m_1, x_2_t_m_1)
        save = (x_1_t_m_1, B_1_t_m_1, x_2_t_m_1, B_2_t_m_1, coupled_t_m_1)
        return next_carry, save

    Mt_params = tree_map(lambda x: x[::-1], Pt.params)

    # xs[-2::-1] is the reversed list of xs[:-1], I know, not readable...
    _, scanned_out = jax.lax.scan(body, (x_1_T, x_2_T),
                                  (keys[1:], xs_1[-2::-1], log_ws_1[-2::-1], xs_2[-2::-1], log_ws_2[-2::-1], Mt_params))
    xs_1, Bs_1, xs_2, Bs_2, coupled_flags = scanned_out

    xs_1 = jnp.insert(xs_1, 0, x_1_T, axis=0)
    xs_2 = jnp.insert(xs_2, 0, x_2_T, axis=0)

    Bs_1 = jnp.insert(Bs_1, 0, B_1_T, axis=0)
    Bs_2 = jnp.insert(Bs_2, 0, B_2_T, axis=0)

    coupled_flags = jnp.insert(coupled_flags, 0, coupled_T, axis=0)
    return xs_1[::-1], Bs_1[::-1], xs_2[::-1], Bs_2[::-1], coupled_flags[::-1]
