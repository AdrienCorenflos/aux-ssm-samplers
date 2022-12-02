"""
Implements the classical cSMC kernel from the seminal pMCMC paper by Andrieu et al. (2010). We also implement the
backward sampling step of Whiteley.
"""


from typing import Optional

import jax
from jax import numpy as jnp, tree_map

from aux_samplers._primitives.csmc.base import Distribution, UnivariatePotential, Dynamics, Potential, CSMCState
from aux_samplers._primitives.csmc.resamplings import multinomial
from aux_samplers._primitives.math.utils import normalize


def get_kernel(M0: Distribution, G0: UnivariatePotential, Mt: Dynamics, Gt: Potential, N: int,
               backward: bool = False, Pt: Optional[Dynamics] = None):
    """
    Get a cSMC kernel.

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

    def kernel(key, state):
        key_fwd, key_bwd = jax.random.split(key)
        w_T, xs, log_ws, As = _csmc(key_fwd, state.x, M0, G0, Mt, Gt, N)
        if not backward:
            x, ancestors = _backward_scanning_pass(key_bwd, w_T, xs, As)
        else:
            x, ancestors = _backward_sampling_pass(key_bwd, Pt, w_T, xs, log_ws)
        return CSMCState(x=x, ancestors=ancestors)

    def init(x_star):
        T, *_ = x_star.shape
        ancestors = jnp.zeros((T,), dtype=jnp.int_)
        return CSMCState(x=x_star, ancestors=ancestors)

    return init, kernel


def _csmc(key, x_star, M0, G0, Mt, Gt, N):
    T = x_star.shape[0]
    keys = jax.random.split(key, T)

    # Sample initial state
    x0 = M0.sample(keys[0], N)
    # Replace the first particle with the star trajectory
    x0 = x0.at[0].set(x_star[0])

    # Compute initial weights and normalize
    log_w0 = G0(x0)
    w0 = normalize(log_w0)

    def body(carry, inp):
        w_t_m_1, x_t_m_1 = carry
        Mt_params, Gt_params, x_star_t, key_t = inp
        resampling_key, sampling_key = jax.random.split(key_t)
        # Conditional resampling
        A_t = multinomial(resampling_key, w_t_m_1)
        x_t_m_1 = jnp.take(x_t_m_1, A_t, axis=0)

        # Sample new particles
        x_t = Mt.sample(sampling_key, x_t_m_1, Mt_params)
        x_t = x_t.at[0].set(x_star_t)

        # Compute weights
        log_w_t = Gt(x_t, x_t_m_1, Gt_params)
        w_t = normalize(log_w_t)
        # Return next step
        next_carry = (w_t, x_t)
        save = (x_t, log_w_t, A_t)

        return next_carry, save

    (w_T, _), (xs, log_ws, As) = jax.lax.scan(body, (w0, x0), (Mt.params, Gt.params, x_star[1:], keys[1:]))

    log_ws = jnp.insert(log_ws, 0, log_w0, axis=0)
    xs = jnp.insert(xs, 0, x0, axis=0)
    return w_T, xs, log_ws, As


def _backward_scanning_pass(key, w_T, xs, As):
    B_T = jax.random.choice(key, w_T.shape[0], p=w_T, shape=())
    x_T = xs[-1, B_T]

    def body(B_t, inp):
        xs_t_m_1, A_t = inp
        B_t_m_1 = A_t[B_t]
        x_t_m_1 = xs_t_m_1[B_t_m_1]
        return B_t_m_1, (x_t_m_1, B_t_m_1)

    # xs[-2::-1] is the reversed list of xs[:-1], I know, not readable...
    _, (xs, Bs) = jax.lax.scan(body, B_T, (xs[-2::-1], As[::-1]))
    xs = jnp.insert(xs, 0, x_T, axis=0)
    Bs = jnp.insert(Bs, 0, B_T, axis=0)
    return xs[::-1], Bs[::-1]


def _backward_sampling_pass(key, Mt: Dynamics, w_T, xs, log_ws):
    T = xs.shape[0]
    keys = jax.random.split(key, T)

    B_T = jax.random.choice(keys[0], w_T.shape[0], p=w_T, shape=())
    x_T = xs[-1, B_T]

    def body(x_t, inp):
        op_key, xs_t_m_1, log_w_t_m_1, Mt_m_1_params = inp
        log_w = Mt.logpdf(x_t, xs_t_m_1, Mt_m_1_params) + log_w_t_m_1
        w = normalize(log_w)
        B_t_m_1 = jax.random.choice(op_key, w.shape[0], p=w, shape=())
        x_t_m_1 = xs_t_m_1[B_t_m_1]
        return x_t_m_1, (x_t_m_1, B_t_m_1)

    Mt_params = tree_map(lambda x: x[::-1], Mt.params)

    # xs[-2::-1] is the reversed list of xs[:-1], I know, not readable... Same for log_ws[:-1].
    inps = keys[1:], xs[-2::-1], log_ws[-2::-1], Mt_params
    _, (xs, Bs) = jax.lax.scan(body, x_T, inps)
    xs = jnp.insert(xs, 0, x_T, axis=0)
    Bs = jnp.insert(Bs, 0, B_T, axis=0)
    return xs[::-1], Bs[::-1]
