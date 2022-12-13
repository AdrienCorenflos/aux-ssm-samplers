"""
Implements the auxiliary particle Gibbs algorithm with generic proposals.
"""
from typing import Optional, Callable

import jax
from chex import Array
from jax import numpy as jnp

from .._primitives.csmc.base import Distribution, UnivariatePotential, Dynamics, Potential, CSMCState
from .._primitives.csmc.standard import get_kernel as get_standard_kernel


def get_kernel(M0_factory: Callable[[Array, float], Distribution],
               G0_factory: Callable[[Array, float], UnivariatePotential],
               Mt_factory: Callable[[Array, float], Dynamics],
               Gt_factory: Callable[[Array, float], Potential],
               N: int,
               backward: bool = False,
               Pt: Optional[Dynamics] = None):
    """
    Get a local auxiliary kernel.
    All factories take as input the current value of the auxiliary variable `u_t` and the value of \sqrt{\delta/2}
    and return the resulting component of the model. Mt and Gt are supposed to take the batched version of u_{1:T}
    as input.

    Parameters
    ----------
    M0_factory:
        Factory for the initial distribution.
    G0_factory:
        Factory for the initial potential.
    Mt_factory:
        Factory for the dynamics of the model.
    Gt_factory:
        Factory for the potential of the model.
    N:
        Total number of particles to use in the cSMC sampler.
    backward: bool
        Whether to perform backward sampling or not. If True, the dynamics must implement a valid logpdf method.
    Pt:
        Dynamics of the true model.

    Returns
    -------
    kernel: Callable
        cSMC kernel.
    init: Callable
        Function to initialize the state of the sampler given a trajectory.
    """

    if backward and Pt is None:
        raise ValueError("If backward is True, the true dynamics `Pt` must be provided.")
    elif backward and not hasattr(Pt, "logpdf"):
        raise ValueError("`Pt` must implement a valid logpdf method.")

    def kernel(key, state, delta):
        # Housekeeping
        x, ancestors = state.x, state.ancestors
        sqrt_half_delta = jnp.sqrt(0.5 * delta)
        auxiliary_key, key = jax.random.split(key)

        # Auxiliary observations
        u = x + sqrt_half_delta * jax.random.normal(auxiliary_key, x.shape)

        m0 = M0_factory(u[0], sqrt_half_delta)
        g0 = G0_factory(u[0], sqrt_half_delta)

        mt = Mt_factory(u[1:], sqrt_half_delta)
        gt = Gt_factory(u[1:], sqrt_half_delta)

        _, auxiliary_kernel = get_standard_kernel(m0, g0, mt, gt, N, backward=backward, Pt=Pt)
        return auxiliary_kernel(key, state)

    def init(x_star):
        T, *_ = x_star.shape
        ancestors = jnp.zeros((T,), dtype=jnp.int_)
        return CSMCState(x=x_star, ancestors=ancestors)

    return init, kernel