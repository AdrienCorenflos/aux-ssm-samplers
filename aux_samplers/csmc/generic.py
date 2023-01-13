"""
Implements the auxiliary particle Gibbs algorithm with generic proposals.
"""
from typing import Optional, Callable, Tuple

import jax
from chex import Array, Numeric
from jax import numpy as jnp

from .._primitives.csmc import get_kernel as get_standard_kernel
from .._primitives.csmc.base import Distribution, UnivariatePotential, Dynamics, Potential, CSMCState


def get_kernel(factory: Callable[[Array, Numeric], Tuple[Distribution, UnivariatePotential,Dynamics, Potential]],
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
    factory:
        Factory that returns the initial distribution, the initial potential, the dynamics,
        and the potential of the models.
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
        x = state.x
        T = x.shape[0]

        sqrt_half_delta = jnp.sqrt(0.5 * delta)
        if jnp.ndim(sqrt_half_delta) == 0:
            sqrt_half_delta = sqrt_half_delta * jnp.ones((T,))
        auxiliary_key, key = jax.random.split(key)

        # Auxiliary observations
        u = x + sqrt_half_delta[:, None] * jax.random.normal(auxiliary_key, x.shape)

        m0, g0, mt, gt = factory(u, sqrt_half_delta)

        _, auxiliary_kernel = get_standard_kernel(m0, g0, mt, gt, N, backward=backward, Pt=Pt)
        return auxiliary_kernel(key, state)

    def init(x_star):
        T, *_ = x_star.shape
        ancestors = jnp.zeros((T,), dtype=jnp.int_)
        return CSMCState(x=x_star, updated=ancestors != 0)

    return init, kernel



def delta_adaptation(delta, target_rate, acceptance_rate, adaptation_rate, min_delta=1e-20):
    """
    A simple adaptation rule for the delta parameter of the auxiliary Kalman sampler.

    Parameters
    ----------
    delta: Array
        Current value of delta per time step.
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
    delta: Array
        Adapted value of delta.

    """

    out = delta * jnp.exp(adaptation_rate * (acceptance_rate - target_rate))
    return jnp.maximum(out, min_delta)