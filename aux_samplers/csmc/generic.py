"""
Implements the auxiliary particle Gibbs algorithm with generic proposals.
"""
from typing import Optional, Callable, Tuple

import jax
from chex import Array, Numeric
from jax import numpy as jnp

from .._primitives.base import CoupledSamplerState
from .._primitives.csmc import get_kernel as get_standard_kernel, get_coupled_kernel as get_standard_coupled_kernel
from .._primitives.csmc.base import Distribution, UnivariatePotential, Dynamics, Potential, CSMCState
from .._primitives.math.mvn.couplings import lindvall_roger


def get_kernel(factory: Callable,
               N: int,
               backward: bool = False,
               Pt: Optional[Dynamics] = None,
               coupled: bool = False):
    """
    Get a local auxiliary kernel.
    All factories take as input the current value of the auxiliary variable `u_t` and the value of \sqrt{\delta/2}
    and return the resulting component of the model. Mt and Gt are supposed to take the batched version of u_{1:T}
    as input.

    Parameters
    ----------
    factory:
        Factory that returns the initial distribution, the initial potential, the dynamics,
        and the potential of the models. Alternatively, a coupled factory can be provided, which returns
        the coupled initial distribution, the initial potential of the first model, the initial potential of the second model,
        the coupled dynamics, the potential of the first model, and the potential of the second model.
    N:
        Total number of particles to use in the cSMC sampler.
    backward: bool
        Whether to perform backward sampling or not. If True, the dynamics must implement a valid logpdf method.
    Pt:
        Dynamics of the true model.
    coupled: bool
        Whether this returns the coupled version of the kernel or not.
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

    if not coupled:
        return _get_kernel(factory, N, backward, Pt)
    else:
        return _get_coupled_kernel(factory, N, backward, Pt)


def _get_kernel(factory: Callable[[Array, Numeric], Tuple[Distribution, UnivariatePotential, Dynamics, Potential]],
                N: int,
                backward: bool,
                Pt: Dynamics):
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

    def init(x):
        T, *_ = x.shape
        ancestors = jnp.zeros((T,), dtype=jnp.int_)
        return CSMCState(x=x, updated=ancestors != 0)

    return init, kernel


def _get_coupled_kernel(
        coupled_factory,
        N: int,
        backward: bool,
        Pt: Dynamics):
    def kernel(key, state: CoupledSamplerState, delta):
        # Housekeeping
        state_1, state_2 = state.state_1, state.state_2
        x_1, x_2 = state_1.x, state_2.x
        T = x_1.shape[0]

        sqrt_half_delta = jnp.sqrt(0.5 * delta)
        if jnp.ndim(sqrt_half_delta) == 0:
            sqrt_half_delta = sqrt_half_delta * jnp.ones((T,))
        auxiliary_key, key = jax.random.split(key)

        # Auxiliary observations
        mvn_coupling = lambda k, a, b: lindvall_roger(k, a, b, sqrt_half_delta, sqrt_half_delta)
        aux_keys = jax.random.split(auxiliary_key, T)
        u_1, u_2, _ = jax.vmap(mvn_coupling)(aux_keys, x_1, x_2)

        cm0, g0_1, g0_2, cmt, gt_1, gt_2 = coupled_factory(u_1, u_2, sqrt_half_delta)

        _, auxiliary_kernel = get_standard_coupled_kernel(cm0, g0_1, g0_2, cmt, gt_1, gt_2, N, backward=backward,
                                                          Pt_1=Pt, Pt_2=Pt)
        return auxiliary_kernel(key, state)

    def init(x_1, x_2):
        T, *_ = x_1.shape
        ancestors = jnp.zeros((T,), dtype=jnp.int_)
        state_1 = CSMCState(x=x_1, updated=ancestors != 0)
        state_2 = CSMCState(x=x_2, updated=ancestors != 0)
        coupled_state = CoupledSamplerState(state_1=state_1, state_2=state_2,
                                            flags=jnp.zeros((T,), dtype=jnp.bool_))
        return coupled_state

    return init, kernel
