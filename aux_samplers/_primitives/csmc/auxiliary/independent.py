"""
Implements the auxiliary particle Gibbs algorithm with independent proposals, i.e.,
the version of Finke and Thiery (2021) but in the auxiliary paradigm.
"""
from typing import Optional

import chex
import jax
from chex import Array
from jax import numpy as jnp
from jax.scipy.stats import norm

from aux_samplers._primitives.csmc.base import Distribution, UnivariatePotential, Dynamics, Potential
from .base import get_kernel as get_base_kernel


def get_kernel(M0: Distribution, G0: UnivariatePotential, Mt: Dynamics, Gt: Potential, N: int,
               backward: bool = False, Pt: Optional[Dynamics] = None):
    """
    Get a local auxiliary kernel with separable proposals.

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

    M0_factory = lambda u, scale: AuxiliaryM0(u=u, sqrt_half_delta=scale)
    G0_factory = lambda u, scale: AuxiliaryG0(M0=M0, G0=G0)
    Mt_factory = lambda u, scale: AuxiliaryMt(params=u, sqrt_half_delta=scale)
    Gt_factory = lambda u, scale: AuxiliaryGt(Mt=Mt, Gt=Gt)

    return get_base_kernel(M0_factory, G0_factory, Mt_factory, Gt_factory, N, backward, Pt)


def get_coupled_kernel(M0: Distribution, G0: UnivariatePotential, Mt: Dynamics, Gt: Potential, N: int,
                       backward: bool = False, Pt: Optional[Dynamics] = None):
    """
    Get a local auxiliary kernel with separable proposals.

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

    M0_factory = lambda u, scale: AuxiliaryM0(u=u, sqrt_half_delta=scale)
    G0_factory = lambda u, scale: AuxiliaryG0(M0=M0, G0=G0)
    Mt_factory = lambda u, scale: AuxiliaryMt(params=u, sqrt_half_delta=scale)
    Gt_factory = lambda u, scale: AuxiliaryGt(Mt=Mt, Gt=Gt)

    return get_base_kernel(M0_factory, G0_factory, Mt_factory, Gt_factory, N, backward, Pt)


@chex.dataclass
class AuxiliaryM0(Distribution):
    u: Array
    sqrt_half_delta: float

    def logpdf(self, x):
        logpdf = norm.logpdf(x, self.u, self.sqrt_half_delta)
        return jnp.sum(logpdf, axis=-1)

    def sample(self, key, n):
        return self.u[None, ...] + self.sqrt_half_delta * jax.random.normal(key, (n, *self.u.shape))


@chex.dataclass
class AuxiliaryG0(UnivariatePotential):
    M0: Distribution
    G0: UnivariatePotential

    def __call__(self, x):
        return self.G0(x) + self.M0.logpdf(x)


@chex.dataclass
class AuxiliaryMt(Dynamics):
    sqrt_half_delta: float = None  # needs a default as Dynamics has params with default None

    def sample(self, key, x_t, u):
        return u[None, ...] + self.sqrt_half_delta * jax.random.normal(key, x_t.shape)


@chex.dataclass
class AuxiliaryGt(Potential):
    Mt: Dynamics = None
    Gt: Potential = None

    def __post_init__(self):
        self.params = (self.Mt.params, self.Gt.params)

    def __call__(self, x_t_p_1, x_t, params):
        Mt_params, Gt_params = params
        return self.Mt.logpdf(x_t_p_1, x_t, Mt_params) + self.Gt(x_t_p_1, x_t, Gt_params)
