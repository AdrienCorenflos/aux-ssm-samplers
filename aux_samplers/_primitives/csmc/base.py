import abc
from typing import Optional

import chex
import jax.numpy as jnp
from chex import ArrayTree, dataclass, Array

from aux_samplers._primitives.base import SamplerState

_MSG = """
The logpdf is not implemented for this {type(self).__name__} but was called.
If you see this message, you likely are using a cSMC method relying on it. 
Please implement this function or choose the standard cSMC with no backward pass.
"""


@dataclass
class CSMCState(SamplerState):
    x: ArrayTree
    updated: Array


@chex.dataclass
class UnivariatePotential(abc.ABC):
    """
    Abstract class for univariate potential functions.
    This is just a callable, but may have parameters.
    """

    def __call__(self, x):
        raise NotImplementedError


@chex.dataclass
class Distribution(abc.ABC):
    """
    Abstract class for densities.
    """

    def sample(self, key, N):
        raise NotImplementedError

    def logpdf(self, x):
        return NotImplemented(_MSG.format(type(self).__name__))


@chex.dataclass
class Potential(abc.ABC):
    """
    Abstract class for potential functions.
    This is just a callable, but may have parameters.
    """
    params: Optional[ArrayTree] = None

    def __call__(self, x_t_p_1, x_t, params):
        raise NotImplementedError


@chex.dataclass
class Dynamics(abc.ABC):
    """
    Abstract class for conditional densities.
    """
    params: Optional[ArrayTree] = None

    def sample(self, key, x_t, params):
        raise NotImplementedError

    def logpdf(self, x_t_p_1, x_t, params):
        return NotImplemented(_MSG.format(type(self).__name__))


@chex.dataclass
class CoupledDistribution(abc.ABC):
    """
    Abstract class for coupled distributions.
    """

    def sample(self, key, N):
        raise NotImplementedError

    def logpdf_1(self, x):
        return NotImplemented(_MSG.format(type(self).__name__))

    def logpdf_2(self, x):
        return NotImplemented(_MSG.format(type(self).__name__))


@chex.dataclass
class CoupledDynamics(abc.ABC):
    """
    Abstract class for conditional densities.
    """
    params: Optional[ArrayTree] = None

    def sample(self, key, x_t_1, x_t_2, params_1, params_2):
        raise NotImplementedError

    def logpdf_1(self, x_t_p_1, x_t, params):
        return NotImplemented(_MSG.format(type(self).__name__))

    def logpdf_2(self, x_t_p_1, x_t, params):
        return NotImplemented(_MSG.format(type(self).__name__))


@chex.dataclass
class CRNDistribution(CoupledDistribution):
    dist_1: Distribution
    dist_2: Distribution

    _EPS: float = 1e-9

    def sample(self, key, N):
        x1 = self.dist_1.sample(key, N)
        x2 = self.dist_2.sample(key, N)

        coupled = jnp.linalg.norm(x1 - x2, axis=-1) < self._EPS

        return x1, x2, coupled

    def logpdf_1(self, x):
        return self.dist_1.logpdf(x)

    def logpdf_2(self, x):
        return self.dist_2.logpdf(x)


@chex.dataclass
class CRNDynamics(CoupledDynamics):
    dynamics_1: Dynamics = None
    dynamics_2: Dynamics = None

    # thresholding for checking if the two samples are coupled
    _EPS = 1e-9

    def __post_init__(self):
        self.params = (self.dynamics_1.params, self.dynamics_2.params)

    def sample(self, key, x1_t, x2_t, params_1, params_2):
        x1_t_p_1 = self.dynamics_1.sample(key, x1_t, params_1)
        x2_t_p_1 = self.dynamics_2.sample(key, x2_t, params_2)
        coupled = jnp.linalg.norm(x1_t_p_1 - x2_t_p_1, axis=-1) < self._EPS
        return x1_t_p_1, x2_t_p_1, coupled

    def logpdf_1(self, x_t_p_1, x_t, params):
        return self.dynamics_1.logpdf(x_t_p_1, x_t, params)

    def logpdf_2(self, x_t_p_1, x_t, params):
        return self.dynamics_2.logpdf(x_t_p_1, x_t, params)
