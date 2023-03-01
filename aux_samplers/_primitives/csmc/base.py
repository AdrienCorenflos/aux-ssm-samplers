import abc
from typing import Optional

import chex
from chex import ArrayTree, dataclass, Array

from aux_samplers._primitives.base import SamplerState

_MSG = """
The logpdf is not implemented for this {type(self).__name__} but was called.
If you see this message, you likely are using a cSMC method relying on it. 
Please implement this function or choose the standard cSMC with no backward pass.
"""

_EPS = 1e-10


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
