import abc
from abc import ABC
from typing import Optional

import chex
import jax.numpy as jnp
from chex import ArrayTree, dataclass, Array

_MSG = """
The logpdf is not implemented for this {type(self).__name__} but was called.
If you see this message, you likely are using a cSMC method relying on it. 
Please implement this function or choose the standard cSMC with no backward pass.
"""


@chex.dataclass
class UnivariatePotential(abc.ABC):
    """
    Abstract class for univariate potential functions.
    """

    def logpdf(self, x):
        raise NotImplementedError


@chex.dataclass
class Distribution(UnivariatePotential, abc.ABC):
    """
    Abstract class for sampling functions.
    """

    def sample(self, key, N):
        raise NotImplementedError

    def logpdf(self, x):
        return NotImplemented(_MSG.format(type(self).__name__))


@chex.dataclass
class Potential(abc.ABC):
    """
    Abstract class for potential functions.

    """
    params: Optional[ArrayTree] = None

    def logpdf(self, x_t_p_1, x_t, params):
        raise NotImplementedError


@chex.dataclass
class Dynamics(Potential, ABC):
    def sample(self, key, x_t, params):
        raise NotImplementedError

    def logpdf(self, x_t_p_1, x_t, params):
        return NotImplemented(_MSG.format(type(self).__name__))


def normalize(log_weights):
    """
    Normalize log weights to obtain unnormalized weights.

    Parameters
    ----------
    log_weights : Array
        Log weights.

    Returns
    -------
    weights : Array
        Unnormalized weights.
    """
    weights = jnp.exp(log_weights - jnp.max(log_weights))
    weights /= jnp.sum(weights)

    return weights


@dataclass
class CSMCState:
    x: ArrayTree
    ancestors: Array
