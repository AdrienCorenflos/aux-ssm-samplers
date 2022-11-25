import abc
from abc import ABC
from typing import Optional

import chex
import jax.numpy as jnp
from chex import ArrayTree


@chex.dataclass
class Distribution(abc.ABC):
    """
    Abstract class for sampling functions.
    """
    params: Optional[ArrayTree] = None

    def sample(self, key, N):
        raise NotImplementedError


@chex.dataclass
class UnivariatePotential(abc.ABC):
    """
    Abstract class for univariate potential functions.
    """
    params: Optional[ArrayTree] = None

    def logpdf(self, x):
        raise NotImplementedError


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
