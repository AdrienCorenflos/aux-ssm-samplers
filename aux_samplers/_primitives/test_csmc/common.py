from dataclasses import field

import chex
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from aux_samplers._primitives.csmc.base import Dynamics, Distribution, UnivariatePotential, Potential


@chex.dataclass
class GaussianDynamics(Dynamics):
    """
    AR dynamics with Gaussian noise.
    """
    rho: float = 0.9
    sig: float = field(init=False)

    def __post_init__(self):
        self.sig = (1 - self.rho ** 2) ** 0.5

    def logpdf(self, x_t_p_1, x_t, _params):
        x_pred = self.rho * x_t
        return norm.logpdf(x_t_p_1, x_pred, self.sig).ravel()

    def sample(self, key, x_t, params):
        x_pred = self.rho * x_t
        return x_pred + self.sig * jax.random.normal(key, x_t.shape)


@chex.dataclass
class GaussianDistribution(Distribution):
    """
    Gaussian distribution.
    """
    mu: float = 0.0
    sig: float = 1.0

    def sample(self, key, N):
        return self.mu + self.sig * jax.random.normal(key, (N, 1))

    def logpdf(self, x):
        return norm.logpdf(x, self.mu, self.sig).ravel()


@chex.dataclass
class GaussianPotential(UnivariatePotential):
    """
    Gaussian potential.
    """
    y_0: float = None
    sig: float = 1.0

    def logpdf(self, x):
        return norm.logpdf(x, self.y0, self.sig).ravel()


@chex.dataclass
class GaussianObservationPotential(Potential):
    sig: float = 1.0

    def logpdf(self, x_t_p_1, _x_t, params):
        y = params
        return norm.logpdf(y, x_t_p_1, self.sig).ravel()


@chex.dataclass
class FlatUnivariatePotential(UnivariatePotential):
    """
    Gaussian potential.
    """

    def logpdf(self, x):
        return jnp.zeros(x.shape[:1], dtype=x.dtype)


@chex.dataclass
class FlatPotential(Potential):

    def logpdf(self, x_t_p_1, _x_t, params):
        return jnp.zeros(x_t_p_1.shape[:1], dtype=x_t_p_1.dtype)
