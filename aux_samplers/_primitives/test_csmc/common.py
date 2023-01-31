from dataclasses import field

import chex
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from aux_samplers._primitives.csmc.base import Dynamics, Distribution, UnivariatePotential, Potential


@chex.dataclass
class GaussianDynamics(Dynamics, Potential):
    """
    AR dynamics with Gaussian noise.
    """
    rho: float = 0.9
    sig: float = field(init=False)

    def __post_init__(self):
        self.sig = (1 - self.rho ** 2) ** 0.5

    def logpdf(self, x_t_p_1, x_t, _params):
        x_pred = self.rho * x_t
        return jnp.sum(norm.logpdf(x_t_p_1, x_pred, self.sig), axis=-1)

    def sample(self, key, x_t, params):
        x_pred = self.rho * x_t
        return x_pred + self.sig * jax.random.normal(key, x_t.shape)

    def __call__(self, x_t_p_1, x_t, params):
        return self.logpdf(x_t_p_1, x_t, params)


@chex.dataclass
class GaussianDistribution(Distribution, UnivariatePotential):
    """
    Gaussian distribution.
    """
    mu: float = 0.0
    sig: float = 1.0

    def sample(self, key, N):
        return self.mu + self.sig * jax.random.normal(key, (N, 1))

    def logpdf(self, x):
        return jnp.sum(norm.logpdf(x, self.mu, self.sig), axis=-1)

    def __call__(self, x):
        return self.logpdf(x)


@chex.dataclass
class GaussianObservationPotential(Potential):
    sig: float = 1.0

    def __call__(self, x_t_p_1, _x_t, params):
        y = params
        return norm.logpdf(y, x_t_p_1, self.sig).ravel()


@chex.dataclass
class FlatUnivariatePotential(UnivariatePotential):
    """
    Gaussian potential.
    """

    def __call__(self, x):
        return jnp.zeros(x.shape[:1], dtype=x.dtype)


@chex.dataclass
class FlatPotential(Potential):

    def __call__(self, x_t_p_1, _x_t, _params):
        return jnp.zeros(x_t_p_1.shape[:1], dtype=x_t_p_1.dtype)


def lgssm_data(key, rho, sig_y, T):
    init_key, key = jax.random.split(key)
    sig_x = (1 - rho ** 2) ** 0.5
    eps = jax.random.normal(key, (T, 2))
    x0 = eps[0, 0]
    y0 = x0 + sig_y * eps[0, 1]

    def body(x, inps):
        eps_x, eps_y = inps
        x = rho * x + sig_x * eps_x
        y = x + sig_y * eps_y
        return x, (x, y)

    _, (xs, ys) = jax.lax.scan(body, x0, eps[1:])
    xs = jnp.insert(xs, 0, x0, 0)
    ys = jnp.insert(ys, 0, y0, 0)
    return xs, ys
