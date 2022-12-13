import chex
import jax
from chex import Array
from jax import numpy as jnp
from jax.scipy.stats import norm

from .._primitives.csmc.base import Distribution, UnivariatePotential, Dynamics, Potential


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
