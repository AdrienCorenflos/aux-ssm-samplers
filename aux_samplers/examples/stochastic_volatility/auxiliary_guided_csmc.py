from functools import partial

import chex
import jax
import jax.numpy as jnp
from chex import Array
from jax.scipy.stats import norm

from aux_samplers import mvn
from aux_samplers.csmc import get_generic_kernel, Distribution, UnivariatePotential, Dynamics, Potential
from aux_samplers.csmc.independent import AuxiliaryM0, AuxiliaryG0
from auxiliary_csmc import get_feynman_kac


def get_kernel(ys, m0, P0, F, Q, b, n_samples, backward, gradient):
    Mt, G0, Mt, Gt = get_feynman_kac(ys, m0, P0, F, Q, b)

    def factory(u, scale):
        if gradient:
            grad_pi = jax.grad(_log_pdf)(u, G0, Gt)


def _log_pdf(u, G0, Gt):
    # Compute the log-pdf of the auxiliary variable
    log_pdf = G0(u[0])
    fn_out = jax.vmap(Gt)(u[1:], u[:-1], Gt.params)
    log_pdf += jnp.sum(fn_out)
    return log_pdf



@chex.dataclass
class AuxiliaryM0(Distribution):
    u: Array
    sqrt_half_delta: float
    grad: Array

    def logpdf(self, x):
        half_delta = self.sqrt_half_delta ** 2
        mean = self.u + half_delta * self.grad
        logpdf = norm.logpdf(x, mean, self.sqrt_half_delta)
        return jnp.sum(logpdf, axis=-1)

    def sample(self, key, n):
        half_delta = self.sqrt_half_delta ** 2
        mean = self.u + half_delta * self.grad
        return mean[None, ...] + self.sqrt_half_delta * jax.random.normal(key, (n, *self.u.shape))


@chex.dataclass
class AuxiliaryG0(UnivariatePotential):
    M0: Distribution
    G0: UnivariatePotential

    def __call__(self, x):
        return self.G0(x) + self.M0.logpdf(x)


@chex.dataclass
class AuxiliaryMt(Dynamics):
    def sample(self, key, x_t, params):
        u_t, sqrt_half_delta, grad_t = params
        half_delta = sqrt_half_delta ** 2
        mean = u_t[None, :] + half_delta * grad_t[None, :]
        return mean + sqrt_half_delta * jax.random.normal(key, x_t.shape)


@chex.dataclass
class AuxiliaryGt(Potential):
    Mt: Dynamics = None
    Gt: Potential = None

    def __post_init__(self):
        self.params = (self.Mt.params, self.Gt.params)

    def __call__(self, x_t_p_1, x_t, params):
        Mt_params, Gt_params = params
        return self.Mt.logpdf(x_t_p_1, x_t, Mt_params) + self.Gt(x_t_p_1, x_t, Gt_params)

@chex.dataclass
class GradientAuxiliaryGt(Potential):
    Mt: Dynamics = None
    Gt: Potential = None

    def __post_init__(self):
        self.params = (self.params, self.Mt.params, self.Gt.params)

    def __call__(self, x_t_p_1, x_t, params):
        (u_t, sqrt_half_delta, grad_t), Mt_params, Gt_params = params
        half_delta = sqrt_half_delta ** 2
        mean = u_t + half_delta * grad_t

        out_1 = self.Mt.logpdf(x_t_p_1, x_t, Mt_params) + self.Gt(x_t_p_1, x_t, Gt_params)
        out_2 = jnp.sum(norm.logpdf(x_t_p_1, u_t, sqrt_half_delta))
        out_2 -= jnp.sum(norm.logpdf(x_t_p_1, mean, sqrt_half_delta))

        return out_1 + out_2
