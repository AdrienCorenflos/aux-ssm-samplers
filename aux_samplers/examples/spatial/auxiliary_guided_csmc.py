from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp
from chex import Array, Numeric
from jax.scipy.stats import norm

from aux_samplers.csmc import Distribution, UnivariatePotential, Dynamics, Potential, get_generic_kernel
from aux_samplers.examples.spatial.model import log_potential_one
from auxiliary_csmc import get_feynman_kac


def get_kernel(ys, sigma_x, nu, prec, n_samples, backward, gradient):
    _, _, Pt, _ = get_feynman_kac(ys, sigma_x, nu, prec)

    def factory(u, scale):
        M0 = AuxiliaryM0(sigma_x=sigma_x, u=u[0], sqrt_half_delta=scale[0], y=ys[0], grad=gradient, nu=nu, prec=prec)
        Mt = AuxiliaryMt(sigma_x=sigma_x, nu=nu, prec=prec, params=(u[1:], scale[1:], ys[1:]), grad=gradient)
        G0 = AuxiliaryG0(sigma_x=sigma_x, nu=nu, prec=prec, u=u[0], sqrt_half_delta=scale[0], y=ys[0], grad=gradient)
        Gt = AuxiliaryGt(sigma_x=sigma_x, nu=nu, prec=prec, params=(u[1:], scale[1:], ys[1:]), grad=gradient)
        return M0, G0, Mt, Gt

    return get_generic_kernel(factory, n_samples, backward, Pt)


@chex.dataclass
class AuxiliaryM0(Distribution):
    nu: Numeric
    sigma_x: Numeric
    prec: Any
    u: Array
    sqrt_half_delta: float
    y: Array
    grad: bool = False

    def sample(self, key, n):
        nu, sigma_x, prec, u = self.nu, self.sigma_x, self.prec, self.u
        d = u.shape[0]
        zero = jnp.zeros((d,))
        mu_t, sqrt_Lambda_t = get_mu_chol_Lambda_t(zero, sigma_x, u, self.sqrt_half_delta, self.y, self.grad, nu, prec)
        out = mu_t[None, ...] + sqrt_Lambda_t * jax.random.normal(key, (n, d))
        return out


@chex.dataclass
class AuxiliaryG0(UnivariatePotential):
    nu: Numeric
    sigma_x: Numeric
    prec: Any
    u: Array
    sqrt_half_delta: float
    y: Array
    grad: bool = False

    def _get_K(self):
        sigma_x, sqrt_half_delta = self.sigma_x, self.sqrt_half_delta
        return sigma_x ** 2 / (sigma_x ** 2 + sqrt_half_delta ** 2)

    def __call__(self, x):
        nu, sigma_x, prec = self.nu, self.sigma_x, self.prec
        d = x.shape[-1]
        zero = jnp.zeros((d,))
        mu_t, sqrt_Lambda_t = get_mu_chol_Lambda_t(zero, sigma_x, self.u, self.sqrt_half_delta, self.y, self.grad, nu,
                                                   prec)

        out = log_potential_one(x, self.y, nu, prec)  # g0
        out += jnp.sum(norm.logpdf(x, 0., sigma_x), -1)  # m0
        out += jnp.sum(norm.logpdf(x, self.u, self.sqrt_half_delta), -1)  # N(x|u, 0.5 * delta)
        out -= jnp.sum(norm.logpdf(x, mu_t, sqrt_Lambda_t), -1)  # N(x|m, P - K @ P)
        return out


@chex.dataclass
class AuxiliaryMt(Dynamics):
    nu: Numeric = None
    sigma_x: Numeric = None
    prec: Any = None
    grad: bool = False

    def sample(self, key, x_t, params):
        n, d = x_t.shape

        nu, sigma_x, prec = self.nu, self.sigma_x, self.prec
        u, scale, y = params
        mu_t, sqrt_Lambda_t = get_mu_chol_Lambda_t(x_t, sigma_x, u, scale, y, self.grad, nu, prec)

        out = mu_t + sqrt_Lambda_t[:, None] * jax.random.normal(key, (n, d))
        return out


@chex.dataclass
class AuxiliaryGt(Potential):
    nu: Numeric = None
    sigma_x: Numeric = None
    prec: Any = None
    grad: bool = False

    def __call__(self, x_t_p_1, x_t, params):
        nu, sigma_x, prec = self.nu, self.sigma_x, self.prec
        u, scale, y = params
        mu_t, sqrt_Lambda_t = get_mu_chol_Lambda_t(x_t, sigma_x, u, scale, y, self.grad, nu, prec)

        gt = log_potential_one(x_t_p_1, y, nu, prec)  # gt
        # jax.debug.print("gt: {}", gt)
        mt = jnp.sum(norm.logpdf(x_t_p_1, x_t, sigma_x), -1)  # mt
        # jax.debug.print("mt: {}", mt)
        ut_part = jnp.sum(norm.logpdf(x_t_p_1, u, scale), -1)  # N(x|u, 0.5 * delta)
        # jax.debug.print("ut: {}", ut_part)
        prop_part = jnp.sum(norm.logpdf(x_t_p_1, mu_t, sqrt_Lambda_t[:, None]), -1)  # N(x|m, P - K @ P)
        # jax.debug.print("prop_part: {}", ut_part)

        out = gt + mt + ut_part - prop_part
        return out


def get_K(sigma_x, scale):
    return sigma_x ** 2 / (sigma_x ** 2 + scale ** 2)


@partial(jnp.vectorize, signature="(d),(),(d),(),(d)->(d),()", excluded=(5, 6, 7))
def get_mu_chol_Lambda_t(x, sigma_x, u, scale, y, grad, nu, prec):
    K = get_K(sigma_x, scale)

    x_pred = x
    Lambda_t = sigma_x ** 2 * (1 - K)

    if grad:
        grad_val = jax.grad(log_potential_one)(x_pred, y, nu, prec)
        u += scale ** 2 * grad_val

    mu_t = x_pred + K * (u - x_pred)
    sqrt_Lambda_t = jnp.sqrt(Lambda_t)
    return mu_t, sqrt_Lambda_t
