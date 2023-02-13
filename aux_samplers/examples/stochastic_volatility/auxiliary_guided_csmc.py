from functools import partial

import chex
import jax
import jax.numpy as jnp
from chex import Array
from jax.scipy.linalg import solve
from jax.scipy.stats import norm

from aux_samplers import mvn
from aux_samplers.csmc import Distribution, UnivariatePotential, Dynamics, Potential, get_generic_kernel
from auxiliary_csmc import get_feynman_kac


def get_kernel(ys, m0, P0, F, Q, b, n_samples, backward, gradient):
    _, _, Pt, _ = get_feynman_kac(ys, m0, P0, F, Q, b)

    def factory(u, scale):
        M0 = AuxiliaryM0(m0=m0, P0=P0, u=u[0], sqrt_half_delta=scale[0], y=ys[0], grad=gradient)
        Mt = AuxiliaryMt(F=F, Q=Q, b=b, params=(u[1:], scale[1:], ys[1:]), grad=gradient)
        G0 = AuxiliaryG0(m0=m0, P0=P0, u=u[0], sqrt_half_delta=scale[0], y=ys[0], grad=gradient)
        Gt = AuxiliaryGt(F=F, Q=Q, b=b, params=(u[1:], scale[1:], ys[1:]), grad=gradient)
        return M0, G0, Mt, Gt

    return get_generic_kernel(factory, n_samples, backward, Pt)


@chex.dataclass
class AuxiliaryM0(Distribution):
    m0: Array
    P0: Array
    u: Array
    sqrt_half_delta: float
    y: Array
    grad: bool = False

    def sample(self, key, n):
        m0, P0, u = self.m0, self.P0, self.u
        d = m0.shape[0]
        zero_F, zero_b = jnp.zeros((d, d)), jnp.zeros((d,))
        mu_t, chol_Lambda_t = get_mu_chol_Lambda_t(zero_b, zero_F, P0, m0, u, self.sqrt_half_delta, self.y, self.grad)
        out = mu_t[None, ...] + jax.random.normal(key, (n, d)) @ chol_Lambda_t.T
        return out


@chex.dataclass
class AuxiliaryG0(UnivariatePotential):
    m0: Array
    P0: Array
    u: Array
    sqrt_half_delta: float
    y: Array
    grad: bool = False

    def _get_K(self):
        P0, sqrt_half_delta = self.P0, self.sqrt_half_delta
        d = P0.shape[0]
        return solve((P0 + sqrt_half_delta ** 2 * jnp.eye(d)), P0, assume_a="pos").T

    def __call__(self, x):
        m0, P0, u = self.m0, self.P0, self.u
        d = m0.shape[0]
        zero_F, zero_b = jnp.zeros((d, d)), jnp.zeros((d,))
        mu_t, chol_Lambda_t = get_mu_chol_Lambda_t(zero_b, zero_F, P0, m0, u, self.sqrt_half_delta, self.y, self.grad)

        out = obs_logpdf(x, self.y)  # g0
        out += mvn.logpdf(x, m0, jnp.linalg.cholesky(P0))  # m0
        out += jnp.sum(norm.logpdf(x, u, self.sqrt_half_delta), -1)  # N(x|u, 0.5 * delta)
        out -= mvn.logpdf(x, mu_t, chol_Lambda_t)  # N(x|m, P - K @ P)
        return out


@chex.dataclass
class AuxiliaryMt(Dynamics):
    F: Array = None
    Q: Array = None
    b: Array = None
    grad: bool = False

    def sample(self, key, x_t, params):
        n, d = x_t.shape

        F, Q, b = self.F, self.Q, self.b
        u, scale, y = params
        mu_t, chol_Lambda_t = get_mu_chol_Lambda_t(x_t, F, Q, b, u, scale, y, self.grad)

        out = mu_t + jnp.einsum("...ij,...j->...i", chol_Lambda_t, jax.random.normal(key, (n, d)))
        return out


@chex.dataclass
class AuxiliaryGt(Potential):
    F: Array = None
    Q: Array = None
    b: Array = None
    grad: bool = False

    def __call__(self, x_t_p_1, x_t, params):
        F, Q, b = self.F, self.Q, self.b
        u, scale, y = params
        mu_t, chol_Lambda_t = get_mu_chol_Lambda_t(x_t, F, Q, b, u, scale, y, self.grad)

        out = obs_logpdf(x_t_p_1, y)  # gt
        out += trans_logpdf(x_t_p_1, x_t, F, Q, b)
        out += jnp.sum(norm.logpdf(x_t_p_1, u, scale), -1)  # N(x|u, 0.5 * delta)
        out -= mvn.logpdf(x_t_p_1, mu_t, chol_Lambda_t)  # N(x|m, P - K @ P)
        return out


def get_K(Q, scale):
    d = Q.shape[0]
    return solve((Q + scale ** 2 * jnp.eye(d)), Q, assume_a="pos").T


@partial(jnp.vectorize, signature="(d),(d,d),(d,d),(d),(d),(),(d)->(d),(d,d)", excluded=(7,))
def get_mu_chol_Lambda_t(x, F, Q, b, u, scale, y, grad):
    d = x.shape[0]
    K = get_K(Q, scale)

    x_pred = F @ x + b
    Lambda_t = Q - K @ Q
    Lambda_t = 0.5 * (Lambda_t + Lambda_t.T)

    if grad:
        grad_val = jax.grad(obs_logpdf)(u, y)
        u += scale ** 2 * grad_val

    mu_t = x_pred + K @ (u - x_pred)
    chol_Lambda_t = jnp.linalg.cholesky(Lambda_t)

    # another option is to use a linearisation of the locally optimal proposal.
    # if grad:
    #     temp = 0.5 * (jnp.exp(-mu_t) * y ** 2 - 2)
    #     mu_t += Lambda_t @ temp

    # When scale is too small, the covariance matrix Lambda_t can numerically be singular.
    chol_Lambda_t = jnp.where(jnp.isfinite(chol_Lambda_t), chol_Lambda_t,
                              scale * jnp.eye(d))

    return mu_t, chol_Lambda_t


@partial(jnp.vectorize, signature="(d),(d)->()")
def obs_logpdf(x, y):
    return jnp.sum(norm.logpdf(y, 0., jnp.exp(0.5 * x)))


@partial(jnp.vectorize, signature="(d),(d),(d,d),(d,d),(d)->()")
def trans_logpdf(x_t_p_1, x_t, F, Q, b):
    pred_x = F @ x_t + b
    return mvn.logpdf(x_t_p_1, pred_x, jnp.linalg.cholesky(Q))
