import chex
import jax
import jax.numpy as jnp
from chex import Array
from jax.scipy.linalg import solve
from jax.scipy.stats import norm

from aux_samplers import mvn
from aux_samplers.csmc import Distribution, UnivariatePotential, Dynamics, Potential, get_generic_kernel
from auxiliary_csmc import get_feynman_kac


def get_kernel(ys, m0, P0, F, Q, b, n_samples, backward, _gradient):
    _, _, Pt, _ = get_feynman_kac(ys, m0, P0, F, Q, b)

    def factory(u, scale):
        M0 = AuxiliaryM0(m0=m0, P0=P0, u=u[0], sqrt_half_delta=scale[0])
        Mt = AuxiliaryMt(F=F, Q=Q, b=b, params=(u[1:], scale[1:]))
        G0 = AuxiliaryG0(m0=m0, P0=P0, u=u[0], sqrt_half_delta=scale[0], y=ys[0])
        Gt = AuxiliaryGt(F=F, Q=Q, b=b, params=(u[1:], scale[1:], ys[1:]))
        return M0, G0, Mt, Gt

    return get_generic_kernel(factory, n_samples, backward, Pt)


@chex.dataclass
class AuxiliaryM0(Distribution):
    m0: Array
    P0: Array
    u: Array
    sqrt_half_delta: float

    def _get_K(self):
        P0, sqrt_half_delta = self.P0, self.sqrt_half_delta
        d = P0.shape[0]
        return solve((P0 + sqrt_half_delta ** 2 * jnp.eye(d)), P0).T

    def logpdf(self, x):
        m0, P0, u = self.m0, self.P0, self.u
        K = self._get_K()
        m = m0 + K @ (u - m0)
        chol_P = jnp.linalg.cholesky(P0 - K @ P0)

        return mvn.logpdf(x, m, chol_P)

    def sample(self, key, n):
        m0, P0, u = self.m0, self.P0, self.u
        d = m0.shape[0]
        K = self._get_K()
        m = m0 + K @ (u - m0)
        S = P0 - K @ P0
        chol_P = jnp.linalg.cholesky(0.5 * (S + S.T))
        chol_P = jnp.where(jnp.isfinite(chol_P), chol_P, self.sqrt_half_delta * jnp.eye(d))  # if delta << 1, we recover N(x | u, delta)
        out = m[None, ...] + jax.random.normal(key, (n, d)) @ chol_P.T
        return out


@chex.dataclass
class AuxiliaryG0(UnivariatePotential):
    m0: Array
    P0: Array
    u: Array
    sqrt_half_delta: float
    y: Array

    def _get_K(self):
        P0, sqrt_half_delta = self.P0, self.sqrt_half_delta
        d = P0.shape[0]
        return solve((P0 + sqrt_half_delta ** 2 * jnp.eye(d)), P0).T

    def __call__(self, x):
        m0, P0, u = self.m0, self.P0, self.u
        d = m0.shape[0]

        K = self._get_K()
        m = m0 + K @ (u - m0)
        S = P0 - K @ P0
        chol_P = jnp.linalg.cholesky(0.5 * (S + S.T))
        chol_P = jnp.where(jnp.isfinite(chol_P), chol_P, self.sqrt_half_delta * jnp.eye(d))  # if delta << 1, we recover N(x | u, delta)

        out = jnp.sum(norm.logpdf(self.y, 0., jnp.exp(0.5 * x)), -1)  # g0
        out += mvn.logpdf(x, m0, jnp.linalg.cholesky(P0))  # m0
        out += jnp.sum(norm.logpdf(x, u, self.sqrt_half_delta), -1)  # N(x|u, 0.5 * delta)
        out -= mvn.logpdf(x, m, chol_P)  # N(x|m, P - K @ P)
        return out


@chex.dataclass
class AuxiliaryMt(Dynamics):
    F: Array = None
    Q: Array = None
    b: Array = None

    def _get_K(self, scale):
        d = self.b.shape[0]
        return solve((self.Q + scale ** 2 * jnp.eye(d)), self.Q).T

    def sample(self, key, x_t, params):
        n, d = x_t.shape

        F, Q, b = self.F, self.Q, self.b
        u, scale = params
        K = self._get_K(scale)

        pred_m = x_t @ F.T + b[None, ...]
        m = pred_m + (u[None, ...] - pred_m) @ K.T

        S = Q - K @ Q
        chol_P = jnp.linalg.cholesky(0.5 * (S + S.T))
        chol_P = jnp.where(jnp.isfinite(chol_P), chol_P, scale * jnp.eye(d))  # if delta << 1, we recover N(x | u, delta)

        out = m + jax.random.normal(key, (n, d)) @ chol_P.T
        return out


@chex.dataclass
class AuxiliaryGt(Potential):
    F: Array = None
    Q: Array = None
    b: Array = None

    def _get_K(self, scale):
        d = self.b.shape[0]
        return solve((self.Q + scale ** 2 * jnp.eye(d)), self.Q).T

    def __call__(self, x_t_p_1, x_t, params):
        F, Q, b = self.F, self.Q, self.b
        d = b.shape[0]
        u, scale, y = params
        K = self._get_K(scale)

        pred_m = x_t @ F.T + b[None, ...]
        m = pred_m + (u[None, ...] - pred_m) @ K.T

        S = Q - K @ Q
        chol_P = jnp.linalg.cholesky(0.5 * (S + S.T))
        chol_P = jnp.where(jnp.isfinite(chol_P), chol_P, scale * jnp.eye(d))  # if delta << 1, we recover N(x | u, delta)

        out = jnp.sum(norm.logpdf(y, 0., jnp.exp(0.5 * x_t_p_1)), -1)  # gt
        out += mvn.logpdf(x_t_p_1, pred_m, jnp.linalg.cholesky(Q))  # mt
        out += jnp.sum(norm.logpdf(x_t_p_1, u, scale), -1)  # N(x|u, 0.5 * delta)
        out -= mvn.logpdf(x_t_p_1, m, chol_P)  # N(x|m, P - K @ P)
        return out
