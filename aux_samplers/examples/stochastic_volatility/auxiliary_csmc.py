import chex
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from aux_samplers import mvn
from aux_samplers.csmc import get_independent_kernel, Distribution, UnivariatePotential, Dynamics, Potential


def get_kernel(ys, m0, P0, F, Q, b, n_samples, backward, parallel):
    d = m0.shape[0]

    chol_P0 = jnp.linalg.cholesky(P0)
    chol_Q = jnp.linalg.cholesky(Q)

    @chex.dataclass
    class M0(Distribution, UnivariatePotential):
        def sample(self, key, N): return m0[None, ...] + jax.random.normal(key, (N, d)) @ chol_P0.T

        def logpdf(self, x): return mvn.logpdf(x, m0, chol_P0).ravel()

        def __call__(self, x): return self.logpdf(x)

    @chex.dataclass
    class Mt(Dynamics):
        @staticmethod
        def _x_pred(x): return x @ F.T + b[None, ...]

        def logpdf(self, x_t_p_1, x_t, _params): return mvn.logpdf(x_t_p_1, self._x_pred(x_t), chol_Q).ravel()

        def sample(self, key, x_t, _params): return self._x_pred(x_t) + jax.random.normal(key, x_t.shape) @ chol_Q.T

    @chex.dataclass
    class G0(UnivariatePotential):
        def __call__(self, x): return jnp.sum(norm.logpdf(ys[0], loc=0, scale=jnp.exp(x)), -1)

    @chex.dataclass
    class Gt(Potential):
        params = ys[1:]

        def __call__(self, x_t_p_1, _x_t, y):
            return jnp.sum(norm.logpdf(y, loc=0, scale=jnp.exp(x_t_p_1)), -1)

    mt = Mt()
    return get_independent_kernel(M0(), G0(), mt, Gt(), n_samples, backward, mt, parallel)
