from functools import partial

import chex
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from aux_samplers.csmc import get_independent_kernel, Distribution, UnivariatePotential, Dynamics, Potential


def get_kernel(y, rho, r2, T, n_samples, backward, parallel, gradient):
    M0, G0, Mt, Gt = get_feynman_kac(y, rho, r2, T)
    return get_independent_kernel(M0, G0, Mt, Gt, n_samples, backward, Mt, gradient, parallel)


def get_feynman_kac(y, rho, r2, T):
    sig_x = (1 - rho ** 2) ** 0.5
    r = jnp.sqrt(r2)

    @chex.dataclass
    class M0(Distribution, UnivariatePotential):
        def sample(self, key, N): return jax.random.normal(key, (N, 1))

        def logpdf(self, x): return norm.logpdf(x[..., 0], 0, 1)

        def __call__(self, x): return self.logpdf(x[..., 0])

    @chex.dataclass
    class Mt(Dynamics):
        @staticmethod
        @partial(jnp.vectorize, signature='(d)->(d)')
        def _x_pred(x):  return rho * x

        def logpdf(self, x_t_p_1, x_t, _params): return norm.logpdf(x_t_p_1[..., 0], self._x_pred(x_t)[..., 0], sig_x)

        def sample(self, key, x_t, _params): return self._x_pred(x_t) + sig_x * jax.random.normal(key, x_t.shape)

    @chex.dataclass
    class G0(UnivariatePotential):
        def __call__(self, x): return (T == 1) * norm.logpdf(x[..., 0], y, r)

    @chex.dataclass
    class Gt(Potential):
        def __call__(self, x_t_p_1, _x_t, t):
            return (t == T - 1) * norm.logpdf(y, x_t_p_1[..., 0], r)

    ts = jnp.arange(1, T)
    return M0(), G0(), Mt(), Gt(params=ts)
