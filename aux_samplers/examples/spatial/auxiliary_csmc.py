import chex
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from aux_samplers.csmc import get_independent_kernel, Distribution, UnivariatePotential, Dynamics, Potential
from model import log_potential_one


def get_kernel(ys, sigma_x, nu, prec, n_samples, backward, parallel, gradient):
    M0, G0, Mt, Gt = get_feynman_kac(ys, sigma_x, nu, prec)
    return get_independent_kernel(M0, G0, Mt, Gt, n_samples, backward, Mt, gradient, parallel)


def get_feynman_kac(ys, sigma_x, nu, prec):
    d = ys.shape[-1]

    @chex.dataclass
    class M0(Distribution, UnivariatePotential):
        def sample(self, key, N): return sigma_x * jax.random.normal(key, (N, d))

        def logpdf(self, x): return jnp.sum(norm.logpdf(x, 0., sigma_x), -1)

        def __call__(self, x): return self.logpdf(x)

    @chex.dataclass
    class Mt(Dynamics):
        def logpdf(self, x_t_p_1, x_t, _params): return jnp.sum(norm.logpdf(x_t_p_1, x_t, sigma_x), -1)

        def sample(self, key, x_t, _params): return x_t + sigma_x * jax.random.normal(key, x_t.shape)

    @chex.dataclass
    class G0(UnivariatePotential):
        def __call__(self, x): return log_potential_one(x, ys[0], nu, prec)

    @chex.dataclass
    class Gt(Potential):
        def __call__(self, x_t_p_1, _x_t, y):
            return log_potential_one(y, x_t_p_1, nu, prec)

    return M0(), G0(), Mt(), Gt(params=ys[1:])
