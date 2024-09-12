from functools import partial

import chex
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from aux_samplers.csmc import Distribution, UnivariatePotential, Dynamics, Potential, get_generic_kernel
from auxiliary_csmc import get_feynman_kac


def get_kernel(y, rho, r2, T, n_samples, backward, gradient):
    p0, g0, pt, gt = get_feynman_kac(y, rho, r2, T)
    sig_x = (1 - rho ** 2) ** 0.5
    r = jnp.sqrt(r2)

    sig_xs = sig_x * jnp.ones((T,))
    sig_xs = sig_xs.at[0].set(1.)

    def factory(u, scale):
        # scale is sqrt_half_delta

        Ks = jax.vmap(get_K, in_axes=(None, 0))(sig_x, scale)
        scale_props = sig_xs * (1 - Ks) ** 0.5

        @chex.dataclass
        class M0(Distribution, UnivariatePotential):
            def sample(self, key, N):
                grad_t = (T == 1) * (y - 0.) / r2
                u_t = u[0] + gradient * scale[0] ** 2 * grad_t
                m0 = Ks[0] * u_t
                return m0 + scale_props[0] * jax.random.normal(key, (N, 1))

            def logpdf(self, x):
                grad_t = (T == 1) * (y - 0.) / r2
                u_t = u[0] + gradient * scale[0] ** 2 * grad_t
                m0 = Ks[0] * u_t
                return norm.logpdf(x[..., 0], m0, scale_props[0])

            def __call__(self, x): return self.logpdf(x[..., 0])

        @chex.dataclass
        class Mt(Dynamics):
            @staticmethod
            @partial(jnp.vectorize, signature='(d)->(d)')
            def _x_pred(x):
                return rho * x

            def logpdf(self, x_t_p_1, x_t, params):
                K, sig, u_t, scale_t, t = params
                x_pred = self._x_pred(x_t)[..., 0]

                grad_t = (t == T - 1) * (y - x_pred) / r2
                u_t += gradient * scale_t ** 2 * grad_t
                mu_t = x_pred + K * (u_t - x_pred)
                return norm.logpdf(x_t_p_1[..., 0], mu_t, sig)

            def sample(self, key, x_t, params_t):
                K, sig, u_t, scale_t, t = params_t
                x_pred = self._x_pred(x_t)
                grad_t = (t == T - 1) * (y - x_pred) / r2
                u_t += gradient * scale_t ** 2 * grad_t
                mu_t = x_pred + K * (u_t - x_pred)
                return mu_t + sig * jax.random.normal(key, x_t.shape)

        @chex.dataclass
        class G0(UnivariatePotential):
            def __call__(self, x):
                grad_t = (T == 1) * (y - 0.) / r2
                u_t = u[0] + gradient * scale[0] ** 2 * grad_t
                m0 = Ks[0] * u_t

                out = norm.logpdf(x[..., 0], 0., 1.)  # m0
                out += norm.logpdf(x[..., 0], u[0], scale[0])  # N(x|u, 0.5 * delta)
                out -= norm.logpdf(x[..., 0], m0, scale_props[0])  # N(x|m, P - K @ P)
                # if T == 1: then we observe y
                out += (T == 1) * norm.logpdf(x[..., 0], y, r)
                return out

        @chex.dataclass
        class Gt(Potential):
            def __call__(self, x_t_p_1, x_t, params_t):
                K, sig, u_t, scale_t, t = params_t
                x_pred = rho * x_t[..., 0]

                grad_t = (t == T-1) * (y - x_pred) / r2
                u_t_prop = u_t + gradient * scale_t ** 2 * grad_t
                mu_t = x_pred + K * (u_t_prop - x_pred)

                out = norm.logpdf(x_t_p_1[..., 0], x_pred, sig_x)  # mt
                out += norm.logpdf(x_t_p_1[..., 0], u_t, scale_t)  # N(x|u, 0.5 * delta)
                out -= norm.logpdf(x_t_p_1[..., 0], mu_t, sig)  # N(x|m, P - K @ P)
                # if t == T - 1: then we observe y
                out += (t == T - 1) * norm.logpdf(y, x_t_p_1[..., 0], r)
                return out

        ts = jnp.arange(1, T)

        params = Ks[1:], scale_props[1:], u[1:], scale[1:], ts

        return M0(), G0(), Mt(params=params), Gt(params=params)

    return get_generic_kernel(factory, n_samples, backward, pt)


def get_K(sigma_x, scale):
    return sigma_x ** 2 / (sigma_x ** 2 + scale ** 2)
