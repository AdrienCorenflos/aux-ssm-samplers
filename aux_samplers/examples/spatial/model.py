from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm


@partial(jax.jit, static_argnums=(4, 5))
def get_data(key, tau, sigma_x, r_y, T, D):
    path_key, obs_key = jax.random.split(key)

    xs = sigma_x * jax.random.normal(path_key, shape=(T, D * D))
    xs = jnp.cumsum(xs)

    return xs


@partial(jax.jit, static_argnums=(4,))
def get_dynamics(nu, phi, tau, rho, dim):
    F = phi * jnp.eye(dim)
    Q = stationary_covariance(phi, tau, rho, dim)
    mu = nu * jnp.ones((dim,))
    b = mu + F @ mu

    m0 = mu
    P0 = Q
    return m0, P0, F, Q, b


@partial(jax.jit, static_argnums=(3,))
def stationary_covariance(phi, tau, rho, dim):
    U = tau * rho * jnp.ones((dim, dim))
    U = U.at[np.diag_indices(dim)].set(tau)
    vec_U = jnp.reshape(U, (dim ** 2, 1))
    vec_U_star = vec_U / (1 - phi ** 2)
    U_star = jnp.reshape(vec_U_star, (dim, dim))
    return U_star


@jax.jit
def log_potential(xs, ys):
    vals = jax.vmap(_log_potential_one)(xs, ys)
    return jnp.sum(vals)


@jax.jit
def grad_log_potential(xs, ys):
    return jax.grad(log_potential)(xs, ys)


@jax.jit
def hess_log_potential(xs, ys):
    out = jax.vmap(_hess_log_potential_one)(xs, ys)
    return jnp.diag(out)


@jax.jit
def _log_potential_one(x, y):
    scale = jnp.exp(0.5 * x)
    val = norm.logpdf(y, scale=scale)
    return jnp.nan_to_num(val)  # in case the scale is infinite, we get nan, but we want 0


@jax.jit
def _hess_log_potential_one(x, y):
    return jax.grad(jax.grad(_log_potential_one))(x, y)
