from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm


@partial(jax.jit, static_argnums=(5, 6))
def get_data(key, nu, phi, tau, rho, dim, T):
    m0, P0, F, Q, b = get_dynamics(nu, phi, tau, rho, dim)

    init_key, sampling_key = jax.random.split(key)

    chol_P0 = jnp.linalg.cholesky(P0)
    chol_Q = jnp.linalg.cholesky(Q)
    x0 = m0 + chol_P0 @ jax.random.normal(init_key, (dim,))

    def body(x_k, key_k):
        state_key, observation_key = jax.random.split(key_k)

        observation_scale = jnp.exp(0.5 * x_k)

        y_k = observation_scale * jax.random.normal(observation_key, shape=(dim,))
        x_kp1 = F @ x_k + b + chol_Q @ jax.random.normal(state_key, shape=(dim,))
        return x_kp1, (x_k, y_k)

    _, (xs, ys) = jax.lax.scan(body, x0, jax.random.split(sampling_key, T))
    return xs, ys


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
