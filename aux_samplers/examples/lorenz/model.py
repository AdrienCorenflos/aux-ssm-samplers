from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve

_EPS = 1e-8


def phi_0(x):
    x1, x2, x3 = x
    return jnp.array([0, -x2 - x1 * x3, x1 * x2])


def phi(x):
    x1, x2, x3 = x
    return jnp.array([x2 - x1, x1, -x3])


def get_dynamics(theta, sigma_x, dt):
    def mean(x, _params):
        return x + dt * phi_0(x) + dt * theta * phi(x)

    Q = dt * sigma_x ** 2 * jnp.eye(3)
    return mean, Q


def observations_model(data, sig_y, n_steps, sample_every):
    ys = data[:, 1:]
    ys_extended = np.ones((n_steps, 2)) * np.nan
    ys_extended[::sample_every] = ys

    H = np.array([[0, 1, 0], [0, 0, 1]], dtype=float)
    Hs = np.ones((*ys_extended.shape, 3)) * np.nan
    Hs[::sample_every, ...] = H

    R = sig_y ** 2 * np.eye(2)
    Rs = np.tile(R[None, ...], (ys_extended.shape[0], 1, 1))

    cs = np.zeros_like(ys_extended)
    return ys_extended, Hs, Rs, cs


def theta_posterior_mean_and_chol(x, prior_cov, dt, sigma_x):
    """Posterior over theta given x and prior covariance."""
    phis = jax.vmap(phi)(x[:-1])
    phis_0 = jax.vmap(phi_0)(x[:-1])
    dx = x[1:] - x[:-1]
    mu = jnp.sum(phis * (dx - dt * phis_0), 0) / sigma_x ** 2
    Gamma = dt * jnp.einsum("ij,ik->jk", phis, phis) / sigma_x ** 2

    Gamma = Gamma + jnp.linalg.inv(prior_cov)
    cov = jnp.linalg.inv(Gamma)
    mean = cov @ mu
    chol = jnp.linalg.cholesky(cov)
    return mean, chol


@partial(jax.jit, static_argnums=(1,))
def init_x_fn(data, n_steps):
    """Initial state and covariance."""
    T = data[-1, 0]
    ts = jnp.linspace(0, T, n_steps)
    xs = jnp.ones((n_steps, 3))
    xs = xs.at[:, 0].set(25)
    xs = xs.at[:, 1].set(jnp.interp(ts, data[:, 0], data[:, 1]))
    xs = xs.at[:, 2].set(jnp.interp(ts, data[:, 0], data[:, 2]))
    return xs



