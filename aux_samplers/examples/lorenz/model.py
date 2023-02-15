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



def observations_model(ys, sig_y, obs_freq, sampling_freq, T):
    ys_extended = np.ones((int(T / sampling_freq + _EPS) + 1, 2)) * np.nan
    ys_extended[::int(obs_freq / sampling_freq + _EPS)] = ys
    ts = np.linspace(0, T, ys_extended.shape[0])


    H = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
    Hs = np.ones((*ys_extended.shape, 3)) * np.nan
    Hs[::int(obs_freq / sampling_freq + _EPS), ...] = H

    R = sig_y ** 2 * np.eye(2)
    Rs = np.tile(R[None, ...], (ys_extended.shape[0], 1, 1))

    cs = np.zeros_like(ys_extended)
    return ys_extended, Hs, Rs, cs, ts


def theta_posterior_mean_and_chol(x, prior_cov, dt, sigma_x):
    """Posterior over theta given x and prior covariance."""
    phis = jax.vmap(phi)(x[:-1])
    phis_0 = jax.vmap(phi_0)(x[:-1])
    dx = x[1:] - x[:-1]

    mu = jnp.sum(phis * (dx - dt * phis_0), 0)
    Gamma = jnp.einsum("ij,ik,jk", phis, phis) / sigma_x ** 2
    Gamma = Gamma + prior_cov
    mean = solve(Gamma, mu, assume_a="pos")
    cov = solve(Gamma, jnp.eye(3), assume_a="pos")
    chol = jnp.linalg.cholesky(cov)
    return mean, chol

