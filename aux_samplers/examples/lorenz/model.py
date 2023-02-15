from functools import partial

import jax.numpy as jnp
import numpy as np

_EPS = 1e-8


def get_dynamics(theta, sigma_x, dt):
    def mean(x, _params):
        x1, x2, x3 = x
        z1 = x1 + dt * (theta[0] * (x2 - x1))
        z2 = x2 + dt * (-theta[1] * x1 - x2 - x1 * x3)
        z3 = x3 + dt * (x1 * x2 - theta[2] * x3)
        return jnp.array([z1, z2, z3])

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

#
# data = np.loadtxt("data.csv", delimiter=",", skiprows=1)
# ys_ex, *_, ts_ex = observations_model(data[:, 1:], 5 ** 0.5, 0.01, 2e-3, 2)
# print(ys_ex[:50, 0])
# print(ts_ex[:50])
