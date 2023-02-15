import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve

from aux_samplers import mvn, extended
from aux_samplers.kalman import get_kernel as get_generic_kernel
from model import get_dynamics, observations_model


def get_kernel(data, theta, sigma_x, sigma_y, sampling_freq, parallel):
    ts, ys = data[:, 0], data[:, 1:]
    obs_freq = ts[1] - ts[0]

    ys,  Hs, Rs, cs, _ = observations_model(ys, sigma_y, obs_freq, sampling_freq, ts[-1])

    eye = jnp.eye(3)

    mean, Q = get_dynamics(theta, sigma_x, sampling_freq)
    cov = lambda _x, _params: Q

    m0 = jnp.array([1.5, -1.5, 25])
    P0 = jnp.diag(jnp.array[400, 20, 20])
    chol_P0 = jnp.linalg.cholesky(P0)

    specialised_extended = lambda x: extended(mean, cov, None, x, None)

    def dynamics_factory(x):
        Fs, Qs, bs = jax.vmap(specialised_extended)(x)
        return m0, P0, Fs, Qs, bs

    def observation_factory(_x, u, delta):

        aux_ys = jnp.concatenate([u, ys], axis=1)


        aux_Hs
        aux_ys = u + 0.5 * delta * grad_x
        Hs = eyes
        Rs = 0.5 * delta * eyes
        cs = zeros
        return aux_ys, Hs, Rs, cs


    def second_order_observations_factory(x, u, delta):
        return jax.vmap(second_order_observations_factory_one, in_axes=(0, 0, 0, None))(ys, x, u, delta)

    def log_likelihood_fn(x):
        out = mvn.logpdf(x[0], m0, chol_P0)
        pred_x = jnp.einsum("ij,tj->ti", F, x[:-1]) + b
        out += jnp.sum(mvn.logpdf(x[1:], pred_x, chol_Q))
        return out + log_potential(x, ys)

    observations_factory = first_order_observations_factory if order == 1 else second_order_observations_factory
    return get_generic_kernel(dynamics_factory, observations_factory, log_likelihood_fn, parallel)
