import jax
import jax.numpy as jnp

from aux_samplers import mvn
from aux_samplers.kalman import get_kernel as get_generic_kernel
from model import log_potential


def get_kernel(ys, m0, P0, F, Q, b, parallel, log_potential_args):
    # We batch the model
    d = m0.shape[0]
    T = ys.shape[0]

    eyes = jnp.ones((T, d, 1, 1))
    zeros = jnp.zeros((T, d, 1))

    chol_P0 = jnp.linalg.cholesky(P0)
    chol_Q = jnp.linalg.cholesky(Q)

    def dynamics_factory(_x):
        Fs = jnp.tile(F[None, ...], (T - 1, 1, 1, 1))
        Qs = jnp.tile(Q[None, ...], (T - 1, 1, 1, 1))
        bs = jnp.tile(b[None, ...], (T - 1, 1, 1))
        return m0, P0, Fs, Qs, bs

    def first_order_observations_factory(x, u, delta):
        grad_x = jax.grad(log_potential)(x.reshape(-1, d), ys, *log_potential_args)
        grad_x = grad_x.reshape(T, d, 1)
        grad_x = jnp.nan_to_num(grad_x)
        aux_ys = u + 0.5 * delta * grad_x
        Hs = eyes
        Rs = 0.5 * delta * eyes
        cs = zeros
        return aux_ys, Hs, Rs, cs

    def log_likelihood_fn(x):
        out = jnp.sum(mvn.logpdf(x[0], m0, chol_P0))
        pred_x = jnp.einsum("...ij,...j->...i", F, x[:-1]) + b[None, ...]
        out += jnp.sum(mvn.logpdf(x[1:], pred_x, chol_Q))
        return out + log_potential(x.reshape(-1, d), ys, *log_potential_args)

    observations_factory = first_order_observations_factory
    init_, kernel = get_generic_kernel(dynamics_factory, observations_factory, log_likelihood_fn, parallel)

    def init(xs):
        return init_(jnp.expand_dims(xs, -1))

    return init, kernel
