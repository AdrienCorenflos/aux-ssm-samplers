import jax
import jax.numpy as jnp

from aux_samplers import mvn
from aux_samplers.kalman import get_kernel as get_generic_kernel
from model import log_potential


def get_kernel(ys, m0, P0, F, Q, b, parallel):
    d = m0.shape[0]
    T = ys.shape[0]

    eyes = jnp.repeat(jnp.eye(d)[None, ...], T, axis=0)
    zeros = jnp.zeros((T, d))

    chol_P0 = jnp.linalg.cholesky(P0)
    chol_Q = jnp.linalg.cholesky(Q)

    def dynamics_factory(_x):
        Fs = jnp.tile(F[None, ...], (T - 1, 1, 1))
        Qs = jnp.tile(Q[None, ...], (T - 1, 1, 1))
        bs = jnp.tile(b[None, ...], (T - 1, 1))
        return m0, P0, Fs, Qs, bs

    def observations_factory(x, u, delta):
        grad_x = jax.grad(log_potential)(x, ys)
        grad_x = jnp.nan_to_num(grad_x)
        aux_ys = u + 0.5 * delta * grad_x
        Hs = eyes
        Rs = 0.5 * delta * eyes
        cs = zeros
        return aux_ys, Hs, Rs, cs

    def log_likelihood_fn(x):
        out = mvn.logpdf(x[0], m0, chol_P0)
        pred_x = jnp.einsum("ij,tj->ti", F, x[:-1]) + b
        out += jnp.sum(mvn.logpdf(x[1:], pred_x, chol_Q))
        return out + log_potential(x, ys)

    return get_generic_kernel(dynamics_factory, observations_factory, log_likelihood_fn, parallel)

