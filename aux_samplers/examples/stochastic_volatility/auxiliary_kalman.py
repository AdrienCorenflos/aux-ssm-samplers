import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve

from aux_samplers import mvn
from aux_samplers.kalman import get_kernel as get_generic_kernel
from model import log_potential, hess_log_potential


def get_kernel(ys, m0, P0, F, Q, b, parallel, order=1):
    d = m0.shape[0]
    T = ys.shape[0]

    eye = jnp.eye(d)
    eyes = jnp.repeat(eye[None, ...], T, axis=0)
    zero = jnp.zeros((d,))
    zeros = jnp.zeros((T, d))

    chol_P0 = jnp.linalg.cholesky(P0)
    chol_Q = jnp.linalg.cholesky(Q)

    def dynamics_factory(_x):
        Fs = jnp.tile(F[None, ...], (T - 1, 1, 1))
        Qs = jnp.tile(Q[None, ...], (T - 1, 1, 1))
        bs = jnp.tile(b[None, ...], (T - 1, 1))
        return m0, P0, Fs, Qs, bs

    def first_order_observations_factory(x, u, delta):
        grad_x = jax.grad(log_potential)(x, ys)
        grad_x = jnp.nan_to_num(grad_x)
        aux_ys = u + 0.5 * delta * grad_x
        Hs = eyes
        Rs = 0.5 * delta * eyes
        cs = zeros
        return aux_ys, Hs, Rs, cs

    def second_order_observations_factory_one(y, x, u, delta):
        # For a single time step!
        grad = jax.grad(log_potential)(x, y)
        hess = hess_log_potential(x, y)
        Omega_inv = -hess + 2 * eye / delta
        chol_Omega_inv = jnp.linalg.cholesky(Omega_inv)
        Omega = cho_solve((chol_Omega_inv, True), eye)
        aux_y = Omega @ (2 * u / delta + grad - hess @ x)
        return aux_y, eye, Omega, zero

    def second_order_observations_factory(x, u, delta):
        return jax.vmap(second_order_observations_factory_one, in_axes=(0, 0, 0, None))(ys, x, u, delta)

    def log_likelihood_fn(x):
        out = mvn.logpdf(x[0], m0, chol_P0)
        pred_x = jnp.einsum("ij,tj->ti", F, x[:-1]) + b
        out += jnp.sum(mvn.logpdf(x[1:], pred_x, chol_Q))
        return out + log_potential(x, ys)

    observations_factory = first_order_observations_factory if order == 1 else second_order_observations_factory
    return get_generic_kernel(dynamics_factory, observations_factory, log_likelihood_fn, parallel)
