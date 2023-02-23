import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag

from aux_samplers import mvn, extended
from aux_samplers.kalman import get_kernel as get_generic_kernel
from model import get_dynamics


def _batched_block_diag(a, b):
    return jax.vmap(block_diag)(a, b)


def get_kernel(ys, Hs, Rs, cs, m0, P0, theta, sigma_x, dt, parallel):
    eye = jnp.eye(3)
    eyes = jnp.tile(eye[None, ...], (ys.shape[0], 1, 1))

    mean, Q = get_dynamics(theta, sigma_x, dt)
    cov = lambda _x, _params: Q

    chol_P0 = jnp.linalg.cholesky(P0)
    chol_Q = jnp.linalg.cholesky(Q)
    chol_Rs = jnp.linalg.cholesky(Rs)
    specialised_extended = lambda x: extended(mean, cov, None, x, None)

    def dynamics_factory(x):
        Fs, Qs, bs = jax.vmap(specialised_extended)(x[:-1])
        return m0, P0, Fs, Qs, bs

    def observations_factory(_x, u, delta):
        aux_ys = jnp.concatenate([u, ys], axis=1)
        aux_Hs = jnp.concatenate([eyes, Hs], axis=1)
        aux_cs = jnp.concatenate([jnp.zeros_like(u), cs], axis=1)
        aux_Rs = _batched_block_diag(0.5 * delta * eyes, Rs)
        return aux_ys, aux_Hs, aux_Rs, aux_cs

    def log_likelihood_fn(x):
        pred_x = jax.vmap(mean)(x[:-1], None)
        prior = mvn.logpdf(x[0], m0, chol_P0)
        prior += jnp.sum(mvn.logpdf(x[1:], pred_x, chol_Q))
        pred_y = jnp.einsum("ijk,ik->ij", Hs, x) + cs
        loglik = jnp.nansum(mvn.logpdf(ys, pred_y, chol_Rs))
        out = prior + loglik
        return out

    return get_generic_kernel(dynamics_factory, observations_factory, log_likelihood_fn, parallel)


def get_coupled_kernel(ys, Hs, Rs, cs, m0, P0, theta_1, theta_2, sigma_x, dt, parallel):
    eye = jnp.eye(3)
    eyes = jnp.tile(eye[None, ...], (ys.shape[0], 1, 1))

    mean_1, Q_1 = get_dynamics(theta_1, sigma_x, dt)
    mean_2, Q_2 = get_dynamics(theta_2, sigma_x, dt)

    cov_1 = lambda _x, _params: Q_1
    cov_2 = lambda _x, _params: Q_2

    chol_P0 = jnp.linalg.cholesky(P0)
    chol_Q_1 = jnp.linalg.cholesky(Q_1)
    chol_Q_2 = jnp.linalg.cholesky(Q_2)
    chol_Rs = jnp.linalg.cholesky(Rs)

    specialised_extended_1 = lambda x: extended(mean_1, cov_1, None, x, None)
    specialised_extended_2 = lambda x: extended(mean_2, cov_2, None, x, None)

    def dynamics_factory_1(x):
        Fs, Qs, bs = jax.vmap(specialised_extended_1)(x[:-1])
        return m0, P0, Fs, Qs, bs

    def dynamics_factory_2(x):
        Fs, Qs, bs = jax.vmap(specialised_extended_2)(x[:-1])
        return m0, P0, Fs, Qs, bs

    def observations_factory(_x, u, delta):
        aux_ys = jnp.concatenate([u, ys], axis=1)
        aux_Hs = jnp.concatenate([eyes, Hs], axis=1)
        aux_cs = jnp.concatenate([jnp.zeros_like(u), cs], axis=1)
        aux_Rs = _batched_block_diag(0.5 * delta * eyes, Rs)
        return aux_ys, aux_Hs, aux_Rs, aux_cs

    def log_likelihood_fn(x, mean, chol_Q):
        pred_x = jax.vmap(mean)(x[:-1], None)
        prior = mvn.logpdf(x[0], m0, chol_P0)
        prior += jnp.sum(mvn.logpdf(x[1:], pred_x, chol_Q))
        pred_y = jnp.einsum("ijk,ik->ij", Hs, x) + cs
        loglik = jnp.nansum(mvn.logpdf(ys, pred_y, chol_Rs))
        out = prior + loglik
        return out

    log_likelihood_fn_1 = lambda x: log_likelihood_fn(x, mean_1, chol_Q_1)
    log_likelihood_fn_2 = lambda x: log_likelihood_fn(x, mean_2, chol_Q_2)

    dynamics_factories = (dynamics_factory_1, dynamics_factory_2)
    log_likelihood_fns = (log_likelihood_fn_1, log_likelihood_fn_2)
    return get_generic_kernel(dynamics_factories, observations_factory, log_likelihood_fns, parallel)
