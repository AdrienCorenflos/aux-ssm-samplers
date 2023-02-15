import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag

from aux_samplers import mvn, extended
from aux_samplers.kalman import get_kernel as get_generic_kernel
from model import get_dynamics, observations_model


def _batched_block_diag(a, b):
    return jax.vmap(block_diag)(a, b)


def get_kernel(data, theta, sigma_x, sigma_y, sampling_freq, parallel):
    ts, ys = data[:, 0], data[:, 1:]
    obs_freq = ts[1] - ts[0]

    ys, Hs, Rs, cs, _ = observations_model(ys, sigma_y, obs_freq, sampling_freq, ts[-1])

    eye = jnp.eye(3)
    eyes = jnp.tile(eye[None, ...], (ys.shape[0], 1, 1))

    mean, Q = get_dynamics(theta, sigma_x, sampling_freq)
    cov = lambda _x, _params: Q

    m0 = jnp.array([1.5, -1.5, 25])
    P0 = jnp.diag(jnp.array[400, 20, 20])
    chol_P0 = jnp.linalg.cholesky(P0)
    chol_Q = jnp.linalg.cholesky(Q)
    chol_Rs = jnp.linalg.cholesky(Rs)
    specialised_extended = lambda x: extended(mean, cov, None, x, None)

    def dynamics_factory(x):
        Fs, Qs, bs = jax.vmap(specialised_extended)(x)
        return m0, P0, Fs, Qs, bs

    def observations_factory(_x, u, delta):
        aux_ys = jnp.concatenate([u, ys], axis=1)
        aux_Hs = jnp.concatenate([eyes, Hs], axis=2)
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
