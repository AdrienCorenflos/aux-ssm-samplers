import jax
import jax.numpy as jnp
import numpy as np

from aux_samplers import mvn
from aux_samplers.kalman import get_kernel as get_generic_kernel
from model import log_potential


def get_kernel(ys, m0, P0, F, Q, b, parallel, log_potential_args, order=1, coupled=False, **coupling_kwargs):
    d = m0.shape[0]
    T = ys.shape[0]
    nu, prec = log_potential_args
    prec_diag = prec[np.diag_indices(d)].todense()

    # We batch the model

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
        grad_x = jax.grad(log_potential)(x.reshape(-1, d), ys, nu, prec)
        grad_x = grad_x.reshape(T, d, 1)
        grad_x = jnp.nan_to_num(grad_x)
        aux_ys = u + 0.5 * delta * grad_x
        Hs = eyes
        Rs = 0.5 * delta * eyes
        cs = zeros
        return aux_ys, Hs, Rs, cs

    def second_order_observations_factory(x, u, delta):
        # For a single time step!
        grad = jax.grad(log_potential)(x.reshape(-1, d), ys, nu, prec)
        grad = grad.reshape(T, d, 1)
        hess_diag_approx = - nu * prec_diag / (nu - 2)
        Omega_inv = -hess_diag_approx[None, ..., None, None] + 2 * eyes / delta
        Omega = 1. / Omega_inv
        aux_y = Omega[..., 0] * (2 * u / delta + grad - hess_diag_approx[None, ..., None] * x)
        return aux_y, eyes, Omega, zeros

    def log_likelihood_fn(x):
        out = jnp.sum(mvn.logpdf(x[0], m0, chol_P0))
        pred_x = jnp.einsum("...ij,...j->...i", F, x[:-1]) + b[None, ...]
        out += jnp.sum(mvn.logpdf(x[1:], pred_x, chol_Q))
        return out + log_potential(x.reshape(-1, d), ys, *log_potential_args)

    observations_factory = first_order_observations_factory if order == 1 else second_order_observations_factory
    init_, kernel = get_generic_kernel(dynamics_factory, observations_factory, log_likelihood_fn, parallel, coupled,
                                       **coupling_kwargs)

    if not coupled:
        def init(xs):
            return init_(jnp.expand_dims(xs, -1))
    else:
        def init(xs_1, xs_2):
            return init_(jnp.expand_dims(xs_1, -1), jnp.expand_dims(xs_2, -1))
    return init, kernel
