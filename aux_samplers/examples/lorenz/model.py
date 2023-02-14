from functools import partial

import jax
import jax.numpy as jnp
import numba as nb
import numpy as np
from jax.experimental.sparse.bcoo import BCOO
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import inv
from scipy.stats import multivariate_t as scipy_t


def get_data(x0, sigma_x, theta, T):
    coo_prec = coo_matrix((data, (indices[:, 0], indices[:, 1])), shape=(d ** 2, d ** 2))
    coo_cov = inv(coo_prec.tocsc())
    # I don't like that, but not sure how to do it otherwise. This won't happen in the real world.
    cov = coo_cov.todense()
    xs = sigma_x * state.normal(size=(T, d ** 2))
    xs = jnp.cumsum(xs, axis=0)
    ys = xs + scipy_t.rvs(shape=cov, df=nu, size=(T,), random_state=state)
    return xs, ys


@partial(jax.jit, static_argnums=(1,))
def get_dynamics(sigma_x, d):
    # Batched version of the dynamics
    F = jnp.ones((d ** 2, 1, 1))
    Q = sigma_x ** 2 * jnp.ones((d ** 2, 1, 1))
    b = jnp.zeros((d ** 2, 1))

    m0 = b
    P0 = Q
    return m0, P0, F, Q, b


@jax.jit
def log_potential(xs, ys, nu, prec):
    vals = jax.vmap(log_potential_one, in_axes=[0, 0, None, None])(xs, ys, nu, prec)
    return jnp.sum(vals)


@jax.jit
def log_potential_one(x, y, nu, prec):
    val = t_log_pdf(y, x, nu, prec)
    return jnp.nan_to_num(val)


def init_x_fn(key, ys, sigma_x, nu, prec, N):
    # run a simple bootstrap filter + backward sampling.
    T, d = ys.shape
    init_key, key = jax.random.split(key)
    x0 = sigma_x * jax.random.normal(init_key, shape=(N, d))
    fwd_key, bwd_key = jax.random.split(key)

    def fwd_body(x, inps):
        y, op_key = inps
        op_key_1, op_key_2 = jax.random.split(op_key)
        log_w = log_potential_one(x, y, nu, prec)
        log_w = log_w - jax.scipy.special.logsumexp(log_w)
        w = jnp.exp(log_w)
        # systematic resampling
        u = jax.random.uniform(op_key_1, shape=())
        linspace = (u + jnp.arange(N)) / N
        ancestors = jnp.searchsorted(jnp.cumsum(w, axis=-1), linspace)
        next_x = x[ancestors] + sigma_x * jax.random.normal(op_key_2, shape=(N, d))
        return next_x, (log_w, x)

    _, (log_ws, xs) = jax.lax.scan(fwd_body, x0, (ys, jax.random.split(fwd_key, T)))

    def bwd_body(x, inps):
        log_w, x_prev, op_key = inps
        log_w += jnp.sum(jax.scipy.stats.norm.logpdf(x, x_prev, sigma_x), -1)
        w = jnp.exp(log_w - jax.scipy.special.logsumexp(log_w))
        x = jax.random.choice(op_key, x_prev, shape=(), p=w)
        return x, x

    bwd_key_init, bwd_key_loop = jax.random.split(bwd_key)
    x_T = jax.random.choice(bwd_key_init, xs[-1], shape=(), p=jnp.exp(log_ws[-1]))
    _, xs = jax.lax.scan(bwd_body, x_T, (log_ws[:-1], xs[:-1], jax.random.split(bwd_key_loop, T - 1)), reverse=True)
    xs = jnp.concatenate([xs, x_T[None, :]], axis=0)
    return xs
