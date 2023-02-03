from functools import partial

import jax
import numba as nb
import numpy as np
import jax.numpy as jnp
from jax.experimental.sparse.bcoo import BCOO
from t_distribution import sample as t_sample, logpdf as t_log_pdf
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import inv
from scipy.stats import multivariate_t as scipy_t

def make_precision(tau, r_y, d):
    """
    Makes the sparse precision matrix of the model.

    The resulting precision will be given as tau^D((i,j),(i',j')) if D((i,j),(i',j')) < r_y, 0 otherwise, for
    D((i,j),(i',j')) = |i-i'| + |j-j'|.

    Parameters
    ----------
    tau : float
        The parameter used for filling the matrix.
    r_y : float
        The radius of the spatial correlation in terms of Hamming ball.
    d : int
        The dimension of the grid. The resulting matrix will be d^2 x d^2 albeit sparse.

    Returns
    -------
    prec : BCOO
        The sparse precision matrix.

    Examples:
    ---------
    >>> tau = -0.25
    >>> r_y = 1
    >>> d = 2
    >>> coo = make_precision(tau, r_y, d)
    >>> coo.todense()
    Array([[ 1.  , -0.25, -0.25,  0.  ],
           [-0.25,  1.  ,  0.  , -0.25],
           [-0.25,  0.  ,  1.  , -0.25],
           [ 0.  , -0.25, -0.25,  1.  ]], dtype=float32)
    """
    data, indices = _make_precision_np_coo(tau, r_y, d)
    prec = BCOO((data, indices), shape=(d ** 2, d ** 2))  # type: ignore
    return prec

@nb.njit
def _make_precision_np_coo(tau: float, r_y: float, d: int):
    data = []
    indices = []

    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    D_val = abs(i - k) + abs(j - l)
                    if D_val <= r_y:
                        data.append(tau ** D_val)
                        indices.append([i * d + j, k * d + l])

    data = np.array(data)
    indices = np.array(indices, dtype=np.int_)
    return data, indices

@nb.njit
def _make_precision_np_csr(tau: float, r_y: float, d: int):
    data = []
    indices = []

    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    D_val = abs(i - k) + abs(j - l)
                    if D_val <= r_y:
                        data.append(tau ** D_val)
                        indices.append([i * d + j, k * d + l])

    data = np.array(data)
    indices = np.array(indices, dtype=np.int_)
    return data, indices


def get_data(state, sigma_x, r_y, tau, nu, d, T):
    data, indices = _make_precision_np_coo(tau, r_y, d)
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
    vals = jax.vmap(_log_potential_one, in_axes=[0, 0, None, None])(xs, ys, nu, prec)
    return jnp.sum(vals)


@jax.jit
def grad_log_potential(xs, ys):
    return jax.grad(log_potential)(xs, ys)



@jax.jit
def _log_potential_one(x, y, nu, prec):
    val = t_log_pdf(y, x, nu, prec)
    return jnp.nan_to_num(val)

