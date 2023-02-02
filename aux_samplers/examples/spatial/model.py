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


def get_data(seed, sigma_x, r_y, tau, nu, d, T):
    state = np.random.RandomState(seed)
    data, indices = _make_precision_np_coo(tau, r_y, d)
    coo_prec = coo_matrix((data, (indices[:, 0], indices[:, 1])), shape=(d ** 2, d ** 2))
    coo_cov = inv(coo_prec.tocsc())
    # I don't like that, but not sure how to do it otherwise. This won't happen in the real world.
    cov = coo_cov.todense()
    xs = sigma_x * state.normal(size=(T, d ** 2))
    xs = jnp.cumsum(xs, axis=0)
    ys = xs + scipy_t.rvs(shape=cov, df=nu, size=(T,), random_state=state)
    return xs, ys

if __name__  == "__main__":
    jax.config.update("jax_platform_name", "cpu")
    XS, YS = get_data(0, 1.0, 1.0, -0.25, 3.0, 16, 1_000)
    print(XS.shape, YS.shape)


