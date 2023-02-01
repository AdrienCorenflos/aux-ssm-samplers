from functools import partial

import jax
import numba as nb
import numpy as np
from jax.experimental.sparse.bcoo import BCOO


def make_chol_precision(tau, r_y, d):
    """
    Makes the sparse Cholesky decomposition for the precision matrix of the model.

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
    >>> d = 4
    >>> coo = make_precision(tau, r_y, d)
    >>> coo.todense().shape
    Array([[ 1.  , -0.25, -0.25,  0.  ],
           [-0.25,  1.  ,  0.  , -0.25],
           [-0.25,  0.  ,  1.  , -0.25],
           [ 0.  , -0.25, -0.25,  1.  ]], dtype=float32)
    """

    data, indices = _make_precision_np(tau, r_y, d)
    prec = BCOO((data, indices), shape=(d ** 2, d ** 2))  # type: ignore
    return prec

@nb.njit
def _make_precision_np(tau: float, r_y: float, d: int):
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


@partial(jax.jit, static_argnums=(4, 5))
def get_data(key, sigma_x, r_y, tau, d, T):
    pass


