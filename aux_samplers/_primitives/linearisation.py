import math
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from chex import ArrayTree, Array
from jax.scipy.linalg import cho_solve


def extended(mean: Callable, cov: Callable, params: Optional[ArrayTree], x_star: Array, _P_star: Optional[Array]):
    """
    First order extended linearisation. TODO: extend to second order.

    Parameters
    ----------
    mean: Callable
        Conditional mean E[X | x*], signature (x, params) -> x
    cov: Callable
        Conditional covariance Cov[X | x*], signature (x, params) -> Q
    params: ArrayTree, optional
        Parameters of the conditional mean and covariance functions, can be None
    x_star: Array
        The point at which the linearisation is performed
    _P_star: Array, optional
        This is not used at the moment and is only there for compatibility of the API.
        Ideally, we could implement a second order linearisation.

    Returns
    -------

    """

    b = mean(x_star, params)

    d_x = x_star.shape[0]
    d_y = b.shape[0]
    if d_y < d_x:
        F = jax.jacrev(mean, 0)(x_star, params)
    else:
        F = jax.jacfwd(mean, 0)(x_star, params)
    Q = cov(x_star, params)
    b = b - F @ x_star
    return F, Q, b


def gauss_hermite(mean: Callable, cov: Callable, params: Optional[ArrayTree], x_star: Array, P_star: Array, order=3):
    """
        Cubature method for linearisation.

        Parameters
        ----------
        mean: Callable
            Conditional mean E[X | x*], signature (x, params) -> x
        cov: Callable
            Conditional covariance Cov[X | x*], signature (x, params) -> Q
        params: ArrayTree
            Parameters of the conditional mean and covariance functions, can be None
        x_star: Array
            The point at which the linearisation is performed
        P_star: Array,
            Covariance used for defining the sigma points
        order: int
            Order of the Hermite polynomial used for the approximation.

        Returns
        -------
        F: Array
            Approximate transition matrix
        Q: Array
            Approximate additive noise
        b: Array
            Approximate bias
        """
    return _generic_sigma_points(mean, cov, params, x_star, P_star, lambda dim: _gauss_hermite_points(dim, order))


def cubature(mean: Callable, cov: Callable, params: Optional[ArrayTree], x_star: Array, P_star: Array):
    """
    Cubature method for linearisation.

    Parameters
    ----------
    mean: Callable
        Conditional mean E[X | x*], signature (x, params) -> x
    cov: Callable
        Conditional covariance Cov[X | x*], signature (x, params) -> Q
    params: ArrayTree
        Parameters of the conditional mean and covariance functions, can be None
    x_star: Array
        The point at which the linearisation is performed
    P_star: Array,
        Covariance used for defining the sigma points

    Returns
    -------
    F: Array
        Approximate transition matrix
    Q: Array
        Approximate additive noise
    b: Array
        Approximate bias
    """
    return _generic_sigma_points(mean, cov, params, x_star, P_star, _cubature_points)


def _generic_sigma_points(mean, cov, params, x_star, P_star, get_sigma_points):
    chol = jnp.linalg.cholesky(P_star)
    dim = x_star.shape[0]
    w, xi = get_sigma_points(dim)
    points = x_star[None, :] + jnp.dot(chol, xi).T

    f_pts = jax.vmap(mean, in_axes=[0, None])(points, params)
    m_f = jnp.dot(w, f_pts)

    Psi_x = _cov(w, points, x_star, f_pts, m_f)
    F_x = cho_solve((chol, True), Psi_x).T

    v_pts = jax.vmap(cov, in_axes=[0, None])(points, params)
    v_f = jnp.sum(w[:, None, None] * v_pts, 0)

    Phi = _cov(w, f_pts, m_f, f_pts, m_f)

    temp = F_x @ chol
    L = Phi - temp @ temp.T + v_f

    return F_x, L, m_f - F_x @ x_star


def _cov(wc, x_pts, x_mean, y_points, y_mean):
    one = (x_pts - x_mean[None, :]).T * wc[None, :]
    two = y_points - y_mean[None, :]
    return jnp.dot(one, two)


def _gauss_hermite_points(n_dim: int, order: int = 3):
    """ Computes the weights associated with the Gauss--Hermite quadrature method.
    The Hermite polynomial is in the physician version.
    This is coded in pure numpy, not JAX, so that it will only be computed once.

    Parameters
    ----------
    n_dim: int
        Dimensionality of the problem
    order: int, optional, default is 3
        The order of Hermite polynomial
    Returns
    -------
    w: np.ndarray
        Weights
    xi: np.ndarray
        Orthogonal vectors
    References
    ----------
    .. [1] Simo Särkkä.
       *Bayesian Filtering and Smoothing.*
       In: Cambridge University Press 2013.
    """
    n = n_dim
    p = order

    hermite_coeff = _hermite_coeff(p)
    hermite_roots = np.flip(np.roots(hermite_coeff[-1]))

    table = np.zeros(shape=(n, p ** n))

    w_1d = np.zeros(shape=(p,))
    for i in range(p):
        w_1d[i] = (2 ** (p - 1) * np.math.factorial(p) * np.sqrt(np.pi) /
                   (p ** 2 * (np.polyval(hermite_coeff[p - 1],
                                         hermite_roots[i])) ** 2))

    # Get roll table
    for i in range(n):
        base = np.ones(shape=(1, p ** (n - i - 1)))
        for j in range(1, p):
            base = np.concatenate([base,
                                   (j + 1) * np.ones(shape=(1, p ** (n - i - 1)))],
                                  axis=1)
        table[n - i - 1, :] = np.tile(base, (1, int(p ** i)))

    table = table.astype("int64") - 1

    s = 1 / (np.sqrt(np.pi) ** n)

    w = s * np.prod(w_1d[table], axis=0)
    xi = math.sqrt(2) * hermite_roots[table]

    return w, xi


def _hermite_coeff(order: int):
    """ Give the 0 to p-th order physician Hermite polynomial coefficients, where p is the
    order from the argument. The returned coefficients is ordered from highest to lowest.
    Also note that this implementation is different from the np.hermite method.
    This is coded in pure numpy, not JAX, so that it will only be computed once.


    Parameters
    ----------
    order:  int
        The order of Hermite polynomial
    Returns
    -------
    H: List
        The 0 to p-th order Hermite polynomial coefficients in a list.
    """
    H0 = np.array([1])
    H1 = np.array([2, 0])

    H = [H0, H1]

    for i in range(2, order + 1):
        H.append(2 * np.append(H[i - 1], 0) -
                 2 * (i - 1) * np.pad(H[i - 2], (2, 0), 'constant', constant_values=0))

    return H


def _cubature_points(n_dim: int):
    """ Computes the weights associated with the spherical cubature method.
    The number of sigma-points is 2 * n_dim
    This is coded in pure numpy, not JAX, so that it will only be computed once.

    Parameters
    ----------
    n_dim: int
        Dimensionality of the problem
    Returns
    -------
    wm: np.ndarray
        Weights means
    wc: np.ndarray
        Weights covariances
    xi: np.ndarray
        Orthogonal vectors
    """
    w = np.ones(shape=(2 * n_dim,)) / (2 * n_dim)
    xi = np.concatenate([np.eye(n_dim), -np.eye(n_dim)], axis=0) * np.sqrt(n_dim)

    return w, xi.T
