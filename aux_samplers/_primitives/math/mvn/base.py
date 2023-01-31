import math
from functools import partial

import jax
from chex import PRNGKey
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular, eigh
from jaxtyping import Float, Array

_MIN_LOGPDF = -300
_MAX_LOGPDF = 300


@partial(jnp.vectorize, signature="(n),(n),(n,n)->()")
def logpdf(
        x: Float[Array, "dim"], m: Float[Array, "dim"], chol: Float[Array, "dim dim"]
) -> Float[Array, ""]:
    """
    Computes the log-likelihood of a multivariate normal distribution.

    Parameters
    ----------
    x : Array
        The data to compute the log-likelihood for.
    m : Array
        The mean of the multivariate normal distribution.
    chol : Array
        The Cholesky decomposition of the covariance matrix of
        the multivariate normal distribution.

    Returns
    -------
    y: Numeric
        The log-likelihood of the multivariate normal distribution.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import multivariate_normal
    >>> z = jnp.array([1, 2, 3])
    >>> mu = jnp.array([2, 3, 4])
    >>> L = jnp.array([[1, 0, 0], [0.2, 1.3, 0], [0.123, -0.5, 1.7]])
    >>> np.allclose(logpdf(z, mu, L), multivariate_normal.logpdf(z, mu, L @ L.T))
    True
    """

    dim = x.shape[0]

    y = solve_triangular(chol, x - m, lower=True)

    normalizing_constant = tril_log_det(chol) + 0.5 * dim * math.log(2 * math.pi)
    norm_y = jnp.sum(y * y)

    return jnp.clip(-0.5 * norm_y - normalizing_constant, _MIN_LOGPDF, _MAX_LOGPDF)


def rvs(key: PRNGKey, m: Float[Array, "dim"], chol: Float[Array, "dim dim"]) -> Float[Array, "dim"]:
    """
    Samples from the multivariate normal distribution.

    Parameters
    ----------
    key: PRNGKey
        Random number generator key.
    m: Array
        Mean of the multivariate normal distribution.
    chol: Array
        Cholesky decomposition of the covariance matrix of the multivariate normal distribution.
    """
    eps = jax.random.normal(key, shape=m.shape)
    return m + jnp.einsum("...ij,...j->...i", chol, eps)


def get_optimal_covariance(chol_P: Float[Array, "dim dim"], chol_Sig: Float[Array, "dim dim"]):
    """
    Get the optimal covariance according to the objective defined in Section 3 of [1].

    The notations roughly follow the ones in the article.

    Parameters
    ----------
    chol_P:
        Square root of the covariance of X. Lower triangular.
    chol_Sig:
        Square root of the covariance of Y. Lower triangular.

    Returns
    -------
    chol_Q: jnp.ndarray
        Cholesky of the resulting dominating matrix.
    """
    d, _ = chol_P.shape
    if d == 1:
        return jnp.maximum(chol_P, chol_Sig)

    right_Y = solve_triangular(chol_P, chol_Sig, lower=True)  # Y = RY.T RY
    w_Y, v_Y = eigh(right_Y.T @ right_Y)  # is there better than this ?
    w_Y = jnp.minimum(w_Y, 1)
    i_w_Y = 1. / jnp.sqrt(w_Y)

    left_Q = chol_Sig @ (v_Y * i_w_Y[None, :])
    return jnp.linalg.cholesky(left_Q @ left_Q.T)


def tril_log_det(chol):
    """
    Computes the log determinant of a lower triangular matrix.

    Parameters
    ----------
    chol: Array
        Lower triangular matrix.

    Returns
    -------
    log_det: Numeric
        Log determinant of the matrix.
    """

    return jnp.sum(jnp.log(jnp.abs(jnp.diag(chol))))
