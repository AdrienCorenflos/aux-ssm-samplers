import math
from functools import partial

import jax
from chex import PRNGKey
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular
from jaxtyping import Float, Array


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

    diag = jnp.abs(jnp.diag(chol))
    normalizing_constant = jnp.sum(jnp.log(diag)) + 0.5 * dim * math.log(2 * math.pi)
    norm_y = jnp.sum(y * y)

    return -0.5 * norm_y - normalizing_constant


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
    return m + chol @ jax.random.normal(key, shape=m.shape)
