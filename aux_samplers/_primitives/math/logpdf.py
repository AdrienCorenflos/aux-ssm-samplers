import math
from functools import partial

import numpy
import numpy as np
import scipy
from jax import numpy as jnp
from jax._src.scipy.linalg import solve_triangular
from jaxtyping import Float, Array
from scipy.stats import multivariate_normal


@partial(jnp.vectorize, signature="(n),(n),(n,n)->()")
def mvn_loglikelihood(
        x: Float[Array, "dim"], mean: Float[Array, "dim"], chol_cov: Float[Array, "dim dim"]
) -> Float[Array, ""]:
    """
    Computes the log-likelihood of a multivariate normal distribution.

    Parameters
    ----------
    x : Array
        The data to compute the log-likelihood for.
    mean : Array
        The mean of the multivariate normal distribution.
    chol_cov : Array
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
    >>> m = jnp.array([2, 3, 4])
    >>> chol = jnp.array([[1, 0, 0], [0.2, 1.3, 0], [0.123, -0.5, 1.7]])
    >>> np.allclose(mvn_loglikelihood(z, m, chol), multivariate_normal.logpdf(z, m, chol @ chol.T))
    True
    """

    dim = x.shape[0]

    y = solve_triangular(chol_cov, x - mean, lower=True)

    diag = jnp.abs(jnp.diag(chol_cov))
    normalizing_constant = jnp.sum(jnp.log(diag)) + 0.5 * dim * math.log(2 * math.pi)
    norm_y = jnp.sum(y * y)

    return -0.5 * norm_y - normalizing_constant
