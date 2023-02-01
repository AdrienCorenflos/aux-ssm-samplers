from functools import partial

import jax
from chex import Array
from jax.experimental.sparse import BCOO
import jax.numpy as jnp
from jax import pure_callback
from jax.scipy.linalg import solve_triangular, cholesky

@partial(jnp.vectorize, signature="(k),(d),()->(d)", excluded=(3,))
def sample(key, mu:Array, nu:float, prec: BCOO):
    """
    Samples from a multivariate t-distribution.

    Parameters
    ----------
    key: PRNGKey
        The key used for sampling.
    mu: Array
        The mode of the distribution.
    nu: float
        The degrees of freedom of the distribution.
    prec: BCOO
        The precision matrix of the distribution in sparse format.

    Returns
    -------
    y: Array
        The sample from the distribution.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import multivariate_t
    >>> key = jax.random.PRNGKey(0)
    >>> mu = jnp.array([2, 3])
    >>> nu = 3
    >>> prec = np.array([[1, 0.2],
    ...                  [0.2, 1]])
    >>> coo_prec = jax.experimental.sparse.COO.fromdense(prec)
    >>> samples = sample(jax.random.split(key, 100_000), mu, nu, coo_prec)
    >>> np.allclose(samples.mean(axis=0), mu, atol=1e-2)
    True
    >>> np.allclose(np.cov(samples, rowvar=False), nu * np.linalg.inv(prec) / (nu - 2), atol=1e-1)
    True


    """
    key, subkey = jax.random.split(key)
    eps = jax.random.normal(key, shape=mu.shape)
    # This "to dense" is not ideal, but it's the only way to do it with the current API.
    chol_prec = cholesky(prec.todense(), lower=False)
    y = solve_triangular(chol_prec, eps, lower=False)
    u = 2 * jax.random.gamma(subkey, 0.5 * nu) / nu
    return mu + jnp.sqrt(1 / u) * y


@jax.jit
def log_pdf(x:Array, mu:Array, nu:float, prec: BCOO):
    """
    Computes the (unnormalised) log-likelihood of a multivariate t-distribution at x.

    Parameters
    ----------
    x: Array
        The key used for sampling.
    mu: Array
        The mode of the distribution.
    nu: float
        The degrees of freedom of the distribution.
    prec: BCOO
        The precision matrix of the distribution in sparse format.

    Returns
    -------
    y: Numeric
        The unnormalised logpdf.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import multivariate_t
    >>> xs = np.random.randn(100, 2)
    >>> mu = jnp.array([2, 3])
    >>> prec = np.array([[1, 0.2],
    ...                  [0.2, 1]])
    >>> coo_prec = jax.experimental.sparse.COO.fromdense(prec)
    >>> actual = log_pdf(xs, mu, 3, coo_prec)
    >>> expected = multivariate_t.logpdf(xs, loc=mu, df=3, shape=np.linalg.inv(prec))
    >>> ratio = np.exp(actual - expected)
    >>> np.allclose(ratio, ratio.mean(), atol=1e-5)
    True
    """
    x, mu = jnp.broadcast_arrays(x, mu)
    d = x.shape[-1]
    z = prec @ (x - mu).T
    norm = jnp.sum(z.T * (x - mu), -1)
    out = jnp.log(1 + norm / nu)
    out *= (nu + d) / 2
    return -out