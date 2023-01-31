import math
from functools import partial

import jax
import jax.numpy as jnp

LOG_HALF = math.log(0.5)


@partial(jnp.vectorize, signature="(),()->()")
def logsubexp(x1, x2):
    amax = jnp.maximum(x1, x2)
    delta = jnp.abs(x1 - x2)
    return amax + log1mexp(-abs(delta))


@jax.jit
def log1mexp(x):
    return jnp.where(x < LOG_HALF, jnp.log1p(-jnp.exp(x)), jnp.log(-jnp.expm1(x)))


@jax.jit
def normalize(log_weights):
    """
    Normalize log weights to obtain unnormalized weights.

    Parameters
    ----------
    log_weights : Array
        Log weights.

    Returns
    -------
    weights : Array
        Unnormalized weights.
    """
    weights = jnp.exp(log_weights - jnp.max(log_weights))
    weights /= jnp.sum(weights)

    return weights


@partial(jnp.vectorize, signature="(d,d)->(d,d)")
def cholesky(P):
    """
    A wrapper for cholesky that handles numerical issues when P is too close to being 0.

    Parameters
    ----------
    P : Array
        A (supposedly) positive definite matrix.

    Returns
    -------
    L : Array
        Cholesky decomposition of P.
    """

    is_gpu = "gpu" in jax.default_backend()

    if is_gpu:
        u, s, vh = jnp.linalg.svd(P, hermitian=True)
        s = jnp.maximum(s, 0.)
        P = (u * s[:, None]) @ vh
        P = 0.5 * (P + P.T)
    chol_P = jnp.linalg.cholesky(P)
    return chol_P
