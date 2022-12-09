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

