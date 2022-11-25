"""
Implementation of conditional resampling algorithms.
At the moment it only has conditional multinomial resampling as this is not the topic of the paper.
It could easily be extended to include other algorithms.
"""
import jax
import jax.numpy as jnp
from chex import PRNGKey
from jaxtyping import Array, Float, Int, PyTree


def multinomial(key: PRNGKey, weights: Float[Array, "dim_x"]) -> Int[Array, "dim_x"]:
    """
    Conditional multinomial resampling. The weights are assumed to be normalised already.
    The index 0 is always left unchanged.

    Parameters
    ----------
    key:
        Random number generator key.
    weights:
        Weights of the particles.

    Returns
    -------
    indices:
        Indices of the resampled particles.
    """
    indices = jax.random.choice(key, weights.shape[0], p=weights, shape=weights.shape, replace=True)
    indices = indices.at[0].set(0)
    return indices


def coupled_multinomial(
        key: PRNGKey, weights_1: Float[Array, "dim_x"], weights_2: Float[Array, "dim_x"]
) -> PyTree[Int[Array, "dim_x"]]:
    """
    Maximal coupling of conditional multinomial resampling schemes. The weights are assumed to be normalised already.
    The index 0 is always left unchanged.

    Parameters
    ----------
    key:
        Random number generator key.
    weights_1:
        First set of weights.
    weights_2:
        Second set of weights.

    Returns
    -------
    indices_1:
        Indices of the first set of resampled particles.
    indices_2:
        Indices of the second set of resampled particles.
    coupled_flag
    """
    key_1, key_2, key_3 = jax.random.split(key, 3)
    n = weights_1.shape[0]

    nu = jnp.minimum(weights_1, weights_2)
    alpha = jnp.sum(nu)

    weights_1, weights_2 = (weights_1 - nu) / (1 - alpha), (weights_2 - nu) / (1 - alpha)
    nu /= alpha

    coupled = jax.random.uniform(key_1, shape=(n,)) < alpha
    where_coupled = jax.random.choice(key_2, n, p=nu, shape=(n,), replace=True)

    where_uncoupled_1 = jax.random.choice(key_3, n, p=weights_1, shape=(n,), replace=True)
    where_uncoupled_2 = jax.random.choice(key_3, n, p=weights_2, shape=(n,), replace=True)

    indices_1 = jnp.where(coupled, where_coupled, where_uncoupled_1)
    indices_2 = jnp.where(coupled, where_coupled, where_uncoupled_2)

    indices_1 = indices_1.at[0].set(0)
    indices_2 = indices_2.at[0].set(0)

    coupled = coupled.at[0].set(True)
    return indices_1, indices_2, coupled
