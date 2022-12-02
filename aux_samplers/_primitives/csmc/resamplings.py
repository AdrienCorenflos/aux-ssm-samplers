"""
Implementation of conditional resampling algorithms.
At the moment it only has conditional multinomial resampling as this is not the topic of the paper.
It could easily be extended to include other algorithms.
"""

import jax
from chex import PRNGKey
from jaxtyping import Array, Float, Int, PyTree

from aux_samplers._primitives.math.generic_couplings import index_max_coupling


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
    coupled_flag:
        Flag indicating whether the coupling was successful or not.
    """
    idx_1, idx_2, coupled = index_max_coupling(key, weights_1, weights_2)

    idx_1 = idx_1.at[0].set(0)
    idx_2 = idx_2.at[0].set(0)
    coupled = coupled.at[0].set(True)
    return idx_1, idx_2, coupled

