"""
Implementation of conditional resampling algorithms.
At the moment it only has conditional multinomial resampling as this is not the topic of the paper.
It could easily be extended to include other algorithms.
"""
from typing import Optional

import jax
import jax.numpy as jnp
from chex import PRNGKey
from jaxtyping import Array, Float, Int, PyTree

from aux_samplers._primitives.math.couplings import index_max_coupling


def multinomial(key: PRNGKey, weights: Float[Array, "dim_x"], N: Optional[int] = None) -> Int[Array, "dim_x"]:
    """
    Conditional multinomial resampling. The weights are assumed to be normalised already.
    The index 0 is always left unchanged.

    Parameters
    ----------
    key:
        Random number generator key.
    weights:
        Weights of the particles.
    N:
        Number of particles to resample.
    Returns
    -------
    indices:
        Indices of the resampled particles.
    """
    M = weights.shape[0]
    N = M if N is None else N

    indices = jax.random.choice(key, M, p=weights, shape=(N,), replace=True)
    indices = indices.at[0].set(0)
    return indices


def systematic(key: PRNGKey, weights: Float[Array, "dim_x"], N: Optional[int] = None) -> Int[Array, "dim_x"]:
    """
    Conditional systematic resampling. The weights are assumed to be normalised already.
    The index 0 is always left unchanged.

    Parameters
    ----------
    key:
        Random number generator key.
    weights:
        Weights of the particles.
    N:
        Number of particles to resample.

    Returns
    -------
    indices:
        Indices of the resampled particles.
    """
    M = weights.shape[0]
    N = M if N is None else N

    tmp = N * weights[0]
    tmp_floor = jnp.floor(tmp)

    U, V, W = jax.random.uniform(key, (3,))

    def _otherwise():
        rem = tmp - tmp_floor
        p_cond = rem * (tmp_floor + 1) / tmp
        return jax.lax.select(V < p_cond,
                              rem * U,
                              rem + (1. - rem) * U)

    uniform = jax.lax.cond(tmp <= 1,
                           lambda: tmp * U,
                           _otherwise)

    linspace = (jnp.arange(N, dtype=weights.dtype) + uniform) / N
    idx = jnp.searchsorted(jnp.cumsum(weights), linspace)

    n_zero = jnp.sum(idx == 0)
    zero_loc = jnp.flatnonzero(idx == 0, size=N, fill_value=-1)
    roll_idx = jnp.floor(n_zero * W).astype(int)

    idx = jax.lax.select(n_zero == 1, idx,  jnp.roll(idx, -zero_loc[roll_idx]))
    return jnp.clip(idx, 0, M - 1)


def coupled_multinomial(
        key: PRNGKey, weights_1: Float[Array, "dim_x"], weights_2: Float[Array, "dim_x"], N: Optional[int] = None
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
    N:
        Number of particles to resample.

    Returns
    -------
    indices_1:
        Indices of the first set of resampled particles.
    indices_2:
        Indices of the second set of resampled particles.
    coupled_flag:
        Flag indicating whether the coupling was successful or not.
    """
    idx_1, idx_2, coupled = index_max_coupling(key, weights_1, weights_2, N)

    idx_1 = idx_1.at[0].set(0)
    idx_2 = idx_2.at[0].set(0)
    coupled = coupled.at[0].set(True)
    return idx_1, idx_2, coupled
