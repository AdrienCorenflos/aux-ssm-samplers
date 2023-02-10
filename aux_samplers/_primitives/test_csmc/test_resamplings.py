"""Test file for resampling methods."""
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from aux_samplers._primitives.csmc.resamplings import multinomial, coupled_multinomial, systematic


def test_multinomial_resampling():
    """Test the conditional multinomial resampling."""
    JAX_KEY = jax.random.PRNGKey(42)
    data_key, key = jax.random.split(JAX_KEY)
    keys = jax.random.split(key, 100_000)

    weights = jax.random.uniform(data_key, shape=(10,))
    weights /= jnp.sum(weights)

    indices = jax.vmap(multinomial, in_axes=[0, None])(keys, weights)
    bincount = np.bincount(indices[:, 1:].ravel(), minlength=weights.shape[0])

    npt.assert_allclose(bincount / np.sum(bincount), weights, atol=1e-3)
    npt.assert_allclose(indices[:, 0], 0, atol=1e-3)


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("M", [100,1000])
def test_systematic_resampling(seed, M):
    """Test the conditional multinomial resampling."""
    N = 100
    JAX_KEY = jax.random.PRNGKey(seed)
    data_key, key = jax.random.split(JAX_KEY)
    n_keys = 100
    keys = jax.random.split(key, n_keys)

    weights = jax.random.uniform(data_key, shape=(M,))
    weights /= jnp.sum(weights)
    indices_test = np.empty((n_keys, N), dtype=np.int32)
    for i, key in enumerate(keys):
        indices_test[i] = _sys_resampling(weights, *jax.random.uniform(key, shape=(3,)), N)
    indices = jax.vmap(systematic, in_axes=[0, None, None])(keys, weights, N)

    npt.assert_allclose(indices, indices_test, atol=1e-3)
    npt.assert_allclose(indices[:, 0], 0, atol=1e-3)


def test_coupled_multinomial():
    """Test the coupled conditional multinomial resampling."""
    JAX_KEY = jax.random.PRNGKey(42)
    data_key_1, data_key_2, key = jax.random.split(JAX_KEY, 3)
    keys = jax.random.split(key, 100_000)

    weights_1 = jax.random.uniform(data_key_1, shape=(10,))
    weights_1 /= jnp.sum(weights_1)

    weights_2 = jax.random.uniform(data_key_2, shape=(10,))
    weights_2 /= jnp.sum(weights_2)

    indices_1, indices_2, coupled = jax.vmap(coupled_multinomial, in_axes=[0, None, None])(
        keys, weights_1, weights_2
    )

    bincount_1 = np.bincount(indices_1[:, 1:].ravel(), minlength=weights_1.shape[0])
    bincount_2 = np.bincount(indices_2[:, 1:].ravel(), minlength=weights_2.shape[0])

    npt.assert_allclose(bincount_1 / np.sum(bincount_1), weights_1, atol=1e-3)
    npt.assert_allclose(bincount_2 / np.sum(bincount_2), weights_2, atol=1e-3)

    npt.assert_allclose(indices_1[:, 0], 0, atol=1e-3)
    npt.assert_allclose(indices_2[:, 0], 0, atol=1e-3)

    coupled_avg = np.mean(coupled[:, 1:])
    npt.assert_allclose(coupled_avg, np.sum(np.minimum(weights_1, weights_2)), atol=1e-3)

    npt.assert_allclose(indices_1[coupled], indices_2[coupled], atol=1e-4)
    assert np.all(np.abs(indices_1[~coupled] - indices_2[~coupled]) >= 1)


def _sys_resampling(W, u, v, w, M):
    """
    Conditional systemic resampling

    This is Algorithm 4 of Chopin & Singh (doi:10.3150/14-BEJ629).
    """
    dim = len(W)
    nW1 = M * W[0]

    # Step (a)
    if nW1 <= 1.0:
        U = nW1 * u
    else:
        r1 = nW1 % 1

        if v < r1 * np.ceil(nW1) / nW1:
            U = r1 * u
        else:
            U = r1 + (1.0 - r1) * u

    # Step (b)
    linspace = (np.arange(M) + U) / M
    a_bar = np.searchsorted(np.cumsum(W), linspace)

    # Step (c)
    zero_loc = np.nonzero(a_bar == 0)
    n_zero = len(zero_loc)

    if n_zero == 1:
        return a_bar

    return np.roll(a_bar, -zero_loc[int(n_zero * w)])
