"""Test file for resampling methods."""
import jax
import jax.numpy as jnp
import numpy.testing as npt
import numpy as np
from aux_samplers._primitives.csmc.resamplings import multinomial, coupled_multinomial


def test_conditional_multinoimal():
    """Test the conditional multinomial resampling."""
    JAX_KEY = jax.random.PRNGKey(0)
    data_key, key = jax.random.split(JAX_KEY)
    keys = jax.random.split(key, 100_000)

    weights = jax.random.uniform(data_key, shape=(10,))
    weights /= jnp.sum(weights)

    indices = jax.vmap(multinomial, in_axes=[0, None])(keys, weights)
    bincount = np.bincount(indices[:, 1:].ravel(), minlength=weights.shape[0])

    npt.assert_allclose(bincount / np.sum(bincount), weights, atol=1e-3)
    npt.assert_allclose(indices[:, 0], 0, atol=1e-3)


def test_coupled_multinomial():
    """Test the coupled conditional multinomial resampling."""
    JAX_KEY = jax.random.PRNGKey(0)
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
