"""Pytest file for MVN couplings."""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from jax import random

from .couplings import thorisson, rejection, lindvall_roger, modified_lindvall_roger


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_platform_name", "cpu")


@pytest.mark.parametrize("method", [thorisson, rejection, lindvall_roger, modified_lindvall_roger])
def test_couplings_full(method):
    """Test coupling."""
    key = random.PRNGKey(0)
    m1 = jnp.array([1., 1.])
    L1 = jnp.array([[1., 0.], [0.5, 1.]])
    m2 = jnp.array([0., 0.])
    L2 = jnp.array([[1., 0.], [0.75, 1.]])

    keys = jax.random.split(key, 500_000)

    x_1, x_2, coupled = jax.vmap(lambda k: method(k, m1, L1, m2, L2))(keys)
    npt.assert_allclose(x_1[coupled], x_2[coupled], atol=1e-6)
    npt.assert_array_less(0, np.abs(x_1[~coupled] - x_2[~coupled]))

    npt.assert_allclose(m1, jnp.mean(x_1, axis=0), atol=1e-2)
    npt.assert_allclose(m2, jnp.mean(x_2, axis=0), atol=1e-2)

    cov_1 = L1 @ L1.T
    cov_2 = L2 @ L2.T

    npt.assert_allclose(cov_1, jnp.cov(x_1.T), atol=1e-1)
    npt.assert_allclose(cov_2, jnp.cov(x_2.T), atol=1e-1)


@pytest.mark.parametrize("method", [thorisson, rejection, lindvall_roger, modified_lindvall_roger])
def test_couplings_diag(method):
    key = random.PRNGKey(0)
    m1 = jnp.array([1., 1.])
    L1 = jnp.array([1.5, 0.5])
    m2 = jnp.array([0., 0.])
    L2 = jnp.array([0.5, 1.])

    keys = jax.random.split(key, 500_000)

    x_1, x_2, coupled = jax.vmap(lambda k: method(k, m1, L1, m2, L2))(keys)
    npt.assert_allclose(x_1[coupled], x_2[coupled], atol=1e-6)
    npt.assert_array_less(0, np.abs(x_1[~coupled] - x_2[~coupled]))

    npt.assert_allclose(m1, jnp.mean(x_1, axis=0), atol=1e-2)
    npt.assert_allclose(m2, jnp.mean(x_2, axis=0), atol=1e-2)

    cov_1 = jnp.diag(L1 ** 2)
    cov_2 = jnp.diag(L2 ** 2)

    npt.assert_allclose(cov_1, jnp.cov(x_1.T), atol=1e-1)
    npt.assert_allclose(cov_2, jnp.cov(x_2.T), atol=1e-1)


@pytest.mark.parametrize("method", [thorisson, rejection, lindvall_roger, modified_lindvall_roger])
def test_couplings_scalars(method):
    key = random.PRNGKey(0)
    m1 = jnp.array([1., 1.])
    L1 = 1.5
    m2 = jnp.array([0., 0.])
    L2 = 0.5

    keys = jax.random.split(key, 500_000)

    x_1, x_2, coupled = jax.vmap(lambda k: method(k, m1, L1, m2, L2))(keys)
    npt.assert_allclose(x_1[coupled], x_2[coupled], atol=1e-6)
    npt.assert_array_less(0, np.abs(x_1[~coupled] - x_2[~coupled]))

    npt.assert_allclose(m1, jnp.mean(x_1, axis=0), atol=1e-2)
    npt.assert_allclose(m2, jnp.mean(x_2, axis=0), atol=1e-2)

    cov_1 = L1 ** 2 * jnp.eye(2)
    cov_2 = L2 ** 2 * jnp.eye(2)

    npt.assert_allclose(cov_1, jnp.cov(x_1.T), atol=1e-1)
    npt.assert_allclose(cov_2, jnp.cov(x_2.T), atol=1e-1)
