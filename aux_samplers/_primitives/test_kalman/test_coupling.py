import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from .common import explicit_kalman_smoothing
from ..kalman.base import LGSSM
from ..kalman.coupled_sampling import divide_and_conquer, progressive
from ..kalman.filtering import filtering


@pytest.fixture(scope="module", autouse=True)
def jax_config():
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("seed", [123])
@pytest.mark.parametrize("T", [3, 5])
@pytest.mark.parametrize("dx", [1, 2])
@pytest.mark.parametrize("dy", [3])
@pytest.mark.parametrize("mode", ["dnc", "sequential"])
@pytest.mark.parametrize("method", ["rejection", "thorisson", "lindvall-roger"])
def test_parallel_vs_sequential(seed, T, dx, dy, mode, method):
    np.random.seed(seed)
    jax_key = jax.random.PRNGKey(seed)
    sampling_keys = jax.random.split(jax_key, 250_000)  # this may be a bit much for a unittest, but meh, it's fast...

    m0 = np.random.randn(dx)
    P0 = np.random.randn(dx, 5 * dx)
    P0 = P0 @ P0.T

    Fs_1 = np.random.randn(T - 1, dx, dx)
    Qs_1 = np.random.randn(T - 1, dx, 5 * dx)
    Qs_1 = Qs_1 @ Qs_1.transpose((0, 2, 1))
    bs_1 = np.random.randn(T - 1, dx)

    Fs_2 = np.random.randn(T - 1, dx, dx)
    Qs_2 = np.random.randn(T - 1, dx, 5 * dx)
    Qs_2 = Qs_2 @ Qs_2.transpose((0, 2, 1))
    bs_2 = np.random.randn(T - 1, dx)

    Hs = np.random.randn(T, dy, dx)
    Rs = np.random.randn(T, dy, 5 * dy)
    Rs = Rs @ Rs.transpose((0, 2, 1))
    cs = np.random.randn(T, dy)

    ys = np.random.randn(T, dy)

    lgssm_1 = LGSSM(m0, P0, Fs_1, Qs_1, bs_1, Hs, Rs, cs)
    lgssm_2 = LGSSM(m0, P0, Fs_2, Qs_2, bs_2, Hs, Rs, cs)
    ms_1, Ps_1, _ = filtering(ys, lgssm_1, False)
    ms_2, Ps_2, _ = filtering(ys, lgssm_2, False)

    if mode == "dnc":
        sampling_fn = lambda key: divide_and_conquer(key, lgssm_1, lgssm_2, ms_1, Ps_1, ms_2, Ps_2, method=method)
    elif mode == "sequential":
        sampling_fn = lambda key: progressive(key, lgssm_1, lgssm_2, ms_1, Ps_1, ms_2, Ps_2, method=method)
    else:
        raise ValueError("Unknown mode.")
    samples_1, samples_2, coupled_index = jax.vmap(sampling_fn)(sampling_keys)

    # check that the samples are actually coupled
    print(np.mean(coupled_index, 0))
    npt.assert_allclose(samples_1[coupled_index], samples_2[coupled_index])
    assert np.all(~np.equal(samples_1[~coupled_index], samples_2[~coupled_index]))

    expected_ms_1, expected_Ps_1 = explicit_kalman_smoothing(ms_1, Ps_1, Fs_1, Qs_1, bs_1)
    expected_ms_2, expected_Ps_2 = explicit_kalman_smoothing(ms_2, Ps_2, Fs_2, Qs_2, bs_2)

    npt.assert_allclose(expected_ms_1, samples_1.mean(0), atol=1e-1, rtol=1e-2)
    npt.assert_allclose(expected_ms_2, samples_2.mean(0), atol=1e-1, rtol=1e-2)

    covs_1 = jax.vmap(lambda z: jnp.cov(z, rowvar=False), in_axes=[1])(samples_1)
    covs_2 = jax.vmap(lambda z: jnp.cov(z, rowvar=False), in_axes=[1])(samples_2)

    covs_1 = covs_1.reshape((T, dx, dx))
    covs_2 = covs_2.reshape((T, dx, dx))

    npt.assert_allclose(expected_Ps_1, covs_1, atol=1e-1, rtol=1e-2)
    npt.assert_allclose(expected_Ps_2, covs_2, atol=1e-1, rtol=1e-2)
