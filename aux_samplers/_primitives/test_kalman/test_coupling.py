from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from scipy.linalg import block_diag

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


@pytest.mark.parametrize("seed", [123])
@pytest.mark.parametrize("T", [3])
@pytest.mark.parametrize("dx", [2])
@pytest.mark.parametrize("dy", [3])
@pytest.mark.parametrize("mode", ["dnc", "sequential"])
@pytest.mark.parametrize("method", ["lindvall-roger"])
def test_batched_model(seed, T, dx, dy, mode, method):
    # Compare the batched version of the coupled sampling to the non-batched version.
    # In the coupled case, contrary to the standard one, we can't expect the same exact samples given the same seed.
    # This is because the Gaussians generated for the marginal couplings will in general not be the same ones...
    B = 2
    np.random.seed(seed)
    jax_key = jax.random.PRNGKey(seed)
    sampling_keys = jax.random.split(jax_key, 500_000)  # this may be a bit much for a unittest, but meh, it's fast...

    bm0 = np.random.randn(B, dx)
    bP0 = np.random.randn(B, dx, 5 * dx)
    bP0 = bP0 @ bP0.transpose((0, 2, 1))

    bFs_1 = np.random.randn(T - 1, B, dx, dx)
    bQs_1 = np.random.randn(T - 1, B, dx, 5 * dx)
    bQs_1 = bQs_1 @ bQs_1.transpose((0, 1, 3, 2))
    bbs_1 = np.random.randn(T - 1, B, dx)

    bFs_2 = np.random.randn(T - 1, B, dx, dx)
    bQs_2 = np.random.randn(T - 1, B, dx, 5 * dx)
    bQs_2 = bQs_2 @ bQs_2.transpose((0, 1, 3, 2))
    bbs_2 = np.random.randn(T - 1, B, dx)

    bHs = np.random.randn(T, B, dy, dx)
    bRs = np.random.randn(T, B, dy, 5 * dy)
    bRs = bRs @ bRs.transpose((0, 1, 3, 2))
    bcs = np.random.randn(T, B, dy)

    bys = np.random.randn(T, B, dy)

    @partial(np.vectorize, signature="(b,i,j)->(k,l)")
    def batched_block_diag(a):
        return block_diag(*a)

    m0 = np.reshape(bm0, (B * dx,))
    P0 = block_diag(*bP0)
    Fs_1 = batched_block_diag(bFs_1)
    Qs_1 = batched_block_diag(bQs_1)
    bs_1 = np.reshape(bbs_1, (T - 1, B * dx))
    Fs_2 = batched_block_diag(bFs_2)
    Qs_2 = batched_block_diag(bQs_2)
    bs_2 = np.reshape(bbs_2, (T - 1, B * dx))
    Hs = batched_block_diag(bHs)
    Rs = batched_block_diag(bRs)
    cs = np.reshape(bcs, (T, B * dy))
    ys = np.reshape(bys, (T, B * dy))

    lgssm_1 = LGSSM(m0, P0, Fs_1, Qs_1, bs_1, Hs, Rs, cs)
    lgssm_2 = LGSSM(m0, P0, Fs_2, Qs_2, bs_2, Hs, Rs, cs)
    batched_lgssm_1 = LGSSM(bm0, bP0, bFs_1, bQs_1, bbs_1, bHs, bRs, bcs)
    batched_lgssm_2 = LGSSM(bm0, bP0, bFs_2, bQs_2, bbs_2, bHs, bRs, bcs)

    ms_1, Ps_1, _ = filtering(ys, lgssm_1, False)
    ms_2, Ps_2, _ = filtering(ys, lgssm_2, False)
    bms_1, bPs_1, _ = filtering(bys, batched_lgssm_1, False)
    bms_2, bPs_2, _ = filtering(bys, batched_lgssm_2, False)

    if mode == "dnc":
        sampling_fn = lambda key: divide_and_conquer(key, lgssm_1, lgssm_2, ms_1, Ps_1, ms_2, Ps_2, method=method)
        batched_sampling_fn = lambda key: divide_and_conquer(key, batched_lgssm_1, batched_lgssm_2, bms_1, bPs_1, bms_2,
                                                             bPs_2, method=method)
    elif mode == "sequential":
        sampling_fn = lambda key: progressive(key, lgssm_1, lgssm_2, ms_1, Ps_1, ms_2, Ps_2, method=method)
        batched_sampling_fn = lambda key: progressive(key, batched_lgssm_1, batched_lgssm_2, bms_1, bPs_1, bms_2, bPs_2,
                                                      method=method)
    else:
        raise ValueError("Unknown mode.")
    samples_1, samples_2, coupled_index = jax.vmap(sampling_fn)(sampling_keys)
    batched_samples_1, batched_samples_2, batched_coupled_index = jax.vmap(batched_sampling_fn)(sampling_keys)
    reshaped_samples_1 = batched_samples_1.reshape((-1, T, B * dx))
    reshaped_samples_2 = batched_samples_2.reshape((-1, T, B * dx))

    # check that the samples are actually coupled
    npt.assert_allclose(reshaped_samples_1[batched_coupled_index], reshaped_samples_2[batched_coupled_index])

    mean_samples_1 = np.mean(samples_1, axis=0)
    mean_samples_2 = np.mean(samples_2, axis=0)
    mean_batched_samples_1 = np.mean(reshaped_samples_1, axis=0)
    mean_batched_samples_2 = np.mean(reshaped_samples_2, axis=0)

    std_samples_1 = np.std(samples_1, axis=0)
    std_samples_2 = np.std(samples_2, axis=0)
    std_batched_samples_1 = np.std(reshaped_samples_1, axis=0)
    std_batched_samples_2 = np.std(reshaped_samples_2, axis=0)

    # We can only hope that the statistics from the batched version are close to the non-batched version.
    npt.assert_allclose(mean_samples_1, mean_batched_samples_1, atol=1e-2, rtol=1e-2)
    npt.assert_allclose(mean_samples_2, mean_batched_samples_2, atol=1e-2, rtol=1e-2)
    npt.assert_allclose(std_samples_1, std_batched_samples_1, atol=1e-2, rtol=1e-2)
    npt.assert_allclose(std_samples_2, std_batched_samples_2, atol=1e-2, rtol=1e-2)
