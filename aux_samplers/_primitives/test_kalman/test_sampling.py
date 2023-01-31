from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from scipy.linalg import block_diag

from .common import explicit_kalman_smoothing
from ..kalman.base import LGSSM
from ..kalman.dnc_sampling import sampling as dnc_sampling
from ..kalman.filtering import filtering
from ..kalman.sampling import sampling


@pytest.fixture(scope="module", autouse=True)
def jax_config():
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("seed", [42, 666])
@pytest.mark.parametrize("T", [3, 5])
@pytest.mark.parametrize("dx", [1, 2])
@pytest.mark.parametrize("dy", [1, 3])
@pytest.mark.parametrize("mode", ["parallel", "dnc", "sequential"])
def test_parallel_vs_sequential(seed, T, dx, dy, mode):
    np.random.seed(seed)
    jax_key = jax.random.PRNGKey(seed)
    sampling_keys = jax.random.split(jax_key, 500_000)  # this may be a bit much for a unittest, but meh, it's fast...

    m0 = np.random.randn(dx)
    P0 = np.random.randn(dx, 5 * dx)
    P0 = P0 @ P0.T

    Fs = np.random.randn(T - 1, dx, dx)
    Qs = np.random.randn(T - 1, dx, 5 * dx)
    Qs = Qs @ Qs.transpose((0, 2, 1))
    bs = np.random.randn(T - 1, dx)

    Hs = np.random.randn(T, dy, dx)
    Rs = np.random.randn(T, dy, 5 * dy)
    Rs = Rs @ Rs.transpose((0, 2, 1))
    cs = np.random.randn(T, dy)

    ys = np.random.randn(T, dy)

    lgssm = LGSSM(m0, P0, Fs, Qs, bs, Hs, Rs, cs)
    ms, Ps, ell = filtering(ys, lgssm, False)

    if mode == "parallel":
        sampling_fn = lambda key: sampling(key, ms, Ps, lgssm, True)
    elif mode == "dnc":
        sampling_fn = lambda key: dnc_sampling(key, ms, Ps, lgssm)
    elif mode == "sequential":
        sampling_fn = lambda key: sampling(key, ms, Ps, lgssm, False)
    else:
        raise ValueError("Unknown mode.")
    samples = jax.vmap(sampling_fn)(sampling_keys)

    expected_ms, expected_Ps = explicit_kalman_smoothing(ms, Ps, Fs, Qs, bs)

    npt.assert_allclose(expected_ms, samples.mean(0), atol=1e-2, rtol=1e-2)

    covs = jax.vmap(lambda z: jnp.cov(z, rowvar=False), in_axes=[1])(samples)
    covs = covs.reshape((T, dx, dx))
    npt.assert_allclose(expected_Ps, covs, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("seed", [42, 666])
@pytest.mark.parametrize("T", [5])
@pytest.mark.parametrize("dx", [2])
@pytest.mark.parametrize("dy", [3])
@pytest.mark.parametrize("mode", ["parallel", "sequential"])
def test_batched_model(seed, T, dx, dy, mode):
    # Test that batched and unbatched models give *exactly* the same samples
    B = 3
    np.random.seed(seed)
    jax_key = jax.random.PRNGKey(seed)
    sampling_keys = jax.random.split(jax_key, 10)

    bm0 = np.random.randn(B, dx)
    bP0 = np.random.randn(B, dx, 5 * dx)
    bP0 = bP0 @ bP0.transpose((0, 2, 1))

    bFs = np.random.randn(T - 1, B, dx, dx)
    bQs = np.random.randn(T - 1, B, dx, 5 * dx)
    bQs = bQs @ bQs.transpose((0, 1, 3, 2))
    bbs = np.random.randn(T - 1, B, dx)

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
    Fs = batched_block_diag(bFs)
    Qs = batched_block_diag(bQs)
    bs = np.reshape(bbs, (T - 1, B * dx))
    Hs = batched_block_diag(bHs)
    Rs = batched_block_diag(bRs)
    cs = np.reshape(bcs, (T, B * dy))
    ys = np.reshape(bys, (T, B * dy))

    batched_lgssm = LGSSM(bm0, bP0, bFs, bQs, bbs, bHs, bRs, bcs)
    lgssm = LGSSM(m0, P0, Fs, Qs, bs, Hs, Rs, cs)
    bms, bPs, bell = filtering(bys, batched_lgssm, False)
    ms, Ps, ell = filtering(ys, lgssm, False)

    if mode == "parallel":
        sampling_fn = lambda key: sampling(key, ms, Ps, lgssm, True)
        batched_sampling_fn = lambda key: sampling(key, bms, bPs, batched_lgssm, True)
    else:
        sampling_fn = lambda key: sampling(key, ms, Ps, lgssm, False)
        batched_sampling_fn = lambda key: sampling(key, bms, bPs, batched_lgssm, False)

    samples = jax.vmap(sampling_fn)(sampling_keys)
    batched_samples = jax.vmap(batched_sampling_fn)(sampling_keys)
    npt.assert_allclose(samples, batched_samples.reshape((-1, T, B * dx)), atol=1e-10, rtol=1e-10)
