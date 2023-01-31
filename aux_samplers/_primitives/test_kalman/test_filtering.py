from functools import partial

import jax
import numpy as np
import numpy.testing as npt
import pytest
from scipy.linalg import block_diag

from .common import explicit_kalman_filter
from ..kalman.base import LGSSM
from ..kalman.filtering import filtering


@pytest.fixture(scope="module", autouse=True)
def jax_config():
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("seed", [0, 1234])
@pytest.mark.parametrize("T", [3, 5])
@pytest.mark.parametrize("dx", [1, 2])
@pytest.mark.parametrize("dy", [1, 3])
@pytest.mark.parametrize("parallel", [True, False])
def test_parallel_vs_sequential(seed, T, dx, dy, parallel):
    np.random.seed(seed)

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
    ms, Ps, ell = filtering(ys, lgssm, parallel)
    expected_ms, expected_Ps, expected_ell = explicit_kalman_filter(ys, m0, P0, Hs, Rs, cs, Fs, Qs, bs)
    npt.assert_allclose(ms, expected_ms)
    npt.assert_allclose(Ps, expected_Ps)
    npt.assert_allclose(ell, expected_ell)


@pytest.mark.parametrize("seed", [0, 1234])
@pytest.mark.parametrize("T", [3, 5])
@pytest.mark.parametrize("dx", [1, 2])
@pytest.mark.parametrize("dy", [1, 3])
@pytest.mark.parametrize("parallel", [True, False])
def test_batched_model(seed, T, dx, dy, parallel):
    B = 3
    np.random.seed(seed)

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
    bms, bPs, bell = filtering(bys, batched_lgssm, parallel)
    ms, Ps, ell = filtering(ys, lgssm, parallel)
    expected_ms, expected_Ps, expected_ell = explicit_kalman_filter(ys, m0, P0, Hs, Rs, cs, Fs, Qs, bs)
    npt.assert_allclose(ms, expected_ms)
    npt.assert_allclose(Ps, expected_Ps)
    npt.assert_allclose(ell, expected_ell)
    npt.assert_allclose(np.reshape(bms, (T, B * dx)), expected_ms)
    npt.assert_allclose(batched_block_diag(bPs), expected_Ps)
    npt.assert_allclose(np.sum(bell), expected_ell)
