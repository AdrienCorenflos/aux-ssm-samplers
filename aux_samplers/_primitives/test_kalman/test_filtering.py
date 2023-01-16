import jax
import numpy as np
import numpy.testing as npt
import pytest

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
