import jax
import numpy as np
import numpy.testing as npt
import pytest
from scipy.stats import multivariate_normal as mvn

from ..kalman.base import LGSSM
from ..kalman.filtering import filtering


@pytest.fixture(scope="module", autouse=True)
def jax_config():
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("seed", [0, 1234])
@pytest.mark.parametrize("T", [3, 7])
@pytest.mark.parametrize("dx", [1, 2, 3])
@pytest.mark.parametrize("dy", [1, 2, 3])
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


def explicit_kalman_filter(ys, m0, P0, Hs, Rs, cs, Fs, Qs, bs):
    """ Explicit Kalman filter implementation for testing purposes.
    Computes the marginal likelihood and the marginals of the state for a model
    X_0 ~ N(m0, P0)
    X_t = F_{t-1} X_{t-1} + b_{t-1} + N(0, Q_{t-1}), t = 1, ..., T
    Y_t = H_t X_t + c_t + N(0, R_t), t = 0, ..., T
    """
    T = len(ys)
    dx, dy = Hs.shape[2], Hs.shape[1]

    # Initialize
    ms = np.zeros((T, dx))
    Ps = np.zeros((T, dx, dx))

    # Initial update
    S0 = Hs[0] @ P0 @ Hs[0].T + Rs[0]
    y0_hat = Hs[0] @ m0 + cs[0]
    ell0 = mvn.logpdf(ys[0], y0_hat, S0)
    K0 = P0 @ Hs[0].T @ np.linalg.inv(S0)
    m0 = m0 + K0 @ (ys[0] - y0_hat)
    P0 = P0 - K0 @ S0 @ K0.T

    ms[0] = m0
    Ps[0] = P0
    ell = ell0

    for t in range(1, T):
        # Prediction
        m = Fs[t-1] @ ms[t - 1] + bs[t-1]
        P = Fs[t-1] @ Ps[t - 1] @ Fs[t-1].T + Qs[t-1]

        # Update
        S = Hs[t] @ P @ Hs[t].T + Rs[t]
        y_hat = Hs[t] @ m + cs[t]
        ell += mvn.logpdf(ys[t], y_hat, S)
        K = P @ Hs[t].T @ np.linalg.inv(S)
        ms[t] = m + K @ (ys[t] - y_hat)
        Ps[t] = P - K @ Hs[t] @ P

    return ms, Ps, ell
