import jax
import numpy as np
import numpy.testing as npt
import pytest
import jax.numpy as jnp

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
@pytest.mark.parametrize("mode", ["_parallel", "dnc", "sequential"])
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

    if mode == "_parallel":
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


def explicit_kalman_smoothing(ms, Ps, Fs, Qs, bs):
    """ Explicit Kalman smoother implementation for testing purposes.
    """
    T, _ = ms.shape
    ms_smoothed = np.zeros_like(ms)
    Ps_smoothed = np.zeros_like(Ps)
    ms_smoothed[-1] = ms[-1]
    Ps_smoothed[-1] = Ps[-1]
    for t in range(T - 2, -1, -1):
        m, P, F, Q, b = ms[t], Ps[t], Fs[t], Qs[t], bs[t]
        m_next = ms_smoothed[t + 1]
        P_next = Ps_smoothed[t + 1]

        P_pred = F @ P @ F.T + Q
        m_pred = F @ m + b

        K = P @ F.T @ np.linalg.inv(P_pred)
        ms_smoothed[t] = m + K @ (m_next - m_pred)
        Ps_smoothed[t] = P + K @ (P_next - P_pred) @ K.T
    return ms_smoothed, Ps_smoothed
