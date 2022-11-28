import jax
import numpy as np
import numpy.testing as npt
import pytest
from matplotlib import pyplot as plt
from scipy.stats import norm

from .common import GaussianDistribution, FlatPotential, FlatUnivariatePotential, GaussianDynamics
from ..csmc.finke import get_kernel


@pytest.mark.parametrize("backward", [False])
@pytest.mark.parametrize("delta", [0.5, 1., 2.5])
def test_flat_potential(backward, delta):
    # Test a flat potential, to check that we recover the prior.
    # The model is a stationary AR process with Gaussian noise.
    JAX_KEY = jax.random.PRNGKey(0)
    init_key, key = jax.random.split(JAX_KEY)

    T = 5  # 5 time steps
    RHO = 0.9  # 90% correlation

    N = 32  # use 32 particles
    M = 50_000  # get 2,500 samples from the particle Gibbs kernel

    M0 = GaussianDistribution(mu=0.0, sig=1.0)
    G0 = FlatUnivariatePotential()
    Gt = FlatPotential()
    Mt = GaussianDynamics(rho=RHO)

    init, kernel = get_kernel(M0, G0, Mt, Gt, N=N, backward=backward)

    x0 = jax.random.normal(init_key, (T, 1))
    init_state = init(x0)

    def body(state, curr_key):
        state = kernel(curr_key, state, delta)
        return state, state.x

    _, xs = jax.lax.scan(body, init_state, jax.random.split(key, M))

    xs = xs[1_000:, :, 0]  # 1,000 burn-in samples
    plt.hist(xs[:, 0], bins=100, density=True)
    sorted_xs = np.sort(xs[:, 0])
    plt.plot(sorted_xs, norm.pdf(sorted_xs))
    plt.show()

    atol = 1e-1
    cov = np.cov(xs, rowvar=False)
    cov = np.atleast_2d(cov)

    rows, cols = np.diag_indices_from(cov)

    cov_diag = cov[rows, cols]  # marginal variances
    sub_cov_diag = cov[rows[:-1], cols[1:]]  # Covariances between adjacent time steps
    print(cov_diag)
    print(sub_cov_diag)

    npt.assert_allclose(xs.mean(axis=0), 0., atol=atol)
    npt.assert_allclose(cov_diag, 1., atol=atol)
    npt.assert_allclose(sub_cov_diag, RHO, atol=atol)
