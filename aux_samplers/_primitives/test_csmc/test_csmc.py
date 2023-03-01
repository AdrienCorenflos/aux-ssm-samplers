import jax
import numpy as np
import numpy.testing as npt
import pytest
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

from .common import GaussianDistribution, FlatPotential, FlatUnivariatePotential, GaussianDynamics, lgssm_data, \
    GaussianObservationPotential
from ..csmc import get_kernel


@pytest.fixture(scope="module", autouse=True)
def jax_config():
    jax.config.update("jax_platform_name", "cpu")


@pytest.mark.parametrize("backward", [True, False])
def test_flat_potential(backward):
    # Test a flat potential, to check that we recover the prior.
    # The model is a stationary AR process with Gaussian noise.
    JAX_KEY = jax.random.PRNGKey(0)

    T = 5  # T time steps
    RHO = 0.9  # correlation

    N = 32  # use N particles in total
    M = 50_000  # get M - B samples from the particle Gibbs kernel
    B = M // 10  # Discard the first 10% of the samples

    M0 = GaussianDistribution(mu=0.0, sig=1.0)
    G0 = FlatUnivariatePotential()
    Gt = FlatPotential()
    Mt = GaussianDynamics(rho=RHO)

    init, kernel = get_kernel(M0, G0, Mt, Gt, N=N, backward=backward, Pt=Mt)

    init_key, key = jax.random.split(JAX_KEY)
    x0 = jax.random.normal(init_key, (T, 1))
    init_state = init(x0)

    def body(state, curr_key):
        state = kernel(curr_key, state)
        return state, (state.x, state.updated)

    _, (xs, ancestors) = jax.lax.scan(body, init_state, jax.random.split(key, M))

    xs = xs[B:, :, 0]

    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
    fig.suptitle("Backward: {}".format(backward))
    plot_acf(xs[:, 0], ax=axes[0])
    axes[0].set_title("ACF of x_0")
    plot_acf(xs[:, T // 2], ax=axes[1])
    axes[1].set_title("ACF of x_T/2")
    plt.show()

    atol = 0.05
    cov = np.cov(xs, rowvar=False)
    cov = np.atleast_2d(cov)

    rows, cols = np.diag_indices_from(cov)

    cov_diag = cov[rows, cols]  # marginal variances
    sub_cov_diag = cov[rows[:-1], cols[1:]]  # Covariances between adjacent time steps

    npt.assert_allclose(xs.mean(axis=0), 0., atol=atol)
    npt.assert_allclose(cov_diag, 1., atol=atol)
    npt.assert_allclose(sub_cov_diag, RHO, atol=atol)


@pytest.mark.parametrize("backward", [True, False])
def test_lgssm(backward):
    # Test a LGSSM model test
    JAX_KEY = jax.random.PRNGKey(0)

    T = 25  # T time steps
    RHO = 0.9  # correlation
    SIG_Y = 0.1  # observation noise

    data_key, init_key, key = jax.random.split(JAX_KEY, 3)
    true_xs, true_ys = lgssm_data(data_key, RHO, SIG_Y, T)

    N = 32  # use N particles in total
    M = 50_000  # get M - B samples from the particle Gibbs kernel
    B = M // 10  # Discard the first 10% of the samples

    M0 = GaussianDistribution(mu=0.0, sig=1.0)
    G0 = GaussianDistribution(mu=true_ys[0], sig=SIG_Y)
    Gt = GaussianObservationPotential(params=true_ys[1:], sig=SIG_Y)
    Mt = GaussianDynamics(rho=RHO)

    init, kernel = get_kernel(M0, G0, Mt, Gt, N=N, backward=backward, Pt=Mt)

    x0 = jax.random.normal(init_key, (T, 1))
    init_state = init(x0)

    def body(state, curr_key):
        state = kernel(curr_key, state)
        return state, (state.x, state.updated)

    _, (xs, ancestors) = jax.lax.scan(body, init_state, jax.random.split(key, M))

    xs = xs[B:, :, 0]

    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
    fig.suptitle("Backward: {}".format(backward))
    plot_acf(xs[:, 0], ax=axes[0])
    axes[0].set_title("ACF of x_0")
    plot_acf(xs[:, T // 2], ax=axes[1])
    axes[1].set_title("ACF of x_T/2")
    plot_acf(xs[:, -1], ax=axes[2])
    axes[2].set_title("ACF of x_T")
    plt.show()

    print(xs.mean(axis=0))
    print(xs.std(axis=0))
