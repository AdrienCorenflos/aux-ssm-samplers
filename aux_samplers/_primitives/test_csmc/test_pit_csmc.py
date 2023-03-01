import jax
import numpy as np
import pytest
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

from .common import GaussianDistribution, GaussianDynamics, lgssm_data
from ..csmc.pit import get_kernel


@pytest.fixture(scope="module", autouse=True)
def jax_config():
    jax.config.update("jax_platform_name", "cpu")


def test_lgssm():
    # Test a flat potential, to check that we recover the prior.
    # The model is a stationary AR process with Gaussian noise.
    JAX_KEY = jax.random.PRNGKey(0)

    T = 25  # T time steps
    RHO = 0.9  # correlation
    SIG_Y = 0.1  # observation noise

    data_key, init_key, key = jax.random.split(JAX_KEY, 3)
    true_xs, true_ys = lgssm_data(data_key, RHO, SIG_Y, T)

    N = 32  # use N particles in total
    M = 50_000  # get M - B samples from the particle Gibbs kernel
    B = M // 10  # Discard the first 10% of the samples

    Mt = GaussianDistribution(mu=true_ys, sig=SIG_Y * np.ones((T,)))
    G0 = GaussianDistribution(mu=0.0, sig=1.0)
    Gt = GaussianDynamics(rho=RHO, params=np.arange(T - 1))

    init, kernel = get_kernel(Mt, G0, Gt, N=N)

    init_key, key = jax.random.split(JAX_KEY)
    x0 = jax.random.normal(init_key, (T, 1))
    init_state = init(x0)

    def body(state, curr_key):
        state = kernel(curr_key, state)
        return state, (state.x, state.updated)

    _, (xs, ancestors) = jax.lax.scan(body, init_state, jax.random.split(key, M))

    xs = xs[B:, :, 0]

    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
    plot_acf(xs[:, 0], ax=axes[0])
    axes[0].set_title("ACF of x_0")
    plot_acf(xs[:, T // 2], ax=axes[1])
    axes[1].set_title("ACF of x_T/2")
    plot_acf(xs[:, -1], ax=axes[2])
    axes[2].set_title("ACF of x_T")
    plt.show()

    print(xs.mean(axis=0))
    print(xs.std(axis=0))
