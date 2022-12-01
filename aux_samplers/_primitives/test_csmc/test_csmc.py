from functools import partial

import jax
import numpy as np
import numpy.testing as npt
import pytest
from matplotlib import pyplot as plt

from .common import GaussianDistribution, FlatPotential, FlatUnivariatePotential, GaussianDynamics, lgssm_data, \
    GaussianObservationPotential

from statsmodels.graphics.tsaplots import plot_acf
from ..csmc.auxiliary import get_independent_kernel as get_auxiliary_kernel
from ..csmc.standard import get_kernel as get_standard_kernel


def get_kernel_fn(method):
    if method == "standard":
        get_kernel = get_standard_kernel
    elif method == "auxiliary":
        get_kernel = get_auxiliary_kernel
    else:
        raise ValueError("Unknown method: {}".format(method))
    return get_kernel


def specialise_kernel(kernel, method, delta):
    if method == "standard":
        return kernel
    else:
        return partial(kernel, delta=delta)


@pytest.mark.parametrize("method", ["standard", "auxiliary"])
@pytest.mark.parametrize("backward", [True, False])
@pytest.mark.parametrize("delta", [0.5, 1.5])
def test_flat_potential(method, backward, delta):
    get_kernel = get_kernel_fn(method)

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
    kernel = specialise_kernel(kernel, method, delta)

    init_key, key = jax.random.split(JAX_KEY)
    x0 = jax.random.normal(init_key, (T, 1))
    init_state = init(x0)

    def body(state, curr_key):
        state = kernel(curr_key, state)
        return state, (state.x, state.ancestors)

    _, (xs, ancestors) = jax.lax.scan(body, init_state, jax.random.split(key, M))

    xs = xs[B:, :, 0]

    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
    fig.suptitle("Method: {}, Backward: {}, Delta: {}".format(method, backward, delta))
    plot_acf(xs[:, 0], ax=axes[0])
    axes[0].set_title("ACF of x_0")
    plot_acf(xs[:, T//2], ax=axes[1])
    axes[1].set_title("ACF of x_T/2")
    plt.show()

    atol = 0.05
    cov = np.cov(xs, rowvar=False)
    cov = np.atleast_2d(cov)

    rows, cols = np.diag_indices_from(cov)

    cov_diag = cov[rows, cols]  # marginal variances
    sub_cov_diag = cov[rows[:-1], cols[1:]]  # Covariances between adjacent time steps
    print(cov_diag)
    print(sub_cov_diag)
    print(xs.mean(axis=0))
    npt.assert_allclose(xs.mean(axis=0), 0., atol=atol)
    npt.assert_allclose(cov_diag, 1., atol=atol)
    npt.assert_allclose(sub_cov_diag, RHO, atol=atol)



@pytest.mark.parametrize("method", ["standard", "auxiliary"])
@pytest.mark.parametrize("backward", [True, False])
@pytest.mark.parametrize("delta", [0.5, 1.5])
def test_lgssm(method, backward, delta):
    get_kernel = get_kernel_fn(method)

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

    M0 = GaussianDistribution(mu=0.0, sig=1.0)
    G0 = GaussianDistribution(mu=true_ys[0], sig=SIG_Y)
    Gt = GaussianObservationPotential(params=true_ys[1:], sig=SIG_Y)
    Mt = GaussianDynamics(rho=RHO)

    init, kernel = get_kernel(M0, G0, Mt, Gt, N=N, backward=backward, Pt=Mt)
    kernel = specialise_kernel(kernel, method, delta)

    x0 = jax.random.normal(init_key, (T, 1))
    init_state = init(x0)

    def body(state, curr_key):
        state = kernel(curr_key, state)
        return state, (state.x, state.ancestors)

    _, (xs, ancestors) = jax.lax.scan(body, init_state, jax.random.split(key, M))

    xs = xs[B:, :, 0]

    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
    fig.suptitle("Method: {}, Backward: {}, Delta: {}".format(method, backward, delta))
    plot_acf(xs[:, 0], ax=axes[0])
    axes[0].set_title("ACF of x_0")
    plot_acf(xs[:, T//2], ax=axes[1])
    axes[1].set_title("ACF of x_T/2")
    plot_acf(xs[:, -1], ax=axes[2])
    axes[2].set_title("ACF of x_T")
    plt.show()

    print(xs.mean(axis=0))
    print(xs.std(axis=0))
