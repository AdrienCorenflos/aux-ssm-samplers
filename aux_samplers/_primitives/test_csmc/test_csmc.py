from functools import partial

import jax
import numpy as np
import numpy.testing as npt
import pytest
from matplotlib import pyplot as plt
from scipy.stats import norm

from .common import GaussianDistribution, FlatPotential, FlatUnivariatePotential, GaussianDynamics
from ..csmc.auxiliary import get_kernel as get_auxiliary_kernel
from ..csmc.finke import get_kernel as get_finke_kernel
from ..csmc.standard import get_kernel as get_standard_kernel


def get_kernel_fn(method):
    if method == "standard":
        get_kernel = get_standard_kernel
    elif method == "finke":
        get_kernel = get_finke_kernel
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


# @pytest.mark.parametrize("method", ["standard", "finke", "auxiliary"])
# @pytest.mark.parametrize("backward", [True, False])
# @pytest.mark.parametrize("delta", [1., 2.5])
@pytest.mark.parametrize("method", ["auxiliary"])
@pytest.mark.parametrize("backward", [True, False])
@pytest.mark.parametrize("delta", [1.])
def test_flat_potential(method, backward, delta):
    get_kernel = get_kernel_fn(method)

    # Test a flat potential, to check that we recover the prior.
    # The model is a stationary AR process with Gaussian noise.
    JAX_KEY = jax.random.PRNGKey(0)
    init_key, key = jax.random.split(JAX_KEY)

    T = 2  # 5 time steps
    RHO = 0.9  # 90% correlation

    N = 5_000  # use 32 particles
    M = 100_000  # get 100,000 samples from the particle Gibbs kernel

    M0 = GaussianDistribution(mu=0.0, sig=1.0)
    G0 = FlatUnivariatePotential()
    Gt = FlatPotential()
    Mt = GaussianDynamics(rho=RHO)

    init, kernel = get_kernel(M0, G0, Mt, Gt, N=N, backward=backward)
    kernel = specialise_kernel(kernel, method, delta)

    x0 = jax.random.normal(init_key, (T, 1))
    init_state = init(x0)


    def body(state, curr_key):
        state = kernel(curr_key, state)
        return state, state.x

    _, xs = jax.lax.scan(body, init_state, jax.random.split(key, M))

    xs = xs[1_000:, :, 0]  # 1,000 burn-in samples

    fig, ax = plt.subplots()
    fig.suptitle("Method: {}, Backward: {}, Delta: {}".format(method, backward, delta))
    ax.hist(xs[:, -1], bins=100, density=True)
    sorted_xs = np.sort(xs[:, -1])
    ax.plot(sorted_xs, norm.pdf(sorted_xs))
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
