import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from .base import CoupledSamplerState, SamplerState
from .math.mvn import reflection
from .rhee_glynn import estimator



def get_coupled_kernel(dim):
    # A coupled MCMC kernel targeting a N-dim Gaussian distribution with mean 2
    log_pdf = lambda z: -jnp.sum(jnp.abs(z - 2.))
    L = 1 / dim ** 0.5

    def kernel(key, state: CoupledSamplerState):
        state_1, state_2 = state.state_1, state.state_2
        x1, x2 = state_1.x, state_2.x
        k1, k2 = jax.random.split(key, 2)
        y1, y2, coupled_prop = reflection(k1, x1, L, x2, L)

        log_alpha_1 = log_pdf(y1) - log_pdf(x1)
        log_alpha_2 = log_pdf(y2) - log_pdf(x2)

        log_u = jnp.log(jax.random.uniform(k2))

        accept_1 = log_u < log_alpha_1
        accept_2 = log_u < log_alpha_2

        coupled = coupled_prop & accept_1 & accept_2
        x1 = jax.lax.select(accept_1, y1, x1)
        x2 = jax.lax.select(accept_2, y2, x2)
        state_1, state_2 = SamplerState(x=x1), SamplerState(x=x2)
        state = CoupledSamplerState(state_1=state_1, state_2=state_2, flags=coupled)
        return state

    return kernel


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="This test should be run locally with a GPU. "
                                                           "Ideally a regression test, but it would be a bit painful to "
                                                           "implement the baseliner etc.")
@pytest.mark.parametrize("dim", [5])
def test_on_laplace(dim):
    # Test the asymptotic normality.
    SEED = 1
    np.random.seed(SEED)
    key = jax.random.PRNGKey(SEED)

    K = 200
    M = K * 10  # large M as in the paper
    kernel = get_coupled_kernel(dim)

    def init_sampler(k):
        x1, x2 = jax.random.normal(k, (2, dim))
        state_1, state_2 = SamplerState(x=x1), SamplerState(x=x2)
        return CoupledSamplerState(state_1=state_1, state_2=state_2, flags=False)

    def test_fn(z):
        return jnp.sum(z ** 2)

    # MCMC estimates
    # We do 100 experiments with 1'000'000 samples each
    n_samples = 250_000
    n_experiments = 25

    coupled_results = np.zeros((n_experiments, n_samples))
    uncoupled_results = np.zeros((n_experiments, n_samples))
    coupling_times = np.zeros((n_experiments, n_samples))
    true_value = 6 * dim  # variance + mean ** 2 = 2 * dim + dim * 2 ** 2

    estimator_here = lambda k: estimator(k, kernel, init_sampler, K, M, test_fn, True)
    vmapped_estimator = jax.jit(jax.vmap(estimator_here))
    arange = np.arange(1, n_samples + 1)
    for i in range(n_experiments):
        key, subkey = jax.random.split(key)
        sampling_keys = jax.random.split(subkey, n_samples)
        results, total_iter, coupling_time, standard_results = vmapped_estimator(sampling_keys)

        coupling_times[i] = np.asarray(coupling_time)
        coupled_results[i] = np.cumsum(results) / arange
        uncoupled_results[i] = np.cumsum(standard_results) / arange

    coupling_times = np.ravel(coupling_times)
    print(np.percentile(coupling_times, [50, 75, 90, 95, 97.5, 99]))

    fig, ax = plt.subplots(figsize=(20, 12))
    fig.suptitle("Mean estimate as a function of samples")
    ax.plot(arange, coupled_results.T, alpha=0.2, color="tab:blue")
    ax.plot(arange, np.mean(coupled_results, 0), alpha=1., color="tab:blue", label="Unbiased estimates", linewidth=3)
    ax.plot(arange, uncoupled_results.T, alpha=0.2, color="tab:orange")
    ax.plot(arange, np.mean(uncoupled_results, 0), alpha=1, color="tab:orange", label="Biased estimates", linewidth=3)
    # ax.plot(arange, ideal_resuls.T, alpha=0.2, color="tab:green")
    # ax.plot(arange, np.mean(ideal_resuls, 0), alpha=0.2, color="tab:green", label="Ideal estimates")
    ax.set_ylim(true_value - 1, true_value + 1)
    ax.set_xlim(1_000, arange[-1])
    ax.set_xscale("log")
    ax.hlines(true_value, arange[0], arange[-1],
              color="k", label="True value", linestyle="-", linewidth=3)
    ax.legend(loc="upper right")
    plt.show()

