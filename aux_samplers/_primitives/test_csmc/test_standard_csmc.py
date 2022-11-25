import jax
import numpy.testing as npt
import pytest

from .common import GaussianDistribution, FlatPotential, FlatUnivariatePotential, GaussianDynamics
from ..csmc.standard import get_kernel


@pytest.mark.parametrize("backward", [False])
def test_flat_potential(backward):
    # Test a flat potential, to check that we recover the prior.
    JAX_KEY = jax.random.PRNGKey(0)
    init_key, key = jax.random.split(JAX_KEY)

    T = 25
    RHO = 0.5
    N = 50
    M = 10_000

    M0 = GaussianDistribution(mu=0.0, sig=1.0)
    G0 = FlatUnivariatePotential()
    Gt = FlatPotential()
    Mt = GaussianDynamics(rho=RHO)

    init, kernel = get_kernel(M0, G0, Mt, Gt, N=N, backward=False)

    x0 = jax.random.normal(init_key, (T, 1))
    init_state = init(x0)

    def body(state, curr_key):
        state = kernel(curr_key, state)
        return state, state.x

    _, xs = jax.lax.scan(body, init_state, jax.random.split(key, M))
    xs = xs[:1_000, :, 0]

    npt.assert_allclose(xs.mean(axis=0), 0., atol=1e-1)
    npt.assert_allclose(xs.std(axis=0), 1., atol=1e-1)
