"""
Implements the auxiliary samplers introduced in the paper.
"""
import chex
import jax
import jax.numpy as jnp
from chex import Array
from jax.scipy.stats import norm

from .base import Distribution, Potential, UnivariatePotential, Dynamics, CSMCState
from .standard import get_kernel as get_standard_kernel


def get_kernel(M0: Distribution, G0: UnivariatePotential, Mt: Dynamics, Gt: Potential, N: int,
               backward=False):
    """
    Get a local auxiliary kernel with separable proposals.

    Parameters:
    -----------
    M0:
        Initial distribution.
    G0:
        Initial potential.
    Mt:
        Dynamics of the model.
    Gt:
        Potential of the model.
    N: int
        Total number of particles to use in the cSMC sampler.
    backward: bool
        Whether to perform backward sampling or not. If True, the dynamics must implement a valid logpdf method.

    Returns:
    --------
    kernel: Callable
        cSMC kernel.
    init: Callable
        Function to initialize the state of the sampler given a trajectory.
    """

    def kernel(key, state, delta):
        # This uses the distributions defined below
        x, ancestors = state.x, state.ancestors
        sqrt_half_delta = jnp.sqrt(0.5 * delta)
        auxiliary_key, key = jax.random.split(key)
        u = x + sqrt_half_delta * jax.random.normal(auxiliary_key, x.shape)

        m0 = AuxiliaryM0(u=u[0], sqrt_half_delta=sqrt_half_delta)
        g0 = AuxiliaryG0(M0=M0, G0=G0)
        mt = AuxiliaryMt(params=u[1:], sqrt_half_delta=sqrt_half_delta)
        gt = AuxiliaryGt(Mt=Mt, Gt=Gt)

        _, auxiliary_kernel = get_standard_kernel(m0, g0, mt, gt, N, backward=backward)
        return auxiliary_kernel(key, state)

    def init(x_star):
        T, *_ = x_star.shape
        ancestors = jnp.zeros((T,), dtype=jnp.int_)
        return CSMCState(x=x_star, ancestors=ancestors)

    return init, kernel


###########################
# Auxiliary distributions #
###########################


@chex.dataclass
class AuxiliaryM0(Distribution):
    u: Array
    sqrt_half_delta: float

    def logpdf(self, x):
        logpdf = norm.logpdf(x, self.u, self.sqrt_half_delta)
        return jnp.sum(logpdf, axis=-1)

    def sample(self, key, n):
        return self.u[None, ...] + self.sqrt_half_delta * jax.random.normal(key, (n, *self.u.shape))


@chex.dataclass
class AuxiliaryG0(UnivariatePotential):
    M0: Distribution
    G0: UnivariatePotential

    def logpdf(self, x):
        return self.G0.logpdf(x) + self.M0.logpdf(x)


@chex.dataclass
class AuxiliaryMt(Dynamics):
    sqrt_half_delta: float = None  # needs a default as Dynamics has params with default None

    def logpdf(self, x_t_p_1, x_t, u):
        out = norm.logpdf(x_t_p_1, u, self.sqrt_half_delta)
        return out.sum(axis=-1)

    def sample(self, key, x_t, u):
        return u[None, ...] + self.sqrt_half_delta * jax.random.normal(key, x_t.shape)


@chex.dataclass
class AuxiliaryGt(Potential):
    Mt: Dynamics = None
    Gt: Potential = None

    def __post_init__(self):
        self.params = (self.Mt.params, self.Gt.params)

    def logpdf(self, x_t_p_1, x_t, params):
        Mt_params, Gt_params = params
        return self.Mt.logpdf(x_t_p_1, x_t, Mt_params) + self.Gt.logpdf(x_t_p_1, x_t, Gt_params)
