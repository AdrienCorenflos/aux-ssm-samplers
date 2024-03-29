"""
Implements the auxiliary particle Gibbs algorithm with independent proposals, i.e.,
the version of Finke and Thiery (2021) but in the auxiliary paradigm.
"""
from typing import Optional

import chex
import jax
from chex import Array
from jax import numpy as jnp
from jax.scipy.stats import norm

from .generic import get_kernel as get_base_kernel
from .._primitives.csmc.base import Distribution, UnivariatePotential, Dynamics, Potential, CSMCState
from .._primitives.csmc.pit import get_kernel as get_pit_kernel


def get_kernel(M0: Distribution, G0: UnivariatePotential, Mt: Dynamics, Gt: Potential, N: int,
               backward: bool = False, Pt: Optional[Dynamics] = None, gradient: bool = False, parallel: bool = False):
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
    Pt:
        Dynamics of the true model. If None, it is assumed to be the same as Mt.
    gradient: bool
        Whether to use the gradient model in the proposal or not.
    parallel: bool
        Whether to use the parallel particle Gibbs or not.

    Returns:
    --------
    kernel: Callable
        cSMC kernel.
    init: Callable
        Function to initialize the state of the sampler given a trajectory.
    """
    if not parallel:
        return _get_classical_kernel(M0, G0, Mt, Gt, N, backward, Pt, gradient)
    else:
        return _get_parallel_kernel(M0, G0, Mt, Gt, N, gradient)


def _get_classical_kernel(M0: Distribution, G0: UnivariatePotential, Mt: Dynamics, Gt: Potential, N: int,
                          backward: bool, Pt: Optional[Dynamics], gradient):
    # This function uses the classes defined below
    def factory(u, scale):
        if gradient:
            grad_pi = jax.grad(_log_pdf)(u, M0, G0, Mt, Gt)
        else:
            grad_pi = 0. * u
        m0 = AuxiliaryM0(u=u[0], sqrt_half_delta=scale[0], grad=grad_pi[0])
        mt = AuxiliaryMtDynamics(params=(u[1:], scale[1:], grad_pi[1:]))
        if gradient:
            g0 = GradientAuxiliaryG0(M0=M0, G0=G0, u=u[0], sqrt_half_delta=scale[0], grad=grad_pi[0])
            gt = GradientAuxiliaryGt(Mt=Mt, Gt=Gt, params=(u[1:], scale[1:], grad_pi[1:]))
        else:
            g0 = AuxiliaryG0(M0=M0, G0=G0)
            gt = AuxiliaryGt(Mt=Mt, Gt=Gt)
        return m0, g0, mt, gt

    return get_base_kernel(factory, N, backward, Pt)


def _get_parallel_kernel(M0: Distribution, G0: UnivariatePotential, Mt: Dynamics, Gt: Potential, N: int,
                         gradient: bool):
    def factory(u, scale):
        if gradient:
            grad_pi = jax.grad(_log_pdf)(u, M0, G0, Mt, Gt)
            mt = AuxiliaryMtDistribution(params=(u, scale, grad_pi))
            qt = AuxiliaryMtDistribution(params=(u, scale, None))

        else:
            mt = AuxiliaryMtDistribution(params=(u, scale, None))
            qt = None

        g0 = AuxiliaryG0(M0=M0, G0=G0)
        gt = AuxiliaryGt(Mt=Mt, Gt=Gt)

        return mt, g0, gt, qt

    def kernel(key, state, delta):
        # Housekeeping
        x = state.x
        T = x.shape[0]

        sqrt_half_delta = jnp.sqrt(0.5 * delta)
        if jnp.ndim(sqrt_half_delta) == 0:
            sqrt_half_delta = sqrt_half_delta * jnp.ones((T,))
        auxiliary_key, key = jax.random.split(key)

        # Auxiliary observations
        u = x + sqrt_half_delta[:, None] * jax.random.normal(auxiliary_key, x.shape)

        mt, g0, gt, qt = factory(u, sqrt_half_delta)

        _, auxiliary_kernel = get_pit_kernel(mt, g0, gt, N, qt)
        return auxiliary_kernel(key, state)

    def init(x):
        T, *_ = x.shape
        ancestors = jnp.zeros((T,), dtype=int)
        return CSMCState(x=x, updated=ancestors != 0)

    return init, kernel


def _log_pdf(u, M0, G0, Mt, Gt):
    # Compute the log-pdf of the auxiliary variable

    log_pdf = M0.logpdf(u[0]) + G0(u[0])

    @jax.vmap
    def fn(u_t_p_1, u_t, Gt_param, Mt_param):
        out = Gt(u_t_p_1, u_t, Gt_param)
        out += Mt.logpdf(u_t_p_1, u_t, Mt_param)
        return out

    fn_out = fn(u[1:], u[:-1], Gt.params, Mt.params)
    log_pdf += jnp.sum(fn_out)
    return log_pdf


###########################
# Auxiliary distributions #
###########################

# M0:

@chex.dataclass
class AuxiliaryM0(Distribution):
    u: Array
    sqrt_half_delta: float
    grad: Array

    def logpdf(self, x):
        half_delta = self.sqrt_half_delta ** 2
        mean = self.u + half_delta * self.grad
        logpdf = norm.logpdf(x, mean, self.sqrt_half_delta)
        return jnp.sum(logpdf, axis=-1)

    def sample(self, key, N):
        half_delta = self.sqrt_half_delta ** 2
        mean = self.u + half_delta * self.grad
        return mean[None, ...] + self.sqrt_half_delta * jax.random.normal(key, (N, *self.u.shape))


# G0:

@chex.dataclass
class AuxiliaryG0(UnivariatePotential):
    M0: Distribution
    G0: UnivariatePotential

    def __call__(self, x):
        return self.G0(x) + self.M0.logpdf(x)


@chex.dataclass
class GradientAuxiliaryG0(UnivariatePotential):
    M0: Distribution
    G0: UnivariatePotential
    u: Array
    sqrt_half_delta: float
    grad: Array

    def __call__(self, x):
        half_delta = self.sqrt_half_delta ** 2
        mean = self.u + half_delta * self.grad

        out = self.G0(x) + self.M0.logpdf(x)
        out += jnp.sum(norm.logpdf(x, self.u, self.sqrt_half_delta), axis=-1)
        out -= jnp.sum(norm.logpdf(x, mean, self.sqrt_half_delta), axis=-1)
        return out


# Mt:

@chex.dataclass
class AuxiliaryMtDynamics(Dynamics):
    def sample(self, key, x_t, params):
        u_t, sqrt_half_delta, grad_t = params
        half_delta = sqrt_half_delta ** 2
        mean = u_t[None, :] + half_delta * grad_t[None, :]
        return mean + sqrt_half_delta * jax.random.normal(key, x_t.shape)


@chex.dataclass
class AuxiliaryMtDistribution(Distribution):
    params: Array

    def sample(self, key, N):
        u_t, sqrt_half_delta, grad_t = self.params
        d = u_t.shape[-1]
        half_delta = sqrt_half_delta ** 2
        if grad_t is None:
            mean = u_t[None, :]
        else:
            mean = u_t[None, :] + half_delta * grad_t[None, :]
        return mean + sqrt_half_delta * jax.random.normal(key, (N, d))

    def logpdf(self, x):
        u_t, sqrt_half_delta, grad_t = self.params
        half_delta = sqrt_half_delta ** 2

        if grad_t is None:
            mean = u_t
        else:
            mean = u_t + half_delta * grad_t
        logpdf = norm.logpdf(x, mean, sqrt_half_delta)
        return jnp.sum(logpdf, axis=-1)


@chex.dataclass
class AuxiliaryMtDynamics(Dynamics):
    def sample(self, key, x_t, params):
        u_t, sqrt_half_delta, grad_t = params
        half_delta = sqrt_half_delta ** 2
        mean = u_t[None, :] + half_delta * grad_t[None, :]
        return mean + sqrt_half_delta * jax.random.normal(key, x_t.shape)


# Gt:

@chex.dataclass
class AuxiliaryGt(Potential):
    Mt: Dynamics = None
    Gt: Potential = None

    def __post_init__(self):
        self.params = (self.Mt.params, self.Gt.params)

    def __call__(self, x_t_p_1, x_t, params):
        Mt_params, Gt_params = params
        return self.Mt.logpdf(x_t_p_1, x_t, Mt_params) + self.Gt(x_t_p_1, x_t, Gt_params)


@chex.dataclass
class GradientAuxiliaryGt(Potential):
    Mt: Dynamics = None
    Gt: Potential = None

    def __post_init__(self):
        self.params = (self.params, self.Mt.params, self.Gt.params)

    def __call__(self, x_t_p_1, x_t, params):
        (u_t, sqrt_half_delta, grad_t), Mt_params, Gt_params = params
        half_delta = sqrt_half_delta ** 2
        mean = u_t + half_delta * grad_t

        out_1 = self.Mt.logpdf(x_t_p_1, x_t, Mt_params) + self.Gt(x_t_p_1, x_t, Gt_params)
        out_2 = jnp.sum(norm.logpdf(x_t_p_1, u_t, sqrt_half_delta))
        out_2 -= jnp.sum(norm.logpdf(x_t_p_1, mean, sqrt_half_delta))

        return out_1 + out_2
