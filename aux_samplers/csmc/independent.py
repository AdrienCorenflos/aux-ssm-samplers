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
from .._primitives.csmc.base import Distribution, UnivariatePotential, Dynamics, Potential


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

    # This function uses the classes defined below
    def factory(u, scale):
        if gradient:
            grad_pi = jax.grad(_log_pdf)(u, M0, G0, Mt, Gt)
        else:
            grad_pi = 0. * u
        m0 = AuxiliaryM0(u=u[0], sqrt_half_delta=scale[0], grad=grad_pi[0])
        mt = AuxiliaryMt(params=(u[1:], scale[1:], grad_pi[1:]))
        if gradient:
            g0 = GradientAuxiliaryG0(M0=M0, G0=G0, u=u[0], sqrt_half_delta=scale[0], grad=grad_pi[0])
            gt = GradientAuxiliaryGt(Mt=Mt, Gt=Gt, params=(u[1:], scale[1:], grad_pi[1:]))
        else:
            g0 = AuxiliaryG0(M0=M0, G0=G0)
            gt = AuxiliaryGt(Mt=Mt, Gt=Gt)
        return m0, g0, mt, gt

    if not parallel:
        return get_base_kernel(factory, N, backward, Pt)
    else:
        raise NotImplementedError("Parallel version not implemented yet.")


def get_coupled_kernel(M0: Distribution, G0: UnivariatePotential, Mt: Dynamics, Gt: Potential, N: int,
                       backward: bool = False, Pt: Optional[Dynamics] = None, gradient: bool = False,
                       parallel: bool = False):
    """
    Get a coupled local auxiliary kernel with separable proposals.

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

    # This function uses the classes defined below
    def factory(u, scale):
        if gradient:
            grad_pi = jax.grad(_log_pdf)(u, M0, G0, Mt, Gt)
        else:
            grad_pi = 0. * u
        m0 = AuxiliaryM0(u=u[0], sqrt_half_delta=scale[0], grad=grad_pi[0])
        mt = AuxiliaryMt(params=(u[1:], scale[1:], grad_pi[1:]))
        if gradient:
            g0 = GradientAuxiliaryG0(M0=M0, G0=G0, u=u[0], sqrt_half_delta=scale[0], grad=grad_pi[0])
            gt = GradientAuxiliaryGt(Mt=Mt, Gt=Gt, params=(u[1:], scale[1:], grad_pi[1:]))
        else:
            g0 = AuxiliaryG0(M0=M0, G0=G0)
            gt = AuxiliaryGt(Mt=Mt, Gt=Gt)
        return m0, g0, mt, gt

    if not parallel:
        return get_base_kernel(factory, N, backward, Pt)
    else:
        raise NotImplementedError("Parallel version not implemented yet.")


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

    def sample(self, key, n):
        half_delta = self.sqrt_half_delta ** 2
        mean = self.u + half_delta * self.grad
        return mean[None, ...] + self.sqrt_half_delta * jax.random.normal(key, (n, *self.u.shape))


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


@chex.dataclass
class AuxiliaryMt(Dynamics):
    def sample(self, key, x_t, params):
        u_t, sqrt_half_delta, grad_t = params
        half_delta = sqrt_half_delta ** 2
        mean = u_t[None, :] + half_delta * grad_t[None, :]
        return mean + sqrt_half_delta * jax.random.normal(key, x_t.shape)


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
