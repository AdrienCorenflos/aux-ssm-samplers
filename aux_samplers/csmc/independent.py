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
from .._primitives.csmc.base import Distribution, UnivariatePotential, Dynamics, Potential, CSMCState, CoupledDynamics, \
    CoupledDistribution
from .._primitives.csmc.pit.csmc import get_kernel as get_pit_kernel
from .._primitives.math.mvn.couplings import reflection_maximal, reflection


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
        ancestors = jnp.zeros((T,), dtype=jnp.int_)
        return CSMCState(x=x, updated=ancestors != 0)

    return init, kernel


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
        mt = AuxiliaryMtDynamics(params=(u[1:], scale[1:], grad_pi[1:]))
        if gradient:
            g0 = GradientAuxiliaryG0(M0=M0, G0=G0, u=u[0], sqrt_half_delta=scale[0], grad=grad_pi[0])
            gt = GradientAuxiliaryGt(Mt=Mt, Gt=Gt, params=(u[1:], scale[1:], grad_pi[1:]))
        else:
            g0 = AuxiliaryG0(M0=M0, G0=G0)
            gt = AuxiliaryGt(Mt=Mt, Gt=Gt)
        return m0, g0, mt, gt

    if not parallel:
        return get_base_kernel(factory, N, backward, Pt, coupled=True)
    else:
        raise NotImplementedError("Parallel version not implemented yet.")


def _get_classical_coupled_kernel(M0: Distribution, G0: UnivariatePotential, Mt: Dynamics, Gt: Potential, N: int,
                                  backward: bool, Pt: Optional[Dynamics], gradient):
    # This function uses the classes defined below
    def coupled_factory(u_1, u_2, scale):
        if gradient:
            grad_pi_1 = jax.grad(_log_pdf)(u_1, M0, G0, Mt, Gt)
            grad_pi_2 = jax.grad(_log_pdf)(u_2, M0, G0, Mt, Gt)
        else:
            grad_pi_1, grad_pi_2 = 0. * u_1, 0. * u_2

        cm0 = CoupledAuxiliaryM0(u_1=u_1[0], u_2=u_2[0], sqrt_half_delta=scale[0], grad_1=grad_pi_1[0],
                                 grad_2=grad_pi_2[0])
        cmt = CoupledAuxiliaryMtDynamics(params=(u_1[1:], u_2[1:], scale[1:], grad_pi_1[1:], grad_pi_2[1:]))

        if gradient:
            g0_1 = GradientAuxiliaryG0(M0=M0, G0=G0, u=u_1[0], sqrt_half_delta=scale[0], grad=grad_pi_1[0])
            g0_2 = GradientAuxiliaryG0(M0=M0, G0=G0, u=u_2[0], sqrt_half_delta=scale[0], grad=grad_pi_2[0])
            gt_1 = GradientAuxiliaryGt(Mt=Mt, Gt=Gt, params=(u_1[1:], scale[1:], grad_pi_1[1:]))
            gt_2 = GradientAuxiliaryGt(Mt=Mt, Gt=Gt, params=(u_2[1:], scale[1:], grad_pi_2[1:]))
        else:

            g0_1 = g0_2 = AuxiliaryG0(M0=M0, G0=G0)

            gt_1 = gt_2 = AuxiliaryGt(Mt=Mt, Gt=Gt)

        return cm0, g0_1, g0_2, cmt, gt_1, gt_2

    return get_base_kernel(coupled_factory, N, backward, Pt, coupled=True)


def _get_parallel_coupled_kernel(M0: Distribution, G0: UnivariatePotential, Mt: Dynamics, Gt: Potential, N: int,
                                 Pt: Optional[Dynamics], gradient):
    # This function uses the classes defined below
    def coupled_factory(u_1, u_2, scale):
        pass


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


@chex.dataclass
class CoupledAuxiliaryM0(CoupledDistribution):
    u_1: Array
    u_2: Array
    sqrt_half_delta: float
    grad_1: Array
    grad_2: Array

    def _logpdf(self, x, m):
        logpdf = norm.logpdf(x, m, self.sqrt_half_delta)
        return jnp.sum(logpdf, axis=-1)

    def logpdf_1(self, x):
        half_delta = self.sqrt_half_delta ** 2
        mean = self.u_1 + half_delta * self.grad_1
        return self._logpdf(x, mean)

    def logpdf_2(self, x):
        half_delta = self.sqrt_half_delta ** 2
        mean = self.u_2 + half_delta * self.grad_2
        return self._logpdf(x, mean)

    def sample(self, key, N):
        half_delta = self.sqrt_half_delta ** 2
        mean_1 = self.u_1 + half_delta * self.grad_1
        mean_2 = self.u_2 + half_delta * self.grad_2
        return reflection_maximal(key, N, mean_1, mean_2, self.sqrt_half_delta)


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
class CoupledAuxiliaryMtDynamics(CoupledDynamics):
    def sample(self, key, x_t_1, x_t_2, params_1, params_2):
        u_t_1, sqrt_half_delta, grad_t_1 = params_1
        u_t_2, _, grad_t_2 = params_2
        half_delta = sqrt_half_delta ** 2
        mean_1 = u_t_1[None, :] + half_delta * grad_t_1[None, :]
        mean_2 = u_t_2[None, :] + half_delta * grad_t_2[None, :]
        keys = jax.random.split(key, x_t_1.shape[0])
        return jax.vmap(reflection, (0, 0, None, 0, None))(keys, mean_1, sqrt_half_delta, mean_2,
                                                           sqrt_half_delta)


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
class CoupledAuxiliaryMtDistribution(CoupledDistribution):
    params_1: Array
    params_2: Array

    def sample(self, key, N):
        u_t_1, sqrt_half_delta, grad_t_1 = self.params_1
        u_t_2, _, grad_t_2 = self.params_2

        half_delta = sqrt_half_delta ** 2
        if grad_t_1 is None:
            mean_1 = u_t_1[None, :]
            mean_2 = u_t_2[None, :]
        else:
            mean_1 = u_t_1[None, :] + half_delta * grad_t_1[None, :]
            mean_2 = u_t_2[None, :] + half_delta * grad_t_2[None, :]

        return reflection_maximal(key, N, mean_1, mean_2, self.sqrt_half_delta)

    @staticmethod
    def _logpdf(x, params):
        u_t, sqrt_half_delta, grad_t = params
        half_delta = sqrt_half_delta ** 2
        if grad_t is None:
            mean = u_t
        else:
            mean = u_t + half_delta * grad_t
        logpdf = norm.logpdf(x, mean, sqrt_half_delta)
        return jnp.sum(logpdf, axis=-1)

    def logpdf_1(self, x):
        return self._logpdf(x, self.params_1)

    def logpdf_2(self, x):
        return self._logpdf(x, self.params_2)


@chex.dataclass
class AuxiliaryMtDynamics(Dynamics):
    def sample(self, key, x_t, params):
        u_t, sqrt_half_delta, grad_t = params
        half_delta = sqrt_half_delta ** 2
        mean = u_t[None, :] + half_delta * grad_t[None, :]
        return mean + sqrt_half_delta * jax.random.normal(key, x_t.shape)


@chex.dataclass
class CoupledAuxiliaryMtDynamics(CoupledDynamics):
    def sample(self, key, x_t_1, x_t_2, params):
        u_t_1, u_t_2, sqrt_half_delta, grad_t_1, grad_t_2 = params
        half_delta = sqrt_half_delta ** 2
        mean_1 = u_t_1[None, :] + half_delta * grad_t_1[None, :]
        mean_2 = u_t_2[None, :] + half_delta * grad_t_2[None, :]
        keys = jax.random.split(key, x_t_1.shape[0])
        return jax.vmap(reflection, (0, 0, None, 0, None))(keys, mean_1, sqrt_half_delta, mean_2,
                                                           sqrt_half_delta)


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
