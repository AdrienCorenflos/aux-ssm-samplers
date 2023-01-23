"""
Implements the parallel-in-time cSMC kernel from Corenflos et al. (2022). Reimplemented from the original code in order
to preserve a common API feel in this library (to some extent).
"""
from typing import Optional

import jax
from jax import numpy as jnp, tree_map
from jax.scipy.special import logsumexp

from .dc_map import dc_map
from .operator import operator
from ..base import Distribution, UnivariatePotential, Potential, CSMCState, CoupledDistribution
from ...base import CoupledSamplerState


def get_coupled_kernel(cMt: CoupledDistribution, G0_1: UnivariatePotential, G0_2: UnivariatePotential, Gt_1: Potential,
                       Gt_2: Potential, N: int, Qt_1: Optional[Distribution] = None,
                       Qt_2: Optional[Distribution] = None):
    """
    Get a parallel-in-time coupled cSMC kernel. For each leg, this will target the model (up to proportionality)
    .. math::
        Mt[0](x_0) G0(x_0)\prod_{t=1}^T Mt[t](x_t) Gt[t](x_t, x_{t-1})
    or, if `Qt` is provided, the model
    .. math::
        Qt[0](x_0) G0(x_0)\prod_{t=1}^T Qt[t](x_t) Gt[t](x_t, x_{t-1})
    but using `Mt` as the proposal distribution.
    Parameters:
    -----------
    cMt: CoupledDistribution
        Proposal distributions per time-step.
        This will be called as `jax.vmap(lambda ms, key: ms.sample(key, N))(cMs, keys)`.
        For example, for a common random number coupling, `cMs` should be defined as an instance of
        ```python
            @chex.dataclass
            class cMs(CoupledDistribution):
                mean_1: Array  # mean of the normal distribution
                mean_2: Array  # mean of the normal distribution
                def sample(self, key, N):
                    d = self.mean_1.shape[0]
                    eps = jax.random.normal(key, (N, d))
                    coupled = jnp.allclose(self.mean_1, self.mean_2)
                    return self.mean_1[None, :] + eps, self.mean_2[None, :] + eps, jnp.repeat(coupled, N)
        ```
        Then `jax.vmap(lambda ms, key: ms.sample(key, N))(Ms, keys)` will return a tensor of shape `(Ms.u.shape[0], N, Ms.u.shape[1])`.
    G0_1, G0_2: UnivariatePotential
        Initial potential for the first time step.
    Gt_1, Gt_2: Potential
        Potential of the model.
    N: int
        Total number of particles to use in the cSMC sampler.
    Qt_1, Qt_2: Distribution, optional
        Optional. If provided, this will be used to compute the importance weights.

    Returns:
    --------
    kernel: Callable
        cSMC kernel.
    init: Callable
        Function to initialize the state of the sampler given a trajectory.
    """
    if Qt_1 is not None or Qt_2 is not None:
        raise NotImplementedError("Qt support is not implemented yet.")

    def kernel(key, coupled_state):
        state_1, state_2, coupled_flags = coupled_state.state_1, coupled_state.state_2, coupled_state.flags
        x_1, ancestors_1, x_2, ancestors_2, coupled_flags = _ccsmc(key, state_1.x, state_2.x, cMt, G0_1, G0_2, Gt_1,
                                                                   Gt_2, N, Qt_1, Qt_2)
        state_1 = CSMCState(x=x_1, updated=ancestors_1 != 0)
        state_2 = CSMCState(x=x_2, updated=ancestors_2 != 0)
        coupled_state = CoupledSamplerState(state_1=state_1, state_2=state_2, flags=coupled_flags)
        return coupled_state

    def init(x_star_1, x_star_2):
        T, *_ = x_star_1.shape
        ancestors = jnp.zeros((T,), dtype=jnp.int_)
        state_1 = CSMCState(x=x_star_1, updated=ancestors == 0)
        state_2 = CSMCState(x=x_star_2, updated=ancestors == 0)
        coupled_state = CoupledSamplerState(state_1=state_1, state_2=state_2, flags=jnp.zeros((T,), dtype=jnp.bool_))
        return coupled_state

    return init, kernel


def _ccsmc(key, x_star_1, x_star_2, cMt, G0_1, G0_2, Gt_1, Gt_2, N, Qt_1, Qt_2):
    T = x_star_1.shape[0]
    sampling_key, resampling_key = jax.random.split(key)
    sampling_keys = jax.random.split(sampling_key, T)
    resampling_keys = jax.random.split(resampling_key, T)

    # Sample all the proposals
    xs_1, xs_2, flags = jax.vmap(lambda ms, k: ms.sample(k, N))(cMt, sampling_keys)

    # Replace the first particle with the star trajectory
    xs_1 = xs_1.at[:, 0].set(x_star_1)
    xs_2 = xs_2.at[:, 0].set(x_star_2)

    # Compute initial weights and normalize
    log_wts = jnp.zeros((T, N))

    log_w0_1, log_w0_2 = G0_1(xs_1[0]), G0_2(xs_2[0])
    log_wts_1, log_wts_2 = log_wts.at[0].add(log_w0_1), log_wts.at[0].add(log_w0_2)

    log_wts_1 -= logsumexp(log_wts_1, axis=1, keepdims=True)
    log_wts_2 -= logsumexp(log_wts_2, axis=1, keepdims=True)

    # Original ancestors:
    origins = jnp.tile(jnp.arange(N), (T, 1))
    states = (xs_1, log_wts_1, xs_2, log_wts_2, origins, origins)

    # Housekeeping for the parameters, we need to add an extra line for the initial state to keep shapes consistent
    params = Gt_1.params, Gt_2.params
    # Take the shape of the parameters but fill with NaNs
    fake_params = tree_map(lambda x: jnp.ones_like(x[0]) * jnp.nan, params)
    # Insert fake parameters at the beginning, they will never be used.
    params = jax.tree_map(lambda x, y: jnp.insert(x, 0, y, axis=0), params, fake_params)

    def log_weight_fn(x_t_1, x_t, params_t):
        return Gt(x_t, x_t_1, params_t)

    csmc_operator = lambda inputs_a, inputs_b: operator(inputs_a, inputs_b, log_weight_fn, N, False)
    last_csmc_operator = lambda inputs_a, inputs_b: operator(inputs_a, inputs_b, log_weight_fn, N, True)

    inputs = states, resampling_keys, params

    state_out, *_ = dc_map(inputs, jax.vmap(csmc_operator), jax.vmap(last_csmc_operator))
    xs, _, ancestors = state_out
    return xs, ancestors
