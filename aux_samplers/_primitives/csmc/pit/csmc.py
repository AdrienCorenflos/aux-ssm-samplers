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
from ..base import Distribution, UnivariatePotential, Potential, CSMCState
from ..resamplings import systematic


def get_kernel(Mt: Distribution, G0: UnivariatePotential, Gt: Potential, N: int, Qt: Optional[Distribution] = None,
               resampling=systematic):
    """
    Get a parallel-in-time cSMC kernel. This will target the model (up to proportionality)
    .. math::
        Mt[0](x_0) G0(x_0)\prod_{t=1}^T Mt[t](x_t) Gt[t](x_t, x_{t-1})
    or, if `Qt` is provided, the model
    .. math::
        Qt[0](x_0) G0(x_0)\prod_{t=1}^T Qt[t](x_t) Gt[t](x_t, x_{t-1})
    but using `Mt` as the proposal distribution.
    Parameters:
    -----------
    Mt:
        Proposal distributions per time-step.
        This will be called as `jax.vmap(lambda ms, key: ms.sample(key, N))(Ms, keys)`.
        For example, `Ms` should be defined as
        ```python
            @chex.dataclass
            class Ms(Distribution):
                mean: Array  # mean of the normal distribution
                def sample(self, key, N):
                    return self.mean[None, :] + jax.random.normal(key, (N, self.u.shape[0]))
        ```
        Then `jax.vmap(lambda ms, key: ms.sample(key, N))(Ms, keys)` will return a tensor of shape `(Ms.u.shape[0], N, Ms.u.shape[1])`.
    G0:
        Initial potential for the first time step.
    Gt:
        Potential of the model.
    N: int
        Total number of particles to use in the cSMC sampler.
    Qt:
        Optional. If provided, this will be used to compute the importance weights.
    resampling:
        Resampling function to use.

    Returns:
    --------
    kernel: Callable
        cSMC kernel.
    init: Callable
        Function to initialize the state of the sampler given a trajectory.
    """

    def kernel(key, state):
        x, ancestors = _csmc(key, state.x, Mt, G0, Gt, N, Qt, resampling)
        return CSMCState(x=x, updated=ancestors != 0)

    def init(x_star):
        T, *_ = x_star.shape
        ancestors = jnp.zeros((T,), dtype=int)
        return CSMCState(x=x_star, updated=ancestors == 0)

    return init, kernel


def _csmc(key, x_star, Mt, G0, Gt, N, Qt, resampling):
    T = x_star.shape[0]
    sampling_key, resampling_key = jax.random.split(key)
    sampling_keys = jax.random.split(sampling_key, T)
    resampling_keys = jax.random.split(resampling_key, T)

    # Sample all the proposals
    xs = jax.vmap(lambda ms, k: ms.sample(k, N))(Mt, sampling_keys)

    # Replace the first particle with the star trajectory
    xs = xs.at[:, 0].set(x_star)

    # Compute initial weights and normalize
    # FIXME: I am not sure where the logweights of the proposal should live. In the operator or here? Theoretically, no diff,
    #        but in practice, the degeneracy behaviour may change.
    if Qt is not None:
        log_wts = jax.vmap(lambda qs, x: qs.logpdf(x))(Qt, xs)
        log_wts -= jax.vmap(lambda ms, x: ms.logpdf(x))(Mt, xs)
    else:
        log_wts = jnp.zeros((T, N))

    log_w0 = G0(xs[0])
    log_wts = log_wts.at[0].add(log_w0)
    log_wts -= logsumexp(log_wts, axis=1, keepdims=True)

    # Original ancestors:
    origins = jnp.tile(jnp.arange(N), (T, 1))
    states = (xs, log_wts, origins)

    # Housekeeping for the parameters, we need to add an extra line for the initial state to keep shapes consistent
    params = Gt.params
    # Take the shape of the parameters but fill with NaNs
    fake_params = tree_map(lambda x: jnp.ones_like(x[0]) * jnp.nan, params)
    # Insert fake parameters at the beginning, they will never be used.
    params = jax.tree_map(lambda x, y: jnp.insert(x, 0, y, axis=0), params, fake_params)

    def log_weight_fn(x_t_1, x_t, params_t):
        return Gt(x_t, x_t_1, params_t)

    csmc_operator = lambda inputs_a, inputs_b: operator(inputs_a, inputs_b, log_weight_fn, N, False, resampling)
    last_csmc_operator = lambda inputs_a, inputs_b: operator(inputs_a, inputs_b, log_weight_fn, N, True, None)

    inputs = states, resampling_keys, params

    state_out, *_ = dc_map(inputs, jax.vmap(csmc_operator), jax.vmap(last_csmc_operator))
    xs, _, ancestors = state_out
    return xs, ancestors
