#  MIT License
#
#  Copyright (c) 2021 Adrien Corenflos
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import math
from functools import partial
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from chex import ArrayTree, Array
from jax import tree_map, vmap
from jax.scipy.special import logsumexp

from ..resamplings import multinomial, coupled_multinomial

STATE = Any


@partial(jax.jit, static_argnums=(2, 3, 4), donate_argnums=(0, 1))
def operator(inputs_a: STATE, inputs_b: STATE, log_weight_fn: Callable[[ArrayTree, ArrayTree, Any], float],
             n_samples: int, last_step: bool):
    """
    Operator corresponding to the stitching operation of the conditional dSMC algorithm.

    Parameters
    ----------
    inputs_a: STATE
        A tuple of three arguments.
        First one is the state of the partial dSMC smoother to the left of the current time step.
        Second are the jax random keys used for resampling at the time steps to the left of the current time step.
        Third are the parameters used to compute the mixing weights to the left of the current time step.
    inputs_b: STATE
        Same as `inputs_a` but to the right of the current time step
    log_weight_fn: callable
        Function that computes the un-normalised stitching N^2 weights, first argument is x_{t-1}, second is x_t, and
        third is the parameters.
        It will be automatically batched so only needs to be expressed elementwise
    n_samples: int
        Number of samples in the resampling
    last_step: bool
        Whether we are at the last time step or not. If so, we only need one trajectory.

    Returns
    -------

    """

    # Unpack the states
    state_a, keys_a, params_a = inputs_a
    state_b, keys_b, params_b = inputs_b
    trajectories_a, log_weights_a, origins_a = state_a
    trajectories_b, log_weights_b, origins_b = state_b

    weights = get_weights_batch(trajectories_a, log_weights_a,
                                trajectories_b, log_weights_b, params_b,
                                log_weight_fn)
    if last_step:
        # If last step
        idx = jax.random.choice(keys_b[0], n_samples ** 2, p=jnp.ravel(weights))
        l_idx, r_idx = jnp.unravel_index(idx, (n_samples, n_samples))
    else:
        idx = multinomial(keys_b[0], jnp.ravel(weights), n_samples)
        l_idx, r_idx = jax.vmap(jnp.unravel_index, in_axes=[0, None])(idx, (n_samples, n_samples))

    return _gather_results(l_idx, r_idx, n_samples,
                           trajectories_a, origins_a, log_weights_a, keys_a, params_a,
                           trajectories_b, origins_b, log_weights_b, keys_b, params_b)


@partial(jax.jit, static_argnums=(2, 3, 4), donate_argnums=(0, 1))
def coupled_operator(inputs_a: Tuple[STATE, STATE], inputs_b: Tuple[STATE, STATE],
                     log_weight_fn_1: Callable[[ArrayTree, ArrayTree, Any], float],
                     log_weight_fn_2: Callable[[ArrayTree, ArrayTree, Any], float],
                     n_samples: int, last_step: bool):
    """
    Operator corresponding to the stitching operation of the conditional dSMC algorithm.

    Parameters
    ----------
    inputs_a: Tuple[STATE, STATE]
        A tuple of two arguments.
        The first two ones correspond to the inputs of the `operator` function.
        The last one corresponds to coupling flags.
    inputs_b: Tuple[STATE, STATE]
        Same as `inputs_b` but to the right of the current time step
    log_weight_fn_1: callable
        Function that computes the un-normalised stitching N^2 weights, first argument is x_{t-1}, second is x_t, and
        third is the parameters.
        It will be automatically batched so only needs to be expressed elementwise
    log_weight_fn_2: callable
        Same as `log_weight_fn_1` but for the second instance of the coupled dSMC algorithm
    n_samples: int
        Number of samples in the resampling
    last_step: bool
        Whether we are at the last time step or not. If so, we only need one trajectory.

    Returns
    -------

    """

    # Unpack the states
    inputs_a_1, inputs_a_2 = inputs_a
    inputs_b_1, inputs_b_2 = inputs_b

    state_a_1, keys_a_1, params_a_1 = inputs_a_1
    state_a_2, keys_a_2, params_a_2 = inputs_a_2
    state_b_1, keys_b_1, params_b_1 = inputs_b_1
    state_b_2, keys_b_2, params_b_2 = inputs_b_2

    trajectories_a_1, log_weights_a_1, origins_a_1 = state_a_1
    trajectories_a_2, log_weights_a_2, origins_a_2 = state_a_2
    trajectories_b_1, log_weights_b_1, origins_b_1 = state_b_1
    trajectories_b_2, log_weights_b_2, origins_b_2 = state_b_2

    weights_1 = get_weights_batch(trajectories_a_1, log_weights_a_1,
                                  trajectories_b_1, log_weights_b_1, params_b_1,
                                  log_weight_fn_1)
    weights_2 = get_weights_batch(trajectories_a_2, log_weights_a_2,
                                  trajectories_b_2, log_weights_b_2, params_b_2,
                                  log_weight_fn_2)
    if last_step:
        # If last step
        p_1, p_2 = jnp.ravel(weights_1), jnp.ravel(weights_2)
        idx_1, idx_2, idx_coupled = coupled_multinomial(keys_b_1[0], p_1, p_2, 1)
        l_idx_1, r_idx_1 = jnp.unravel_index(idx_1[0], (n_samples, n_samples))
        l_idx_2, r_idx_2 = jnp.unravel_index(idx_2[0], (n_samples, n_samples))

    else:
        p_1, p_2 = jnp.ravel(weights_1), jnp.ravel(weights_2)
        idx_1, idx_2, _ = coupled_multinomial(keys_b_1[0], p_1, p_2, n_samples)
        l_idx_1, r_idx_1 = jax.vmap(jnp.unravel_index, in_axes=[0, None])(idx_1, (n_samples, n_samples))
        l_idx_2, r_idx_2 = jax.vmap(jnp.unravel_index, in_axes=[0, None])(idx_2, (n_samples, n_samples))


    outputs_1 = _gather_results(l_idx_1, r_idx_1, n_samples,
                                trajectories_a_1, origins_a_1, log_weights_a_1, keys_a_1, params_a_1,
                                trajectories_b_1, origins_b_1, log_weights_b_1, keys_b_1, params_b_1)

    outputs_2 = _gather_results(l_idx_2, r_idx_2, n_samples,
                                trajectories_a_2, origins_a_2, log_weights_a_2, keys_a_2, params_a_2,
                                trajectories_b_2, origins_b_2, log_weights_b_2, keys_b_2, params_b_2)

    # Update the coupling flags
    return outputs_1, outputs_2


def _gather_results(left_idx, right_idx, n_samples,
                    trajectories_a, origins_a, log_weights_a, keys_a, params_a,
                    trajectories_b, origins_b, log_weights_b, keys_b, params_b):
    # If we are using conditional dSMC, we need to make sure to preserve the first trajectory.

    # Resample the trajectories

    trajectories_a = tree_map(lambda z: jnp.take(z, left_idx, 1), trajectories_a)
    trajectories_b = tree_map(lambda z: jnp.take(z, right_idx, 1), trajectories_b)

    # Keep track of the trajectories origins for analysis down the line (not used in the algo)
    origins_a = jnp.take(origins_a, left_idx, 1)
    origins_b = jnp.take(origins_b, right_idx, 1)

    # Gather the results
    keys = jnp.concatenate([keys_a, keys_b])
    params = tree_map(lambda a, b: jnp.concatenate([a, b]), params_a, params_b)
    origins = jnp.concatenate([origins_a, origins_b])
    trajectories = tree_map(lambda a, b: jnp.concatenate([a, b]), trajectories_a, trajectories_b)

    log_weights = jnp.concatenate([jnp.full_like(log_weights_a, -math.log(n_samples)),
                                   jnp.full_like(log_weights_b, -math.log(n_samples))])

    return (trajectories, log_weights, origins), keys, params


def get_weights_batch(trajectories_a, log_weights_a,
                      trajectories_b, log_weights_b, params_b,
                      log_weight_fn: Callable[[ArrayTree, ArrayTree, Any], float]):
    # House keeping to get the required inputs.
    params_t = tree_map(lambda z: z[0], params_b)
    x_t_1 = tree_map(lambda z: z[-1], trajectories_a)
    x_t = tree_map(lambda z: z[0], trajectories_b)
    log_w_t_1 = log_weights_a[-1]
    log_w_t = log_weights_b[0]

    log_weights = get_log_weights(x_t_1, log_w_t_1,
                                  x_t, log_w_t, params_t,
                                  log_weight_fn)

    ell_inc = logsumexp(log_weights)
    weights = jnp.exp(log_weights - ell_inc)
    return weights


def get_log_weights(x_t_1, log_w_t_1,
                    x_t, log_w_t, params_t,
                    log_weight_fn):
    # House keeping to get the required inputs.

    # This nested vmap allows to define log_weight_fn more easily at the API level. This is to create a
    # (N,N) -> N^2 function while only having to care about elementwise formulas.
    # if log_weight_fn = lambda a, b: u * v, then this corresponds to np.outer.
    vmapped_log_weight_fn = vmap(vmap(log_weight_fn,
                                      in_axes=[None, 0, None], out_axes=0),
                                 in_axes=[0, None, None], out_axes=0)
    log_weight_increment = vmapped_log_weight_fn(x_t_1, x_t, params_t)  # shape = M, N

    # Take the corresponding time step and reshape to allow for adding residual weights in parallel

    log_weights = log_weight_increment + log_w_t_1[:, None] + log_w_t[None, :]
    return log_weights
