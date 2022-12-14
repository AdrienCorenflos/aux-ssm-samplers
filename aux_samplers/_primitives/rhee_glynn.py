from functools import partial
from typing import Callable

import chex
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from aux_samplers._primitives.base import CoupledSamplerState


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5, 6))
def estimator(key,
              coupled_kernel: Callable[[chex.PRNGKey, CoupledSamplerState], CoupledSamplerState],
              init_sampler: Callable[[chex.PRNGKey], CoupledSamplerState],
              k,
              m,
              test_fn,
              return_standard_term=False):
    if not 1 <= k <= m:
        raise ValueError("We must have 1 <= k <= m")

    key, init_key_1, init_key_2 = jax.random.split(key, 3)
    coupled_state_init = init_sampler(init_key_1)

    # Dephase the chains by one step
    coupled_state = coupled_kernel(init_key_2, coupled_state_init)
    coupled_state.state_2, coupled_state.flags = coupled_state_init.state_2, coupled_state_init.flags

    den = m - k + 1

    # Burnin loop: discard all samples until step k
    def first_body(_, carry):
        key_t, coupled_state_t, coupling_time_t = carry
        key_t_p_1, sample_key = jax.random.split(key_t)
        coupled_state_t_p_1 = coupled_kernel(sample_key, coupled_state_t)

        coupling_time_t = jax.lax.select(coupled_state_t.is_coupled, coupling_time_t, 1 + coupling_time_t)
        return key_t_p_1, coupled_state_t_p_1, coupling_time_t

    key, coupled_state, coupling_time = jax.lax.fori_loop(0, k - 1,
                                                          first_body, (key, coupled_state, 1))

    # Second loop: compute the standard mcmc estimate, and the bias correction term.
    x_k = coupled_state.state_1.x
    init_one = tree_map(lambda z: z / den, test_fn(x_k))  # standard mcmc estimate
    init_two = tree_map(lambda z: 0. * z, test_fn(x_k))  # bias correction term
    init_loop = k, key, coupled_state, init_one, init_two, coupling_time

    def cond(carry):
        t, _, coupled_state_t, *_ = carry
        continue_ = t < m
        continue_ |= ~coupled_state_t.is_coupled
        return continue_

    def second_body(carry):
        t, key_t, coupled_state_t, one_t, two_t, coupling_time_t = carry
        key_t_p_1, sample_key = jax.random.split(key_t, 2)
        coupled_state_t_p_1 = coupled_kernel(sample_key, coupled_state_t)
        t_p_1 = t + 1

        coupled_t_p_1 = coupled_state_t_p_1.is_coupled | coupled_state_t.is_coupled

        # accumulate the standard mcmc estimate
        first_cond = t_p_1 < m

        f_x_t_p_1 = test_fn(coupled_state_t_p_1.state_1.x)

        def if_first(): return tree_map(lambda u, v: u + v / den, one_t, f_x_t_p_1)

        def else_first(): return one_t

        one_t_p_1 = jax.lax.cond(first_cond, if_first, else_first)

        # accumulate the bias correction term
        def if_not_coupled():
            factor = jnp.minimum(1., (t_p_1 - k) / den)
            # correction = correction + factor * (h(x) - h(y)) but with tree handling
            f_y_t = test_fn(coupled_state_t_p_1.state_2.x)
            return tree_map(lambda u, v, w: u + factor * (v - w), two_t, f_x_t_p_1, f_y_t)

        def if_coupled():
            return two_t

        two_t_p_1 = jax.lax.cond(coupled_t_p_1, if_coupled, if_not_coupled)
        coupling_time_t = jax.lax.select(coupled_state_t.is_coupled, coupling_time_t, 1 + coupling_time_t)

        return t_p_1, key_t_p_1, coupled_state_t_p_1, one_t_p_1, two_t_p_1, coupling_time_t

    total_t, *_, standard_term, bias_correction, coupling_time = jax.lax.while_loop(cond, second_body, init_loop)
    h_km = standard_term + bias_correction
    if return_standard_term:
        return h_km, total_t, coupling_time, standard_term
    return h_km, total_t, coupling_time
