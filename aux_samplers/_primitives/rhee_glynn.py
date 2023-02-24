from functools import partial
from typing import Callable, Any

import chex
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map


@partial(jax.jit, static_argnums=(1, 3, 4, 5, 6))
def estimator(key,
              coupled_kernel: Callable[[chex.PRNGKey, Any], Any],
              coupled_state_init: Any,
              k,
              m,
              test_fn,
              return_standard_term=False):
    """
    Rhee & Glynn estimator.
    This returns the standard mcmc estimate, and the bias correction term. It is computed online and in a single pass.

    The coupled_state_init is the initial state of the coupled sampler. It is assumed that the sampler has already been
    delayed for one step.

    Parameters
    ----------
    key
    coupled_kernel
    coupled_state_init
    k
    m
    test_fn
    return_standard_term

    Returns
    -------

    """

    if not 1 <= k <= m:
        raise ValueError("We must have 1 <= k <= m")

    den = m - k

    # Burnin loop: discard all samples until step k
    def first_body(_, carry):
        key_t, coupled_state_t, coupling_time_t = carry
        key_t_p_1, sample_key = jax.random.split(key_t)
        coupled_state_t_p_1 = coupled_kernel(sample_key, coupled_state_t)

        coupling_time_t = jax.lax.select(coupled_state_t.is_coupled, coupling_time_t, 1 + coupling_time_t)
        return key_t_p_1, coupled_state_t_p_1, coupling_time_t

    key, coupled_state, coupling_time = jax.lax.fori_loop(0, k - 1,
                                                          first_body, (key, coupled_state_init, 1))

    # Second loop: compute the standard mcmc estimate, and the bias correction term.
    init_one = tree_map(lambda z: z / den, test_fn(coupled_state.state_1))  # standard mcmc estimate
    init_two = tree_map(lambda z: 0. * z, test_fn(coupled_state.state_1))  # bias correction term
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

        jax.debug.print("coupled: {}", coupled_state_t_p_1.flags.mean())
        jax.debug.print("theta_coupled: {}", coupled_state_t_p_1.theta_coupled)
        # accumulate the standard mcmc estimate
        first_cond = t_p_1 < m

        f_x_t_p_1 = test_fn(coupled_state_t_p_1.state_1)

        def if_first(): return tree_map(lambda u, v: u + v / den, one_t, f_x_t_p_1)  # one_t + f_x_t_p_1 / den

        def else_first(): return one_t

        one_t_p_1 = jax.lax.cond(first_cond, if_first, else_first)

        # accumulate the bias correction term
        def if_not_coupled():
            factor = jnp.minimum(1., (t_p_1 - k) / den)
            # correction = correction + factor * (h(x) - h(y)) but with tree handling
            f_y_t = test_fn(coupled_state_t_p_1.state_2)
            return tree_map(lambda u, v, w: u + factor * (v - w), two_t, f_x_t_p_1, f_y_t)

        def if_coupled():
            return two_t

        two_t_p_1 = jax.lax.cond(coupled_t_p_1, if_coupled, if_not_coupled)
        coupling_time_t = jax.lax.select(coupled_state_t.is_coupled, coupling_time_t, 1 + coupling_time_t)

        return t_p_1, key_t_p_1, coupled_state_t_p_1, one_t_p_1, two_t_p_1, coupling_time_t

    total_t, *_, standard_term, bias_correction, coupling_time = jax.lax.while_loop(cond, second_body, init_loop)

    h_km = tree_map(lambda u, v: u + v, standard_term, bias_correction)
    if return_standard_term:
        return h_km, total_t, coupling_time, standard_term
    return h_km, total_t, coupling_time
