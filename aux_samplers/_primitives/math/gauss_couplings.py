"""Coupling utilities"""
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from .logpdf import mvn_loglikelihood

_EPS = 0.01  # this is a small float to make sure that log2(2**k) = k exactly


def thorisson(key, p, q, log_p, log_q, C=1.):
    """
    Modified version of Thorisson coupling algorithm [4] as suggested in [5].

    Parameters
    ----------
    key: PRNGKey
       JAX random key
    p, q: callable
        Sample from the marginals. Take a JAX key and returns a sample.
    log_p, log_q: callable
        The log densities of the dominating marginals and the target ones. They take arrays (d) and return a float.
    C: float, optional
        Constant to control the variance of the run time, this will be clipped between 0 and 1. Default is 1.

    Returns
    -------
    X: jnp.ndarray
        The resulting sample for p
    Y: jnp.ndarray
        The resulting sampled for q
    is_coupled: bool
        Do we have X = Y? Note that if the distributions are not continuous this may be False even if X=Y.
    n_trials: int
        The number of trials before acceptance

    """

    C = jnp.clip(C, 0., 1.)
    key, init_key, init_accept_key = jax.random.split(key, 3)

    log_w = lambda x: log_q(x) - log_p(x)

    def log_phi(x):
        return jnp.minimum(log_w(x), jnp.log(C))

    X = p(init_key)
    log_u = jnp.log(jax.random.uniform(init_accept_key))

    # P(accept) = phi(X)
    accept_X_init = log_u < log_phi(X)

    def cond(carry):
        accepted, *_ = carry
        return ~accepted

    def body(carry):
        *_, i, current_key = carry
        next_key, sample_key, accept_key = jax.random.split(current_key, 3)
        Y = q(sample_key)
        log_v = jnp.log(jax.random.uniform(accept_key))

        # P(accept) = 1 - phi(Y)/w(Y)
        accept = log_v > log_phi(Y) - log_w(Y)
        return accept, Y, i + 1, next_key

    _, Z, n_trials, _ = jax.lax.while_loop(cond, body, (accept_X_init, X, 1, key))

    return X, Z, accept_X_init, n_trials


def _thorisson_mvn_coupling(k, m1, L1, m2, L2, C=1.):
    p = lambda k_: m1 + L1 @ jax.random.normal(k_, m1.shape)
    q = lambda k_: m2 + L2 @ jax.random.normal(k_, m2.shape)

    log_p = lambda x: mvn_loglikelihood(x, m1, L1)
    log_q = lambda x: mvn_loglikelihood(x, m2, L2)

    return thorisson(k, p, q, log_p, log_q, C)


def _rejection_mvn_coupling(k, m1, L1, m2, L2, N=1):
    return coupled_mvns(k, m1, L1, m2, L2, N)


def _reflection_mvn_coupling(k, m1, L1, m2, _L2):
    res_1, res_2, do_accept = reflection_maximal(k, 1, m1, m2, L1)
    return res_1[0], res_2[0], do_accept[0], 1


def _lindvall_roger(key, m1, L1, m2, L2):
    dim = m1.shape[0]
    z = solve_triangular(L2, (m1 - m2))
    e = z / jnp.linalg.norm(z)
    norm_1 = jax.random.normal(key, dim)
    norm_2 = norm_1 - 2 * jnp.dot(norm_1, e) * e

    x_1 = m1 + L1 @ norm_1
    x_2 = m2 + L2 @ norm_2
    return x_1, x_2


def _lindvall_roger_diagonal(key, m1, L1, m2, L2):
    dim = m1.shape[0]
    z = (m1 - m2) / L1
    e = z / jnp.linalg.norm(z)
    norm_1 = jax.random.normal(key, dim)
    norm_2 = norm_1 - 2 * jnp.dot(norm_1, e) * e

    x_1 = m1 + L1 * norm_1
    x_2 = m2 + L2 * norm_2
    return x_1, x_2


def _modified_lindvall_roger(k, m1, L1, m2, L2):
    dim = m1.shape[0]
    k1, k2, k3, k4 = jax.random.split(k, 4)
    x_1, x_2 = _lindvall_roger(k1, m1, L1, m2, L2)

    log_u = jnp.log(jax.random.uniform(k2))

    eps_y = jax.random.normal(k4, (dim,))
    y = m1 + L1 @ eps_y
    log_v = jnp.log(jax.random.uniform(k3))

    def if_true():
        flag_1 = log_u < mvn_loglikelihood(x_1, m2, L2) - mvn_loglikelihood(x_1, m1, L1)
        flag_2 = log_u < mvn_loglikelihood(x_2, m1, L1) - mvn_loglikelihood(x_2, m2, L2)
        z_1 = jax.lax.select(flag_1, y, x_1)
        z_2 = jax.lax.select(flag_2, y, x_2)
        return z_1, z_2, flag_1 & flag_2, 1

    def if_false():
        return x_1, x_2, False, 1

    cond = log_v < mvn_loglikelihood(y, m2, L2) - mvn_loglikelihood(y, m1, L1)
    return jax.lax.cond(cond, if_true, if_false)
