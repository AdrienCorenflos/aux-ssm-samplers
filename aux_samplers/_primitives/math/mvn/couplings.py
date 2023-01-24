"""Coupling utilities"""
from functools import partial
from typing import Union

import chex
import jax
import jax.numpy as jnp
from chex import Numeric
from jax.scipy.linalg import solve_triangular
from jaxtyping import Float, Array

from .base import logpdf, get_optimal_covariance, rvs, tril_log_det
from ..couplings import thorisson as thorisson_gen, coupled_sampler

_EPS = 0.01  # this is a small float to make sure that log2(2**k) = k exactly


def thorisson(key, m1, L1, m2, L2, C=1.):
    """
    Thorisson coupling between two multivariate normal distributions with
    the same covariance matrix.

    Parameters
    ----------
    key: jax.random.PRNGKey
        The random key for JAX
    m1: jnp.ndarray
        The mean of the first normal distribution
    L1: jnp.ndarray
        The Cholesky factor of the covariance matrix of the first normal distribution
    m2: jnp.ndarray
        The mean of the second normal distribution
    L2: jnp.ndarray
        The Cholesky factor of the covariance matrix of the second normal distribution
    C: float
        The constant that controls the desired relative level of coupling

    Returns
    -------
    jnp.ndarray
        The first sample
    jnp.ndarray
        The second sample
    bool
        Whether the samples are coupled
    """
    p = lambda k_: m1 + L1 @ jax.random.normal(k_, m1.shape)
    q = lambda k_: m2 + L2 @ jax.random.normal(k_, m2.shape)

    log_p = lambda z: logpdf(z, m1, L1)
    log_q = lambda z: logpdf(z, m2, L2)

    return thorisson_gen(key, p, q, log_p, log_q, C)


def rejection(key: chex.PRNGKey,
              m1: Float[Array, "dim"], L1: Float[Array, "dim dim"],
              m2: Float[Array, "dim"], L2: Float[Array, "dim dim"], N: int = 1):
    """
    Rejection coupling between two multivariate normal distributions with
    different covariance matrices. It uses the reflection maximal coupling as a proposal.

    Parameters
    ----------
    key : jax.random.PRNGKey
        The random key for JAX
    m1 : jnp.ndarray
        The mean of the first normal distribution
    L1 : jnp.ndarray
        The Cholesky factor of the covariance matrix of the first normal distribution
    m2 : jnp.ndarray
        The mean of the second normal distribution
    L2 : jnp.ndarray
        The Cholesky factor of the covariance matrix of the second normal distribution
    N : int
        The number of samples used in the rejection sampler

    Returns
    -------
    jnp.ndarray
        The first sample
    jnp.ndarray
        The second sample
    bool
        Whether the samples are coupled
    """

    chol_Q = get_optimal_covariance(L1, L2)
    log_det_chol_P = tril_log_det(L1)
    log_det_chol_Sig = tril_log_det(L2)
    log_det_chol_Q = tril_log_det(chol_Q)

    log_M_P_Q = jnp.maximum(log_det_chol_Q - log_det_chol_P, 0.)
    log_M_Sigma_Q = jnp.maximum(log_det_chol_Q - log_det_chol_Sig, 0.)

    Gamma_hat = partial(reflection_maximal, m=m1, mu=m2, chol_Q=chol_Q)
    log_p = lambda x: logpdf(x, m1, L1)
    log_q = lambda x: logpdf(x, m2, L2)

    log_p_hat = lambda x: logpdf(x, m1, chol_Q)
    log_q_hat = lambda x: logpdf(x, m2, chol_Q)

    p = lambda kk: rvs(kk, m1, L1)
    q = lambda kk: rvs(kk, m2, L2)

    return coupled_sampler(key, Gamma_hat, p, q, log_p_hat, log_q_hat, log_p, log_q, log_M_P_Q, log_M_Sigma_Q, N)


def reflection(key, m1, L1, m2, _L2):
    """
    Reflection maximal coupling between two multivariate normal distributions with the same covariance matrix.
    This will spit out garbage if the covariance matrices are not the same.

    Parameters
    ----------
    key : jax.random.PRNGKey
        The random key for JAX
    m1 : jnp.ndarray
        The mean of the first normal distribution
    L1 : jnp.ndarray
        The Cholesky factor of the covariance matrix of the first normal distribution
    m2 : jnp.ndarray
        The mean of the second normal distribution
    _L2 : jnp.ndarray
        The Cholesky factor of the covariance matrix of the second normal distribution

    Returns
    -------
    jnp.ndarray
        The first sample
    jnp.ndarray
        The second sample
    bool
        Whether the samples are coupled
    """
    x1, x2, coupled = reflection_maximal(key, 1, m1, m2, L1)
    return x1[0], x2[0], coupled[0]


def lindvall_roger(key, m1, L1, m2, L2):
    """
    Linvall Roger (reflection) coupling between two multivariate normal distributions with different covariance matrix.
    This will spit out garbage if the covariance matrices are not the same.

    Parameters
    ----------
    key : jax.random.PRNGKey
        The random key for JAX
    m1 : jnp.ndarray
        The mean of the first normal distribution
    L1 : jnp.ndarray
        The Cholesky factor of the covariance matrix of the first normal distribution
    m2 : jnp.ndarray
        The mean of the second normal distribution
    L2 : jnp.ndarray
        The Cholesky factor of the covariance matrix of the second normal distribution

    Returns
    -------
    jnp.ndarray
        The first sample
    jnp.ndarray
        The second sample
    bool
        Whether the samples are coupled. This will only be true if the covariance matrices and the means are the same.
    """

    dim = m1.shape[0]
    if jnp.ndim(L1) == 2:
        z = solve_triangular(L2, (m1 - m2))
    else:
        z = (m1 - m2) / L2
    e = z / jnp.linalg.norm(z)
    norm_1 = jax.random.normal(key, (dim,))
    norm_2 = norm_1 - 2 * jnp.dot(norm_1, e) * e

    if jnp.ndim(L1) == 2:
        x_1 = m1 + L1 @ norm_1
        x_2 = m2 + L2 @ norm_2
    else:
        x_1 = m1 + L1 * norm_1
        x_2 = m2 + L2 * norm_2
    return x_1, x_2, jnp.allclose(x_1, x_2)


def modified_lindvall_roger(key, m1, L1, m2, L2):
    """
    Modified Lindvall Roger (reflection) coupling between two multivariate normal distributions with different covariance matrix.

    Parameters
    ----------
    key : jax.random.PRNGKey
        The random key for JAX
    m1 : jnp.ndarray
        The mean of the first normal distribution
    L1 : jnp.ndarray
        The Cholesky factor of the covariance matrix of the first normal distribution
    m2 : jnp.ndarray
        The mean of the second normal distribution
    L2 : jnp.ndarray
        The Cholesky factor of the covariance matrix of the second normal distribution

    Returns
    -------
    jnp.ndarray
        The first sample
    jnp.ndarray
        The second sample
    bool
        Whether the samples are coupled. This will only be true if the covariance matrices and the means are the same.
    """
    dim = m1.shape[0]
    k1, k2, k3, k4 = jax.random.split(key, 4)
    x_1, x_2, _ = lindvall_roger(k1, m1, L1, m2, L2)

    log_u = jnp.log(jax.random.uniform(k2))

    eps_y = jax.random.normal(k4, (dim,))
    y = m1 + L1 @ eps_y
    log_v = jnp.log(jax.random.uniform(k3))

    def if_true():
        flag_1 = log_u < logpdf(x_1, m2, L2) - logpdf(x_1, m1, L1)
        flag_2 = log_u < logpdf(x_2, m1, L1) - logpdf(x_2, m2, L2)
        z_1 = jax.lax.select(flag_1, y, x_1)
        z_2 = jax.lax.select(flag_2, y, x_2)
        return z_1, z_2, flag_1 & flag_2

    def if_false():
        return x_1, x_2, False

    cond = log_v < logpdf(y, m2, L2) - logpdf(y, m1, L1)
    return jax.lax.cond(cond, if_true, if_false)


def reflection_maximal(key, N: int, m: Array, mu: Array, chol_Q: Union[Array, Numeric]):
    """
    Sample N pairs of points from the reflection maximal coupling of two multivariate normal distributions.
    The first distribution is N(mu, chol_Q^T chol_Q) and the second is N(m, chol_Q^T chol_Q).

    Parameters
    ----------
    key : jax.random.PRNGKey
        The random key for JAX
    N : int
        The number of samples to draw
    m : jnp.ndarray
        The mean of the first normal distribution
    mu : jnp.ndarray
        The mean of the second normal distribution
    chol_Q : jnp.ndarray
        The Cholesky factor of the covariance matrix of the first and second normal distribution

    Returns
    -------
    jnp.ndarray
        The first samples
    jnp.ndarray
        The second samples
    jnp.ndarray
        The acceptance flags
    """
    dim = m.shape[0]

    if jnp.ndim(chol_Q) == 2:
        z = solve_triangular(chol_Q, m - mu, lower=True)
    else:
        z = (m - mu) / chol_Q

    e = z / jnp.linalg.norm(z)

    normal_key, uniform_key = jax.random.split(key, 2)
    norm = jax.random.normal(normal_key, (N, dim))
    log_u = jnp.log(jax.random.uniform(uniform_key, (N,)))

    temp = norm + z[None, :]

    mvn_loglikelihood = lambda x: - 0.5 * jnp.sum(x ** 2, -1)

    do_accept = log_u + mvn_loglikelihood(norm) < mvn_loglikelihood(temp)

    reflected_norm = jnp.where(do_accept[:, None], temp, norm - 2 * jnp.outer(jnp.dot(norm, e), e))

    if jnp.ndim(chol_Q) == 2:
        res_1 = m[None, :] + norm @ chol_Q.T
        res_2 = mu[None, :] + reflected_norm @ chol_Q.T
    else:
        res_1 = m[None, :] + norm * chol_Q
        res_2 = mu[None, :] + reflected_norm * chol_Q

    return res_1, res_2, do_accept
