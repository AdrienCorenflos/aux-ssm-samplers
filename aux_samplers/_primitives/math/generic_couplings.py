import math
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.random
from chex import PRNGKey
from jax import numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Float, Array, PyTree, Int

from .utils import logsubexp, normalize


def coupled_sampler(key: jnp.ndarray,
                    Gamma_hat: Callable,
                    p: Callable, q: Callable,
                    log_p_hat: Callable, log_q_hat: Callable,
                    log_p: Callable, log_q: Callable,
                    log_M_p: float, log_M_q: float,
                    N: int = 1):
    """
    This is the general code for Algorithm 3 of the coupled rejection method paper [1]. In the case when `N` it
    reduces to Algorithm 1.
    Parameters
    ----------
    key: jnp.ndarray
       JAX random key
    Gamma_hat: callable
        Sample from the coupling of p_hat and q_hat. Takes a random key from JAX, the number of desired samples M,
        and returns a pair of samples from the coupling, both with shape (M, d) and a flag saying if the sample returned is
        successfully coupled.
    p, q: callable
        Sample from the marginals. Takes a JAX key and returns a sample. This is used when one of the
        acceptance flags is false.
    log_p_hat, log_q_hat, log_p, log_q: callable
        The log densities of the dominating marginals and the target ones. They take arrays (N, d) and return an array (N,).
    log_M_p: float
        Logarithm of the dominating constant for log_p and log_p_hat: log_p < log_M_p + log_p_hat
    log_M_q: float
        Logarithm of the dominating constant for log_q and log_q_hat: log_q < log_M_q + log_q_hat
    N: int, optional
        The number of particles to be used in the ensemble. Default is 1, which reduces to simple coupled
        rejection sampling.

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
    References
    ----------
    .. [1]
    """

    def _accept_proposal_and_acceptance_ratio(op_key, Xs_hat, Ys_hat, are_coupled):
        # unnormalised log weights
        select_key, accept_key = jax.random.split(op_key, 2)
        log_w_X = log_p(Xs_hat) - log_p_hat(Xs_hat)
        log_w_Y = log_q(Ys_hat) - log_q_hat(Ys_hat)

        if N == 1:
            X_hat, Y_hat = Xs_hat[0], Ys_hat[0]
            coupled_proposal = are_coupled[0]
            X_acceptance_proba = log_w_X[0] - log_M_p
            Y_acceptance_proba = log_w_Y[0] - log_M_q
        else:
            log_N = math.log(N)

            # log likelihood of the samples
            log_Z_X_hat = logsumexp(log_w_X) - log_N
            log_Z_Y_hat = logsumexp(log_w_Y) - log_N

            # Normalised log weights
            log_W_X = log_w_X - log_N - log_Z_X_hat
            log_W_Y = log_w_Y - log_N - log_Z_Y_hat
            W_X, W_Y = normalize(log_W_X), normalize(log_W_Y)
            # Coupled index sampling
            I, J, same_index = index_max_coupling(select_key, log_W_X, log_W_Y, 1)

            # Select the proposal
            X_hat, Y_hat = Xs_hat[I], Ys_hat[J]
            coupled_proposal = same_index & are_coupled[I]

            # Compute the upper bounds
            log_Z_X_bar = jnp.logaddexp(log_Z_X_hat, logsubexp(log_M_p, log_w_X[I]) - log_N)
            log_Z_Y_bar = jnp.logaddexp(log_Z_Y_hat, logsubexp(log_M_q, log_w_Y[J]) - log_N)

            X_acceptance_proba = log_Z_X_hat - log_Z_X_bar
            Y_acceptance_proba = log_Z_Y_hat - log_Z_Y_bar

        log_u = jnp.log(jax.random.uniform(accept_key))
        accept_X = log_u < X_acceptance_proba
        accept_Y = log_u < Y_acceptance_proba
        return accept_X, accept_Y, X_hat, Y_hat, coupled_proposal

    def cond(carry):
        accept_X, accept_Y, *_ = carry
        return ~accept_X & ~accept_Y

    def body(carry):
        *_, i, curr_key = carry
        next_key, sample_key, accept_key = jax.random.split(curr_key, 3)
        Xs_hat, Ys_hat, are_coupled = Gamma_hat(sample_key, N)

        accept_X, accept_Y, X_hat, Y_hat, coupled_proposal = _accept_proposal_and_acceptance_ratio(accept_key,
                                                                                                   Xs_hat, Ys_hat,
                                                                                                   are_coupled)
        return accept_X, accept_Y, X_hat, Y_hat, coupled_proposal, i + 1, next_key

    # initialisation
    init_key, key = jax.random.split(key)
    X_init = p(init_key)
    Y_init = q(init_key)

    output = jax.lax.while_loop(cond,
                                lambda carry: body(carry),
                                (False, False, X_init, Y_init, False, 0, key))

    is_X_accepted, is_Y_accepted, X, Y, is_coupled, n_trials, _ = output

    X = jax.lax.select(is_X_accepted, X, X_init)
    Y = jax.lax.select(is_Y_accepted, Y, Y_init)

    is_coupled = is_coupled & is_X_accepted & is_Y_accepted

    return X, Y, is_coupled, n_trials


# def index_max_coupling(key, log_W_X, log_W_Y):
#     """
#     Maximum coupling of multinomials.
#     Parameters
#     ----------
#     key
#     log_W_X, log_W_Y
#     Returns
#     -------
#     I: int
#         Chosen index for X
#     J: int
#         Chosen index for Y
#     is_coupled:
#         Do we have I = J
#     """
#     N = log_W_X.shape[0]
#
#     key_1, key_2 = jax.random.split(key)
#     # compute the overlap
#     log_nu = jnp.minimum(log_W_X, log_W_Y)
#     log_alpha = logsumexp(log_nu)
#
#     # sample to know if we are coupled
#     log_u = jnp.log(jax.random.uniform(key_1))
#     is_coupled = log_u < log_alpha
#
#     # sample
#     def if_coupled(k):
#         nu = jnp.exp(log_nu - log_alpha)
#         idx_X = idx_Y = jax.random.choice(k, N, p=nu)
#         return idx_X, idx_Y
#
#     def otherwise(k):
#         # compute the residuals
#         log_1_minus_alpha = log1mexp(log_alpha)
#         log_r_1 = logsubexp(log_W_X, log_nu) - log_1_minus_alpha
#         log_r_2 = logsubexp(log_W_Y, log_nu) - log_1_minus_alpha
#
#         # Note that the fact the same key is used does not matter here as we are sampling from
#         # the uncoupled bit, so we don't care.
#         idx_X = jax.random.choice(k, N, p=jnp.exp(log_r_1))
#         idx_Y = jax.random.choice(k, N, p=jnp.exp(log_r_2))
#
#         return idx_X, idx_Y
#
#     I, J = jax.lax.cond(is_coupled, if_coupled, otherwise, key_2)
#     return I, J, is_coupled
def index_max_coupling(
        key: PRNGKey, weights_1: Float[Array, "dim_x"], weights_2: Float[Array, "dim_x"], N: Optional[int] = None
) -> PyTree[Int[Array, "dim_x"]]:
    """
    Maximal coupling of unconditional multinomial resampling schemes. The weights are assumed to be normalised already.
    The index 0 is always left unchanged.

    Parameters
    ----------
    key:
        Random number generator key.
    weights_1:
        First set of weights.
    weights_2:
        Second set of weights.
    N:
        Number of indices to sample.

    Returns
    -------
    indices_1:
        Indices of the first set of resampled particles.
    indices_2:
        Indices of the second set of resampled particles.
    coupled_flag:
        Flag indicating whether the coupling was successful or not.
    """
    key_1, key_2, key_3 = jax.random.split(key, 3)
    if N is None:
        N = weights_1.shape[0]

    nu = jnp.minimum(weights_1, weights_2)
    alpha = jnp.sum(nu)

    weights_1, weights_2 = (weights_1 - nu) / (1 - alpha), (weights_2 - nu) / (1 - alpha)
    nu /= alpha

    coupled = jax.random.uniform(key_1, shape=(N,)) < alpha
    where_coupled = jax.random.choice(key_2, N, p=nu, shape=(N,), replace=True)

    where_uncoupled_1 = jax.random.choice(key_3, N, p=weights_1, shape=(N,), replace=True)
    where_uncoupled_2 = jax.random.choice(key_3, N, p=weights_2, shape=(N,), replace=True)

    indices_1 = jnp.where(coupled, where_coupled, where_uncoupled_1)
    indices_2 = jnp.where(coupled, where_coupled, where_uncoupled_2)

    indices_1 = indices_1.at[0].set(0)
    indices_2 = indices_2.at[0].set(0)
    if N == 1:
        return indices_1[0], indices_2[0], coupled[0]
    return indices_1, indices_2, coupled
