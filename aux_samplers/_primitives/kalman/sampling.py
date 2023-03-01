from functools import partial

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from jax.scipy.linalg import solve

from .base import LGSSM


def sampling(key: PRNGKey, ms: Array, Ps: Array, lgssm: LGSSM, parallel: bool) -> Array:
    """
    Samples from the pathwise smoothing distribution a LGSSM.

    Parameters
    ----------
    key: PRNGKey
        Random number generator key.
    ms: Array
        Filtering means of the LGSSM.
    Ps: Array
        Filtering covariances of the LGSSM.
    lgssm: LGSSM
        LGSSM model to be sampled from.
    parallel: bool
        Whether to run the sampling in _parallel.
    """
    Fs, Qs, bs = lgssm.Fs, lgssm.Qs, lgssm.bs
    gains, increments = _sampling_init(key, ms, Ps, Fs, Qs, bs)  # noqa: bad static type checking
    if parallel:
        _, samples = jax.lax.associative_scan(jax.vmap(_sampling_op),
                                              (gains, increments), reverse=True)
    else:
        def body(carry, inputs):
            carry = _sampling_op(carry, inputs)
            return carry, carry

        _, (_, samples) = jax.lax.scan(body, (gains[-1], increments[-1]), (gains[:-1], increments[:-1]), reverse=True)
        samples = jnp.append(samples, increments[None, -1, ...], 0)
    return samples


# Operator
def _sampling_op(elem1, elem2):
    G1, e1 = elem1
    G2, e2 = elem2

    return _sampling_op_impl(G1, e1, G2, e2)


@partial(jnp.vectorize, signature='(dx,dx),(dx),(dx,dx),(dx)->(dx,dx),(dx)')
def _sampling_op_impl(G1, e1, G2, e2):
    G = G2 @ G1
    e = G2 @ e1 + e2
    return G, e


# Initialization

@partial(jnp.vectorize, signature="(dx,dx),(dx,dx),(dx),(dx),(dx,dx)->(dx),(dx,dx),(dx,dx)")
def mean_and_chol(F, Q, b, m, P):
    """
    Computes the increments means and Cholesky decompositions for the backward sampling steps.

    Parameters
    ----------
    F: Array
        Transition matrix for time t to t+1.
    Q: Array
        Transition covariance matrix for time t to t+1.
    b:
        Transition offset for time t to t+1.
    m: Array
        Filtering mean at time t
    P: Array
        Filtering covariance at time t
    Returns
    -------
    m: Array
        Increment mean to go from time t+1 to t.
    chol: Array
        Cholesky decomposition of the increment covariance to go from time t+1 to t.
    gain: Array
        Gain to go from time t+1 to t.
    """
    dim = m.shape[0]
    S = F @ P @ F.T + Q  # noqa: bad static type checking
    S = 0.5 * (S + S.T)

    if dim == 1:
        gain = P * F / S
    else:
        gain = P @ solve(S, F, assume_a="pos").T

    inc_Sig = P - gain @ S @ gain.T
    inc_Sig = 0.5 * (inc_Sig + inc_Sig.T)

    inc_m = m - gain @ (F @ m + b)
    if dim == 1:
        L = jnp.sqrt(inc_Sig)
    else:
        L = jnp.linalg.cholesky(inc_Sig)
    # When there is 0 uncertainty, the Cholesky decomposition is not defined.
    L = jnp.nan_to_num(L)
    return inc_m, L, gain


@partial(jnp.vectorize, signature="(dx,dx),(dx,dx),(dx),(dx),(dx,dx),(dx)->(dx,dx),(dx)")
def _sampling_init_one(F, Q, b, m, P, eps):
    inc_m, L, gain = mean_and_chol(F, Q, b, m, P)
    inc = inc_m + L @ eps
    return gain, inc


@partial(jnp.vectorize, signature="(dx),(dx,dx),(dx)->(dx,dx),(dx)")
def _sample_last_step(m, P, eps):
    if P.shape[0] == 1:
        L = jnp.sqrt(P)
    else:
        L = jnp.linalg.cholesky(P)
    L = jnp.nan_to_num(L)
    last_sample = m + L @ eps
    gain = jnp.zeros_like(P)
    return gain, last_sample


def _sampling_init(key, ms, Ps, Fs, Qs, bs):
    epsilons = jax.random.normal(key, shape=ms.shape)

    gains, increments = jax.vmap(_sampling_init_one)(Fs, Qs, bs, ms[:-1], Ps[:-1], epsilons[:-1])

    # When we condition on the last step this is 0 and Cholesky ain't liking this.
    last_gain, last_increment = _sample_last_step(ms[-1], Ps[-1], epsilons[-1])
    gains = jnp.append(gains, last_gain[None, ...], 0)
    increments = jnp.append(increments, last_increment[None, ...], 0)
    return gains, increments
