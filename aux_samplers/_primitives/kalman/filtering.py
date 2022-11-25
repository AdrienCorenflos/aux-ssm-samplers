from typing import Tuple, Callable

import jax
import jax.numpy as jnp
from chex import Numeric, Array
from jax.scipy.linalg import solve, cho_solve
from jax.tree_util import tree_map

from .base import LGSSM
from ..base import mvn_loglikelihood


def filtering(ys, lgssm: LGSSM, parallel: bool) -> Tuple[Array, Array, Numeric]:
    """ Kalman filtering algorithm.
    Parameters
    ----------
    ys : Array
        Observations of shape (T, D_y).
    lgssm: LGSSM
        LGSSM parameters.
    parallel : bool
        Whether to run the parallel-in-time version or not.
    Returns
    -------
    ms : Array
        Filtered state means.
    Ps : Array
        Filtered state covariances.
    ell : Numeric
        Log-likelihood of the observations.
    """
    m0, P0, Fs, Qs, bs, Hs, Rs, cs, _ = lgssm

    if parallel:
        return _parallel_filtering(m0, P0, ys, Fs, Qs, bs, Hs, Rs, cs)  # noqa: bad static type checking
    else:
        return _sequential_filtering(m0, P0, ys, Fs, Qs, bs, Hs, Rs, cs)


def _parallel_filtering(m0, P0, ys, Fs, Qs, bs, Hs, Rs, cs):
    y0, H0, c0, R0 = jax.tree_map(lambda x: x[0], (ys, Hs, cs, Rs))
    y1T, H1T, c1T, R1T = jax.tree_map(lambda x: x[1:], (ys, Hs, cs, Rs))
    m0, P0, ell0 = sequential_update(y0, m0, P0, H0, c0, R0)

    init_elems = _filtering_init(Fs, Qs, bs, H1T, R1T, c1T, m0, P0, y1T)
    _, ms, Ps, _, _ = jax.lax.associative_scan(jax.vmap(_filtering_op),
                                               init_elems)

    ms, Ps = tree_map(lambda z, y: jnp.insert(z, 0, y, axis=0), (ms, Ps), (m0, P0))

    *_, ell_increments = jax.vmap(sequential_predict_update)(ms[:-1], Ps[:-1], Fs, bs, Qs, y1T, H1T, c1T,
                                                             R1T)
    ell = ell0 + jnp.sum(ell_increments)
    return ms, Ps, ell


def _sequential_filtering(m0, P0, ys, Fs, Qs, bs, Hs, Rs, cs):
    y0, H0, c0, R0 = jax.tree_map(lambda x: x[0], (ys, Hs, cs, Rs))
    y1T, H1T, c1T, R1T = jax.tree_map(lambda x: x[1:], (ys, Hs, cs, Rs))
    m0, P0, ell0 = sequential_update(y0, m0, P0, H0, c0, R0)

    def body(carry, inputs):
        m, P, curr_ell = carry
        F, Q, b, H, R, c, y = inputs
        m, P, ell_inc = sequential_predict_update(m, P, F, b, Q, y, H, c, R)
        return (m, P, curr_ell + ell_inc), (m, P)

    (*_, ell), (ms, Ps) = jax.lax.scan(body, (m0, P0, ell0), (Fs, Qs, bs, H1T, R1T, c1T, y1T))
    ms, Ps = tree_map(lambda z, y: jnp.insert(z, 0, y, axis=0), (ms, Ps), (m0, P0))
    return ms, Ps, ell


# Sequential filtering ops
def sequential_update(y, m, P, H, c, R):
    if isinstance(y, tuple):
        ell = 0.
        for y_, H_, C_, R_ in zip(y, H, c, R):
            m, P, ell_inc = sequential_update_one(y_, m, P, H_, C_, R_)
            ell += ell_inc
    else:
        m, P, ell = sequential_update_one(y, m, P, H, c, R)
    return m, P, ell


def sequential_update_one(y, m, P, H, c, R):
    y_hat = H @ m + c
    y_diff = y - y_hat
    S = R + H @ P @ H.T

    chol_S = jnp.linalg.cholesky(S)
    ell_inc = mvn_loglikelihood(y, y_hat, chol_S)
    G = cho_solve((chol_S, True), H @ P).T
    m = m + G @ y_diff
    P = P - G @ S @ G.T
    P = 0.5 * (P + P.T)
    return m, P, jnp.nan_to_num(ell_inc)


def sequential_predict(m, P, F, b, Q):
    m = F @ m + b
    P = Q + F @ P @ F.T
    return m, P


def sequential_predict_update(m, P, F, b, Q, y, H, c, R):
    m, P = sequential_predict(m, P, F, b, Q)
    m, P, ell_inc = sequential_update(y, m, P, H, c, R)
    return m, P, ell_inc


# Associative operator for the parallel filter

def _filtering_op(elem1, elem2):
    A1, b1, C1, eta1, J1 = elem1
    A2, b2, C2, eta2, J2 = elem2
    dim = b1.shape[0]

    I_dim = jnp.eye(dim)

    IpCJ = I_dim + jnp.dot(C1, J2)
    IpJC = I_dim + jnp.dot(J2, C1)

    AIpCJ_inv = solve(IpCJ.T, A2.T).T
    AIpJC_inv = solve(IpJC.T, A1).T

    A = jnp.dot(AIpCJ_inv, A1)
    b = jnp.dot(AIpCJ_inv, b1 + jnp.dot(C1, eta2)) + b2
    C = jnp.dot(AIpCJ_inv, jnp.dot(C1, A2.T)) + C2
    eta = jnp.dot(AIpJC_inv, eta2 - jnp.dot(J2, b1)) + eta1
    J = jnp.dot(AIpJC_inv, jnp.dot(J2, A1)) + J1
    return A, b, 0.5 * (C + C.T), eta, 0.5 * (J + J.T)


# Initialization of the parallel filter

def _filtering_init(Fs, Qs, bs, Hs, Rs, cs, m0, P0, ys):
    T = bs.shape[0]
    ms = jnp.concatenate([m0[None, ...], jnp.zeros_like(m0, shape=(T - 1,) + m0.shape)])
    Ps = jnp.concatenate([P0[None, ...], jnp.zeros_like(P0, shape=(T - 1,) + P0.shape)])
    return _filtering_init_one(Fs, Qs, bs, Hs, Rs, cs, ys, ms, Ps)


@jax.vmap
def _filtering_init_one(F, Q, b, H, R, c, y, m, P):
    m = F @ m + b
    P = F @ P @ F.T + Q

    S = H @ P @ H.T + R
    S_invH_T = solve(S, H, assume_a="pos").T
    K = P @ S_invH_T
    A = F - K @ H @ F

    b_std = m + K @ (y - H @ m - c)
    C = P - K @ S @ K.T

    temp = F.T @ S_invH_T
    eta = temp @ (y - H @ b - c)
    J = temp @ H @ F

    return A, b_std, C, eta, J
