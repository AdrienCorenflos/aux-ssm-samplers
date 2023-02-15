from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from chex import Numeric
from jax.scipy.linalg import solve, cho_solve
from jax.scipy.stats import norm
from jax.tree_util import tree_map

from .base import LGSSM
from ..base import Array
from ..math.mvn import logpdf


_INF = 1e100

def filtering(ys: Array, lgssm: LGSSM, parallel: bool) -> Tuple[Array, Array, Numeric]:
    """ Kalman filtering algorithm.
    Parameters
    ----------
    ys : Array
        Observations of shape (T, D_y).
    lgssm: LGSSM
        LGSSM parameters.
    parallel : bool
        Whether to run the _parallel-in-time version or not.
    Returns
    -------
    ms : Array
        Filtered state means.
    Ps : Array
        Filtered state covariances.
    ell : Numeric
        Log-likelihood of the observations.
    """
    m0, P0, Fs, Qs, bs, Hs, Rs, cs = lgssm

    if parallel:
        ms, Ps, ell = _parallel_filtering(m0, P0, ys, Fs, Qs, bs, Hs, Rs, cs)  # noqa: bad static type checking
    else:
        ms, Ps, ell = _sequential_filtering(m0, P0, ys, Fs, Qs, bs, Hs, Rs, cs)
    if jnp.ndim(ell) == 1:
        # batched case
        ell = jnp.sum(ell)
    return ms, Ps, ell


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
    ell = ell0 + jnp.nansum(ell_increments, 0)
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


#                                   y,    m,     P,     H,    c,    R,  ->  m,     P,  ell
@partial(jnp.vectorize, signature='(dy),(dx),(dx,dx),(dy,dx),(dy),(dy,dy)->(dx),(dx,dx),()')
def sequential_update(y, m, P, H, c, R):
    def _update(m_, P_):
        # When some of the observations are missing, we make the covariance infinite. This is because JAX
        # does not allow us to remove dimensions from arrays, so we need to keep the same shape.
        # In practice the difference is negligible as shown in unittests.

        y_hat = H @ m_ + c
        y_ = jnp.nan_to_num(y, nan=y_hat)
        y_diff = y_ - y_hat

        where_nan = ~jnp.isfinite(y)
        R_ = jnp.where(where_nan[:, None], jnp.inf, R)
        R_ = jnp.where(where_nan[None, :], jnp.inf, R_)

        S = R_ + H @ P_ @ H.T

        if y_.shape[-1] == 1:
            chol_S = S ** 0.5
            ell_inc = norm.logpdf(y_[0], y_hat[0], chol_S[0, 0])
            G = (P_ @ H.T) / S
        else:
            chol_S = jnp.linalg.cholesky(S)
            ell_inc = logpdf(y_, y_hat, chol_S)
            chol_S = jnp.nan_to_num(chol_S, nan=_INF, posinf=_INF, neginf=-_INF)
            G = cho_solve((chol_S, True), H @ P_).T

        m_ = m_ + G @ y_diff

        # Ignore infinities in S
        S = jnp.nan_to_num(S, nan=0., posinf=0., neginf=0.)
        P_ = P_ - G @ S @ G.T
        P_ = 0.5 * (P_ + P_.T)
        return m_, P_, jnp.nan_to_num(ell_inc)

    def _passthrough(m_, P_):
        return m_, P_, 0.

    return jax.lax.cond(jnp.any(jnp.isfinite(y)), _update, _passthrough, m, P)


#                                   m,     P,      F,     b,    Q,  ->  m,    P,
@partial(jnp.vectorize, signature='(dx),(dx,dx),(dx,dx),(dx),(dx,dx)->(dx),(dx,dx)')
def sequential_predict(m, P, F, b, Q):
    m = F @ m + b
    P = Q + F @ P @ F.T
    P = 0.5 * (P + P.T)
    return m, P


#                                   m,     P,      F,     b,    Q,     y,    H,     c,    R   ->  m,    P,   ell
@partial(jnp.vectorize, signature='(dx),(dx,dx),(dx,dx),(dx),(dx,dx),(dy),(dy,dx),(dy),(dy,dy)->(dx),(dx,dx),()')
def sequential_predict_update(m, P, F, b, Q, y, H, c, R):
    m, P = sequential_predict(m, P, F, b, Q)
    m, P, ell_inc = sequential_update(y, m, P, H, c, R)
    return m, P, ell_inc


# Associative operator for the _parallel filter

def _filtering_op(elem1, elem2):
    A1, b1, C1, eta1, J1 = elem1
    A2, b2, C2, eta2, J2 = elem2
    return _filtering_op_impl(A1, b1, C1, eta1, J1, A2, b2, C2, eta2, J2)


#                      A,    b,     C,   eta,   J
_elem_signature = "(dx,dx),(dx),(dx,dx),(dx),(dx,dx)"
_op_signature = _elem_signature + "," + _elem_signature + "->" + _elem_signature


@partial(jnp.vectorize, signature=_op_signature)
def _filtering_op_impl(A1, b1, C1, eta1, J1, A2, b2, C2, eta2, J2):
    dim = b1.shape[0]

    I_dim = jnp.eye(dim)

    IpCJ = I_dim + jnp.dot(C1, J2)
    IpJC = I_dim + jnp.dot(J2, C1)
    if dim == 1:
        AIpCJ_inv = A2 / IpCJ
        AIpJC_inv = A1 / IpJC
    else:
        AIpCJ_inv = solve(IpCJ.T, A2.T).T
        AIpJC_inv = solve(IpJC.T, A1).T

    A = jnp.dot(AIpCJ_inv, A1)
    b = jnp.dot(AIpCJ_inv, b1 + jnp.dot(C1, eta2)) + b2
    C = jnp.dot(AIpCJ_inv, jnp.dot(C1, A2.T)) + C2
    eta = jnp.dot(AIpJC_inv, eta2 - jnp.dot(J2, b1)) + eta1
    J = jnp.dot(AIpJC_inv, jnp.dot(J2, A1)) + J1
    return A, b, 0.5 * (C + C.T), eta, 0.5 * (J + J.T)


# Initialization of the _parallel filter

def _filtering_init(Fs, Qs, bs, Hs, Rs, cs, m0, P0, ys):
    T = bs.shape[0]
    ms = jnp.concatenate([m0[None, ...], jnp.zeros_like(m0, shape=(T - 1,) + m0.shape)])
    Ps = jnp.concatenate([P0[None, ...], jnp.zeros_like(P0, shape=(T - 1,) + P0.shape)])
    return _filtering_init_one(Fs, Qs, bs, Hs, Rs, cs, ys, ms, Ps)


#                                     F,      Q,    b,     H,      R,     c,   y,  m,    P,
@partial(jnp.vectorize, signature='(dx,dx),(dx,dx),(dx),(dy,dx),(dy,dy),(dy),(dy),(dx),(dx,dx)->' + _elem_signature)
def _filtering_init_one(F, Q, b, H, R, c, y, m, P):

    def _update(m_, P_):
        m_ = F @ m_ + b
        P_ = F @ P_ @ F.T + Q

        is_nan = ~jnp.isfinite(y)
        R_ = jnp.where(is_nan[None, :], jnp.inf, R)
        R_ = jnp.where(is_nan[:, None], jnp.inf, R_)

        S = H @ P_ @ H.T + R_
        if y.shape[0] == 1:
            S_invH_T = H.T / S[0, 0]
        else:
            # This is needed as JAX doesn't allow us to delete rows/columns from a matrix
            chol_S_inf = jnp.linalg.cholesky(S)
            chol_S_inf = jnp.where(jnp.isfinite(chol_S_inf), chol_S_inf, _INF)
            S_invH_T = cho_solve((chol_S_inf, True), H).T

        K = P_ @ S_invH_T
        A = F - K @ H @ F

        y_diff_b = jnp.where(is_nan, 0., y - H @ b - c)
        y_diff_m_ = jnp.where(is_nan, 0., y - H @ m_ - c)

        b_std = m_ + K @ y_diff_m_
        S_0 = jnp.where(jnp.isfinite(S), S, 0.)
        C = P_ - K @ S_0 @ K.T

        temp = F.T @ S_invH_T
        eta = temp @ y_diff_b
        J = temp @ H @ F
        return A, b_std, 0.5 * (C + C.T), eta, 0.5 * (J + J.T)

    def _passthrough(m_, P_):
        m_ = F @ m_ + b
        P_ = F @ P_ @ F.T + Q

        A = F
        b_std = m_
        C = P_
        eta = jnp.zeros_like(b)
        J = jnp.zeros_like(F)
        return A, b_std, 0.5 * (C + C.T), eta, J

    return jax.lax.cond(jnp.any(jnp.isfinite(y)), _update, _passthrough, m, P)

