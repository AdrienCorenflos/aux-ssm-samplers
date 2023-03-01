from typing import NamedTuple

import jax
import jax.numpy as jnp
from chex import Numeric
from jax.scipy.stats import norm

from ..base import Array
from ..math.mvn import logpdf


class LGSSM(NamedTuple):
    # initial state
    m0: Array
    P0: Array

    # dynamics
    Fs: Array
    Qs: Array
    bs: Array

    # emission
    Hs: Array
    Rs: Array
    cs: Array

    """
    NamedTuple encapsulating the parameters of the LGSSM.
    For a generic LGSSM, the shapes of the parameters are:
    m0: (dx,)
    P0: (dx, dx)
    Fs: (T-1, dx, dx)
    Qs: (T-1, dx, dx)
    bs: (T-1, dx)
    Hs: (T, dy, dx)
    Rs: (T, dy, dy)
    cs: (T, dy)
    and the corresponding observation will need to be given as (T, dy).
    
    For the special case of batched (independent along the batch dimension) LGSSMs, the shapes of the parameters are:
    m0: (B, dx)
    P0: (B, dx, dx)
    Fs: (T-1, B, dx, dx)
    Qs: (T-1, B, dx, dx)
    bs: (T-1, B, dx)
    Hs: (T, B, dy, dx)
    Rs: (T, B, dy, dy)
    cs: (T, B, dy)
    and the corresponding observation will need to be given as (T, B, dy).
    
    Attributes
    ----------
    m0 : Array
        The initial state mean.
    P0 : Array
        The initial state covariance.
    Fs : Array
        The transition matrices.
    Qs : Array
        The transition covariance matrices.
    bs : Array
        The transition offsets.
    Hs : Array
        The observation matrices.
    Rs : Array
        The observation noise covariance matrices.
    cs : Array
        The observation offsets.
    """


@jax.jit
def posterior_logpdf(ys: Array, xs: Array, ell: Numeric, lgssm: LGSSM) -> Numeric:
    """
    Computes the posterior log-likelihood :math:`p(x_{0:T} \mid y_{0:T})`
    of a trajectory of a linear Gaussian state space model.

    Parameters
    ----------
    ys : Array
        The data.
    xs : Array
        The trajectory considered for the linear Gaussian state space model.
    ell : Numeric
        The marginal log-likelihood :math:`p(y_{1:T})` obtained from running a Kalman filter.
    lgssm : LGSSM
        The linear Gaussian state space model.

    Returns
    -------
    ell: Numeric
        The posterior logpdf of the trajectory of the linear Gaussian state space model.
    """
    out = log_likelihood(ys, xs, lgssm) - ell
    out += prior_logpdf(xs, lgssm)
    return out


@jax.jit
def prior_logpdf(xs: Array, lgssm: LGSSM) -> Numeric:
    """
    Computes the prior log-likelihood :math:`p(x_{0:T})`
    of a trajectory of a linear Gaussian state space model.

    Parameters
    ----------
    xs : Array
        The trajectory considered for the linear Gaussian state space model.
    lgssm : LGSSM
        The linear Gaussian state space model.

    Returns
    -------
    ell: Numeric
        The prior logpdf of the trajectory of the linear Gaussian state space model.
    """
    m0, P0, Fs, Qs, bs, *_ = lgssm

    pred_xs = jnp.einsum("...ij,...j->...i", Fs, xs[:-1]) + bs

    if m0.shape[-1] == 1:
        chol_P0 = jnp.sqrt(P0)
        chol_Qs = jnp.sqrt(Qs)
        out = jnp.nansum(norm.logpdf(xs[0, ..., 0], m0[..., 0], chol_P0[..., 0, 0]))
        trans_log_liks = norm.logpdf(xs[1:, ..., 0], pred_xs[..., 0], chol_Qs[..., 0, 0])

    else:
        chol_P0 = jnp.linalg.cholesky(P0)
        chol_Qs = jnp.linalg.cholesky(Qs)
        out = jnp.nansum(logpdf(xs[0], m0, chol_P0))
        trans_log_liks = logpdf(xs[1:], pred_xs, chol_Qs)

    out += jnp.nansum(trans_log_liks)
    return out


@jax.jit
def log_likelihood(ys: Array, xs: Array, lgssm: LGSSM) -> Numeric:
    """
    Computes the log-likelihood :math:`p(y_{0:T} | x_{0:T})`
    of a set of observations under a given trajectory of a linear Gaussian state space model.

    Parameters
    ----------
    ys : Array
        The data.
    xs : Array
        The trajectory considered for the linear Gaussian state space model.
    lgssm : LGSSM
        The linear Gaussian state space model.

    Returns
    -------
    ell: Numeric
        The log likelihood of the observations for a trajectory.
    """
    *_, Hs, Rs, cs = lgssm
    pred_ys = jnp.einsum("...ij,...j->...i", Hs, xs) + cs

    if cs.shape[-1] == 1:
        chol_Rs = jnp.sqrt(Rs)
        out = norm.logpdf(ys[..., 0], pred_ys[..., 0], chol_Rs[..., 0, 0])
    else:
        chol_Rs = jnp.linalg.cholesky(Rs)
        out = logpdf(ys, pred_ys, chol_Rs)
    return jnp.nansum(out)
