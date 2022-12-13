"""
Divide and conquer sampling for LGSSMs.
"""
import warnings
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from chex import Array, PRNGKey, Numeric
from jax.scipy.linalg import solve

from .base import LGSSM
from ..math.mvn import rvs, logpdf


def sampling(key: PRNGKey, ms: Array, Ps: Array, lgssm: LGSSM) -> Array:
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
        LGSSM parameters.

    Returns
    -------
    xs: Array
        Sampled trajectory.
    """

    warnings.warn(
        "`dnc_sampling.sampling` is a proof-of-concept (and not efficient) feature."
        "Use mks.kalman.sampling with the argument `_parallel=True` instead.",
        UserWarning)

    key, key_0, key_T = jax.random.split(key, 3)

    # Sample from the last time step
    xs = jnp.zeros_like(ms)
    chol_T = jnp.linalg.cholesky(Ps[-1])
    x_T = rvs(key_T, ms[-1], chol_T)
    xs = xs.at[-1].set(x_T)

    # Make the tree from head to leaves, then form the parameters backwards.
    (E_0T, g_0T, L_0T), aux_tree, left_parents, to_sample, right_parents = make_dnc_tree(ms, Ps, lgssm)

    # Sample from x0 | xT
    m0 = E_0T[0] @ x_T + g_0T[0]
    chol_0T = jnp.linalg.cholesky(L_0T[0])
    x0 = rvs(key_0, m0, chol_0T)
    xs = xs.at[0].set(x0)

    # Loop over the tree in reverse order to sample from the mid points conditionally on the parents
    for aux, idx_left, idx, idx_right in zip(aux_tree, left_parents, to_sample, right_parents):
        key, subkey = jax.random.split(key)
        sampling_keys = jax.random.split(subkey, idx.shape[0])
        samples = jax.vmap(_sample)(sampling_keys, xs[idx_left], xs[idx_right], aux)
        xs = xs.at[idx].set(samples)
    return xs


def _sample(key, x1, x2, aux_elem):
    G, Gamma, w, V = aux_elem
    chol = jnp.linalg.cholesky(V)
    mean = G @ x1 + Gamma @ x2 + w
    out = rvs(key, mean, chol)
    return out


def _combination_operator(elem1, elem2):
    E1, g1, L1 = elem1
    E2, g2, L2 = elem2

    E = E1 @ E2
    g = g1 + E1 @ g2
    L = L1 + E1 @ L2 @ E1.T

    G = solve(L, E1 @ L2, assume_a="pos").T
    Gamma = E2 - G @ E
    w = g2 - G @ g
    V = L2 - G @ L @ G.T

    return (E, g, L), (G, Gamma, w, V)


@jax.vmap
def _init_elems(m, P, F, Q, b):
    E = solve(F @ P @ F.T + Q, F @ P, assume_a="pos").T
    g = m - E @ (F @ m + b)
    L = P - E @ F @ P
    return E, g, L


def _combine_elements(elems, n_elems, elems_indices):
    # split even and odd to then combine them
    even = jax.tree_util.tree_map(lambda x: x[::2], elems)
    odd = jax.tree_util.tree_map(lambda x: x[1::2], elems)
    even_indices, odd_indices = elems_indices[::2], elems_indices[1::2]

    # if n_elems is odd, we have a single element left out at the end
    if n_elems % 2:
        remainder = jax.tree_util.tree_map(lambda x: x[-1, None], even)
        even = jax.tree_util.tree_map(lambda x: x[:-1], even)

        even_indices, remainder_index = even_indices[:-1], even_indices[-1, None]
        n_elems = 1 + n_elems // 2
    else:
        remainder = remainder_index = None
        n_elems = n_elems // 2

    new_elems, aux_outputs = jax.vmap(_combination_operator)(even, odd)

    # to_sample can either be even_indices[:, 1] or odd_indices[:, 0] as this describes the same array.
    left_parent, to_sample, right_parent = even_indices[:, 0], even_indices[:, 1], odd_indices[:, 1]
    new_indices = np.stack([left_parent, right_parent], axis=1)

    if remainder is not None:
        new_elems = jax.tree_util.tree_map(lambda x, y: jnp.append(x, y, 0), new_elems, remainder)
        new_indices = np.append(new_indices, remainder_index, 0)

    return n_elems, new_elems, aux_outputs, new_indices, left_parent, to_sample, right_parent


def make_dnc_tree(ms: Array, Ps: Array, lgssm: LGSSM):
    *_, Fs, Qs, bs, _, _, _ = lgssm
    T = len(ms) - 1

    # Initialise the reverse conditional distributions parameters
    Es, gs, Ls = _init_elems(ms[:-1], Ps[:-1], Fs, Qs, bs)

    # Create the tree of parameters
    aux_tree = []
    left_parents_tree = []
    right_parents_tree = []
    to_sample_tree = []

    elems = Es, gs, Ls
    indices = np.stack([np.arange(T, dtype=int),
                        np.arange(1, T + 1, dtype=int)], 1)
    n_elems = T

    while n_elems > 1:
        n_elems, elems, aux_output, indices, left_parent, to_sample, right_parent = _combine_elements(elems, n_elems,
                                                                                                      indices)
        aux_tree.append(aux_output)
        to_sample_tree.append(to_sample)
        left_parents_tree.append(left_parent)
        right_parents_tree.append(right_parent)

    return elems, aux_tree[::-1], left_parents_tree[::-1], to_sample_tree[::-1], right_parents_tree[::-1]
