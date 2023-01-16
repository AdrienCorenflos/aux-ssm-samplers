from functools import partial

import jax.numpy as jnp
import jax.random
from chex import PRNGKey

from .base import LGSSM
from .dnc_sampling import make_dnc_tree
from .sampling import mean_and_chol
from ..math.mvn import rejection, thorisson, modified_lindvall_roger

_EPS = 0.01  # this is a small float to make sure that log2(2**k) = k exactly


# Wrapper for progressive and divide and conquer
# noinspection PyIncorrectDocstring
def sampling(key: PRNGKey, lgssm_1: LGSSM, lgssm_2: LGSSM, ms_1, Ps_1, ms_2, Ps_2, parallel,
             method: str = "rejection", **coupling_params):
    """
    Coupled sampling between two models. If the

    Parameters
    ----------
    key: PRNGKey
        Random number generator key.
    lgssm_1: LGSSM
        LGSSM model 1.
    ms_1, Ps_1:
        Filtering means and covariance matrices for LGSSM 1
    ms_2, Ps_2:
        Filtering means and covariance matrices for LGSSM 2
    parallel: bool
        If True, use parallel sampling (divide and conquer), otherwise, sequential backward sampling.
    lgssm_2: LGSSM
        LGSSM model 2.
    method: str
        Method to use for coupling. Can be "rejection", "thorisson", or "lindvall-roger".

    Other Parameters
    ----------------
    N: int
        If using `method="rejection"`, the parameter is N=<number of particles in the ensemble>
    C: Numeric
        If using "thorisson", the parameter is C=<coupling suboptimality>.

    Returns
    -------
    xs_1: Array
        State samples from model 1.
    xs_2: Array
        State samples from model 2.
    coupled: Array
        If True, the samples are coupled.

    """
    if parallel:
        return divide_and_conquer(key, lgssm_1, lgssm_2, ms_1, Ps_1, ms_2, Ps_2, method, **coupling_params)
    return progressive(key, lgssm_1, lgssm_2, ms_1, Ps_1, ms_2, Ps_2, method, **coupling_params)


######################
# The actual methods #
######################

def progressive(key: PRNGKey, lgssm_1: LGSSM, lgssm_2: LGSSM, ms_1, Ps_1, ms_2, Ps_2,
                method: str = "rejection", **coupling_params):
    """
    Progressive coupling between two LGSSM models.

    Parameters
    ----------
    key: PRNGKey
        Random number generator key.
    lgssm_1: LGSSM
        LGSSM model 1.
    ms_1, Ps_1:
        Filtering means and covariance matrices for LGSSM 1
    ms_2, Ps_2:
        Filtering means and covariance matrices for LGSSM 2
    lgssm_2: LGSSM
        LGSSM model 2.
    method: str
        Method to use for coupling. Can be "rejection", "thorisson", or "lindvall-roger".

    Other Parameters
    ----------------
    N: int
        If using `method="rejection"`, the parameter is N=<number of particles in the ensemble>
    C: Numeric
        If using "thorisson", the parameter is C=<coupling suboptimality>.

    Returns
    -------
    xs_1: Array
        State samples from model 1.
    xs_2: Array
        State samples from model 2.
    coupled: Array
        If True, the samples are coupled.

    """

    if method == "rejection":
        coupling_method_fn = partial(rejection, **coupling_params)
    elif method == "thorisson":
        coupling_method_fn = partial(thorisson, **coupling_params)
    elif method == "lindvall-roger":
        coupling_method_fn = modified_lindvall_roger
    else:
        raise NotImplementedError

    def body(carry, inps):
        x1, x2, op_k = carry

        next_k, op_k = jax.random.split(op_k)
        m1, P1, F1, Q1, b1, m2, P2, F2, Q2, b2 = inps
        inc_m1, inc_L1, gain_1 = mean_and_chol(F1, Q1, b1, m1, P1)
        inc_m2, inc_L2, gain_2 = mean_and_chol(F2, Q2, b2, m2, P2)

        inc_m1 = inc_m1 + gain_1 @ x1
        inc_m2 = inc_m2 + gain_2 @ x2

        x1, x2, coupled = coupling_method_fn(op_k, inc_m1, inc_L1, inc_m2, inc_L2)
        return (x1, x2, next_k), (x1, x2, coupled)

    sub_key, key = jax.random.split(key, 2)

    xT_1, xT_2, init_coupled = coupling_method_fn(sub_key,
                                                  ms_1[-1], jnp.linalg.cholesky(Ps_1[-1]),
                                                  ms_2[-1], jnp.linalg.cholesky(Ps_2[-1]))

    inputs = (
        ms_1[:-1], Ps_1[:-1], lgssm_1.Fs, lgssm_1.Qs, lgssm_1.bs, ms_2[:-1], Ps_2[:-1], lgssm_2.Fs, lgssm_2.Qs,
        lgssm_2.bs)

    _, (xs_1, xs_2, are_coupled) = jax.lax.scan(body, (xT_1, xT_2, key), inputs, reverse=True)
    xs_1, xs_2, are_coupled = jax.tree_util.tree_map(lambda zs, z: jnp.append(zs, z[None], 0),
                                                     (xs_1, xs_2, are_coupled),
                                                     (xT_1, xT_2, init_coupled))
    return xs_1, xs_2, are_coupled


def divide_and_conquer(key: PRNGKey, lgssm_1: LGSSM, lgssm_2: LGSSM, ms_1, Ps_1, ms_2, Ps_2,
                       method: str = "rejection", **coupling_params):
    """
    Progressive coupling between two LGSSM models.

    Parameters
    ----------
    key: PRNGKey
        Random number generator key.
    lgssm_1: LGSSM
        LGSSM model 1.
    lgssm_2: LGSSM
        LGSSM model 2.
    ms_1, Ps_1:
        Filtering means and covariance matrices for LGSSM 1
    ms_2, Ps_2:
        Filtering means and covariance matrices for LGSSM 2
    method: str
        Method to use for coupling. Can be "rejection", "thorisson", or "lindvall-roger".

    Other Parameters
    ----------------
    N: int
        If using `method="rejection"`, the parameter is N=<number of particles in the ensemble>
    C: Numeric
        If using "thorisson", the parameter is C=<coupling suboptimality>.

    Returns
    -------
    xs_1: Array
        State samples from model 1.
    xs_2: Array
        State samples from model 2.
    coupled: Array
        If True, the samples are coupled.

    """

    if method == "rejection":
        coupling_method_fn = partial(rejection, **coupling_params)
    elif method == "thorisson":
        coupling_method_fn = partial(thorisson, **coupling_params)
    elif method == "lindvall-roger":
        coupling_method_fn = modified_lindvall_roger
    else:
        raise NotImplementedError

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
        """
    key, key_0, key_T = jax.random.split(key, 3)

    # Sample from the last time step
    xs_1 = jnp.zeros_like(ms_1)
    xs_2 = jnp.zeros_like(ms_2)
    are_coupled = jnp.zeros((xs_1.shape[0],), dtype=bool)

    xT_1, xT_2, xT_coupled = coupling_method_fn(key_T,
                                                ms_1[-1], jnp.linalg.cholesky(Ps_1[-1]),
                                                ms_2[-1], jnp.linalg.cholesky(Ps_2[-1]))

    xs_1, xs_2, are_coupled = jax.tree_util.tree_map(lambda z, y: z.at[-1].set(y), (xs_1, xs_2, are_coupled),
                                                     (xT_1, xT_2, xT_coupled))

    # Make the tree from head to leaves, then form the parameters backwards.
    (E_0T_1, g_0T_1, L_0T_1), aux_tree_1, left_parents, to_sample, right_parents = make_dnc_tree(ms_1, Ps_1, lgssm_1)
    (E_0T_2, g_0T_2, L_0T_2), aux_tree_2, *_ = make_dnc_tree(ms_2, Ps_2, lgssm_2)

    # Sample from x0 | xT
    m0_1 = E_0T_1[0] @ xT_1 + g_0T_1[0]
    m0_2 = E_0T_2[0] @ xT_2 + g_0T_2[0]
    x0_1, x0_2, x0_coupled = coupling_method_fn(key_0,
                                                m0_1, jnp.linalg.cholesky(L_0T_1[0]),
                                                m0_2, jnp.linalg.cholesky(L_0T_2[0]))

    xs_1, xs_2, are_coupled = jax.tree_util.tree_map(lambda z, y: z.at[0].set(y), (xs_1, xs_2, are_coupled),
                                                     (x0_1, x0_2, x0_coupled))

    # Loop over the tree in reverse order to sample from the mid-points conditionally on the parents
    for aux_1, aux_2, idx_left, idx, idx_right in zip(aux_tree_1, aux_tree_2, left_parents, to_sample, right_parents):
        key, subkey = jax.random.split(key)
        sampling_keys = jax.random.split(subkey, idx.shape[0])
        m1s, chol1s = jax.vmap(_mean_chol)(xs_1[idx_left], xs_1[idx_right], aux_1)
        m2s, chol2s = jax.vmap(_mean_chol)(xs_2[idx_left], xs_2[idx_right], aux_2)

        samples_1, samples_2, coupling_flags = jax.vmap(coupling_method_fn)(sampling_keys,
                                                                            m1s, chol1s,
                                                                            m2s, chol2s)

        xs_1, xs_2, are_coupled = jax.tree_util.tree_map(lambda z, y: z.at[idx].set(y), (xs_1, xs_2, are_coupled),
                                                         (samples_1, samples_2, coupling_flags))

    return xs_1, xs_2, are_coupled


def _mean_chol(x1, x2, aux_elem):
    G, Gamma, w, V = aux_elem
    chol = jnp.linalg.cholesky(V)
    mean = G @ x1 + Gamma @ x2 + w
    return mean, chol
