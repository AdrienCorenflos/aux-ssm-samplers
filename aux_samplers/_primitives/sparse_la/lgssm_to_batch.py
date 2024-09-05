"""
Transform a LGSSM into a batched joint distribution over (X, Y).
"""

from jax.scipy.linalg import cho_solve
from jax.experimental.sparse import BCOO, BCSR, eye as speye
from jax import numpy as jnp
import jax
from chex import Array
from jax.scipy.linalg import solve
import jax.experimental.sparse.linalg as sparse_linalg

from aux_samplers._primitives.kalman import filtering
from aux_samplers.kalman import LGSSM


def get_batched_model(lgssm: LGSSM):
    """
    Get the batched model for the LGSSM.

    Parameters
    ------------
    lgssm: LGSSM
        The LGSSM model.

    Returns
    --------
    A_inv: BCOO
        The inverse of the linear dynamics matrix.
    Q_inv: BCOO
        The inverse of the prior covariance of the LGSSM.
    H: BCOO
        The observation matrix.
    R_inv: BCOO
        The inverse of the observation covariance.
    Py_inv: BCOO
        The inverse of the marginal observation covariance.
    """

    Fs, Qs, bs, Hs, Rs, cs = lgssm.Fs, lgssm.Qs, lgssm.bs, lgssm.Hs, lgssm.Rs, lgssm.cs
    m0, P0 = lgssm.m0, lgssm.P0
    # mx = get_prior_mean(m0, Fs, bs)

    T, dy, dx = Hs.shape

    Qs = jnp.concatenate([P0[None, :], Qs], 0)
    # bs = jnp.concatenate([m0[None, :], bs], 0)

    chol_Qs = jnp.linalg.cholesky(Qs)
    chol_Rs = jnp.linalg.cholesky(Rs)

    # Get util matrices

    Qinvs = jax.vmap(lambda M: cho_solve((M, True), jnp.eye(M.shape[0])))(chol_Qs)
    Rinvs = jax.vmap(lambda M: cho_solve((M, True), jnp.eye(M.shape[0])))(chol_Rs)

    A_inv_indices = _block_diag_row_and_cols(T, dx, dx, -1)
    A_inv = BCOO((-Fs.flatten(), A_inv_indices), shape=(T * dx, T * dx), unique_indices=True, indices_sorted=True)
    A_inv = A_inv + speye(T * dx)

    Q_inv_indices = _block_diag_row_and_cols(T, dx, dx, 0)
    Q_inv = BCOO((Qinvs.flatten(), Q_inv_indices), shape=(T * dx, T * dx), unique_indices=True, indices_sorted=True)

    H_indices = _block_diag_row_and_cols(T, dx, dy, 0)
    H = BCOO((Hs.flatten(), H_indices), shape=(T * dx, T * dy), unique_indices=True, indices_sorted=True)

    R_inv_indices = _block_diag_row_and_cols(T, dy, dy, 0)
    R_inv = BCOO((Rinvs.flatten(), R_inv_indices), shape=(T * dy, T * dy), unique_indices=True, indices_sorted=True)

    Py_inv = R_inv - R_inv @ H @ (Q_inv + H.T @ R_inv @ H) @ H.T @ R_inv

    return A_inv, Q_inv, H, R_inv, Py_inv


def _block_diag_row_and_cols(N, M_r, M_c, k):
    min_row = max(0, -k * M_r)
    max_row = min(N, N - k) * M_r

    min_col = max(0, k * M_c)
    max_col = min(N, N + k) * M_c

    rows = jnp.arange(min_row, max_row)
    cols = jnp.arange(min_col, max_col)

    rows = jnp.reshape(rows, (1, -1))
    rows = jnp.tile(rows, (M_c, 1)).flatten(order="F")  # noqa

    cols = jnp.reshape(cols, (-1, M_c))
    cols = jnp.tile(cols, (1, M_r)).flatten()
    indices = jnp.stack((rows, cols), axis=1)
    return indices


# def _block_diag_idx_and_ptr(N, M_r, M_c, k):
#     min_col = max(0, k * M_c)
#     max_col = min(N, N + k) * M_c
#
#     cols = jnp.arange(min_col, max_col)
#
#     cols = jnp.reshape(cols, (-1, M_c))
#     cols = jnp.tile(cols, (1, M_r)).flatten()
#
#     ptr = jnp.arange(0, (N - abs(k)) * M_c * M_r + 1, M_c)
#     if k > 0:
#         ptr = jnp.pad(ptr, (k * M_r, 0), mode="constant", constant_values=0)
#     else:
#         ptr = jnp.pad(ptr, (0, -k * M_r), mode="constant", constant_values=0)
#     return cols, ptr


def get_prior_mean(m0, Fs, bs):
    """
    Get the prior mean of the LGSSM.

    Parameters
    ----------
    m0: Array
        The initial mean.
    Fs:
        The transition matrices.
    bs:
        The transition biases.

    Returns
    -------
    Array:
        The prior mean.

    """
    _, d = bs.shape

    eye = jnp.eye(d)

    ms = jnp.concatenate([m0[None, :], bs], 0)
    Fs = jnp.concatenate([eye[None, :], Fs], 0)

    @jax.vmap
    def operator(elem1, elem2):
        m1, F_1 = elem1
        m2, F_2 = elem2
        return m2 + F_2 @ m1, F_2 @ F_1

    ms, _ = jax.lax.associative_scan(operator, (ms, Fs))
    return ms


def sparse_map(ys, lgssm: LGSSM):
    """
    Compute the posterior mean using batched linear algebra.
    Also returns the marginal

    Parameters
    ------------
    ys: Array
        The observations.
    lgssm: LGSSM
        The LGSSM model.

    Returns
    --------
    ms: Array
        The posterior means.
    """
    m0, P0, Fs, Qs, bs, Hs, Rs, cs = lgssm
    T, dy, dx = Hs.shape
    bs = jnp.concatenate([m0[None, :], bs], 0)
    Qs = jnp.concatenate([P0[None, :], Qs], 0)

    A_inv_subdiag = -Fs
    Q_inv_diag = jax.vmap(lambda Q: solve(Q, jnp.eye(dx), assume_a="pos"))(Qs)
    R_inv_diag = jax.vmap(lambda R: solve(R, jnp.eye(dy), assume_a="pos"))(Rs)



    H_diag = Hs

    b = jnp.einsum("ijk,ik->ij", Q_inv_diag, bs)
    b = b.at[:-1].add(jnp.einsum("ikj,ik->ij", A_inv_subdiag, b[1:]))
    P_inv_diag_one = Q_inv_diag + jnp.einsum("ijk,ijl,ilm->ikm", H_diag, R_inv_diag, H_diag)
    P_inv_subdiag = jnp.einsum("ikj,ikl->ijl", A_inv_subdiag, Q_inv_diag[1:])

    P_inv_diag = P_inv_diag_one.at[:-1].add(
        jnp.einsum("ijk,ikl->ijl", P_inv_subdiag, A_inv_subdiag))

    P_inv_subdiag_bcoo = BCOO(
        (
            P_inv_subdiag.flatten(),
            _block_diag_row_and_cols(T, dx, dx, -1)
        ),
        shape=(T * dx, T * dx))
    P_inv_superdiag_bcoo = BCOO(
        (
            jnp.transpose(P_inv_subdiag, (0, 2, 1)).flatten(),
            _block_diag_row_and_cols(T, dx, dx, 1)
        ),
        shape=(T * dx, T * dx))

    P_inv_diag_bcoo = BCOO(
        (
            P_inv_diag.flatten(),
            _block_diag_row_and_cols(T, dx, dx, 0)
        ),
        shape=(T * dx, T * dx))
    P_inv_csr = _bcoo_to_bcsr(P_inv_diag_bcoo + P_inv_subdiag_bcoo + P_inv_superdiag_bcoo)

    scaled_ys = jnp.einsum("ijk,ik->ij", R_inv_diag, ys - cs)
    b += jnp.einsum("ikj,ik->ij", H_diag, scaled_ys)

    # Compute E[X | Y]
    ms = sparse_linalg.spsolve(P_inv_csr.data, P_inv_csr.indices, P_inv_csr.indptr, b.flatten())
    return ms.reshape(*bs.shape)


def _bcoo_to_bcsr(mat):
    """
    Convert a BCOO matrix to a BCSR matrix.

    Parameters
    ------------
    mat: BCOO
        The BCOO matrix.

    Returns
    --------
    BCSR:
        The BCSR matrix.
    """
    data = mat.data
    m, n = mat.shape
    indices = mat.indices.T
    argsort = jnp.argsort(indices[0], stable=True)
    rows, cols = indices[:, argsort]
    indptr = jnp.zeros(m + 1, dtype=int)
    indptr = indptr.at[rows + 1].add(1, indices_are_sorted=True, unique_indices=False)
    indptr = jnp.cumsum(indptr)
    return BCSR((data[argsort], cols, indptr), shape=(m, n))


def create_csr_matrix(block_subdiagonals, T, dx):
    """
    Create a CSR matrix using canonical indexing for given block subdiagonals.

    Parameters
    ----------
    block_subdiagonals : jnp.ndarray
        Array of block subdiagonals of shape (T-1, dx, dx).
    T : int
        Number of blocks.
    dx : int
        Dimension of each block.

    Returns
    -------
    BCOO
        The resulting CSR matrix.
    """
    # Flatten the block subdiagonals
    data = block_subdiagonals.flatten()
    indices, indptr = _block_diag_idx_and_ptr(T, dx, dx, -1)
    return BCSR((data, indices, indptr), shape=(T * dx, T * dx))


# Example usage


if __name__ == "__main__":
    import numpy as np

    np.random.seed(0)
    T, D = 0, 1
    m0 = np.random.rand(D)
    P0 = np.random.rand(D, 10)
    P0 = P0 @ P0.T
    Fs = np.random.rand(T, D, D)
    Qs = np.random.rand(T, D, 10)
    Qs = np.einsum("...ij,...kj->...ik", Qs, Qs)
    bs = np.random.rand(T, D)
    Hs = np.random.rand(T + 1, D, D)
    Rs = np.random.rand(T + 1, D, 10)
    Rs = np.einsum("...ij,...kj->...ik", Rs, Rs)
    cs = np.random.rand(T + 1, D)
    ys = np.random.rand(T + 1, D)

    # data = Fs.flatten()
    # indices, indptr = _block_diag_idx_and_ptr(T + 1, D, D, 1)
    #
    # Fs_sub = BCSR((data, indices, indptr), shape=((T + 1) * D, (T + 1) * D))
    # print(Fs_sub.todense()._value)
    # indices = _block_diag_row_and_cols(T + 1, D, D, 1)
    # Fs_sub_bcoo = BCOO((data, indices), shape=((T + 1) * D, (T + 1) * D))
    # print(Fs_sub_bcoo.todense()._value)
    # _ = +1 + 1

    lgssm = LGSSM(m0, P0, Fs, Qs, bs, Hs, Rs, cs)
    A_inv, Q_inv, H, R_inv, Py_inv = get_batched_model(lgssm)

    ms = sparse_map(ys, lgssm)

    prior_mean = get_prior_mean(m0, Fs, bs)

    print(filtering(ys, lgssm, False))

    print()
    print(prior_mean)
    print()
    print(ms)
    print()
    print(m0)

    #
    # A = speye(3)
    # B = speye(3)
    # C = A + B
    # print(C.nse)
    # print(C.data)
    # print(C.indices)

    # print(ms)
    # print()
    # print(prior_mean)
    # print()
    # print(m0)
