#   MIT License
#
#  Copyright (c) Adrien Corenflos 2021
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import math
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from chex import ArrayTree
from jax.tree_util import tree_flatten, tree_map

_EPS = 0.1  # this is a small float to make sure that log2(2**k) = k exactly
_TYP = Callable[[ArrayTree, ArrayTree], ArrayTree]


def dc_map(elems: ArrayTree, operator: _TYP, last_operator: Optional[_TYP] = None) -> ArrayTree:
    """
    Divide and conquer routine for a generic operator. The algorithm essentially goes as follows:
    1. Pad `elems` of shape (T, ...) in the first dimension to the nearest power of 2, resulting in length T=2^K
    2. For k=0...K-1
        a. Reshape `elems` to (T/2^k, 2^k, ...)
        b. Split `elems` between even and odd indices
            `even_elems = elems[2t, t=0...T/2^k-2]`
            `odd_elems = elems[2t+1, t=0...T/2^k-2]`
        c. Combine `even_elems` and `odd_elems` in parallel along the first dimension
            In parallel do:
            `elems[t] = operator(even_elems[t], odd_elems[t]), t=0...T/2^k-1`
    3. Return `elems`
    The parallelisation at each recursion level is done manually by the user, for example using `vmap`.
    Parameters
    ----------
    elems: PyTree
        The elements to be combined with operator
    operator: Callable[[PyTree, PyTree], PyTree]
        The combination operator.
        It has to preserve the first (batch) dimension: e.g., in the case when elems is an array,
        `operator(elems_1, elems_2).shape[0] == elems_1.shape[0] + elems_2.shape[0]`.
        In general this has to be verified for all leaves in the PyTree.
    last_operator: Callable[[PyTree, PyTree], PyTree], optional
        The combination operator for the last recursion level. If not provided, `operator` is used.
    
    Returns
    -------
    combined_elems: PyTree
        The result of the operator being applied to the data in a divide and conquer manner.
    """
    if last_operator is None:
        last_operator = operator
    return jax.jit(_dc_map, static_argnums=(1, 2))(elems, operator, last_operator)


def _dc_map(elems, operator, last_operator):
    """
    This function is coded in a suboptimal way: the tree_map functions essentially make us go back and forth between
    the PyTree structure and the flat structure. However, it has the benefit of being substantially more readable than
    the optimised version which operates mostly in the flattened space. It does not seem that the overhead reduction
    is worth the readability trade-off to me. If anyone is ever interested I would be more than happy to discuss the
    details, but for the time being I will essentially consider this implementation to be better.
    """
    flat_tree, _ = tree_flatten(elems)
    shapes = tree_map(jnp.shape, elems)

    T = flat_tree[0].shape[0]
    pow_2 = _next_power_of_2(T)
    K = int(math.log2(pow_2 + _EPS))

    indices = np.arange(pow_2)
    padded_elems = tree_map(lambda elem: _pad(elem, pow_2, T), elems)  # pad with operator zeros (algebraic sense)

    def combine(tree_a, indices_a, tree_b, indices_b, op):
        # Compute the location where we can use the operator.
        # Instead of a boolean mask we could also compute the slice of indices explicitly.
        # Not sure which one would be best.
        mask = np.logical_and(indices_a[..., -1] < T, indices_b[..., 0] < T)

        tree_a_use = tree_map(lambda z: z[mask], tree_a)
        tree_b_use = tree_map(lambda z: z[mask], tree_b)

        tree_a_no_use = tree_map(lambda z: z[~mask], tree_a)
        tree_b_no_use = tree_map(lambda z: z[~mask], tree_b)

        combined = op(tree_a_use, tree_b_use)
        unchanged = _passthrough(tree_a_no_use, tree_b_no_use)

        return tree_map(lambda u, v: jnp.concatenate([u, v], 0), combined, unchanged)

    for k in range(K):
        two_k = 2 ** k
        reshaped_tree = tree_map(lambda elem, shape: _reshape(elem, two_k, shape), padded_elems, shapes)
        indices = np.reshape(indices, (-1, two_k))

        even_elems = tree_map(lambda z: z[::2], reshaped_tree)
        odd_elems = tree_map(lambda z: z[1::2], reshaped_tree)

        even_indices, odd_indices = indices[::2], indices[1::2]

        if k == K - 1:
            padded_elems = combine(even_elems, even_indices, odd_elems, odd_indices, last_operator)
        else:
            padded_elems = combine(even_elems, even_indices, odd_elems, odd_indices, operator)

    return tree_map(lambda z: z[0, :T, ...], padded_elems)


def _next_power_of_2(n):
    q, mod = n, 0
    k = 0
    while q > 1:
        q, mod_ = divmod(q, 2)
        mod += mod_
        k += 1
    if mod:
        k += 1
    return 2 ** k


@jax.vmap
@partial(jax.jit, donate_argnums=(0, 1))
def _passthrough(tree_a, tree_b):
    return tree_map(lambda x, y: jnp.concatenate([x, y], 0),
                    tree_a, tree_b)


@partial(jax.jit, static_argnums=(1, 2), donate_argnums=(0,))
def _pad(elem, pow_2, T):
    dtype = elem.dtype
    if jnp.issubdtype(dtype, jnp.integer):
        constant_val = 0
    else:
        constant_val = jnp.nan
    pad_width = [(0, pow_2 - T)] + [(0, 0)] * (elem.ndim - 1)
    return jnp.pad(elem, pad_width=pad_width, constant_values=constant_val)


@partial(jax.jit, static_argnums=(1, 2), donate_argnums=(0,))
def _reshape(elem, intermediary_shape, original_shape):
    shape = -1, intermediary_shape, *original_shape[1:]
    return jnp.reshape(elem, shape)
