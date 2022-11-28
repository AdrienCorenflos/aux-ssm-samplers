"""
Implements Algorithm 3: i-RW-CSMC appearing in "Conditional smc in high dimension" by Finke and Thiery (2021).
Similar to them, we only implement the multinomial resampling scheme and do not consider the "forced move" extension.
"""
import jax
import jax.numpy as jnp
from chex import ArrayTree, dataclass, Array
from jax.tree_util import tree_map

from .base import Distribution, Potential, UnivariatePotential, Dynamics, normalize
from .resamplings import multinomial
from .standard import _backward_scanning_pass


@dataclass
class CSMCState:
    x: ArrayTree
    ancestors: Array


def get_kernel(M0: Distribution, G0: UnivariatePotential, Mt: Dynamics, Gt: Potential, N: int,
               backward=False):
    """
    Get a local cSMC kernel as per Finke and Thiery  (2021).

    Parameters:
    -----------
    M0:
        Initial distribution.
    G0:
        Initial potential.
    Mt:
        Dynamics of the model.
    Gt:
        Potential of the model.
    N: int
        Total number of particles to use in the cSMC sampler.
    backward: bool
        Whether to perform backward sampling or not. If True, the dynamics must implement a valid logpdf method.

    Returns:
    --------
    kernel: Callable
        cSMC kernel.
    init: Callable
        Function to initialize the state of the sampler given a trajectory.
    """

    def kernel(key, state, delta):
        key_fwd, key_bwd = jax.random.split(key)
        w_T, xs, log_ws, As = _csmc(key_fwd, state.x, M0, G0, Mt, Gt, N, delta)
        if not backward:
            x, ancestors = _backward_scanning_pass(key_bwd, w_T, xs, As)
        else:
            x, ancestors = _backward_sampling_pass(key_bwd, Mt, w_T, xs, log_ws)

        return CSMCState(x=x, ancestors=ancestors)

    def init(x_star):
        T, *_ = x_star.shape
        ancestors = jnp.zeros((T,), dtype=jnp.int_)
        return CSMCState(x=x_star, ancestors=ancestors)

    return init, kernel


def _csmc(key, x_star, M0, G0, Mt, Gt, N, delta):
    T, d = x_star.shape
    keys = jax.random.split(key, T+1)

    # Sample proposal particles
    sqrt_half_delta = jnp.sqrt(0.5 * delta)
    center = x_star + sqrt_half_delta * jax.random.normal(keys[0], (T, d))
    prop_xs = center[:, None, :] + sqrt_half_delta * jax.random.normal(keys[1], (T, N, d))
    prop_xs = prop_xs.at[:, 0].set(x_star)

    log_w0 = G0.logpdf(prop_xs[0]) + M0.logpdf(prop_xs[0])
    w0 = normalize(log_w0)

    def body(w_t_m_1, inp):
        x_t_m_1, x_t, Mt_params, Gt_params, key_t = inp
        # Conditional resampling

        A_t = multinomial(key_t, w_t_m_1)
        x_t_m_1 = jnp.take(x_t_m_1, A_t, axis=0)

        # Compute weights
        log_w_t = Gt.logpdf(x_t, x_t_m_1, Gt_params) + Mt.logpdf(x_t, x_t_m_1, Mt_params)
        w_t = normalize(log_w_t)
        # Return next step
        return w_t, (log_w_t, A_t)

    prop_xs_t_m_1, prop_xs_t = prop_xs[:-1], prop_xs[1:]
    inps = prop_xs_t_m_1, prop_xs_t, Mt.params, Gt.params, keys[2:]

    w_T, (log_ws, As) = jax.lax.scan(body, w0, inps)
    log_ws = jnp.insert(log_ws, 0, log_w0, axis=0)
    return w_T, prop_xs, log_ws, As


def _backward_sampling_pass(key, Mt: Dynamics, w_T, xs, log_ws):
    T = xs.shape[0]
    keys = jax.random.split(key, T)

    B_T = jax.random.choice(keys[0], w_T.shape[0], p=w_T, shape=())
    x_T = xs[-1, B_T]

    def body(x_t, inp):
        op_key, xs_t_m_1, log_w_t_m_1, Mt_m_1_params = inp
        log_w = Mt.logpdf(x_t, xs_t_m_1, Mt_m_1_params) + log_w_t_m_1
        w = normalize(log_w)
        B_t_m_1 = jax.random.choice(op_key, w.shape[0], p=w, shape=())
        x_t_m_1 = xs_t_m_1[B_t_m_1]
        return x_t_m_1, (x_t_m_1, B_t_m_1)

    Mt_params = tree_map(lambda x: x[:0:-1], Mt.params)

    # xs[-2::-1] is the reversed list of xs[:-1], I know, not readable... Same for log_ws[:-1].
    inps = keys[1:], xs[-2::-1], log_ws[-2::-1], Mt_params
    _, (xs, Bs) = jax.lax.scan(body, x_T, inps)
    xs = jnp.insert(xs, 0, x_T, axis=0)
    Bs = jnp.insert(Bs, 0, B_T, axis=0)
    return xs[::-1], Bs[::-1]
