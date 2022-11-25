"""
Implements the classical cSMC kernel from the seminal pMCMC paper by Andrieu et al. (2010). We also implement the
backward sampling step of Whiteley.
"""
import jax
import jax.numpy as jnp
from chex import ArrayTree, dataclass, Array

from .base import Distribution, Potential, UnivariatePotential, Dynamics, normalize
from .resamplings import multinomial


@dataclass
class CSMCState:
    x: ArrayTree
    ancestors: Array


def get_kernel(M0: Distribution, G0: UnivariatePotential, Mt: Dynamics, Gt: Potential, N: int,
               backward=False):
    def kernel(key, state):
        key_fwd, key_bwd = jax.random.split(key)
        w_T, xs, log_ws, As = _csmc(key_fwd, state.x, M0, G0, Mt, Gt, N)
        if not backward:
            x, ancestors = _backward_scanning_pass(key_bwd, w_T, xs, As)
            return CSMCState(x=x, ancestors=ancestors)
        else:
            raise NotImplementedError

    def init(x_star):
        T, *_ = x_star.shape
        ancestors = jnp.zeros((T,), dtype=jnp.int_)
        return CSMCState(x=x_star, ancestors=ancestors)

    return init, kernel


def _csmc(key, x_star, M0, G0, Mt, Gt, N):
    T = x_star.shape[0]
    keys = jax.random.split(key, T)

    # Sample initial state
    x0 = M0.sample(keys[0], N)
    # Replace the first particle with the star trajectory
    x0 = x0.at[0].set(x_star[0])

    # Compute initial weights and normalize
    log_w0 = G0.logpdf(x0)
    w0 = normalize(log_w0)

    def body(carry, inp):
        w_t_m_1, x_t_m_1 = carry
        Mt_params, Gt_params, x_star_t, key_t = inp
        resampling_key, sampling_key = jax.random.split(key_t)
        # Conditional resampling
        A_t = multinomial(resampling_key, w_t_m_1)
        x_t_m_1 = jnp.take(x_t_m_1, A_t, axis=0)

        # Sample new particles
        x_t = Mt.sample(sampling_key, x_t_m_1, Mt_params)
        x_t = x_t.at[0].set(x_star_t)

        # Compute weights
        log_w_t = Gt.logpdf(x_t, x_t_m_1, Gt_params)
        w_t = normalize(log_w_t)
        # Return next step
        next_carry = (w_t, x_t)
        save = (x_t, log_w_t, A_t)

        return next_carry, save

    (w_T, _), (xs, log_ws, As) = jax.lax.scan(body, (w0, x0), (Mt.params, Gt.params, x_star[1:], keys[1:]))

    log_ws = jnp.insert(log_ws, 0, log_w0, axis=0)
    xs = jnp.insert(xs, 0, x0, axis=0)
    return w_T, xs, log_ws, As


def _backward_scanning_pass(key, w_T, xs, As):
    B_T = jax.random.choice(key, w_T.shape[0], p=w_T, shape=())
    x_T = xs[-1, B_T]

    def body(B_t, inp):
        xs_t_m_1, A_t = inp
        B_t_m_1 = A_t[B_t]
        x_t_m_1 = xs_t_m_1[B_t_m_1]
        return B_t_m_1, (x_t_m_1, B_t_m_1)

    _, (xs, Bs) = jax.lax.scan(body, B_T, (xs[:0:-1], As[::-1]))
    xs = jnp.insert(xs, 0, x_T, axis=0)
    Bs = jnp.insert(Bs, 0, B_T, axis=0)
    return xs[::-1], Bs[::-1]


def _backward_sampling_pass():
    pass
