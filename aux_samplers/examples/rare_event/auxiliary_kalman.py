import jax.numpy as jnp
from jax.scipy.stats import norm

from aux_samplers.kalman import get_kernel as get_generic_kernel


def get_kernel(y, rho, r2, T, parallel, grad):
    m0 = jnp.zeros((1,))
    P0 = jnp.eye(1)

    F = rho * jnp.eye(1)
    Q = (1 - rho ** 2) * jnp.eye(1)
    b = jnp.zeros((1,))

    Fs = jnp.tile(F[None, ...], (T - 1, 1, 1))
    Qs = jnp.tile(Q[None, ...], (T - 1, 1, 1))
    bs = jnp.tile(b[None, ...], (T - 1, 1))

    r = jnp.sqrt(r2)

    Hs = jnp.ones((T, 1, 1))
    Rs = jnp.ones((T, 1, 1))
    cs = jnp.zeros((T, 1))

    def dynamics_factory(_x):
        return m0, P0, Fs, Qs, bs

    def observations_factory(x, u, delta):
        grad_x = jnp.zeros((T, 1))
        if grad:
            grad_x_m_1 = (x[-1] - y) / r2
            grad_x = grad_x.at[-1].set(grad_x_m_1)
        aux_ys = u + 0.5 * delta * grad_x
        return aux_ys, Hs, 0.5 * delta * Rs, cs

    def log_likelihood_fn(x):
        out = norm.logpdf(x[0, 0], m0, 1).sum()
        pred_x = rho * x[:-1, 0]
        out += norm.logpdf(x[1:, 0], pred_x, (1 - rho ** 2) ** 0.5).sum()
        out += norm.logpdf(y, x[-1, 0], r)
        return out

    init_, kernel = get_generic_kernel(dynamics_factory, observations_factory, log_likelihood_fn, parallel)

    def init(xs):
        if jnp.ndim(xs) == 1:
            return init_(jnp.expand_dims(xs, -1))
        return init_(xs)

    return init, kernel
