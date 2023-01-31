import jax.numpy as jnp


def delta_adaptation(delta, target_rate, acceptance_rate, adaptation_rate, min_delta=1e-20):
    """
    A simple adaptation rule for the delta parameter of the auxiliary Kalman sampler.

    Parameters
    ----------
    delta:
        Current value of delta.
    target_rate:
        Target acceptance rate.
    acceptance_rate:
        Current average acceptance rate.
    adaptation_rate: float
        Adaptation rate.
    min_delta: float
        Minimum value of delta.

    Returns
    -------
    delta:
        Adapted value of delta.

    """
    rate = jnp.exp(adaptation_rate * (acceptance_rate - target_rate))
    # rate = 1 - (target_rate - acceptance_rate) * adaptation_rate
    out = delta * rate
    return jnp.maximum(out, min_delta)
