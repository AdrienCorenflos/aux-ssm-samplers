# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
#
#  The following code is a derivative work of the code from BlackJAX
#  where we simply added the possibility to divide by the true variance instead
#  of the estimated empirical variance


import jax
import jax.numpy as jnp
import numpy as np
from scipy.fftpack import next_fast_len  # type: ignore

def effective_sample_size(
    input_array, var, chain_axis: int = 0, sample_axis: int = 1):
    """Compute estimate of the effective sample size (ess).

    Parameters
    ----------
    input_array:
        An array representing multiple chains of MCMC samples. The array must
        contains a chain dimension and a sample dimension.
    var:
        Known variance of the target distribution.
    chain_axis
        The axis indicating the multiple chains. Default to 0.
    sample_axis
        The axis indicating a single chain of MCMC samples. Default to 1.

    Returns
    -------
    NDArray of the resulting statistics (ess), with the chain and sample dimensions squeezed.

    Notes
    -----
    The basic ess (:math:`N_{\\mathit{eff}}`) diagnostic is computed by:

    .. math:: \\hat{N}_{\\mathit{eff}} = \\frac{MN}{\\hat{\\tau}}

    .. math:: \\hat{\\tau} = -1 + 2 \\sum_{t'=0}^K \\hat{P}_{t'}

    where :math:`M` is the number of chains, :math:`N` the number of draws,
    :math:`\\hat{\\rho}_t` is the estimated _autocorrelation at lag :math:`t`, and
    :math:`K` is the last integer for which :math:`\\hat{P}_{K} = \\hat{\\rho}_{2K} +
    \\hat{\\rho}_{2K+1}` is still positive :cite:p:`stan_ess,gelman1995bayesian`.

    The current implementation is similar to Stan, which uses Geyer's initial monotone sequence
    criterion :cite:p:`geyer1992practical,geyer2011introduction`.

    """
    input_shape = input_array.shape
    sample_axis = sample_axis if sample_axis >= 0 else len(input_shape) + sample_axis
    num_chains = input_shape[chain_axis]
    num_samples = input_shape[sample_axis]

    mean_across_chain = input_array.mean(axis=sample_axis, keepdims=True)
    # Compute autocovariance estimates for every lag for the input array using FFT.
    centered_array = input_array - mean_across_chain
    m = next_fast_len(2 * num_samples)
    ifft_ary = jnp.fft.rfft(centered_array, n=m, axis=sample_axis)
    ifft_ary *= jnp.conjugate(ifft_ary)
    autocov_value = jnp.fft.irfft(ifft_ary, n=m, axis=sample_axis)
    autocov_value = (
        jnp.take(autocov_value, jnp.arange(num_samples), axis=sample_axis) / num_samples
    )
    mean_autocov_var = autocov_value.mean(chain_axis, keepdims=True)
    mean_var0 = (
        jnp.take(mean_autocov_var, jnp.array([0]), axis=sample_axis)
        * num_samples
        / (num_samples - 1.0)
    )
    weighted_var = mean_var0 * (num_samples - 1.0) / num_samples
    weighted_var = jax.lax.cond(
        num_chains > 1,
        lambda mean_across_chain: weighted_var
        + mean_across_chain.var(axis=chain_axis, ddof=1, keepdims=True),
        lambda _: weighted_var,
        operand=mean_across_chain,
    )
    if var is not None:
        weighted_var *= 0.
        weighted_var += var
        mean_var0 *= 0.
        mean_var0 += var
    # Geyer's initial positive sequence
    num_samples_even = num_samples - num_samples % 2
    mean_autocov_var_tp1 = jnp.take(
        mean_autocov_var, jnp.arange(1, num_samples_even), axis=sample_axis
    )
    rho_hat = jnp.concatenate(
        [
            jnp.ones_like(mean_var0),
            1.0 - (mean_var0 - mean_autocov_var_tp1) / weighted_var,
        ],
        axis=sample_axis,
    )

    rho_hat = jnp.moveaxis(rho_hat, sample_axis, 0)
    rho_hat_even = rho_hat[0::2]
    rho_hat_odd = rho_hat[1::2]

    mask0 = (rho_hat_even + rho_hat_odd) > 0.0
    carry_cond = jnp.ones_like(mask0[0])
    max_t = jnp.zeros_like(mask0[0], dtype=int)

    def positive_sequence_body_fn(state, mask_t):
        t, carry_cond, max_t = state
        next_mask = carry_cond & mask_t
        next_max_t = jnp.where(next_mask, jnp.ones_like(max_t) * t, max_t)
        return (t + 1, next_mask, next_max_t), next_mask

    (*_, max_t_next), mask = jax.lax.scan(
        positive_sequence_body_fn, (0, carry_cond, max_t), mask0
    )
    indices = jnp.indices(max_t_next.shape)
    indices = tuple([max_t_next + 1] + [indices[i] for i in range(max_t_next.ndim)])
    rho_hat_odd = jnp.where(mask, rho_hat_odd, jnp.zeros_like(rho_hat_odd))
    # improve estimation
    mask_even = mask.at[indices].set(rho_hat_even[indices] > 0)
    rho_hat_even = jnp.where(mask_even, rho_hat_even, jnp.zeros_like(rho_hat_even))

    # Geyer's initial monotone sequence
    def monotone_sequence_body_fn(rho_hat_sum_tm1, rho_hat_sum_t):
        update_mask = rho_hat_sum_t > rho_hat_sum_tm1
        next_rho_hat_sum_t = jnp.where(update_mask, rho_hat_sum_tm1, rho_hat_sum_t)
        return next_rho_hat_sum_t, (update_mask, next_rho_hat_sum_t)

    rho_hat_sum = rho_hat_even + rho_hat_odd
    _, (update_mask, update_value) = jax.lax.scan(
        monotone_sequence_body_fn, rho_hat_sum[0], rho_hat_sum
    )

    rho_hat_even_final = jnp.where(update_mask, update_value / 2.0, rho_hat_even)
    rho_hat_odd_final = jnp.where(update_mask, update_value / 2.0, rho_hat_odd)

    # compute effective sample size
    ess_raw = num_chains * num_samples
    tau_hat = (
        -1.0
        + 2.0 * jnp.sum(rho_hat_even_final + rho_hat_odd_final, axis=0)
        - rho_hat_even_final[indices]
    )

    tau_hat = jnp.maximum(tau_hat, 1 / np.log10(ess_raw))
    ess = ess_raw / tau_hat

    return ess.squeeze()