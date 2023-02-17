# This specific script failed randomly with some XLA/CUDA errors at start-up for my computer.
# If you want to run it, you may have to just try again a few times. I am not sure what is causing this. It may just be
# a problem with my computer. Will try on another one soon.
# I am typically getting one of these errors:
# - jaxlib.xla_extension.XlaRuntimeError: INTERNAL: cuSolver internal error
# - jaxlib.xla_extension.XlaRuntimeError: INTERNAL: Failed to execute XLA Runtime executable: run time error: custom call 'xla.gpu.custom_call' failed: jaxlib/gpu/prng_kernels.cc:33: operation gpuGetLastError() failed: out of memory.
# - jaxlib/gpu/solver_kernels.cc:45: operation gpusolverDnCreate(&handle) failed: cuSolver internal error

import argparse
import time
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
from chex import dataclass
from jax.tree_util import tree_map

from aux_samplers import SamplerState
from aux_samplers.common import delta_adaptation
from aux_samplers.kalman.generic import KalmanSampler
from auxiliary_kalman import get_kernel as get_kalman_kernel
from model import init_x_fn, observations_model, theta_posterior_mean_and_chol

# ARGS PARSING

parser = argparse.ArgumentParser("Run a Lorenz experiment")
# General arguments
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--no-parallel', dest='parallel', action='store_false')
parser.set_defaults(parallel=True)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=False)
parser.add_argument('--debug-nans', action='store_true')
parser.add_argument('--no-debug-nans', dest='debug_nans', action='store_false')
parser.set_defaults(debug_nans=False)
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--no-gpu', dest='gpu', action='store_false')
parser.set_defaults(gpu=True)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--no-verbose', dest='verbose', action='store_false')
parser.set_defaults(verbose=True)

# Experiment arguments
parser.add_argument("--n-samples", dest="n_samples", type=int, default=10_000)
parser.add_argument("--burnin", dest="burnin", type=int, default=2_500)
parser.add_argument("--target-alpha", dest="target_alpha", type=float, default=0.234)
parser.add_argument("--lr", dest="lr", type=float, default=0.5)
parser.add_argument("--beta", dest="beta", type=float, default=0.05)
parser.add_argument("--delta-init", dest="delta_init", type=float, default=1e-5)
parser.add_argument("--seed", dest="seed", type=int, default=1234)
parser.add_argument("--style", dest="style", type=str, default="kalman")
parser.add_argument("--gradient", action='store_true')
parser.add_argument('--no-gradient', dest='gradient', action='store_false')
parser.set_defaults(gradient=True)
parser.add_argument("--backward", action='store_true')
parser.add_argument('--no-backward', dest='backward', action='store_false')
parser.set_defaults(backward=True)
parser.add_argument("--N", dest="N", type=int, default=25)

args = parser.parse_args()

# BACKEND CONFIG
NOW = time.time()

jax.config.update("jax_enable_x64", False)
if not args.gpu:
    jax.config.update("jax_platform_name", "cpu")
else:
    jax.config.update("jax_platform_name", "gpu")

# PARAMETERS
# we use the exact same parameters as Mider et al.
theta_prior_cov = 1e3 * jnp.eye(3)
true_theta = jnp.array([10., 28., 8. / 3.])
true_xs = np.loadtxt("true_xs.csv", delimiter=",", skiprows=1)
M0 = jnp.array([1.5, -1.5, 25.])
P0 = jnp.diag(jnp.array([400., 20., 20.]))
SIGMA_X = 3.
SIGMA_Y = 5 ** 0.5
THETA_0 = jnp.array([5.0, 15.0, 6.0])
DATA = np.loadtxt("data.csv", delimiter=",", skiprows=1)
T = DATA[-1, 0]
OBS_FREQ = DATA[1, 0] - DATA[0, 0]
SMOOTH_FREQ = 2e-4
N_STEPS = int(T / SMOOTH_FREQ + 1e-6) + 1
SAMPLE_EVERY = int(OBS_FREQ / SMOOTH_FREQ + 1e-6)

# Observation model can be computed offlien
YS, HS, RS, CS = observations_model(DATA, SIGMA_Y, N_STEPS, SAMPLE_EVERY)


# Gibbs kernel and state
@dataclass
class GibbsState:
    kalman_state: KalmanSampler
    theta: chex.Array


# STATS FN
def stats_fn(x_1, x_2):
    # squared jumping distance averaged across dimensions, and first and second moments
    return (x_2 - x_1) ** 2, x_2, x_2 ** 2


# KERNEL
def loop(key, init_delta, init_state, delta_fn, n_iter, verbose=False, return_samples=False, update_theta=False):
    from datetime import datetime
    keys = jax.random.split(key, n_iter)

    print_func = lambda z, *_: print(f"\riteration: {z[0]}, time: {datetime.now().strftime('%H:%M:%S')}, "
                                     f"min_δ: {z[1]:.2e}, max_δ: {z[2]:.2e}, "
                                     f"min_window_accept: {z[3]:.1%},  max_window_accept: {z[4]:.1%}, "
                                     f"min_accept: {z[5]:.1%}, max_accept: {z[6]:.1%}, "
                                     f"min_esjd: {z[7]:.2e}, max_esjd: {z[8]:.2e}, "
                                     f"theta_0: {z[9]:.2f}, theta_1: {z[10]:.2f}, theta_2: {z[11]:.2f}",
                                     end="")

    def body_fn(carry, key_inp):
        from jax.experimental.host_callback import id_tap
        i, stats, state, delta, window_avg_acceptance, avg_acceptance = carry
        if verbose:
            id_tap(print_func, (i,
                                jnp.min(delta), jnp.max(delta),
                                jnp.min(window_avg_acceptance), jnp.max(window_avg_acceptance),
                                jnp.min(avg_acceptance), jnp.max(avg_acceptance),
                                jnp.min(stats[0]), jnp.max(stats[0]),
                                state.theta[0], state.theta[1], state.theta[2]
                                ), result=None)
        kalman_state = state.kalman_state
        theta = state.theta
        key_kalman, key_theta = jax.random.split(key_inp, 2)

        _, kernel_fn = get_kalman_kernel(YS, HS, RS, CS, M0, P0, theta, SIGMA_X, SMOOTH_FREQ, args.parallel)
        next_kalman_state = kernel_fn(key_kalman, kalman_state, delta)
        next_stats = stats_fn(kalman_state.x, next_kalman_state.x)

        # moving average.
        avg_acceptance = (i * avg_acceptance + 1. * next_kalman_state.updated) / (i + 1)
        window_avg_acceptance = args.beta * next_kalman_state.updated + (1 - args.beta) * window_avg_acceptance
        stats = tree_map(lambda u, v: (i * u + v) / (i + 1), stats, next_stats)

        if delta_fn is not None:
            lr = (n_iter - i) * args.lr / n_iter
            delta = delta_fn(delta, args.target_alpha, window_avg_acceptance, lr)
        if update_theta:
            theta_mean, theta_chol = theta_posterior_mean_and_chol(next_kalman_state.x, theta_prior_cov, SMOOTH_FREQ,
                                                                   SIGMA_X)
            next_theta = theta_mean + theta_chol @ jax.random.normal(key_theta, (3,))
        else:
            next_theta = theta

        next_state = GibbsState(kalman_state=next_kalman_state, theta=next_theta)
        carry = i + 1, stats, next_state, delta, window_avg_acceptance, avg_acceptance
        if return_samples:
            return carry, (next_kalman_state.x[::SAMPLE_EVERY], next_theta)
        return carry, None

    init_stats = stats_fn(init_state.kalman_state.x, init_state.kalman_state.x)
    init = 0, init_stats, init_state, init_delta, 1. * init_state.kalman_state.updated, 1. * init_state.kalman_state.updated

    out, xs = jax.lax.scan(body_fn, init, keys)
    return out, xs


# @partial(jax.jit, static_argnums=(1,))
def one_experiment(exp_key, verbose=args.verbose):
    # DATA
    data_key, init_key, burnin_key, sample_key = jax.random.split(exp_key, 4)

    # INIT
    xs_init = init_x_fn(DATA, N_STEPS)
    theta_init_mean, theta_init_chol = theta_posterior_mean_and_chol(xs_init, theta_prior_cov, SMOOTH_FREQ, SIGMA_X)
    init_kalman_state = KalmanSampler(x=xs_init, updated=True)  # noqa

    init_state = GibbsState(kalman_state=init_kalman_state, theta=true_theta)
    # BURNIN
    (_, _, burnin_state, burnin_delta, burnin_avg_acceptance, _), _ = loop(burnin_key,
                                                                           args.delta_init,
                                                                           init_state,
                                                                           delta_adaptation,
                                                                           args.burnin,
                                                                           verbose)
    from jax.experimental.host_callback import call

    def tic_fn(arr):
        time_elapsed = time.time() - NOW
        return np.array(time_elapsed, dtype=arr.dtype), arr

    output_shape = (jax.ShapeDtypeStruct((), burnin_delta.dtype),
                    jax.ShapeDtypeStruct(burnin_delta.shape, burnin_delta.dtype))

    tic, burnin_delta = call(tic_fn, burnin_delta,
                             result_shape=output_shape)
    (_, stats, _, out_delta, _, pct_accepted), (traj_samples, theta_samples) = loop(sample_key, burnin_delta,
                                                                                    burnin_state, None,
                                                                                    args.n_samples, verbose, True,
                                                                                    False)

    output_shape = (jax.ShapeDtypeStruct((), pct_accepted.dtype),
                    jax.ShapeDtypeStruct(pct_accepted.shape, pct_accepted.dtype))

    toc, _ = call(tic_fn, pct_accepted,
                  result_shape=output_shape)
    return toc - tic, stats, xs_init, pct_accepted, burnin_delta, traj_samples, theta_samples


def full_experiment():
    key = jax.random.PRNGKey(args.seed)

    time_taken, (
        esjd, traj, squared_exp), xs_init, pct_accepted, burnin_delta, traj_samples, theta_samples = one_experiment(key)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(ncols=3, figsize=(20, 3))
    axes[0].plot(theta_samples[:, 0])
    axes[1].plot(theta_samples[:, 1])
    axes[2].plot(theta_samples[:, 2])

    plt.show()
    fig, axes = plt.subplots(ncols=3, figsize=(20, 3))
    axes[0].plot(traj_samples[:, 150, 0])
    axes[1].plot(traj_samples[:, 150, 1])
    axes[2].plot(traj_samples[:, 150, 2])

    plt.show()

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    for rot in range(0, 360, 10):
        fig = plt.figure()
        fig.suptitle(f"Rot: {rot}")
        ax = plt.axes(projection='3d')
        ax.set_box_aspect(aspect=(1, 1, 1))
        for sample in traj_samples[::200]:
            sample = sample.reshape(-1, 1, 3)
            segments = np.concatenate([sample[:-1], sample[1:]], axis=1)
            lc = Line3DCollection(segments, cmap=plt.get_cmap('winter'))
            lc.set_array(DATA[:, 0] / T)
            lc.set_alpha(0.05)
            ax.add_collection3d(lc)
        ax.set_xlim(traj_samples[..., 0].min(), traj_samples[..., 0].max())
        ax.set_ylim(traj_samples[..., 1].min(), traj_samples[..., 1].max())
        ax.set_zlim(traj_samples[..., 2].min(), traj_samples[..., 2].max())
        ax.plot(true_xs[:, 1], true_xs[:, 2], true_xs[:, 3], 'k', alpha=1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(15, rot)
        plt.show()

    return ejsd_per_key, acceptance_rate_per_key, delta_per_key, time_per_key


def main():
    import os
    with jax.disable_jit(args.debug):
        with jax.debug_nans(args.debug_nans):
            ejsd_per_key, acceptance_rate_per_key, delta_per_key, time_per_key = full_experiment()
    file_name = f"results/{args.style}-{args.D}-{args.T}-{args.N}-{args.parallel}-{args.gradient}.npz"
    if not os.path.isdir("results"):
        os.mkdir("results")
    np.savez(file_name,
             ejsd_per_key=ejsd_per_key,
             acceptance_rate_per_key=acceptance_rate_per_key,
             delta_per_key=delta_per_key,
             time_per_key=time_per_key)


if __name__ == "__main__":
    main()
