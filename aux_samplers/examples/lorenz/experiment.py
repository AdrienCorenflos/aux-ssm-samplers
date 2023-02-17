import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import tqdm.auto as tqdm
from jax.tree_util import tree_map

from aux_samplers.common import delta_adaptation
from auxiliary_kalman import get_kernel as get_kalman_kernel
from model import get_dynamics, init_x_fn, observations_model

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
parser.add_argument("--target-alpha", dest="target_alpha", type=float, default=0.5)
parser.add_argument("--lr", dest="lr", type=float, default=0.1)
parser.add_argument("--beta", dest="beta", type=float, default=0.01)
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

jax.config.update("jax_enable_x64", True)
if not args.gpu:
    jax.config.update("jax_platform_name", "cpu")
else:
    jax.config.update("jax_platform_name", "gpu")

# PARAMETERS
# we use the exact same parameters as Mider et al.
theta_prior_cov = 1e3 * jnp.eye(3)
true_theta = jnp.array([10., 28., 8. / 3.])
M0 = jnp.array([1.5, -1.5, 25.])
P0 = jnp.diag(jnp.array([400., 20., 20.]))
SIGMA_X = 3.
SIGMA_Y = 5 ** 0.5

DATA = np.loadtxt("data.csv", delimiter=",", skiprows=1)
T = DATA[-1, 0]
OBS_FREQ = DATA[1, 0] - DATA[0, 0]
SMOOTH_FREQ = 2e-4
N_STEPS = int(T / SMOOTH_FREQ + 1e-6) + 1
SAMPLE_EVERY = int(OBS_FREQ / SMOOTH_FREQ + 1e-6)

# Observation model can be computed offlien
YS, HS, RS, CS = observations_model(DATA, SIGMA_Y, N_STEPS, SAMPLE_EVERY)


# STATS FN
def stats_fn(x_1, x_2):
    # squared jumping distance averaged across dimensions, and first and second moments
    return (x_2 - x_1) ** 2, x_2, x_2 ** 2


# KERNEL
def loop(key, init_delta, init_state, kernel_fn, delta_fn, n_iter, verbose=False):
    from datetime import datetime
    keys = jax.random.split(key, n_iter)

    print_func = lambda z, *_: print(f"\riteration: {z[0]}, time: {datetime.now().strftime('%H:%M:%S')}, "
                                     f"min_δ: {z[1]:.2e}, max_δ: {z[2]:.2e}, "
                                     f"min_window_accept: {z[3]:.1%},  max_window_accept: {z[4]:.1%}, "
                                     f"min_accept: {z[5]:.1%}, max_accept: {z[6]:.1%}, "
                                     f"min_esjd: {z[7]:.2e}, max_esjd: {z[8]:.2e}", end="")

    def body_fn(carry, key_inp):
        from jax.experimental.host_callback import id_tap
        i, stats, state, delta, window_avg_acceptance, avg_acceptance = carry
        if verbose:
            id_tap(print_func, (i,
                                jnp.min(delta), jnp.max(delta),
                                jnp.min(window_avg_acceptance), jnp.max(window_avg_acceptance),
                                jnp.min(avg_acceptance), jnp.max(avg_acceptance),
                                jnp.min(stats[0]), jnp.max(stats[0]),
                                ), result=None)

        next_state = kernel_fn(key_inp, state, delta)
        next_stats = stats_fn(state.x, next_state.x)

        # moving average.
        avg_acceptance = (i * avg_acceptance + 1. * next_state.updated) / (i + 1)
        window_avg_acceptance = args.beta * next_state.updated + (1 - args.beta) * window_avg_acceptance
        stats = tree_map(lambda u, v: (i * u + v) / (i + 1), stats, next_stats)

        if delta_fn is not None:
            lr = (n_iter - i) * args.lr / n_iter
            delta = delta_fn(delta, args.target_alpha, window_avg_acceptance, lr)

        carry = i + 1, stats, next_state, delta, window_avg_acceptance, avg_acceptance
        return carry, None

    init_stats = stats_fn(init_state.x, init_state.x)
    init = 0, init_stats, init_state, init_delta, 1. * init_state.updated, 1. * init_state.updated

    out, _ = jax.lax.scan(body_fn, init, keys)
    return out


@jax.jit
def one_experiment(exp_key, verbose=args.verbose):
    # DATA
    data_key, init_key, burnin_key, sample_key = jax.random.split(exp_key, 4)

    # INIT
    xs_init = init_x_fn(DATA, N_STEPS)

    # KERNEL
    if args.style == "kalman":
        init_fn, kernel_fn = get_kalman_kernel(YS, HS, RS, CS, M0, P0, true_theta, SIGMA_X, SMOOTH_FREQ, args.parallel)
        delta_init = args.delta_init
    # elif args.style == "kalman-2":
    #     init_fn, kernel_fn = get_kalman_kernel(ys, m0, P0, F, Q, b, args.parallel, order=2)
    #     delta_init = args.delta_init
    # elif args.style == "csmc":
    #     init_fn, kernel_fn = get_csmc_kernel(ys, m0, P0, F, Q, b, args.N, args.backward, args.parallel, args.gradient)
    #     delta_init = args.delta_init * jnp.ones((args.T,))
    # elif args.style == "csmc-guided":
    #     init_fn, kernel_fn = get_guided_csmc_kernel(ys, m0, P0, F, Q, b, args.N, args.backward, args.gradient)
    #     delta_init = args.delta_init * jnp.ones((args.T,))
    else:
        raise NotImplementedError

    init_state = init_fn(xs_init)

    # BURNIN
    _, _, burnin_state, burnin_delta, burnin_avg_acceptance, _ = loop(burnin_key,
                                                                      delta_init,
                                                                      init_state,
                                                                      kernel_fn,
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
    _, stats, _, out_delta, _, pct_accepted = loop(sample_key, burnin_delta, burnin_state, kernel_fn, None,
                                                   args.n_samples, verbose)

    output_shape = (jax.ShapeDtypeStruct((), pct_accepted.dtype),
                    jax.ShapeDtypeStruct(pct_accepted.shape, pct_accepted.dtype))

    toc, _ = call(tic_fn, pct_accepted,
                  result_shape=output_shape)
    return toc - tic, stats, xs_init, pct_accepted, burnin_delta


def full_experiment():
    key = jax.random.PRNGKey(args.seed)

    time_taken, (esjd, traj, squared_exp), xs_init, pct_accepted, burnin_delta = one_experiment(key)

    from matplotlib import pyplot as plt

    ax = plt.axes(projection='3d')
    ax.set_box_aspect(aspect=(4, 1, 1))
    ax.plot(np.linspace(0, args.T, N_STEPS), traj[..., 1], traj[..., 2], color="tab:orange")
    plt.show()

    ax = plt.axes(projection='3d')
    ax.set_box_aspect(aspect=(1, 1, 1))
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2])
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
