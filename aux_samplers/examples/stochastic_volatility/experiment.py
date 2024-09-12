import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import tqdm.auto as tqdm
from jax.tree_util import tree_map

from aux_samplers.common import delta_adaptation
from auxiliary_csmc import get_kernel as get_csmc_kernel
from auxiliary_guided_csmc import get_kernel as get_guided_csmc_kernel
from auxiliary_kalman import get_kernel as get_kalman_kernel
from model import get_data, get_dynamics, init_x_fn

# ARGS PARSING

parser = argparse.ArgumentParser("Run a Stochastic volatility experiment")
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
parser.set_defaults(verbose=False)
parser.add_argument("--precision", dest="precision", type=str, default="single")

# Experiment arguments
parser.add_argument("--n-experiments", dest="n_experiments", type=int, default=10)
parser.add_argument("--T", dest="T", type=int, default=250)
parser.add_argument("--D", dest="D", type=int, default=30)
parser.add_argument("--n-samples", dest="n_samples", type=int, default=10_000)
parser.add_argument("--burnin", dest="burnin", type=int, default=2_500)
parser.add_argument("--target-alpha", dest="target_alpha", type=float, default=0.5)
parser.add_argument("--lr", dest="lr", type=float, default=0.1)
parser.add_argument("--beta", dest="beta", type=float, default=0.01)
parser.add_argument("--delta-init", dest="delta_init", type=float, default=1e-8)
parser.add_argument("--seed", dest="seed", type=int, default=1234)
parser.add_argument("--style", dest="style", type=str, default="csmc")
parser.add_argument("--gradient", action='store_true')
parser.add_argument('--no-gradient', dest='gradient', action='store_false')
parser.set_defaults(gradient=False)
parser.add_argument("--backward", action='store_true')
parser.add_argument('--no-backward', dest='backward', action='store_false')
parser.set_defaults(backward=True)
parser.add_argument("--N", dest="N", type=int, default=25)

args = parser.parse_args()

# BACKEND CONFIG
NOW = time.time()

if args.precision.lower() in {"single", "float32", "32"}:
    jax.config.update("jax_enable_x64", False)
elif args.precision.lower() in {"double", "float64", "64"}:
    jax.config.update("jax_enable_x64", True)
else:
    raise ValueError(
        f"Unknown precision {args.precision}, note that jax doesn't support half precision. Stick to single or double.")

if not args.gpu:
    jax.config.update("jax_platform_name", "cpu")
else:
    jax.config.update("jax_platform_name", "gpu")

# PARAMETERS
# we use the same parameters as Finke and Thiery.
NU, PHI, TAU, RHO = 0., .9, 2., .25
m0, P0, F, Q, b = get_dynamics(NU, PHI, TAU, RHO, args.D)


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
        from jax.debug import callback
        i, stats, state, delta, window_avg_acceptance, avg_acceptance = carry
        if verbose:
            callback(print_func, (i,
                                jnp.min(delta), jnp.max(delta),
                                jnp.min(window_avg_acceptance), jnp.max(window_avg_acceptance),
                                jnp.min(avg_acceptance), jnp.max(avg_acceptance),
                                jnp.min(jnp.sum(stats[0], -1)), jnp.max(jnp.sum(stats[0], -1)),
                                ), ordered=True)

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
    true_xs, ys = get_data(data_key, NU, PHI, TAU, RHO, args.D, args.T)

    # INIT
    xs_init = init_x_fn(init_key, ys, NU, PHI, TAU, RHO, 10_000)

    # KERNEL
    if args.style == "kalman-1":
        init_fn, kernel_fn = get_kalman_kernel(ys, m0, P0, F, Q, b, args.parallel, order=1)
        delta_init = args.delta_init
    elif args.style == "kalman-2":
        init_fn, kernel_fn = get_kalman_kernel(ys, m0, P0, F, Q, b, args.parallel, order=2)
        delta_init = args.delta_init
    elif args.style == "csmc":
        init_fn, kernel_fn = get_csmc_kernel(ys, m0, P0, F, Q, b, args.N, args.backward, args.parallel, args.gradient)
        delta_init = args.delta_init * jnp.ones((args.T,))
    elif args.style == "csmc-guided":
        init_fn, kernel_fn = get_guided_csmc_kernel(ys, m0, P0, F, Q, b, args.N, args.backward, args.gradient)
        delta_init = args.delta_init * jnp.ones((args.T,))
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
    from jax.experimental import io_callback as call
    from jax.debug import callback

    def tic_fn(arr):
        time_elapsed = time.time() - NOW
        return arr, np.array(time_elapsed, dtype=arr.dtype)


    burnin_delta, tic = call(tic_fn, (burnin_delta, jnp.sum(burnin_delta)), burnin_delta, ordered=False)
    # callback(lambda z, *_:  print(time.time() - NOW), jnp.sum(burnin_delta), ordered=True)

    _, stats, _, out_delta, _, pct_accepted = loop(sample_key, burnin_delta, burnin_state, kernel_fn, None,
                                                   args.n_samples, verbose)

    pct_accepted, toc = call(tic_fn, (pct_accepted, jnp.sum(pct_accepted)), pct_accepted , ordered=False)

    return toc - tic, stats, true_xs, xs_init, ys, pct_accepted, burnin_delta


def full_experiment():
    keys = jax.random.split(jax.random.key(args.seed), args.n_experiments)

    ejsd_per_key = np.ones((args.n_experiments, args.T, args.D)) * np.nan
    acceptance_rate_per_key = np.ones((args.n_experiments, args.T)) * np.nan
    delta_per_key = np.ones((args.n_experiments, args.T)) * np.nan
    time_per_key = np.ones((args.n_experiments,)) * np.nan
    for i in tqdm.trange(args.n_experiments,
                         desc=f"Style: {args.style}, T: {args.T}, N: {args.N}, D: {args.D}, gpu: {args.gpu}"):
        try:
            time_taken, (esjd, traj, squared_exp), true_xs, xs_init, _, pct_accepted, burnin_delta = one_experiment(
                keys[i])

            ejsd_per_key[i, :] = esjd.block_until_ready()
            acceptance_rate_per_key[i, :] = pct_accepted
            delta_per_key[i, :] = burnin_delta
            time_per_key[i] = time_taken
        except Exception as e:  # noqa
            print(f"Iteration {i} failed with error: ", e)
            continue
        # try:
        #     start = time.time()
        #     (esjd, *_), _, _, _, pct_accepted, burnin_delta = one_experiment(keys[i])
        #
        #     ejsd_per_key[i, :] = esjd.block_until_ready()
        #     acceptance_rate_per_key[i, :] = pct_accepted
        #     delta_per_key[i, :] = burnin_delta
        #     time_per_key[i] = time.time() - start
        # except:  # noqa
        #     continue
        # from matplotlib import pyplot as plt
        # fig, ax = plt.subplots(figsize=(15, 5))
        #
        # ax.plot(np.arange(args.T), traj[..., -1], color="tab:orange")
        # std = np.sqrt(squared_exp[..., -1] - traj[..., -1] ** 2)
        # ax.fill_between(np.arange(args.T), traj[..., -1] + 2 * std, traj[..., -1] - 2 * std,
        #                 color="tab:orange", alpha=0.2)
        # ax.plot(np.arange(args.T), true_xs[:, -1], color="tab:blue")
        # ax.plot(np.arange(args.T), xs_init[:, -1], color="k", linestyle="--")
        # plt.show()
        #
        # fig, ax = plt.subplots(figsize=(15, 5))
        # ax.plot(np.arange(args.T), esjd, color="tab:orange")
        # plt.show()

    return ejsd_per_key, acceptance_rate_per_key, delta_per_key, time_per_key


def main():
    import os
    with jax.disable_jit(args.debug):
        with jax.debug_nans(args.debug_nans):
            ejsd_per_key, acceptance_rate_per_key, delta_per_key, time_per_key = full_experiment()
    file_name = f"results/{args.style}-{args.D}-{args.T}-{args.N}-{args.parallel}-{args.gradient}.npz"
    if not os.path.isdir("results"):
        os.mkdir("results")
    print(time_per_key)
    np.savez(file_name,
             ejsd_per_key=ejsd_per_key,
             acceptance_rate_per_key=acceptance_rate_per_key,
             delta_per_key=delta_per_key,
             time_per_key=time_per_key)


if __name__ == "__main__":
    main()
