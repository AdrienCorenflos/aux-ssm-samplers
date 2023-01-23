import argparse

import jax
import jax.numpy as jnp
import numpy as np
import tqdm.auto as tqdm
from jax.tree_util import tree_map

from aux_samplers.common import delta_adaptation
from auxiliary_csmc import get_kernel as get_csmc_kernel
from auxiliary_guided_csmc import get_kernel as get_guided_csmc_kernel
from auxiliary_kalman import get_kernel as get_kalman_kernel
from model import get_data, get_dynamics

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
parser.set_defaults(gpu=False)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--no-verbose', dest='verbose', action='store_false')
parser.set_defaults(verbose=True)

# Experiment arguments
parser.add_argument("--n-experiments", dest="n_experiments", type=int, default=10)
parser.add_argument("--T", dest="T", type=int, default=50)
parser.add_argument("--D", dest="D", type=int, default=5)
parser.add_argument("--n-samples", dest="n_samples", type=int, default=10_000)
parser.add_argument("--burnin", dest="burnin", type=int, default=2_500)
parser.add_argument("--target-alpha", dest="target_alpha", type=float, default=0.5)
parser.add_argument("--lr", dest="lr", type=float, default=1.)
parser.add_argument("--beta", dest="beta", type=int, default=0.1)
parser.add_argument("--delta-init", dest="delta_init", type=float, default=1e-20)
parser.add_argument("--seed", dest="seed", type=int, default=1234)
parser.add_argument("--style", dest="style", type=str, default="kalman-2")
parser.add_argument("--gradient", action='store_true')
parser.add_argument('--no-gradient', dest='gradient', action='store_false')
parser.set_defaults(gradient=False)
parser.add_argument("--backward", action='store_true')
parser.add_argument('--no-backward', dest='backward', action='store_false')
parser.set_defaults(backward=True)
parser.add_argument("--N", dest="N", type=int, default=25)

args = parser.parse_args()

# BACKEND CONFIG
jax.config.update("jax_enable_x64", True)
if not args.gpu:
    jax.config.update("jax_platform_name", "cpu")
else:
    jax.config.update("jax_platform_name", "gpu")

# PARAMETERS
# we use the same parameters as Finke and Thiery.
NU, PHI, TAU, RHO = 0., .9, 2., .25
m0, P0, F, Q, b = get_dynamics(NU, PHI, TAU, RHO, args.D)

if args.style != "kalman":
    args.target_alpha = 1 - (1 + args.N) ** (-1 / 3)
elif args.style == "csmc" and args.parallel:
    # More conservative target alpha for parallel cSMC. This is perhaps an artifact of the warmup, but it's not clear.
    args.target_alpha = 0.33


# STATS FN
def stats_fn(x_1, x_2):
    # squared jumping distance averaged across dimensions, and first and second moments
    return jnp.mean((x_2 - x_1) ** 2, -1), x_2, x_2 ** 2


# KERNEL
def loop(key, init_delta, init_state, kernel_fn, delta_fn, n_iter, verbose=False):
    from datetime import datetime
    keys = jax.random.split(key, n_iter)

    print_func = lambda z, *_: print(f"\riteration: {z[0]}, current time: {datetime.now().strftime('%H:%M:%S')}, "
                                     f"min_delta: {z[1]:.2e}, max_delta: {z[2]:.2e}, "
                                     f"min_window_accept: {z[3]:.1%},  max_window_accept: {z[4]:.1%}, "
                                     f"min_accept: {z[5]:.1%}, max_accept: {z[6]:.1%},", end="")

    def body_fn(carry, key_inp):
        from jax.experimental.host_callback import id_tap
        i, stats, state, delta, window_avg_acceptance, avg_acceptance = carry
        if verbose:
            id_tap(print_func, (i,
                                jnp.min(delta), jnp.max(delta),
                                jnp.min(window_avg_acceptance), jnp.max(window_avg_acceptance),
                                jnp.min(avg_acceptance), jnp.max(avg_acceptance),
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
    true_xs, ys = get_data(data_key, NU, PHI, TAU, RHO, args.D, args.T)

    # INIT

    xs_init, _ = get_data(init_key, NU, PHI, TAU, RHO, args.D, args.T)

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

    _, stats, _, _, _, pct_accepted = loop(sample_key, burnin_delta, burnin_state, kernel_fn, None, args.n_samples,
                                           verbose)
    return stats, true_xs, ys, xs_init, pct_accepted, burnin_delta


def full_experiment():
    import time
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_experiments)

    ejsd_per_key = np.ones((args.n_experiments, args.T)) * np.nan
    acceptance_rate_per_key = np.ones((args.n_experiments, args.T)) * np.nan
    delta_per_key = np.ones((args.n_experiments, args.T)) * np.nan
    time_per_key = np.ones((args.n_experiments,)) * np.nan
    for i in tqdm.trange(args.n_experiments,
                         desc=f"Style: {args.style}, T: {args.T}, N: {args.N}, D: {args.D}, gpu: {args.gpu}"):
        start = time.time()
        (esjd, traj, squared_exp), true_xs, _, xs_init, pct_accepted, burnin_delta = one_experiment(keys[i])

        ejsd_per_key[i, :] = esjd.block_until_ready()
        acceptance_rate_per_key[i, :] = pct_accepted
        delta_per_key[i, :] = burnin_delta
        time_per_key[i] = time.time() - start
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
        # plt.plot(np.arange(args.T), traj[..., -1], color="tab:orange")
        # std = np.sqrt(squared_exp[..., -1] - traj[..., -1] ** 2)
        # plt.fill_between(np.arange(args.T), traj[..., -1] + 2 * std, traj[..., -1] - 2 * std,
        #                  color="tab:orange", alpha=0.2)
        # plt.plot(np.arange(args.T), true_xs[:, -1], color="tab:blue")
        # plt.plot(np.arange(args.T), xs_init[:, -1], color="k", linestyle="--")
        # plt.show()
        #
        # fig, ax = plt.subplots(figsize=(10, 5))
        # ax.plot(np.arange(args.T), burnin_delta, color="tab:blue")
        # ax.twinx().plot(np.arange(args.T), pct_accepted, color="tab:orange")
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
    np.savez(file_name,
             ejsd_per_key=ejsd_per_key,
             acceptance_rate_per_key=acceptance_rate_per_key,
             delta_per_key=delta_per_key,
             time_per_key=time_per_key)


if __name__ == "__main__":
    main()
