import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tqdm.auto as tqdm

from aux_samplers._primitives.kalman import LGSSM, filtering, sampling
from aux_samplers.common import delta_adaptation
from auxiliary_csmc import get_kernel as get_csmc_kernel
from auxiliary_guided_csmc import get_kernel as get_guided_csmc_kernel
from auxiliary_kalman import get_kernel as get_kalman_kernel

# ARGS PARSING

parser = argparse.ArgumentParser("Run a simple rare final event experiment.")
# General arguments
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--no-parallel', dest='parallel', action='store_false')
parser.set_defaults(parallel=False)
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
parser.add_argument("--precision", dest="precision", type=str, default="double")

# Experiment arguments
parser.add_argument("--n-experiments", dest="n_experiments", type=int, default=21)
parser.add_argument("--T", dest="T", type=int, default=2)
parser.add_argument("--rho", dest="rho", type=int, default=0.75)
parser.add_argument("--y", dest="y", type=float, default=5)
parser.add_argument("--r2", dest="r2", type=float, default=0.1)

parser.add_argument("--n-samples", dest="n_samples", type=int, default=50_000)
parser.add_argument("--burnin", dest="burnin", type=int, default=5_000)
parser.add_argument("--target-alpha", dest="target_alpha", type=float,
                    default=0.5)
parser.add_argument("--lr", dest="lr", type=float, default=0.5)
parser.add_argument("--beta", dest="beta", type=float, default=0.01)
parser.add_argument("--delta-init", dest="delta_init", type=float, default=1e-5)
parser.add_argument("--seed", dest="seed", type=int, default=42)
parser.add_argument("--style", dest="style", type=str, default="csmc-guided")
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
jax.config.update("jax_enable_x64", args.precision == "double")
if not args.gpu:
    jax.config.update("jax_platform_name", "cpu")
else:
    jax.config.update("jax_platform_name", "gpu")


def init_x_fn(key, n_samples=1):
    m0 = jnp.zeros((1,))
    P0 = jnp.eye(1)
    Fs = args.rho * jnp.ones((args.T - 1, 1, 1))
    Qs = (1 - args.rho ** 2) * jnp.ones((args.T - 1, 1, 1))
    bs = jnp.zeros((args.T - 1, 1))

    Hs = jnp.zeros((args.T, 1, 1))
    Rs = args.r2 * jnp.ones((args.T, 1, 1))
    cs = jnp.zeros((args.T, 1))
    Hs = Hs.at[-1].set(jnp.ones((1, 1)))

    ys = args.y * jnp.ones((args.T, 1))

    lgssm = LGSSM(m0, P0, Fs, Qs, bs, Hs, Rs, cs)
    fms, fPs, _ = filtering(ys, lgssm, args.parallel)
    if n_samples == 1:
        x_traj = sampling(key, fms, fPs, lgssm, args.parallel)
    else:
        keys = jax.random.split(key, n_samples)
        x_traj = jax.vmap(lambda k: sampling(k, fms, fPs, lgssm, args.parallel))(keys)
    return x_traj


# KERNEL
def loop(key, init_delta, init_state, kernel_fn, delta_fn, n_iter, verbose=False):
    from datetime import datetime
    keys = jax.random.split(key, n_iter)

    print_func = lambda z, *_: print(f"\riteration: {z[0]}, time: {datetime.now().strftime('%H:%M:%S')}, "
                                     f"min_δ: {z[1]:.2e}, max_δ: {z[2]:.2e}, "
                                     f"min_w_accept: {z[3]:.1%},  max_w_accept: {z[4]:.1%}, "
                                     f"min_accept: {z[5]:.1%}, max_accept: {z[6]:.1%}, ", end="")

    def body_fn(carry, key_inp):
        i, state, delta, window_avg_acceptance, avg_acceptance = carry
        if verbose:
            jax.debug.callback(print_func, (i,
                                            jnp.min(delta), jnp.max(delta),
                                            jnp.min(window_avg_acceptance), jnp.max(window_avg_acceptance),
                                            jnp.min(avg_acceptance), jnp.max(avg_acceptance),
                                            ), ordered=True)

        next_state = kernel_fn(key_inp, state, delta)
        # next_stats = stats_fn(state.x, next_state.x)

        # moving average.
        avg_acceptance = (i * avg_acceptance + 1. * next_state.updated) / (i + 1)
        window_avg_acceptance = args.beta * next_state.updated + (1 - args.beta) * window_avg_acceptance

        if delta_fn is not None:
            lr = (n_iter - i) * args.lr / n_iter
            delta = delta_fn(delta, args.target_alpha, window_avg_acceptance, lr)

        carry = i + 1, next_state, delta, window_avg_acceptance, avg_acceptance
        return carry, next_state.x[0, 0]

    init = 0, init_state, init_delta, 1. * init_state.updated, 1. * init_state.updated

    out, xs_0 = jax.lax.scan(body_fn, init, keys)
    return out, xs_0


@partial(jax.jit, static_argnums=(3,))
def _one_experiment(init_key, burnin_key, sample_key, verbose=args.verbose):
    # KERNEL
    if args.style in {"kalman"}:
        init_fn, kernel_fn = get_kalman_kernel(args.y, args.rho, args.r2, args.T, args.parallel)
        delta_init = args.delta_init
    # elif args.style == "kalman-2":
    #     init_fn, kernel_fn = get_kalman_kernel(ys, m0, P0, F, Q, b, args.parallel, (args.NU, PREC), 2)
    #     delta_init = args.delta_init
    elif args.style == "csmc":
        init_fn, kernel_fn = get_csmc_kernel(args.y, args.rho, args.r2, args.T, args.N, args.backward, args.parallel,
                                             args.gradient)
        delta_init = args.delta_init * jnp.ones((args.T,))
    elif args.style == "csmc-guided":
        init_fn, kernel_fn = get_guided_csmc_kernel(args.y, args.rho, args.r2, args.T, args.N, args.backward,
                                                    args.gradient)
        delta_init = args.delta_init * jnp.ones((args.T,))
    else:
        raise NotImplementedError

    init_xs = init_x_fn(init_key)

    init_state = init_fn(init_xs)

    # BURNIN
    (*_, burnin_state, burnin_delta, burnin_avg_acceptance, _), _ = loop(burnin_key,
                                                                         delta_init,
                                                                         init_state,
                                                                         kernel_fn,
                                                                         delta_adaptation,
                                                                         args.burnin,
                                                                         verbose)

    from jax.experimental import io_callback as call

    # The return is needed by the host callback to ensure that the two tics are taken at the right moment.
    def tic_fn(inp_):
        time_elapsed = time.time() - NOW
        return inp_, np.array(time_elapsed, dtype=inp_.dtype)

    burnin_delta, tic = call(tic_fn, (burnin_delta, burnin_delta.sum()), burnin_delta, ordered=True)

    (_, _, out_delta, _, pct_accepted), xs_0 = loop(sample_key, burnin_delta, burnin_state, kernel_fn, None,
                                                    args.n_samples, verbose)

    pct_accepted, toc = call(tic_fn, (pct_accepted, pct_accepted.sum()), pct_accepted, ordered=True)

    return toc - tic, init_xs, pct_accepted, burnin_delta, xs_0


def one_experiment(exp_key, verbose=args.verbose):
    # DATA
    # Fix "epoch" to now to avoid floating point issues.

    data_key, init_key, burnin_key, sample_key = jax.random.split(exp_key, 4)
    sampling_time, init_xs, pct_accepted, burnin_delta, xs_0 = _one_experiment(init_key, burnin_key,
                                                                               sample_key, verbose)
    return sampling_time, init_xs, pct_accepted, burnin_delta, xs_0


def full_experiment():
    np_random_state = np.random.RandomState(args.seed)
    keys = jax.random.split(jax.random.key(args.seed), args.n_experiments)

    acceptance_rate_per_key = np.ones((args.n_experiments, args.T)) * np.nan
    delta_per_key = np.ones((args.n_experiments, args.T)) * np.nan
    time_per_key = np.ones((args.n_experiments,)) * np.nan
    for i in tqdm.trange(args.n_experiments,
                         desc=f"Style: {args.style}, T: {args.T}, N: {args.N}, gpu: {args.gpu}, grad: {args.gradient}, y: {args.y:.1f}, rho: {args.rho:.2f}"):
        try:

            sampling_time, true_init, pct_accepted, burnin_delta, xs_0 = one_experiment(
                keys[i])
            xs_0.block_until_ready()

            bunch_of_trajs = init_x_fn(keys[i], 10000)

            acceptance_rate_per_key[i, :] = pct_accepted
            delta_per_key[i, :] = burnin_delta
            time_per_key[i] = sampling_time

            plt.plot(xs_0)
            plt.show()

            print("mean")
            print(bunch_of_trajs[:, 0].mean())
            print(xs_0.mean())

            print("std")
            print(bunch_of_trajs[:, 0].std())
            print(xs_0.std())

            plt.hist(bunch_of_trajs[:, 0, 0], bins=100, density=True, color="tab:blue", alpha=0.5)
            plt.hist(xs_0, bins=100, density=True, color="tab:orange", alpha=0.5)
            plt.show()


        except Exception as e:  # noqa
            raise e
            print(f"Experiment {i} failed for reason {e}")
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
        # fig, ax = plt.subplots(figsize=(25, 10))
        #
        # ax.plot(np.arange(args.T), traj[..., -1], color="tab:orange")
        # std = np.sqrt(squared_exp[..., -1] - traj[..., -1] ** 2)
        # ax.fill_between(np.arange(args.T), traj[..., -1] + 2 * std, traj[..., -1] - 2 * std,
        #                 color="tab:orange", alpha=0.2)
        # ax.plot(np.arange(args.T), true_xs[:, -1], color="tab:blue")
        # ax.plot(np.arange(args.T), true_init[:, -1], color="gray", alpha=0.5, linestyle="--")
        # lims = ax.get_ylim()
        # ax.scatter(np.arange(args.T), true_ys[:, -1], alpha=0.5)
        # ax.set_ylim(*lims)
        # plt.show()
        #
        # fig, ax = plt.subplots(figsize=(10, 5))
        # fig.suptitle(f"Style: {args.style}, grad: {args.gradient}, parallel: {args.parallel}")
        # ax.semilogy(np.arange(args.T), esjd, color="tab:blue", label="EJSD")
        # twinx = ax.twinx()
        # if "kalman" in args.style:
        #     twinx.plot(np.arange(args.T), pct_accepted * np.ones((args.T,)), color="tab:orange",
        #                label="acceptance rate")
        # else:
        #     twinx.plot(np.arange(args.T), pct_accepted, color="tab:orange", label="acceptance rate")
        # twinx.set_ylim(0, 1)
        # ax.legend()
        # plt.show()
    return acceptance_rate_per_key, delta_per_key, time_per_key


def main():
    import os
    with jax.disable_jit(args.debug):
        with jax.debug_nans(args.debug_nans):
            acceptance_rate_per_key, delta_per_key, time_per_key = full_experiment()
    file_name = f"results/{args.style}-{args.T}-{args.N}-{args.parallel}-{args.gradient}.npz"
    if not os.path.isdir("results"):
        os.mkdir("results")
    np.savez(file_name,
             acceptance_rate_per_key=acceptance_rate_per_key,
             delta_per_key=delta_per_key,
             time_per_key=time_per_key)


if __name__ == "__main__":
    main()
