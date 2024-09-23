import argparse
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tqdm.auto as tqdm

from aux_samplers._primitives.kalman import LGSSM, filtering, sampling
from aux_samplers.common import delta_adaptation
from auxiliary_csmc import get_kernel as get_csmc_kernel
from auxiliary_guided_csmc import get_kernel as get_guided_csmc_kernel
from auxiliary_kalman import get_kernel as get_kalman_kernel
from ess import effective_sample_size as ess

# ARGS PARSING

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=20"

parser = argparse.ArgumentParser("Run a simple rare final event experiment.")
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
parser.add_argument("--precision", dest="precision", type=str, default="double")

# Experiment arguments
parser.add_argument("--n-experiments", dest="n_experiments", type=int, default=10)
parser.add_argument("--grid-size", dest="grid_size", type=int, default=10)
parser.add_argument("--batch-size", dest="batch_size", type=int, default=8)  # number of parallel chains to compute ESS
parser.add_argument("--T", dest="T", type=int, default=2)
parser.add_argument("--y", dest="y", type=float, default=5)

parser.add_argument("--n-samples", dest="n_samples", type=int, default=20_000)
parser.add_argument("--burnin", dest="burnin", type=int, default=5_000)
parser.add_argument("--target-alpha", dest="target_alpha", type=float,
                    default=0.5)
parser.add_argument("--lr", dest="lr", type=float, default=0.25)
parser.add_argument("--beta", dest="beta", type=float, default=0.01)
parser.add_argument("--delta-init", dest="delta_init", type=float, default=1e-5)
parser.add_argument("--seed", dest="seed", type=int, default=42)
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
jax.config.update("jax_enable_x64", args.precision == "double")
if not args.gpu:
    jax.config.update("jax_platform_name", "cpu")
else:
    jax.config.update("jax_platform_name", "gpu")

RHOS = np.linspace(0., 0.999, args.grid_size)
R2S = np.logspace(-3, 0, args.grid_size)


@partial(jax.jit, static_argnums=(3, 4))
def init_x_fn(key, rho, r2, n_samples=1, prior=False):
    m0 = jnp.zeros((1,))
    P0 = jnp.eye(1)
    Fs = rho * jnp.ones((args.T - 1, 1, 1))
    Qs = (1 - rho ** 2) * jnp.ones((args.T - 1, 1, 1))
    bs = jnp.zeros((args.T - 1, 1))

    Hs = jnp.zeros((args.T, 1, 1))
    Rs = r2 * jnp.ones((args.T, 1, 1))
    cs = jnp.zeros((args.T, 1))
    if not prior:
        Hs = Hs.at[-1].set(jnp.ones((1, 1)))

    ys = args.y * jnp.ones((args.T, 1))

    lgssm = LGSSM(m0, P0, Fs, Qs, bs, Hs, Rs, cs)
    fms, fPs, _ = filtering(ys, lgssm, args.parallel)
    if n_samples == 1:
        x_traj = sampling(key, fms, fPs, lgssm, args.parallel)
    else:
        keys = jax.random.split(key, n_samples)
        if args.gpu:
            x_traj = jax.vmap(lambda k: sampling(k, fms, fPs, lgssm, args.parallel))(keys)
        else:
            x_traj = jax.vmap(lambda k: sampling(k, fms, fPs, lgssm, args.parallel))(keys)
    return x_traj


# KERNEL
@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def loop(key, init_delta, init_state, kernel_fn, delta_fn, n_iter, verbose=False):
    from datetime import datetime
    keys = jax.random.split(key, n_iter)

    print_func = lambda z, *_: print(f"\riteration: {z[0]}, time: {datetime.now().strftime('%H:%M:%S')}, "
                                     f"δ_0: {z[1]:.2e}, δ_1: {z[2]:.2e}, "
                                     f"w_a_0: {z[3]:.1%},  w_a_1: {z[4]:.1%}, "
                                     f"a_0: {z[5]:.1%}, a_1: {z[6]:.1%}, ", end="")

    def body_fn(carry, key_inp):
        i, state, delta, window_avg_acceptance, avg_acceptance = carry
        if verbose:
            jax.debug.callback(print_func, (i,
                                            delta[0], delta[-1],
                                            window_avg_acceptance[0], window_avg_acceptance[-1],
                                            avg_acceptance[0], avg_acceptance[-1],
                                            ), ordered=False)

        next_state = kernel_fn(key_inp, state, delta)
        # next_stats = stats_fn(state.x, next_state.x)

        # moving average.
        avg_acceptance = (i * avg_acceptance + 1. * next_state.updated) / (i + 1)
        window_avg_acceptance = args.beta * next_state.updated + (1 - args.beta) * window_avg_acceptance

        if delta_fn is not None:
            lr = (n_iter - i) * args.lr / n_iter
            delta = delta_fn(delta, args.target_alpha, window_avg_acceptance, lr)

        carry = i + 1, next_state, delta, window_avg_acceptance, avg_acceptance
        return carry, (next_state.x[0, 0], next_state.x[-1, 0])

    init = 0, init_state, init_delta, 1. * init_state.updated, 1. * init_state.updated

    out, (xs_0, xs_T) = jax.lax.scan(body_fn, init, keys)
    return out, xs_0, xs_T


# @partial(jax.jit, static_argnums=(5,))
def _one_experiment(rho, r2, init_key, burnin_key, sample_key, verbose=args.verbose):
    # KERNEL
    if args.style in {"kalman"}:
        init_fn, kernel_fn = get_kalman_kernel(args.y, rho, r2, args.T, args.parallel, args.gradient)
        delta_init = args.delta_init
    elif args.style == "csmc":
        init_fn, kernel_fn = get_csmc_kernel(args.y, rho, r2, args.T, args.N, args.backward, args.parallel,
                                             args.gradient)
        delta_init = args.delta_init * jnp.ones((args.T,))
    elif args.style == "csmc-guided":
        init_fn, kernel_fn = get_guided_csmc_kernel(args.y, rho, r2, args.T, args.N, args.backward,
                                                    args.gradient)
        delta_init = args.delta_init * jnp.ones((args.T,))
    else:
        raise NotImplementedError

    init_xs = init_x_fn(init_key, rho, r2)

    init_state = init_fn(init_xs)

    # BURNIN
    (*_, burnin_state, burnin_delta, burnin_avg_acceptance, _), _, _ = loop(burnin_key,
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

    burnin_delta, tic = call(tic_fn, (burnin_delta, burnin_delta.sum()), burnin_delta, ordered=False)

    @jax.jit
    def batched_loop(k):

        (_, _, out_delta, _, pct_accepted), xs_0, xs_T = loop(k, burnin_delta, burnin_state, kernel_fn, None,
                                                              args.n_samples, verbose)
        return out_delta, pct_accepted, xs_0, xs_T

    sample_keys = jax.random.split(sample_key, args.batch_size)
    out_delta, pct_accepted, xs_0, xs_T = jax.vmap(batched_loop)(sample_keys)

    pct_accepted, toc = call(tic_fn, (pct_accepted, pct_accepted.sum()), pct_accepted, ordered=False)

    return toc - tic, init_xs, pct_accepted, burnin_delta, xs_0, xs_T


def one_experiment(rho, r2, exp_key, verbose=args.verbose):
    # DATA
    # Fix "epoch" to now to avoid floating point issues.

    data_key, init_key, burnin_key, sample_key = jax.random.split(exp_key, 4)
    sampling_time, init_xs, pct_accepted, burnin_delta, xs_0, xs_T = _one_experiment(rho, r2, init_key, burnin_key,
                                                                                     sample_key, verbose)
    return sampling_time, init_xs, pct_accepted, burnin_delta, xs_0, xs_T


def full_experiment(rho, r2):
    keys = jax.random.split(jax.random.key(args.seed), args.n_experiments)

    mean_0_per_key = np.ones((args.n_experiments,)) * np.nan
    std_0_per_key = np.ones((args.n_experiments,)) * np.nan
    ess_0_per_key = np.ones((args.n_experiments,)) * np.nan
    delta_0_per_key = np.ones((args.n_experiments,)) * np.nan
    acc_0_per_key = np.ones((args.n_experiments,)) * np.nan

    mean_T_per_key = np.ones((args.n_experiments,)) * np.nan
    std_T_per_key = np.ones((args.n_experiments,)) * np.nan
    ess_T_per_key = np.ones((args.n_experiments,)) * np.nan
    delta_T_per_key = np.ones((args.n_experiments,)) * np.nan
    acc_T_per_key = np.ones((args.n_experiments,)) * np.nan

    rho_0_T = rho ** (args.T - 1)
    conditional_mean_T = 1. / (1. + r2) * args.y
    conditional_cov_T = r2 / (1. + r2)
    conditional_mean_0 = rho_0_T * conditional_mean_T
    conditional_cov_0 = rho_0_T ** 2 * conditional_cov_T + 1 - rho_0_T ** 2
    conditional_std_0 = conditional_cov_0 ** 0.5

    for i in tqdm.trange(args.n_experiments,
                         desc=f"rho: {rho:.2f}, r2: {r2:.2f}", leave=False):
        try:
            sampling_time, true_init, pct_accepted, burnin_delta, xs_0, xs_T = one_experiment(
                rho, r2,
                keys[i])
            xs_0.block_until_ready()

            # ess_0_per_key[i] = tfp.mcmc.effective_sample_size(xs_0, cross_chain_dims=0)
            ess_0_per_key[i] = ess(xs_0, None, 0, 1)
            mean_0_per_key[i] = jnp.mean(xs_0)
            std_0_per_key[i] = jnp.std(xs_0)

            ess_T_per_key[i] = ess(xs_T, None, 0, 1)
            # ess_T_per_key[i] = tfp.mcmc.effective_sample_size(xs_T, cross_chain_dims=0)
            mean_T_per_key[i] = jnp.mean(xs_T)
            std_T_per_key[i] = jnp.std(xs_T)
            pct_accepted = jnp.mean(pct_accepted, axis=0)

            if jnp.ndim(burnin_delta) == 0:
                delta_0_per_key[i] = burnin_delta
                delta_T_per_key[i] = burnin_delta
                acc_0_per_key[i] = pct_accepted
                acc_T_per_key[i] = pct_accepted
            else:
                delta_0_per_key[i] = burnin_delta[0]
                delta_T_per_key[i] = burnin_delta[-1]
                acc_0_per_key[i] = pct_accepted[0]
                acc_T_per_key[i] = pct_accepted[-1]

            # plt.hist(bunch_of_trajs[:, 0, 0], bins=100, density=True, color="tab:blue", alpha=0.5)
            # plt.hist(xs_0, bins=100, density=True, color="tab:orange", alpha=0.5)
            # arange = jnp.linspace(conditional_mean_0 - 4 * conditional_std_0, conditional_mean_0 + 4 * conditional_std_0, 10000)
            # plt.plot(arange, norm.pdf(arange, conditional_mean_0, conditional_std_0), color="tab:blue")
            # plt.show()


        except Exception as e:  # noqa
            raise e
            print(f"Experiment {i} failed for reason {e}")
            continue

    mean_0_series = pd.Series(mean_0_per_key, name="mean")
    std_0_series = pd.Series(std_0_per_key, name="std")
    ess_0_series = pd.Series(ess_0_per_key, name="ess")
    delta_0_series = pd.Series(delta_0_per_key, name="delta")
    acc_0_series = pd.Series(acc_0_per_key, name="acc")

    mean_T_series = pd.Series(mean_T_per_key, name="mean")
    std_T_series = pd.Series(std_T_per_key, name="std")
    ess_T_series = pd.Series(ess_T_per_key, name="ess")
    delta_T_series = pd.Series(delta_T_per_key, name="delta")
    acc_T_series = pd.Series(acc_T_per_key, name="acc")

    df_0 = pd.concat([mean_0_series, std_0_series, ess_0_series, delta_0_series, acc_0_series], axis=1)
    df_T = pd.concat([mean_T_series, std_T_series, ess_T_series, delta_T_series, acc_T_series], axis=1)

    out = pd.concat([df_0, df_T], axis=1, keys=["0", "T"])

    out_true = pd.DataFrame({"mean": [conditional_mean_0, conditional_mean_T],
                             "std": [conditional_std_0, conditional_cov_T ** 0.5],
                             "ess": [np.nan, np.nan],
                             "delta": [np.nan, np.nan],
                             "acc": [np.nan, np.nan]}, index=["0", "T"])

    return out, out_true.T


def main():
    import os
    results = {}
    true_vals = {}
    with jax.disable_jit(args.debug):
        with jax.debug_nans(args.debug_nans):
            for rho in tqdm.tqdm(RHOS, desc=f"style: {args.style}, grad: {args.gradient}"):
                for r2 in tqdm.tqdm(R2S, leave=False):
                    exp_res, true = full_experiment(rho, r2)
                    results[(rho, r2)] = exp_res
                    true_vals[(rho, r2)] = true

    results_df = pd.concat(results, axis=1).T
    results_df.index.names = ["rho", "r2", "timestep", "statistic"]
    results_df = results_df.T

    true_df = pd.concat(true_vals, axis=1).T
    true_df = true_df.stack()
    true_df.index.names = ["rho", "r2", "timestep", "statistic"]

    file_name = f"results/{args.style}-{args.T}-{args.N}-{args.parallel}-{args.gradient}.csv"
    file_name_true = f"results/{args.T}-true.csv"
    if not os.path.isdir("results"):
        os.mkdir("results")
    results_df.to_csv(file_name)
    true_df.to_csv(file_name_true)


if __name__ == "__main__":
    main()
