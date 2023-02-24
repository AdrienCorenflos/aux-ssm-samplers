import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import tqdm.auto as tqdm

from aux_samplers import delta_adaptation
from aux_samplers import rhee_glynn
from auxiliary_kalman import get_kernel as get_kalman_kernel
from model import get_data, get_dynamics, make_precision, init_x_fn

# ARGS PARSING

parser = argparse.ArgumentParser("Run a Spatio-temporal experiment")
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

# Experiment arguments
parser.add_argument("--n-experiments", dest="n_experiments", type=int, default=5)
parser.add_argument("--T", dest="T", type=int, default=2 ** 10)
parser.add_argument("--D", dest="D", type=int, default=8)
parser.add_argument("--NU", dest="NU", type=int, default=1)
parser.add_argument("--burnin", dest="burnin", type=int, default=5_000)
parser.add_argument("--decoupling", dest="decoupling", type=int, default=5)
parser.add_argument("--n-estimators", dest="n_estimators", type=int, default=100)
parser.add_argument("--K", dest="K", type=int, default=1)
parser.add_argument("--M", dest="M", type=int, default=1_000)
# We use the reflection as the covariances are the same for all the experiments.
parser.add_argument("--coupling", dest="coupling", type=str, default="reflection")
parser.add_argument("--target-alpha", dest="target_alpha", type=float,
                    default=0.5)
parser.add_argument("--lr", dest="lr", type=float, default=0.1)
parser.add_argument("--beta", dest="beta", type=float, default=0.01)
parser.add_argument("--delta-init", dest="delta_init", type=float, default=1e-5)
parser.add_argument("--seed", dest="seed", type=int, default=42)

args = parser.parse_args()

# BACKEND CONFIG
NOW = time.time()
jax.config.update("jax_enable_x64", False)
if not args.gpu:
    jax.config.update("jax_platform_name", "cpu")
else:
    jax.config.update("jax_platform_name", "gpu")

# PARAMETERS

SIGMA_X = 1.
RY = 1.
TAU = -0.25
m0, P0, F, Q, b = get_dynamics(SIGMA_X, args.D)
PREC = make_precision(TAU, RY, args.D)


# STATS FN
def stats_fn(x):
    return x[..., 0], x[..., 0] ** 2


# KERNEL
def adaptation_loop(key, init_delta, init_state, kernel_fn, delta_fn, n_iter, verbose=False):
    from datetime import datetime
    keys = jax.random.split(key, n_iter)

    print_func = lambda z, *_: print(
        f"\riteration: {z[0] + 1} / {n_iter}, time: {datetime.now().strftime('%H:%M:%S')}, "
        f"min_δ: {z[1]:.2e}, max_δ: {z[2]:.2e}, "
        f"min_w_accept: {z[3]:.1%},  max_w_accept: {z[4]:.1%}, "
        f"min_accept: {z[5]:.1%}, max_accept: {z[6]:.1%}", end="")

    def body_fn(carry, key_inp):
        from jax.experimental.host_callback import id_tap
        i, state, delta, window_avg_acceptance, avg_acceptance = carry
        if verbose:
            id_tap(print_func, (i,
                                jnp.min(delta), jnp.max(delta),
                                jnp.min(window_avg_acceptance), jnp.max(window_avg_acceptance),
                                jnp.min(avg_acceptance), jnp.max(avg_acceptance),
                                ), result=None)

        next_state = kernel_fn(key_inp, state, delta)
        # moving average.
        avg_acceptance = (i * avg_acceptance + 1. * next_state.updated) / (i + 1)
        window_avg_acceptance = args.beta * next_state.updated + (1 - args.beta) * window_avg_acceptance

        lr = (n_iter - i) * args.lr / n_iter
        delta = delta_fn(delta, args.target_alpha, window_avg_acceptance, lr)

        carry = i + 1, next_state, delta, window_avg_acceptance, avg_acceptance
        return carry, None

    init = 0, init_state, init_delta, 1. * init_state.updated, 1. * init_state.updated

    out, _ = jax.lax.scan(body_fn, init, keys)
    return out


def decoupling_loop(key, delta, state, kernel_fn, n_iter, verbose=False):
    from datetime import datetime
    key_1, key_2 = jax.random.split(key, 2)
    keys_1 = jax.random.split(key_1, n_iter)
    keys_2 = jax.random.split(key_2, n_iter)

    print_func = lambda z, *_: print(f"\riteration: {z[0] + 1} / {n_iter}, time: {datetime.now().strftime('%H:%M:%S')}",
                                     end="")

    def body_fn(carry, inp):
        key_inp_1, key_inp_2 = inp
        from jax.experimental.host_callback import id_tap
        i, state_1, state_2 = carry
        if verbose:
            id_tap(print_func, (i,), result=None)

        next_state_1 = kernel_fn(key_inp_1, state_1, delta)
        next_state_2 = kernel_fn(key_inp_2, state_2, delta)
        # moving average.

        carry = i + 1, next_state_1, next_state_2
        return carry, None

    init = 0, state, state

    out, _ = jax.lax.scan(body_fn, init, (keys_1, keys_2))
    return out


@partial(jax.jit, static_argnums=(2,))
def get_burnin_delta_and_state(ys, key, verbose=args.verbose):
    # KERNEL
    init_key, burnin_key = jax.random.split(key, 2)
    init_fn, kernel_fn = get_kalman_kernel(ys, m0, P0, F, Q, b, args.parallel, (args.NU, PREC))

    delta_init = args.delta_init

    init_xs = init_x_fn(init_key, ys, SIGMA_X, args.NU, PREC, 1_000)

    init_state = init_fn(init_xs)

    # BURNIN
    *_, burnin_state, burnin_delta, burnin_avg_acceptance, _ = adaptation_loop(burnin_key,
                                                                               delta_init,
                                                                               init_state,
                                                                               kernel_fn,
                                                                               delta_adaptation,
                                                                               args.burnin,
                                                                               verbose)
    return burnin_delta, burnin_state


@partial(jax.jit, static_argnums=(4,))
def get_one_adapted_estimate(key, ys, delta, state, verbose=args.verbose):
    from jax.experimental.host_callback import call

    def tic_fn(arr):
        time_elapsed = time.time() - NOW
        return np.array(time_elapsed, dtype=delta.dtype), arr

    output_shape = (jax.ShapeDtypeStruct((), delta.dtype),
                    state)

    tic, state = call(tic_fn, state,
                      result_shape=output_shape)

    decoupling_key, delay_key, sampling_key = jax.random.split(key, 3)
    _, kernel_fn = get_kalman_kernel(ys, m0, P0, F, Q, b, args.parallel, (args.NU, PREC))
    coupled_init_fn, coupled_kernel_fn = get_kalman_kernel(ys, m0, P0, F, Q, b, args.parallel, (args.NU, PREC),
                                                           coupled=True, method=args.coupling)
    # DECOUPLING
    _, state_1, state_2 = decoupling_loop(decoupling_key, delta, state, kernel_fn, args.decoupling, verbose)
    # DESYNCHRONIZE
    state_1 = kernel_fn(delay_key, state_1, delta)
    # Rhee & Glynn estimator
    coupled_kernel_fn_spec = lambda k_, state_: coupled_kernel_fn(k_, state_, delta)
    coupled_state = coupled_init_fn(state_1.x, state_2.x)

    estimate, _, coupling_time = rhee_glynn(sampling_key, coupled_kernel_fn_spec, coupled_state, args.K, args.M,
                                            stats_fn)

    output_shape = (jax.ShapeDtypeStruct((), delta.dtype),
                    coupling_time)

    toc, coupling_time = call(tic_fn, coupling_time,
                              result_shape=output_shape)

    return toc - tic, *estimate, coupling_time


def one_experiment(random_state, exp_key, verbose=args.verbose):
    # DATA

    true_xs, ys = get_data(random_state, SIGMA_X, RY, TAU, args.NU, args.D, args.T)
    adaptation_key, estimator_key = jax.random.split(exp_key, 2)

    delta, state = get_burnin_delta_and_state(ys, adaptation_key, verbose)
    estimator_keys = jax.random.split(estimator_key, args.n_experiments)

    experiment_times = np.ones((args.n_estimators,)) * np.nan
    means = np.ones((args.n_estimators, args.T, args.D ** 2)) * np.nan
    squares = np.ones((args.n_estimators, args.T, args.D ** 2)) * np.nan
    coupling_times = np.ones((args.n_estimators,)) * np.nan

    for i in tqdm.trange(args.n_estimators, desc="Inner estimators:", leave=False):
        clock_time, mean, square, coupling_time = get_one_adapted_estimate(estimator_keys[i], ys, delta, state, verbose)
        experiment_times[i] = clock_time
        means[i] = mean
        squares[i] = square
        coupling_times[i] = coupling_time.block_until_ready()

    return true_xs, experiment_times, means, squares, coupling_times


def full_experiment():
    np_random_state = np.random.RandomState(args.seed)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_experiments)

    time_per_key = np.ones((args.n_experiments, args.n_estimators)) * np.nan
    true_xs_per_key = np.ones((args.n_experiments, args.T, args.D ** 2)) * np.nan
    mean_traj_per_key = np.ones((args.n_experiments, args.n_estimators, args.T, args.D ** 2)) * np.nan
    square_traj_per_key = np.ones((args.n_experiments, args.n_estimators, args.T, args.D ** 2)) * np.nan
    coupling_time_per_key = np.ones((args.n_experiments, args.n_estimators)) * np.nan
    for i in tqdm.trange(args.n_experiments,
                         desc=f"Style: T: {args.T}, D: {args.D}, gpu: {args.gpu}"):
        true_xs, experiment_times, means, squares, coupling_times = one_experiment(np_random_state, keys[i])

        time_per_key[i] = experiment_times
        true_xs_per_key[i] = true_xs
        mean_traj_per_key[i] = means
        square_traj_per_key[i] = squares
        coupling_time_per_key[i] = coupling_times

        # except Exception as e:  # noqa
        #     print(f"Experiment {i} failed for reason {e}")
        #     continue
    return time_per_key, true_xs_per_key, mean_traj_per_key, square_traj_per_key, coupling_time_per_key


def main():
    import os
    with jax.disable_jit(args.debug):
        with jax.debug_nans(args.debug_nans):
            time_per_key, true_xs_per_key, mean_traj_per_key, square_traj_per_key, coupling_time_per_key = full_experiment()
    file_name = f"results/coupled-{args.D}-{args.T}-{args.parallel}.npz"
    if not os.path.isdir("results"):
        os.mkdir("results")
    np.savez(file_name,
             time_per_key=time_per_key, true_xs_per_key=true_xs_per_key, mean_traj_per_key=mean_traj_per_key,
             square_traj_per_key=square_traj_per_key, coupling_time_per_key=coupling_time_per_key)


if __name__ == "__main__":
    main()
