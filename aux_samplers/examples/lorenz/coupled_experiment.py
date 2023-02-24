import argparse
import time
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
import tqdm.auto as tqdm

from aux_samplers import mvn
from aux_samplers import rhee_glynn
from aux_samplers.common import delta_adaptation
from aux_samplers.examples.lorenz.experiment import GibbsState
from aux_samplers.examples.lorenz.model import observations_model, theta_posterior_mean_and_chol, sample_trajectory
from aux_samplers.kalman.generic import KalmanSampler, CoupledKalmanSampler
from auxiliary_kalman import get_coupled_kernel as get_coupled_kalman_kernel, get_kernel as get_kalman_kernel

# ARGS PARSING

parser = argparse.ArgumentParser("Run a coupled Lorenz experiment")
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
parser.set_defaults(verbose=False)
parser.add_argument("--precision", dest="precision", type=str, default="single")

# Experiment arguments
parser.add_argument("--burnin", dest="burnin", type=int, default=5_000)
parser.add_argument("--decoupling", dest="decoupling", type=int, default=10)
parser.add_argument("--n-estimators", dest="n_estimators", type=int, default=100)
parser.add_argument("--K", dest="K", type=int, default=2_000)
parser.add_argument("--M", dest="M", type=int, default=5_000)
# We use the reflection as the covariances are the same for all the experiments.
parser.add_argument("--coupling", dest="coupling", type=str, default="reflection")
parser.add_argument("--lr", dest="lr", type=float, default=1.)
parser.add_argument("--target-alpha", dest="target_alpha", type=float, default=0.234)
parser.add_argument("--delta-init", dest="delta_init", type=float, default=1e-5)
parser.add_argument("--beta", dest="beta", type=float, default=0.05)
parser.add_argument("--seed", dest="seed", type=int, default=42)
parser.add_argument("--freq", dest="freq", type=int, default=2)

args = parser.parse_args()

# BACKEND CONFIG
NOW = time.time()

if args.precision.lower() in {"single", "float32", "32"}:
    jax.config.update("jax_enable_x64", False)
elif args.precision.lower() in {"double", "float64", "64"}:
    jax.config.update("jax_enable_x64", True)
if not args.gpu:
    jax.config.update("jax_platform_name", "cpu")
else:
    jax.config.update("jax_platform_name", "gpu")

args = parser.parse_args()

# BACKEND CONFIG
NOW = time.time()

if args.precision.lower() in {"single", "float32", "32"}:
    jax.config.update("jax_enable_x64", False)
elif args.precision.lower() in {"double", "float64", "64"}:
    jax.config.update("jax_enable_x64", True)
if not args.gpu:
    jax.config.update("jax_platform_name", "cpu")
else:
    jax.config.update("jax_platform_name", "gpu")

# PARAMETERS
# we use the exact same parameters as Mider et al.
sigma_theta = 1e3 ** 0.5
true_theta = jnp.array([10., 28., 8. / 3.])
true_xs = np.loadtxt("true_xs.csv", delimiter=",", skiprows=1)[:, 1:]

M0 = jnp.array([1.5, -1.5, 25.])
P0 = jnp.diag(jnp.array([400., 20., 20.]))
SIGMA_X = 3.
SIGMA_Y = 5. ** 0.5
THETA_0 = jnp.array([5.0, 15.0, 6.0])
DATA = np.loadtxt("data.csv", delimiter=",", skiprows=1)
T = DATA[-1, 0]
OBS_FREQ = DATA[1, 0] - DATA[0, 0]
SMOOTH_FREQ = args.freq * 1e-4
N_STEPS = int(T / SMOOTH_FREQ + 1e-6) + 1
SAMPLE_EVERY = int(OBS_FREQ / SMOOTH_FREQ + 1e-6)

# Observation model can be computed offlien
YS, HS, RS, CS = observations_model(DATA, SIGMA_Y, N_STEPS, SAMPLE_EVERY)


# STATS FN
def stats_fn(state):
    x = state.kalman_state.x
    return x, state.theta, x ** 2


@chex.dataclass
class CoupledGibbsState:
    state_1: GibbsState
    state_2: GibbsState

    flags: chex.Array
    theta_coupled: bool

    @property
    def is_coupled(self):
        return jnp.all(self.flags) & self.theta_coupled


def gibbs_step(rng_key, state, delta):
    kalman_state, theta = state.kalman_state, state.theta
    key_kalman, key_theta = jax.random.split(rng_key, 2)

    _, kernel_fn = get_kalman_kernel(YS, HS, RS, CS, M0, P0, theta, SIGMA_X, SMOOTH_FREQ, args.parallel)
    next_kalman_state = kernel_fn(key_kalman, kalman_state, delta)
    theta_mean, theta_chol = theta_posterior_mean_and_chol(next_kalman_state.x, sigma_theta, SMOOTH_FREQ, SIGMA_X)
    next_theta = theta_mean + theta_chol * jax.random.normal(key_theta, (3,))
    return GibbsState(kalman_state=next_kalman_state, theta=next_theta)


def coupled_gibbs_step(rng_key, coupled_gibbs_state, delta):
    kalman_key, theta_key = jax.random.split(rng_key, 2)
    coupled_kalman_state = CoupledKalmanSampler(state_1=coupled_gibbs_state.state_1.kalman_state,
                                                state_2=coupled_gibbs_state.state_2.kalman_state,
                                                flags=coupled_gibbs_state.flags)
    theta_1, theta_2 = coupled_gibbs_state.state_1.theta, coupled_gibbs_state.state_2.theta
    _, coupled_kernel_fn = get_coupled_kalman_kernel(YS, HS, RS, CS, M0, P0, theta_1, theta_2, SIGMA_X, SMOOTH_FREQ,
                                                     args.parallel)
    coupled_kalman_state = coupled_kernel_fn(rng_key, coupled_kalman_state, delta)
    state_1, state_2, flags = coupled_kalman_state.state_1, coupled_kalman_state.state_2, coupled_kalman_state.flags
    theta_1_mean, theta_1_chol = theta_posterior_mean_and_chol(state_1.x, sigma_theta, SMOOTH_FREQ, SIGMA_X)
    theta_2_mean, theta_2_chol = theta_posterior_mean_and_chol(state_2.x, sigma_theta, SMOOTH_FREQ, SIGMA_X)

    theta_1, theta_2, theta_coupled = mvn.rejection(theta_key, theta_1_mean, theta_1_chol, theta_2_mean, theta_2_chol)
    state_1 = GibbsState(kalman_state=state_1, theta=theta_1)
    state_2 = GibbsState(kalman_state=state_2, theta=theta_2)
    return CoupledGibbsState(state_1=state_1, state_2=state_2, flags=flags, theta_coupled=theta_coupled)


# KERNEL
@partial(jax.jit, static_argnums=(3, 6))
def adaptation_loop(key, init_delta, init_state, n_iter, target_alpha=None, lr=None, verbose=False):
    from datetime import datetime
    keys = jax.random.split(key, n_iter)

    print_func = lambda z, *_: print(f"\riteration: {z[0]}, time: {datetime.now().strftime('%H:%M:%S')}, "
                                     f"min_δ: {z[1]:.2e}, max_δ: {z[2]:.2e}, "
                                     f"min_window_accept: {z[3]:.1%},  max_window_accept: {z[4]:.1%}, "
                                     f"theta_0: {z[9]:.2f}, theta_1: {z[10]:.2f}, theta_2: {z[11]:.2f}",
                                     end="")

    def body_fn(carry, key_inp):
        from jax.experimental.host_callback import id_tap
        i, state, delta, window_avg_acceptance = carry
        if verbose:
            id_tap(print_func, (i,
                                jnp.min(delta), jnp.max(delta),
                                jnp.min(window_avg_acceptance), jnp.max(window_avg_acceptance),
                                state.theta[0], state.theta[1], state.theta[2]
                                ), result=None)

        # moving average.
        next_state = gibbs_step(key_inp, state, delta)
        window_avg_acceptance = args.beta * next_state.kalman_state.updated + (1 - args.beta) * window_avg_acceptance

        lr_ = (n_iter - i) * lr / n_iter
        delta = delta_adaptation(delta, target_alpha, window_avg_acceptance, lr_)

        carry = i + 1, next_state, delta, window_avg_acceptance

        return carry, None

    init = 0, init_state, init_delta, 1. * init_state.kalman_state.updated

    out, _ = jax.lax.scan(body_fn, init, keys)
    return out


def decoupling_loop(key, delta, state, n_iter, verbose=False):
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

        next_state_1 = gibbs_step(key_inp_1, state_1, delta)
        next_state_2 = gibbs_step(key_inp_2, state_2, delta)

        carry = i + 1, next_state_1, next_state_2
        return carry, None

    init = 0, state, state

    out, _ = jax.lax.scan(body_fn, init, (keys_1, keys_2))
    return out


@partial(jax.jit, static_argnums=(1,))
def get_burnin_delta_and_state(key, verbose=args.verbose):
    # KERNEL
    init_key, burnin_key = jax.random.split(key, 2)

    # INIT
    xs_init = sample_trajectory(init_key, M0, P0, THETA_0, SIGMA_X, SMOOTH_FREQ, N_STEPS)

    init_state = GibbsState(kalman_state=KalmanSampler(x=xs_init, updated=True),
                            theta=THETA_0)
    delta_init = args.delta_init

    # BURNIN
    _, burnin_state, burnin_delta, burnin_avg_acceptance = adaptation_loop(burnin_key, delta_init, init_state,
                                                                           n_iter=args.burnin,
                                                                           target_alpha=args.target_alpha,
                                                                           lr=args.lr,
                                                                           verbose=verbose)
    return burnin_delta, burnin_state


@partial(jax.jit, static_argnums=(3,))
def get_one_adapted_estimate(key, delta, state, verbose=args.verbose):
    from jax.experimental.host_callback import call

    def tic_fn(arr):
        time_elapsed = time.time() - NOW
        return np.array(time_elapsed, dtype=delta.dtype), arr

    output_shape = (jax.ShapeDtypeStruct((), delta.dtype),
                    state)

    tic, state = call(tic_fn, state,
                      result_shape=output_shape)

    decoupling_key, delay_key, sampling_key = jax.random.split(key, 3)

    # DECOUPLING
    _, state_1, state_2 = decoupling_loop(decoupling_key, delta, state, args.decoupling, verbose)
    # DESYNCHRONIZE

    state_1 = gibbs_step(delay_key, state_1, delta)
    # Rhee & Glynn estimator
    coupled_kernel_fn_spec = lambda k_, state_: coupled_gibbs_step(k_, state_, delta)

    coupled_state = CoupledGibbsState(state_1=state_1,
                                      state_2=state_2,
                                      flags=jnp.zeros((N_STEPS,), dtype=bool),
                                      theta_coupled=False)

    estimate, _, coupling_time = rhee_glynn(sampling_key, coupled_kernel_fn_spec, coupled_state, args.K, args.M,
                                            stats_fn)

    output_shape = (jax.ShapeDtypeStruct((), delta.dtype),
                    coupling_time)

    toc, coupling_time = call(tic_fn, coupling_time,
                              result_shape=output_shape)

    return toc - tic, *estimate, coupling_time


def one_experiment(exp_key, verbose=args.verbose):
    # DATA

    adaptation_key, estimator_key = jax.random.split(exp_key, 2)

    delta, state = get_burnin_delta_and_state(adaptation_key, verbose)
    estimator_keys = jax.random.split(estimator_key, args.n_estimators)

    experiment_times = np.ones((args.n_estimators,)) * np.nan
    means = np.ones((args.n_estimators, N_STEPS, 3)) * np.nan
    theta_means = np.ones((args.n_estimators, 3)) * np.nan
    squares = np.ones((args.n_estimators, N_STEPS, 3)) * np.nan
    coupling_times = np.ones((args.n_estimators,)) * np.nan

    for i in tqdm.trange(args.n_estimators, desc="Inner estimators:"):
        clock_time, mean, theta_mean, square, coupling_time = get_one_adapted_estimate(estimator_keys[i], delta, state,
                                                                                       verbose)
        experiment_times[i] = clock_time
        means[i] = mean
        theta_means[i] = theta_mean
        squares[i] = square
        coupling_times[i] = coupling_time.block_until_ready()

    return experiment_times, means, theta_means, squares, coupling_times


def full_experiment():
    key = jax.random.PRNGKey(args.seed)
    experiment_times, means, theta_means, squares, coupling_times = one_experiment(key)
    return experiment_times, means, theta_means, squares, coupling_times


def main():
    import os
    with jax.disable_jit(args.debug):
        with jax.debug_nans(args.debug_nans):
            time_taken, mean_traj, theta_mean, square_traj, coupling_time = full_experiment()
    file_name = f"results/coupled-{args.D}-{args.T}-{args.parallel}.npz"
    if not os.path.isdir("results"):
        os.mkdir("results")
    np.savez(file_name, time_taken=time_taken, mean_traj=mean_traj, theta_mean=theta_mean, square_traj=square_traj,
             coupling_time=coupling_time)


if __name__ == "__main__":
    main()
