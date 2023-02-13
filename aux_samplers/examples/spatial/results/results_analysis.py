import matplotlib.pyplot as plt
import numpy as np

N_SAMPLES = 20_000
T = 1_024
D = 8
PLOT = True

csmc_grad = np.load(f"csmc-{D}-{T}-25-True-True.npz")
csmc = np.load(f"csmc-{D}-{T}-25-True-False.npz")
kalman = np.load(f"kalman-{D}-{T}-25-True-True.npz")

csmc_grad_mean = csmc_grad["mean_traj_per_key"]
csmc_grad_std = csmc_grad["std_traj_per_key"]
csmc_grad_true = csmc_grad["true_xs_per_key"]
csmc_grad_time = csmc_grad["time_per_key"]
csmc_grad_delta = csmc_grad["delta_per_key"]
csmc_grad_rate = csmc_grad["acceptance_rate_per_key"]
csmc_grad_esjd = csmc_grad["ejsd_per_key"]

csmc_mean = csmc["mean_traj_per_key"]
csmc_std = csmc["std_traj_per_key"]
csmc_true = csmc["true_xs_per_key"]
csmc_time = csmc["time_per_key"]
csmc_delta = csmc["delta_per_key"]
csmc_rate = csmc["acceptance_rate_per_key"]
csmc_esjd = csmc["ejsd_per_key"]

kalman_mean = kalman["mean_traj_per_key"]
kalman_std = kalman["std_traj_per_key"]
kalman_true = kalman["true_xs_per_key"]
kalman_time = kalman["time_per_key"]
kalman_delta = kalman["delta_per_key"]
kalman_rate = kalman["acceptance_rate_per_key"]
kalman_esjd = kalman["ejsd_per_key"]

# Plot some trajectories for visual inspection
ts = np.arange(T)
component = np.random.randint(D)
rand_key = np.random.randint(10)
plt.figure(figsize=(30, 15))
plt.plot(ts, csmc_grad_mean[rand_key, :, D], label="CSMC Grad", color="tab:blue")
plt.plot(ts, csmc_mean[rand_key, :, D], label="CSMC", color="tab:orange")
plt.plot(ts, kalman_mean[rand_key, :, D], label="Kalman", color="tab:green")
plt.plot(ts, csmc_grad_true[rand_key, :, D], label="True", linestyle="--", color="k")
plt.fill_between(ts,
                 csmc_grad_mean[rand_key, :, D] - 2 * csmc_grad_std[rand_key, :, D],
                 csmc_grad_mean[rand_key, :, D] + 2 * csmc_grad_std[rand_key, :, D], alpha=0.2,
                 color="tab:blue")
plt.fill_between(ts,
                 csmc_mean[rand_key, :, D] - 2 * csmc_std[rand_key, :, D],
                 csmc_mean[rand_key, :, D] + 2 * csmc_std[rand_key, :, D],
                 alpha=0.2, color="tab:orange")
plt.fill_between(ts,
                 kalman_mean[rand_key, :, D] - 2 * kalman_std[rand_key, :, D],
                 kalman_mean[rand_key, :, D] + 2 * kalman_std[rand_key, :, D],
                 alpha=0.2, color="tab:green")
plt.legend()
plt.show()


# Plot the EJSD
plt.figure(figsize=(30, 15))

csmc_time_per_iter = csmc_time[..., None, None] / N_SAMPLES
csmc_grad_time_per_iter = csmc_grad_time[..., None, None] / N_SAMPLES
kalman_time_per_iter = kalman_time[..., None, None] / N_SAMPLES

print(np.mean(csmc_time_per_iter, 0), np.mean(csmc_grad_time_per_iter, 0), np.mean(kalman_time_per_iter, 0))


csmc_esjd_per_time = csmc_esjd ** 0.5 / csmc_time_per_iter
csmc_grad_esjd_per_time = csmc_grad_esjd ** 0.5 / csmc_grad_time_per_iter
kalman_esjd_per_time = kalman_esjd ** 0.5 / kalman_time_per_iter
plt.plot(np.median(csmc_grad_esjd_per_time, 0), label="CSMC Grad", color="tab:blue")
plt.plot(np.median(csmc_esjd_per_time, 0), label="CSMC", color="tab:orange")
plt.plot(np.median(kalman_esjd_per_time, 0), label="Kalman", color="tab:green")
# plt.legend()
plt.show()