import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_SAMPLES = 10_000
T = 250
D = 30
PLOT = True
GPU = True

csmc_grad = np.load(f"csmc-{D}-{T}-25-{GPU}-True.npz")
csmc = np.load(f"csmc-{D}-{T}-25-{GPU}-False.npz")
kalman_1 = np.load(f"kalman-1-{D}-{T}-25-{GPU}-False.npz")
kalman_2 = np.load(f"kalman-2-{D}-{T}-25-{GPU}-False.npz")

csmc_grad_time = csmc_grad["time_per_key"]
csmc_grad_delta = csmc_grad["delta_per_key"]
csmc_grad_rate = csmc_grad["acceptance_rate_per_key"]
csmc_grad_esjd = csmc_grad["ejsd_per_key"]

csmc_time = csmc["time_per_key"]
csmc_delta = csmc["delta_per_key"]
csmc_rate = csmc["acceptance_rate_per_key"]
csmc_esjd = csmc["ejsd_per_key"]

kalman_1_time = kalman_1["time_per_key"]
kalman_1_delta = kalman_1["delta_per_key"]
kalman_1_rate = kalman_1["acceptance_rate_per_key"]
kalman_1_esjd = kalman_1["ejsd_per_key"]

kalman_2_time = kalman_2["time_per_key"]
kalman_2_delta = kalman_2["delta_per_key"]
kalman_2_rate = kalman_2["acceptance_rate_per_key"]
kalman_2_esjd = kalman_2["ejsd_per_key"]

# print(csmc_rate.mean(0))
# print(csmc_grad_rate.mean(0))
# print(kalman_1_rate.mean(0))
# print(kalman_2_rate.mean(0))
# print()

# Plot the EJSD
plt.figure(figsize=(30, 15))

# print(csmc_time.mean(), csmc_grad_time.mean(), kalman_1_time.mean(), kalman_2_time.mean())
# print(np.median(csmc_grad_esjd, 0), np.median(csmc_esjd, 0), np.mean(kalman_1_esjd, 0), np.mean(kalman_2_esjd, 0))

csmc_time_per_iter = csmc_time[1:, None] / N_SAMPLES
csmc_grad_time_per_iter = csmc_grad_time[1:, None] / N_SAMPLES
kalman_1_time_per_iter = kalman_1_time[1:, None] / N_SAMPLES
kalman_2_time_per_iter = kalman_2_time[1:, None] / N_SAMPLES

csmc_esjd = csmc_esjd[1:].sum(-1)
csmc_grad_esjd = csmc_grad_esjd[1:].sum(-1)
kalman_1_esjd = kalman_1_esjd[1:].sum(-1)
kalman_2_esjd = kalman_2_esjd[1:].sum(-1)

csmc_esjd_per_time = csmc_esjd / csmc_time_per_iter
csmc_grad_esjd_per_time = csmc_grad_esjd / csmc_grad_time_per_iter
kalman_1_esjd_per_time = kalman_1_esjd / kalman_1_time_per_iter
kalman_2_esjd_per_time = kalman_2_esjd / kalman_2_time_per_iter

esjd_time_df = pd.DataFrame([np.arange(T),
                             csmc_esjd_per_time.mean(0),
                             csmc_grad_esjd_per_time.mean(0),
                             kalman_1_esjd_per_time.mean(0),
                             kalman_2_esjd_per_time.mean(0)],
                            index=["t", "cSMC", "cSMC_grad", "Kalman_1", "Kalman_2"]).T
esjd_df = pd.DataFrame([np.arange(T),
                        csmc_esjd.mean(0),
                        csmc_grad_esjd.mean(0),
                        kalman_1_esjd.mean(0),
                        kalman_2_esjd.mean(0)],
                       index=["t", "cSMC", "cSMC_grad", "Kalman_1", "Kalman_2"]).T

with pd.option_context('display.max_rows', 6, 'display.max_columns', None):
    print(esjd_df.head())
    print()
    print(esjd_time_df.head())


esjd_df.to_csv(f"ESJD_{GPU}.csv")
esjd_time_df.to_csv(f"ESJD_time_{GPU}.csv")
