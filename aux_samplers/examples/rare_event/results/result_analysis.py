import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
import matplotlib.ticker as mtick

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Helvetica",
})

N_SAMPLES = 20_000
BATCH_SIZE = 8
PLOT = True
GRID_SIZE = 10
RHOS = np.linspace(0., 1., GRID_SIZE)
R2S = np.logspace(-3, 0., GRID_SIZE)

true_values = pd.read_csv("2-true.csv", header=0, index_col=[0, 1, 2, 3], dtype={"rho": str, "r2": str})
true_values = true_values["0"].unstack().unstack()

results_list = [
    # name, parallel, grad

    # parallel
    ("csmc", True, True),
    ("csmc", True, False),
    ("kalman", True, True),

    # sequential
    ("csmc", False, True),
    ("csmc", False, False),
    ("kalman", False, True),
    ("csmc-guided", False, True),
    ("csmc-guided", False, False),
]

name_to_pretty = {
    "csmc": "cSMC",
    "kalman": "Kalman",
    "csmc-guided": "Guided cSMC",
}

data_dict = {}

template = "{name}-2-25-{parallel}-{grad}.csv"
level_names = ["rho", "r2", "timestep", "statistic"]
pretty_level_names = [r"$\rho$", r"$r^2$", "timestep", "statistic"]

level_names_order = ["statistic", "timestep", r"$\rho$", r"$r^2$"]

true_values.index.names = pretty_level_names[:2]

fig_0, axes_0 = plt.subplots(2, 4, figsize=(10, 5), sharex=True, sharey=True)
cbar_ax_0 = fig_0.add_axes((0.93, 0.15, 0.035, 0.7))

fig_T, axes_T = plt.subplots(2, 4, figsize=(10, 5), sharex=True, sharey=True)
cbar_ax_T = fig_T.add_axes((0.92, 0.15, 0.035, 0.7))

i = -1
flat_axes_0 = axes_0.flatten()
flat_axes_T = axes_T.flatten()

for name, parallel, grad in results_list:
    i += 1
    data = pd.read_csv(template.format(name=name, parallel=parallel, grad=grad), header=[0, 1, 2, 3], index_col=0)
    data.loc[:, pd.IndexSlice[:, :, :, "ess"]] /= BATCH_SIZE
    data.columns.names = pretty_level_names
    idx = data.columns
    # data.columns = idx.set_levels([idx.levels[0].astype(float), idx.levels[1].astype(float), *idx.levels[2:]])

    data = data.reorder_levels(level_names_order, axis=1)

    true_means = true_values["mean"].unstack(level=[0, 1])
    true_std = true_values["std"].unstack(level=[0, 1])

    data["mean"] = data["mean"].sub(true_means, axis=1).div(true_std, axis=1) ** 2
    data["std"] = data["std"].sub(true_std, axis=1).div(true_std, axis=1)

    for statistic in data.columns.levels[0]:
        if statistic != "mean":
            continue

        mean_stats_0 = data[statistic]["0"].mean().unstack()
        mean_stats_0.index = mean_stats_0.index.astype(float)
        mean_stats_0.columns = mean_stats_0.columns.astype(float)

        mean_stats_T = data[statistic]["T"].mean().unstack()
        mean_stats_T.index = mean_stats_T.index.astype(float)
        mean_stats_T.columns = mean_stats_T.columns.astype(float)

        sns.heatmap(mean_stats_0, ax=flat_axes_0[i], cbar_ax=cbar_ax_0, norm=LogNorm())
        sns.heatmap(mean_stats_T, ax=flat_axes_T[i], cbar_ax=cbar_ax_T, norm=LogNorm())

        title = f"{name_to_pretty[name]}"
        if grad and parallel:
            title += " (g/p)"
        elif grad:
            title += " (g)"
        elif parallel:
            title += " (p)"

        flat_axes_0[i].set_title(title)
        flat_axes_T[i].set_title(title)

        flat_axes_0[i].set_xlabel("")
        flat_axes_T[i].set_xlabel("")
        flat_axes_0[i].set_ylabel("")
        flat_axes_T[i].set_ylabel("")

n_y_labels = [0, 4]
yticklabels = ["{:.2f}".format(x) for x in RHOS]
for n in n_y_labels:
    axes_n = [flat_axes_0[n], flat_axes_T[n]]
    for ax in [flat_axes_0[n], flat_axes_T[n]]:
        ax.set_yticklabels(yticklabels, rotation=45)
        ax.set_ylabel(r"$\rho$", rotation=0)

        for i, label in enumerate(ax.get_yticklabels()):
            if i % 3 != 0:
                label.set_visible(False)

n_x_labels = [4, 5, 6, 7]
xticklabels = [f"1e-{int(np.log10(x))}" for x in R2S[:-1]] + ["1e+0"]
for n in n_x_labels:
    axes_n = [flat_axes_0[n], flat_axes_T[n]]
    for ax in axes_n:
        ax.set_xticklabels(xticklabels, rotation=45)
        ax.set_xlabel(r"$r^2$")
        for i, label in enumerate(ax.get_xticklabels()):
            if i % 3 != 0:
                label.set_visible(False)

fig_0.savefig("mean_0.pdf", bbox_inches='tight')
fig_T.savefig("mean_T.pdf", bbox_inches='tight')
