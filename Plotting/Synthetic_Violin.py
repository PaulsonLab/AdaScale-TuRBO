import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

replicate = 10
fontsize = 12
Ds = [50, 100]
ls = [0.05, 0.1, 0.2]

# per-row horizons
T_by_D = {50: 501, 100: 1001}

methods = [
    ("TuRBO",               "Turbo_{D}D_l_{l}_GPprior_regret.pt"),
    ("D-scaled TuRBO",      "Turbo_lognormal_{D}D_l_{l}_GPprior_regret.pt"),
    ("AdaScale-TuRBO",      "Turbo_lognormal_new_{D}D_l_{l}_GPprior_regret.pt"),
    ("D-scaled LogEI",      "Vanilla_{D}D_l_{l}_GPprior_regret.pt"),
    ("Linear BO",              "linear_BO_{D}D_l_{l}_GPprior_regret.pt"),
]

base_dir = "/fs/ess/PAS2983/jontwt/Lengthscale_init/GP prior/Results"

# Consistent colors across all subplots (tab10-like)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
method_colors = {label: colors[k % len(colors)] for k, (label, _) in enumerate(methods)}

# ---- Plot ----
fig, axes = plt.subplots(
    2, 3, figsize=(10.5, 3.2), dpi=300,
    sharex=True, sharey=False, constrained_layout=True
)

legend_patches = [mpatches.Patch(color=method_colors[label], label=label) for label, _ in methods]

for i, D in enumerate(Ds):
    T = T_by_D[D]
    t_idx = T - 2  # because you slice :T-1

    for j, ell in enumerate(ls):
        ax = axes[i, j]

        data = []
        labels = []
        for label, pattern in methods:
            path = f"{base_dir}/{pattern.format(D=D, l=ell)}"
            y = -torch.load(path)[:replicate, :T-1]
            vals = y[:, t_idx].detach().cpu().numpy()  # one scalar per replicate
            data.append(vals)
            labels.append(label)

        positions = np.arange(1, len(labels) + 1)

        vp = ax.violinplot(
            data,
            positions=positions,
            widths=0.85,
            showmeans=False,
            showmedians=False,   # we'll draw medians ourselves (so we can style them)
            showextrema=False
        )

        # Color each violin body
        for body, label in zip(vp["bodies"], labels):
            body.set_facecolor(method_colors[label])
            body.set_edgecolor("black")
            body.set_alpha(0.85)
            body.set_linewidth(0.8)

        # Median marker (white dot) + optional IQR bar
        for x0, vals in zip(positions, data):
            med = np.median(vals)
            q1, q3 = np.quantile(vals, [0.25, 0.75])

            ax.scatter([x0], [med], s=18, c="white", edgecolors="black", linewidths=0.6, zorder=3)
            ax.vlines(x0, q1, q3, colors="black", linewidth=1.2, zorder=2)

            # Optional min/max whiskers like some violin styles:
            # ax.vlines(x0, np.min(vals), np.max(vals), colors="black", linewidth=0.8, alpha=0.6, zorder=1)

        # Match the “legend-only x-axis” look
        ax.set_xticks([])
        ax.set_xlim(0.5, len(labels) + 0.5)

        ax.set_title(rf"$D={D}$, $\ell={ell}$", fontsize=fontsize)
        ax.grid(True, axis="y", alpha=0.5)
        ax.tick_params(axis="y", labelsize=10)

        # If you want log scale like your screenshot, uncomment:
        # ax.set_yscale("log")

# Global labels (like your screenshot)
fig.supylabel("Best-found value", fontsize=fontsize)
fig.supxlabel("")  # keep empty; legend explains methods

# Bottom legend
fig.legend(
    handles=legend_patches,
    loc="lower center",
    ncol=len(methods),
    fontsize=fontsize,
    bbox_to_anchor=(0.5, -0.1),
    frameon=True
)

plt.savefig("Synthetic_violin_GP.png", bbox_inches="tight")
# plt.show()
