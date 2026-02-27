#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 21:08:56 2025

@author: jontwt
"""


import torch
import matplotlib.pyplot as plt
import numpy as np

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

fig, axes = plt.subplots(
    2, 3,
    figsize=(7.0, 6),
    dpi=300,
    sharey=False,   # y comparable
    sharex=False,   # x differs by row
    constrained_layout=True
)

legend_handles = None
legend_labels = None

for i, D in enumerate(Ds):
    T = T_by_D[D]

    for j, ell in enumerate(ls):
        ax = axes[i, j]
        x = np.arange(T-1)

        for label, pattern in methods:
            path = f"{base_dir}/{pattern.format(D=D, l=ell)}"

            y = -torch.load(path)[:replicate, :T-1]   # shape: (replicate, T)
            # if y might be on GPU, uncomment:
            # y = y.cpu()

            med = torch.quantile(y, q=0.5, dim=0)
            se  = y.std(dim=0) / (replicate ** 0.5)

            line, = ax.plot(x, med.numpy(), linewidth=3, label=label)
            ax.fill_between(
                x,
                (med - se).numpy(),
                (med + se).numpy(),
                alpha=0.2
            )

        ax.set_title(rf"$D={D}$, $\ell={ell}$", fontsize=fontsize)
        ax.grid(True, alpha=0.5)
        ax.tick_params(axis="both", labelsize=10)

        # store legend once
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

# Global labels
fig.supxlabel("Function evaluations", fontsize=fontsize)
fig.supylabel("Best value", fontsize=fontsize)
# plt.tight_layout()
# Bottom-center legend
fig.legend(
    legend_handles, legend_labels,
    loc="lower center",
    ncol=3,
    fontsize=fontsize,
    # frameon=False,
    bbox_to_anchor=(0.5, -0.12)
)

# plt.tight_layout()
fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)

# plt.subplots_adjust(bottom=0.12)
plt.savefig("GPprior.png", bbox_inches="tight")
# plt.show()
