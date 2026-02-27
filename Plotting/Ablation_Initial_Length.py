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
Ds = [0.4, 0.2]
ls = ['Michalewicz', 'Schwefel', 'Rastrigin']

# per-row horizons
T_by_D = {0.4: 501, 0.2: 501}

methods = [
    ("TuRBO",               "Turbo_{D}L_50D_{l}_regret.pt"),
    ("AdaScale-TuRBO",      "Turbo_lognormal_new_{D}L_50D_{l}_regret.pt"),
]

base_dir = "/fs/ess/PAS2983/jontwt/AdaScale-TuRBO/Results"

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

        ax.set_title(rf"$L_0={D}$, {ell}", fontsize=fontsize)
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
    bbox_to_anchor=(0.5, -0.08)
)

# plt.tight_layout()
fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
axes[0,0].set_ylim(-20,-7)
axes[1,0].set_ylim(-20,-7)
axes[0,1].set_ylim(8000,19000)
axes[1,1].set_ylim(8000,19000)
axes[0,2].set_ylim(200,850)
axes[1,2].set_ylim(200,850)
# plt.subplots_adjust(bottom=0.12)
plt.savefig("Ablation.png", bbox_inches="tight")
# plt.show()
