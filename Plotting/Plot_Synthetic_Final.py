#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 15:17:52 2025

@author: jontwt
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import matplotlib.ticker as mticker
# -------------------------
# 1. Setup dummy problems & methods
# -------------------------
problems = ["Rastrigin, D=100", "Schwefel, D=100", "Michalewicz, D=100", "Rastrigin, D=100"]
methods = ["TuRBO", "D-scaled TuRBO", "AdaScale-TuRBO", "D-scaled LogEI","LinearBO"]

colors = dict(zip(methods, plt.cm.tab10.colors * 2))  # repeat colors if >10
replicate = 10
n_runs = 5      # number of repeated trials per method
n_iter = 150    # number of function evaluations

# -------------------------
# 2. Generate synthetic results
# Shape: results[problem][method] -> (n_runs, n_iter)
# -------------------------
rng = np.random.default_rng(0)
results = {}
for problem in problems:
    results[problem] = {}
    for method in methods:
        # Simulate decreasing curves (best value found improving with iterations)
        # base = np.linspace(15, 8, n_iter) + rng.normal(0, 0.2, n_iter)
        # runs = np.array([base + rng.normal(0, 0.5, n_iter) for _ in range(n_runs)])
        # results[problem][method] = np.maximum(runs, 0.0)  # avoid negatives
        if problem == 'Rastrigin, D=100' and method == "TuRBO":   
            results[problem][method] = (torch.abs(torch.load("/fs/ess/PAS2983/jontwt/Lengthscale_init/Rastrigin/Results/Turbo_100D_Rastrigin_regret.pt")))
        if problem == 'Rastrigin, D=100' and method == "D-scaled TuRBO":   
            results[problem][method] = (torch.abs(torch.load("/fs/ess/PAS2983/jontwt/Lengthscale_init/Rastrigin/Results/Turbo_lognormal_100D_Rastirgin_regret.pt")))
        if problem == 'Rastrigin, D=100' and method == "AdaScale-TuRBO":   
            results[problem][method] = (torch.abs(torch.load("/fs/ess/PAS2983/jontwt/Lengthscale_init/Rastrigin/Results/Turbo_lognormal_new_100D_Rastrigin_regret.pt")))
        if problem == 'Rastrigin, D=100' and method == "D-scaled LogEI":   
            results[problem][method] = (torch.load("/fs/ess/PAS2983/jontwt/Lengthscale_init/Rastrigin/Results/100D_Rastrigin_Vanilla_regret.pt"))
        if problem == 'Rastrigin, D=100' and method == "LinearBO":   
            results[problem][method] = (torch.load("/fs/ess/PAS2983/jontwt/Lengthscale_init/Rastrigin/Results/100D_Rastrigin_linear_BO_regret.pt"))
        
        if problem == 'Schwefel, D=100' and method == "TuRBO":   
            results[problem][method] = (torch.abs(torch.load("/fs/ess/PAS2983/jontwt/Lengthscale_init/Schwefel/Results/Turbo_100D_Schwefel_regret.pt")))
        if problem == 'Schwefel, D=100' and method == "D-scaled TuRBO":   
            results[problem][method] = (torch.abs(torch.load("/fs/ess/PAS2983/jontwt/Lengthscale_init/Schwefel/Results/Turbo_lognormal_100D_Schwefel_regret.pt")))
        if problem == 'Schwefel, D=100' and method == "AdaScale-TuRBO":   
            results[problem][method] = (torch.abs(torch.load("/fs/ess/PAS2983/jontwt/Lengthscale_init/Schwefel/Results/Turbo_lognormal_new_100D_Schwefel_regret.pt")))
        if problem == 'Schwefel, D=100' and method == "D-scaled LogEI":   
            results[problem][method] = (torch.load("/fs/ess/PAS2983/jontwt/Lengthscale_init/Schwefel/Results/100D_Schwefel_Vanilla_regret.pt"))
        if problem == 'Schwefel, D=100' and method == "LinearBO":   
            results[problem][method] = (torch.load("/fs/ess/PAS2983/jontwt/Lengthscale_init/Schwefel/Results/100D_Schwefel_linear_BO_regret.pt"))
       
        if problem == 'Michalewicz, D=100' and method == "TuRBO":   
            results[problem][method] = torch.load("/fs/ess/PAS2983/jontwt/Lengthscale_init/Michalewicz/Results/Turbo_100D_Michalewicz_regret.pt")
        if problem == 'Michalewicz, D=100' and method == "D-scaled TuRBO":   
            results[problem][method] = torch.load("/fs/ess/PAS2983/jontwt/Lengthscale_init/Michalewicz/Results/Turbo_lognormal_100D_Michalewicz_regret.pt")
        if problem == 'Michalewicz, D=100' and method == "AdaScale-TuRBO":   
            results[problem][method] = torch.load("/fs/ess/PAS2983/jontwt/Lengthscale_init/Michalewicz/Results/Turbo_lognormal_new_100D_Michalewicz_regret.pt")
        if problem == 'Michalewicz, D=100' and method == "D-scaled LogEI":   
            results[problem][method] = (torch.load("/fs/ess/PAS2983/jontwt/Lengthscale_init/Michalewicz/Results/100D_Michalewicz_Vanilla_regret.pt"))
        if problem == 'Michalewicz, D=100' and method == "LinearBO":   
            results[problem][method] = (torch.load("/fs/ess/PAS2983/jontwt/Lengthscale_init/Michalewicz/Results/100D_Michalewicz_linear_BO_regret.pt"))
        
        if problem == 'Rastrigin, D=100' and method == "TuRBO":   
            results[problem][method] = (torch.abs(torch.load("/fs/ess/PAS2983/jontwt/Lengthscale_init/Rastrigin/Results/Turbo_100D_Rastrigin_regret.pt")))
        if problem == 'Rastrigin, D=100' and method == "D-scaled TuRBO":   
            results[problem][method] = (torch.abs(torch.load("/fs/ess/PAS2983/jontwt/Lengthscale_init/Rastrigin/Results/Turbo_lognormal_100D_Rastirgin_regret.pt")))
        if problem == 'Rastrigin, D=100' and method == "AdaScale-TuRBO":   
            results[problem][method] = (torch.abs(torch.load("/fs/ess/PAS2983/jontwt/Lengthscale_init/Rastrigin/Results/Turbo_lognormal_new_100D_Rastrigin_regret.pt")))
        if problem == 'Rastrigin, D=100' and method == "D-scaled LogEI":   
            results[problem][method] = (torch.load("/fs/ess/PAS2983/jontwt/Lengthscale_init/Rastrigin/Results/100D_Rastrigin_Vanilla_regret.pt"))
        if problem == 'Rastrigin, D=100' and method == "LinearBO":   
            results[problem][method] = (torch.load("/fs/ess/PAS2983/jontwt/Lengthscale_init/Rastrigin/Results/100D_Rastrigin_linear_BO_regret.pt"))
       

# -------------------------
# 3. Plot multi-panel figure
# -------------------------
fig, axes = plt.subplots(2, len(problems), figsize=(18, 10))
fontsize = 18

for j, problem in enumerate(problems):
    # --- Top row: convergence curves ---
    for method in methods:
        vals = np.array(results[problem][method])  # (n_runs, n_iter)
        # mean = vals.median(axis=0)
        mean = np.median(vals, axis=0)
        std = vals.std(axis=0) / replicate**0.5
        iters = np.arange(mean.shape[0])

        axes[0, j].plot(mean, label=method, color=colors[method], linewidth=4)
        axes[0, j].fill_between(iters, mean-std, mean+std, alpha=0.2, color=colors[method])
        axes[0, j].set_title(problem, fontsize = fontsize)
        
    # axes[0, 0].set_ylim(-0.5,14)
    # axes[0, 1].set_ylim(2,14)
    # axes[0, 2].set_ylim(-3,5)
    axes[0, 3].set_ylim(-0.1,2.5)
        
    axes[0, j].set_xlabel("Function evaluations", fontsize = fontsize)
    if j == 0:
        axes[0, j].set_ylabel("Simple regret (log scale)", fontsize = fontsize)
    axes[0, j].tick_params(axis="both", which="major", labelsize=14)
    
    
    # --- Bottom row: violin plots of final values ---
    final_vals = []
    for method in methods:
        vals = np.array(results[problem][method])[:, -1]  # last iteration
        for v in vals:
            final_vals.append({"Method": method, "FinalValue": v})
    df = pd.DataFrame(final_vals)
    sns.violinplot(x="Method", y="FinalValue", data=df, palette=colors, ax=axes[1, j], cut=0, inner="quartile")

    
    axes[0, j].grid(True, linewidth = 0.3)
    axes[1, j].grid(True, linewidth = 0.3)
    axes[1, 0].set_ylabel("Best-found value (log scale)", fontsize = fontsize)
    axes[1, j].set_ylabel("", fontsize = fontsize)
    
    axes[1, j].tick_params(axis="x", rotation=90)
    axes[1, j].set_xlabel("")
    axes[1, j].set_xticklabels([]) 
    axes[1, j].tick_params(axis="both", which="major", labelsize=14)

    # final_vals = []
    # for m_idx, method in enumerate(methods):
    #     vals = np.array(results[problem][method])[:, -1]
    #     final_vals.append(vals)

    # ax = axes[1, j]

    # --- plot violin plot manually (one per method index)
    # parts = ax.violinplot(final_vals, positions=np.arange(len(methods)),
    #                       showmeans=False, showmedians=True, showextrema=False)

    # # color violins to match top-row curves
    # for pc, method in zip(parts['bodies'], methods):
    #     pc.set_facecolor(colors[method])
    #     pc.set_edgecolor("black")
    #     pc.set_alpha(0.6)

    # --- formatting
    # ax.set_ylabel("Final value" if j == 0 else "")
    # ax.set_xlabel("")
    # ax.set_xticks(np.arange(len(methods)))
    # ax.set_xticklabels([])   # remove x tick names (numbers remain)
    # ax.grid(True, axis="y", alpha=0.2)

# -------------------------
# 4. Add global legend
# -------------------------
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.004), ncol=5, fontsize=fontsize)

# plt.tight_layout(rect=[0.05, 0.05, 1, 1])  # leave space for legend
# plt.tight_layout(pad=1.0)
plt.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.1)
# plt.show()
plt.savefig('Synthetic_final.png',dpi=300)



