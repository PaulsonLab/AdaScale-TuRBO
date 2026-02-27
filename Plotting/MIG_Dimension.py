#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 16:47:11 2025

@author: jontwt
"""


# import torch
# import matplotlib.pyplot as plt

# # Load shared x and Independent once
# x = torch.load('/fs/ess/PAS2983/jontwt/Lengthscale_init/MIG/N_list.pt')
# Independent = torch.load('/fs/ess/PAS2983/jontwt/Lengthscale_init/MIG/MIG_Independent.pt')

# # Create 1x3 layout
# fig, axes = plt.subplots(
#     1, 3,
#     figsize=(18,8),   # wide layout
#     dpi=300,
#     sharey=True
# )

# L_values = [0.8, 0.4, 0.2]

# for idx, L in enumerate(L_values):

#     Turbo20  = torch.load(f'/fs/ess/PAS2983/jontwt/Lengthscale_init/MIG/MIG_Turbo_D20_L{L}_l0.5.pt')
#     Turbo40  = torch.load(f'/fs/ess/PAS2983/jontwt/Lengthscale_init/MIG/MIG_Turbo_D40_L{L}_l0.5.pt')
#     Turbo60  = torch.load(f'/fs/ess/PAS2983/jontwt/Lengthscale_init/MIG/MIG_Turbo_D60_L{L}_l0.5.pt')
#     Turbo80  = torch.load(f'/fs/ess/PAS2983/jontwt/Lengthscale_init/MIG/MIG_Turbo_D80_L{L}_l0.5.pt')
#     Turbo100 = torch.load(f'/fs/ess/PAS2983/jontwt/Lengthscale_init/MIG/MIG_Turbo_D100_L{L}_l0.5.pt')

#     ax = axes[idx]

#     ax.plot(x, Turbo20,  label='D = 20',  linewidth=3)
#     ax.plot(x, Turbo40,  label='D = 40',  linewidth=3)
#     ax.plot(x, Turbo60,  label='D = 60',  linewidth=3)
#     ax.plot(x, Turbo80,  label='D = 80',  linewidth=3)
#     ax.plot(x, Turbo100, label='D = 100', linewidth=3)
#     ax.plot(x, Independent, label='Indep.', linestyle='dotted', color='black', linewidth=3)

#     ax.set_title(f'L = {L}', fontsize=18)
#     ax.set_xlabel('Number of data', fontsize=18)
#     ax.tick_params(axis='both', labelsize=18)
    
#     ax.grid()

# # Only left panel gets ylabel
# axes[0].set_ylabel(r'Approximate MIG $\gamma_n$', fontsize=18)

# # Single legend (recommended for papers)
# handles, labels = axes[0].get_legend_handles_labels()
# plt.tight_layout()
# fig.legend(handles, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.08),fontsize=18)

# # plt.tight_layout(rect=[0, 0, 1, 0.92])  # leave space for legend
# # plt.show()
# plt.savefig('MIG_D.png')


import torch
import matplotlib.pyplot as plt
import numpy as np

x = torch.load('/fs/ess/PAS2983/jontwt/Lengthscale_init/MIG/N_list.pt').cpu().numpy()
Independent = torch.load('/fs/ess/PAS2983/jontwt/Lengthscale_init/MIG/MIG_Independent.pt').cpu().numpy()

L_values = [0.8, 0.4, 0.2,0.1]
Ds = [20, 40, 60, 80, 100]
linewidth = 4
fontsize = 14
# Single-column friendly size (typical ~3.25-3.5 inches wide)
fig, axes = plt.subplots(2,2, figsize=(8.0, 6), dpi=300, sharey=True)
axes = axes.flatten()

for idx, L in enumerate(L_values):
    ax = axes[idx]

    # Ys = []
    # for D in Ds:
    #     y = torch.load(
    #         f'/fs/ess/PAS2983/jontwt/Lengthscale_init/MIG/MIG_Turbo_D{D}_L{L}_l0.5.pt'
    #     ).cpu().numpy()
    #     ax.plot(x, y, linewidth=1.6)
    #     Ys.append((D, y))

    # ax.plot(x, Independent, linestyle=':', color='black', linewidth=1.6)
    
    Turbo20  = torch.load(f'/fs/ess/PAS2983/jontwt/Lengthscale_init/MIG/MIG_Turbo_D20_L{L}_l0.5.pt')
    Turbo40  = torch.load(f'/fs/ess/PAS2983/jontwt/Lengthscale_init/MIG/MIG_Turbo_D40_L{L}_l0.5.pt')
    Turbo60  = torch.load(f'/fs/ess/PAS2983/jontwt/Lengthscale_init/MIG/MIG_Turbo_D60_L{L}_l0.5.pt')
    Turbo80  = torch.load(f'/fs/ess/PAS2983/jontwt/Lengthscale_init/MIG/MIG_Turbo_D80_L{L}_l0.5.pt')
    Turbo100 = torch.load(f'/fs/ess/PAS2983/jontwt/Lengthscale_init/MIG/MIG_Turbo_D100_L{L}_l0.5.pt')

    ax = axes[idx]

    ax.plot(x, Turbo20,  label='D = 20',  linewidth=linewidth)
    ax.plot(x, Turbo40,  label='D = 40',  linewidth=linewidth)
    ax.plot(x, Turbo60,  label='D = 60',  linewidth=linewidth)
    ax.plot(x, Turbo80,  label='D = 80',  linewidth=linewidth)
    ax.plot(x, Turbo100, label='D = 100', linewidth=linewidth)
    ax.plot(x, Independent, label='Indep.', linestyle='dotted', color='black', linewidth=linewidth)

    ax.set_title(f"L={L}", fontsize=fontsize)
    ax.set_xlabel("Number of data", fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.grid(True, alpha=0.5)

    # Direct labels on the right end of curves (instead of legend)
    # Put labels near the last x position
    x_last = x[-1]
    # for D, y in Ys:
    #     ax.text(x_last * 1.01, y[-1], f"{D}", fontsize=7, va='center', clip_on=False)

    # Label Indep once
    # if idx == 2:
    #     ax.text(x_last * 1.01, Independent[-1], "Indep", fontsize=7, va='center', clip_on=False)
    
    
handles, labels = axes[0].get_legend_handles_labels()
plt.tight_layout()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.12),fontsize=fontsize)
# axes[0].set_ylabel(r"MIG $\gamma_n$", fontsize=fontsize)
fig.supylabel(r"MIG $\gamma_n$", fontsize=fontsize)
# axes[0].legend()
# Make room on the right for the direct labels
plt.tight_layout()
# plt.subplots_adjust(right=0.90)
# plt.show()

plt.savefig('MIG_D.png')