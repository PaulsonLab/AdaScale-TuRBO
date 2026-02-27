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
fontsize=18
linewidth=4
x = np.arange(0,1000,1)

turbo_original = -torch.load('/fs/ess/PAS2983/jontwt/Lengthscale_init/Rover/Results/Turbo_Rover_regret.pt')[0:replicate]
turbo_lognormal = -torch.load('/fs/ess/PAS2983/jontwt/Lengthscale_init/Rover/Results/Turbo_lognormal_Rover_regret.pt')[0:replicate]
turbo_lognormal_new = -torch.load('/fs/ess/PAS2983/jontwt/Lengthscale_init/Rover/Results/Turbo_lognormal_new_Rover_regret.pt')[0:replicate]
vanilla = -torch.load('/fs/ess/PAS2983/jontwt/Lengthscale_init/Rover/Results/Rover_Vanilla_regret.pt')[0:replicate,0:1000]
linear = -torch.load('/fs/ess/PAS2983/jontwt/Lengthscale_init/Rover/Results/Rover_linear_BO_regret.pt')[0:replicate,0:1000]


turbo_original_mean = torch.quantile(turbo_original, q=0.5, dim=0)
turbo_lognormal_mean = torch.quantile(turbo_lognormal, q=0.5, dim=0)
turbo_lognormal_new_mean = torch.quantile(turbo_lognormal_new, q=0.5, dim=0)
vanilla_mean = torch.quantile(vanilla, q=0.5, dim=0)
linear_mean = torch.quantile(linear, q=0.5, dim=0)

turbo_original_std = turbo_original.std(dim=0) / replicate**0.5
turbo_lognormal_std = turbo_lognormal.std(dim=0)/ replicate**0.5
turbo_lognormal_new_std = turbo_lognormal_new.std(dim=0)/ replicate**0.5
vanilla_std = vanilla.std(dim=0)/ replicate**0.5
linear_std = linear.std(dim=0)/ replicate**0.5


plt.figure(figsize = (8,5), dpi = 300)

plt.plot(turbo_original_mean, label='TuRBO', linewidth=linewidth)
plt.plot(turbo_lognormal_mean, label = 'D-scaled TuRBO', linewidth=linewidth)
plt.plot(turbo_lognormal_new_mean, label='AdaScale-TuRBO', linewidth=linewidth)
plt.plot(vanilla_mean, label='D-scaled LogEI', linewidth=linewidth)


plt.plot(linear_mean, label='Linear BO', linewidth=linewidth)



plt.fill_between(x, turbo_original_mean - turbo_original_std, turbo_original_mean + turbo_original_std,  alpha=0.2)
plt.fill_between(x, turbo_lognormal_mean - turbo_lognormal_std, turbo_lognormal_mean + turbo_lognormal_std,  alpha=0.2)
plt.fill_between(x, turbo_lognormal_new_mean - turbo_lognormal_new_std, turbo_lognormal_new_mean + turbo_lognormal_new_std,  alpha=0.2)
plt.fill_between(x, vanilla_mean - vanilla_std, vanilla_mean + vanilla_std,  alpha=0.2)
# plt.fill_between(x, vanilla_RAASP_mean - vanilla_RAASP_std, vanilla_RAASP_mean + vanilla_RAASP_std,  alpha=0.2)
plt.fill_between(x, linear_mean - linear_std, linear_mean + linear_std,  alpha=0.2)
# plt.fill_between(x, turbo_mixed_mean - turbo_mixed_std, turbo_mixed_mean + turbo_mixed_std,  alpha=0.2)


plt.legend(fontsize=fontsize)
plt.xlabel('Function evaluations',fontsize=fontsize)
plt.ylabel('Best value',fontsize=fontsize)
plt.ylim(-4,4)
plt.title('Rover Trajectory, D = 60', fontsize=fontsize)
plt.grid(alpha=0.5)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.tight_layout()
plt.savefig('Rover.png')