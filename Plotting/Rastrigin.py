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
x = np.arange(0,1000,1)
fontsize = 18
linewidth = 4

turbo_original = torch.abs(torch.load('/fs/ess/PAS2983/jontwt/Lengthscale_init/Rastrigin/Results/Turbo_50D_Rastrigin_regret.pt')[0:replicate,0:1000])
turbo_lognormal = torch.abs(torch.load('/fs/ess/PAS2983/jontwt/Lengthscale_init/Rastrigin/Results/Turbo_lognormal_50D_Rastrigin_regret.pt')[0:replicate,0:1000])
turbo_lognormal_new = torch.abs(torch.load('/fs/ess/PAS2983/jontwt/Lengthscale_init/Rastrigin/Results/Turbo_lognormal_new_50D_Rastrigin_regret.pt')[0:replicate,0:1000])
vanilla = torch.load('/fs/ess/PAS2983/jontwt/Lengthscale_init/Rastrigin/Results/50D_Rastrigin_Vanilla_regret.pt')[0:replicate,0:1000]
linear = torch.load('/fs/ess/PAS2983/jontwt/Lengthscale_init/Rastrigin/Results/50D_Rastrigin_linear_BO_regret.pt')[0:replicate,0:1000]


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

plt.figure(dpi = 300, figsize = (14,12))

plt.plot(turbo_original_mean, label='Turbo', linewidth=linewidth)
plt.plot(turbo_lognormal_mean, label='Turbo-LogNormal', linewidth=linewidth)
plt.plot(turbo_lognormal_new_mean, label='AdaScale-Turbo', linewidth=linewidth)
plt.plot(vanilla_mean, label='D-scaled LogEI-ARD', linewidth=linewidth)
plt.plot(linear_mean, label='Linear', linewidth=linewidth)

plt.fill_between(x, turbo_original_mean - turbo_original_std, turbo_original_mean + turbo_original_std,  alpha=0.2)
plt.fill_between(x, turbo_lognormal_mean - turbo_lognormal_std, turbo_lognormal_mean + turbo_lognormal_std,  alpha=0.2)
plt.fill_between(x, turbo_lognormal_new_mean - turbo_lognormal_new_std, turbo_lognormal_new_mean + turbo_lognormal_new_std,  alpha=0.2)
plt.fill_between(x, vanilla_mean - vanilla_std, vanilla_mean + vanilla_std,  alpha=0.2)
plt.fill_between(x, linear_mean - linear_std, linear_mean + linear_std,  alpha=0.2)


plt.legend(loc='best', fontsize = fontsize)
plt.xlabel('function evaluations', fontsize = fontsize)
plt.ylabel('Best value', fontsize = fontsize)
plt.title('50D Rastrigin', fontsize = fontsize)
plt.savefig('50D Rastrigin.png')