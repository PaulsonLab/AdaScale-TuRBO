#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 13:25:19 2026

@author: jontwt
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

dim_list = [50,50,50]
L_list = [0.8,0.4,0.2]
replicate = 20
metric = [r'$NLPD$', r'$R^2$', r'$RMSE$']
fun_name = 'Prior'
fig, axes = plt.subplots(3,3, figsize=(14,14), dpi=300)
gamma = 0.7
fontsize = 18
for j in range(3): # each column
    
    # gamma = gamma_list[j]
    for i, dim in enumerate(dim_list): # each row
        L = L_list[i]
        NLPD_gamma_prior_list_ARD, NLPD_list_ARD = [], []
        NLPD_gamma_prior_list_Iso, NLPD_list_Iso = [], []
        
        NLPD_MLE_list_ARD, NLPD_MLE_Scaled_list_ARD = [], []
        NLPD_MLE_list_Iso, NLPD_MLE_Scaled_list_Iso = [], []
        
        x = np.array([1, 5, 10, 15, 20])*dim
        for a in [1, 5, 10, 15, 20]:
            Ninit = a*dim
            
            if j == 0:
                
                NLPD_Iso_gamma_prior = torch.load('/fs/ess/PAS2983/jontwt/AdaScale-TuRBO/GP_Accuracy/Results/'+fun_name+'_AdaTuRBO_nlpd_N'+str(Ninit)+'_D'+str(dim)+'_L'+str(L)+'.pt')
                NLPD_ARD_gamma_prior = torch.load('/fs/ess/PAS2983/jontwt/AdaScale-TuRBO/GP_Accuracy/Results/'+fun_name+'_TuRBO_nlpd_N'+str(Ninit)+'_D'+str(dim)+'_L'+str(L)+'.pt')
                
                # NLPD_Iso = torch.load('/fs/ess/PAS2983/jontwt/AdaScale-TuRBO/GP_Accuracy/Results/'+fun_name+'_AdaTuRBO_nlpd_N'+str(Ninit)+'_D'+str(dim)+'_L'+str(L)+'.pt')
                # NLPD_ARD = torch.load('/fs/ess/PAS2983/jontwt/AdaScale-TuRBO/GP_Accuracy/Results/'+fun_name+'_TuRBO_nlpd_N'+str(Ninit)+'_D'+str(dim)+'_L'+str(L)+'.pt')
            elif j==1:
                
                NLPD_Iso_gamma_prior = torch.load('/fs/ess/PAS2983/jontwt/AdaScale-TuRBO/GP_Accuracy/Results/'+fun_name+'_AdaTuRBO_r2_N'+str(Ninit)+'_D'+str(dim)+'_L'+str(L)+'.pt')
                NLPD_ARD_gamma_prior = torch.load('/fs/ess/PAS2983/jontwt/AdaScale-TuRBO/GP_Accuracy/Results/'+fun_name+'_TuRBO_r2_N'+str(Ninit)+'_D'+str(dim)+'_L'+str(L)+'.pt')
                
                # NLPD_Iso = torch.load('/fs/ess/PAS2983/jontwt/AdaScale-TuRBO/GP_Accuracy/Results/'+fun_name+'_AdaTuRBO_r2_N'+str(Ninit)+'_D'+str(dim)+'_L'+str(L)+'.pt')
                # NLPD_ARD = torch.load('/fs/ess/PAS2983/jontwt/AdaScale-TuRBO/GP_Accuracy/Results/'+fun_name+'_TuRBO_r2_N'+str(Ninit)+'_D'+str(dim)+'_L'+str(L)+'.pt')
            elif j==2:
                
                NLPD_Iso_gamma_prior = torch.load('/fs/ess/PAS2983/jontwt/AdaScale-TuRBO/GP_Accuracy/Results/'+fun_name+'_AdaTuRBO_rmse_N'+str(Ninit)+'_D'+str(dim)+'_L'+str(L)+'.pt')
                NLPD_ARD_gamma_prior = torch.load('/fs/ess/PAS2983/jontwt/AdaScale-TuRBO/GP_Accuracy/Results/'+fun_name+'_TuRBO_rmse_N'+str(Ninit)+'_D'+str(dim)+'_L'+str(L)+'.pt')
                
                # NLPD_Iso = torch.load('/fs/ess/PAS2983/jontwt/AdaScale-TuRBO/GP_Accuracy/Results/'+fun_name+'_AdaTuRBO_rmse_N'+str(Ninit)+'_D'+str(dim)+'_L'+str(L)+'.pt')
                # NLPD_ARD = torch.load('/fs/ess/PAS2983/jontwt/AdaScale-TuRBO/GP_Accuracy/Results/'+fun_name+'_TuRBO_rmse_N'+str(Ninit)+'_D'+str(dim)+'_L'+str(L)+'.pt')
            
           
            # NLPD_list_ARD.append(NLPD_ARD.tolist())
            # NLPD_list_Iso.append(NLPD_Iso.tolist())
            
            NLPD_gamma_prior_list_ARD.append(NLPD_ARD_gamma_prior.tolist())
            NLPD_gamma_prior_list_Iso.append(NLPD_Iso_gamma_prior.tolist())

            
        # MAP
        NLPD_list_ARD = torch.tensor(NLPD_list_ARD)
        # NLPD_ARD_mean = NLPD_list_ARD.mean(dim=1)
        # NLPD_ARD_std = NLPD_list_ARD.std(dim=1)/replicate**0.5
        
        NLPD_list_Iso = torch.tensor(NLPD_list_Iso)
        # NLPD_Iso_mean = NLPD_list_Iso.mean(dim=1)
        # NLPD_Iso_std = NLPD_list_Iso.std(dim=1)/replicate**0.5
        
        NLPD_gamma_prior_list_ARD = torch.tensor(NLPD_gamma_prior_list_ARD)
        NLPD_gamma_prior_ARD_mean = NLPD_gamma_prior_list_ARD.mean(dim=1)
        NLPD_gamma_prior_ARD_std = NLPD_gamma_prior_list_ARD.std(dim=1)/replicate**0.5
        
        NLPD_gamma_prior_list_Iso = torch.tensor(NLPD_gamma_prior_list_Iso)
        NLPD_gamma_prior_Iso_mean = NLPD_gamma_prior_list_Iso.mean(dim=1)
        NLPD_gamma_prior_Iso_std = NLPD_gamma_prior_list_Iso.std(dim=1)/replicate**0.5
     
       
        
        # plt.figure(dpi=300)
        
        #MAP
        # axes[i,j].plot(x,NLPD_ARD_mean , label = 'ARD (MAP: LogNormal Prior)', color = 'blue')
        # axes[i,j].plot(x,NLPD_Iso_mean, label = 'Isotropic (MAP: LogNormal Prior)', color = 'red')
        # axes[i,j].scatter(x, NLPD_ARD_mean, color = 'blue')
        # axes[i,j].scatter(x, NLPD_Iso_mean, color = 'red')
        # axes[i,j].fill_between(x, NLPD_ARD_mean - NLPD_ARD_std, NLPD_ARD_mean + NLPD_ARD_std, alpha = 0.2, color = 'blue')
        # axes[i,j].fill_between(x, NLPD_Iso_mean - NLPD_Iso_std, NLPD_Iso_mean + NLPD_Iso_std, alpha = 0.2, color = 'red')
        
        axes[i,j].plot(x,NLPD_gamma_prior_list_ARD.mean(dim=1), label = 'TuRBO', color = 'green')
        axes[i,j].plot(x,NLPD_gamma_prior_list_Iso.mean(dim=1), label = 'AdaScale-TuRBO', color = 'orange')
        axes[i,j].scatter(x, NLPD_gamma_prior_ARD_mean, color = 'green')
        axes[i,j].scatter(x, NLPD_gamma_prior_Iso_mean, color = 'orange')
        axes[i,j].fill_between(x, NLPD_gamma_prior_ARD_mean - NLPD_gamma_prior_ARD_std, NLPD_gamma_prior_ARD_mean + NLPD_gamma_prior_ARD_std, alpha = 0.2, color = 'green')
        axes[i,j].fill_between(x, NLPD_gamma_prior_Iso_mean - NLPD_gamma_prior_Iso_std, NLPD_gamma_prior_Iso_mean + NLPD_gamma_prior_Iso_std, alpha = 0.2, color = 'orange')
        
        # plt.ylim(0.5,1.4)
        axes[i,j].set_xlabel('N', fontsize=fontsize)
        axes[i,j].set_ylabel(metric[j], fontsize=fontsize)
        axes[i,j].legend()
        axes[i,j].grid()
        axes[i,j].set_title(str(dim)+'D GP Prior', fontsize=fontsize)
        
plt.tight_layout()
# plt.savefig('Gamma_0.3.png')