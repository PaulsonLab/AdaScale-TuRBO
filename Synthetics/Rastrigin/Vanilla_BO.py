#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:32:57 2025

@author: tang.1856
"""
from botorch.test_functions.synthetic import Ackley, Rosenbrock, StyblinskiTang, Powell, Griewank, Rastrigin
import torch
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch
import botorch 
from botorch.models import SingleTaskGP
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_lognormal_prior,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

dim = 50
fun = Rastrigin(dim=dim, negate=True).to(dtype=dtype, device=device)
Ninit = 10
lb = -5.12
ub = 5.12
NUM_RESTARTS = 5 
RAW_SAMPLES = 20
bo_iter = 1000
replicate = 10

regret_y = [[] for _ in range(replicate)]
for seed in range(replicate):
    # torch.set_num_threads(4)

    train_X = torch.quasirandom.SobolEngine(dimension=dim,  scramble=True, seed=seed).draw(Ninit).to(dtype=dtype, device=device)
    train_Y = fun(lb+(ub-lb)*train_X).unsqueeze(1)
    
    regret_y[seed].append(float(min(-train_Y)))
    for bo in range(bo_iter):
        print('iter = ', bo)
        train_Y_scaled = (train_Y - train_Y.mean(dim=0)) / train_Y.std(dim=0)
        
        covar_module = get_covar_module_with_dim_scaled_prior(
            ard_num_dims=dim,
            use_rbf_kernel=False,
        )
            
        if bo%10==0:
            gp = SingleTaskGP(
              train_X=train_X.to(torch.float64),
              train_Y=train_Y_scaled.to(torch.float64),
              covar_module=covar_module    
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            
            try:
                fit_gpytorch_mll(mll)
            except:
                print('cant fit GP')
        else:
            gp.set_train_data(
                inputs=train_X,
                targets=train_Y_scaled.flatten(),
                strict=False,
            )
            
        logEI = LogExpectedImprovement(model=gp, best_f=train_Y_scaled.max())
        bounds = torch.stack([torch.zeros(dim), torch.ones(dim)]).to(dtype=dtype, device=device)
        x_next, acq_value = optimize_acqf(
          logEI, bounds=bounds, q=1, num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,
        )
        Y_next = torch.cat([fun(lb + (ub - lb) *x_next)]).unsqueeze(1)   
        regret_y[seed].append(min(-float(Y_next[0]), regret_y[seed][-1]))
        
        train_X = torch.cat((x_next,train_X))
        train_Y = torch.cat((Y_next,train_Y))   
    

    torch.save(torch.tensor(regret_y[0:seed+1]), str(dim)+'D_Rastrigin_Vanilla_regret.pt')
