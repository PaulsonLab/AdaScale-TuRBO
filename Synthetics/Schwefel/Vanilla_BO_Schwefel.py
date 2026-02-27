#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:32:57 2025

@author: tang.1856
"""
from botorch.test_functions.synthetic import Ackley, Rosenbrock, StyblinskiTang, Powell, Griewank, Rastrigin
import torch
# from Acquisition_Newton import GradientInformation, optimize_acqf_custom_bo
from botorch.models.transforms import Normalize, Standardize
# from src.cholesky import one_step_cholesky
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
# from src.model import DerivativeExactGPSEModel
import gpytorch
import botorch 
from botorch.models import SingleTaskGP
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf

def fun(x):
    return -(418.9829*dim - torch.sum(x*torch.sin(torch.sqrt(torch.abs(x))), dim=1))

dim = 50

Ninit = 10
lb = -500
ub = 500
bo_iter = 1000
replicate =10
# fun = StyblinskiTang(dim=dim, negate=True)
regret_y, cost_list = [[] for _ in range(replicate)], [[] for _ in range(replicate)]
for seed in range(replicate):
    # torch.set_num_threads(4)
    # torch.manual_seed(seed)
    # params = torch.rand(1, dim).to(torch.float64)
    train_X = torch.quasirandom.SobolEngine(dimension=dim,  scramble=True, seed=seed).draw(Ninit).to(torch.float64)
    # train_X = torch.cat((params, train_X))  
    train_Y = fun(lb+(ub-lb)*train_X).unsqueeze(1)
    
    cost_list[seed].append(train_X.shape[0])
    regret_y[seed].append(float(min(-train_Y)))
    for bo in range(bo_iter):
        print('iter = ', bo)
        train_Y_scaled = (train_Y - train_Y.mean(dim=0)) / train_Y.std(dim=0)
        
            
        if bo%10==0:
            gp = SingleTaskGP(
              train_X=train_X.to(torch.float64),
              train_Y=train_Y_scaled.to(torch.float64),
              # outcome_transform=Standardize(m=1),
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
        bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
        x_next, acq_value = optimize_acqf(
          logEI, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
        )
        Y_next = torch.cat([fun(lb + (ub - lb) *x_next)]).unsqueeze(1)   
        regret_y[seed].append(min(-float(Y_next[0]), regret_y[seed][-1]))
        
        train_X = torch.cat((x_next,train_X))
        train_Y = torch.cat((Y_next,train_Y))   
        cost_list[seed].append(train_X.shape[0])

    torch.save(torch.tensor(regret_y[0:seed+1]), str(dim)+'D_Schwefel_Vanilla_regret.pt')
