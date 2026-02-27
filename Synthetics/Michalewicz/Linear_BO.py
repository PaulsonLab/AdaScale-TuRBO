#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:32:57 2025

@author: tang.1856
"""
from botorch.test_functions.synthetic import Michalewicz
import torch
# from Acquisition_Newton import GradientInformation, optimize_acqf_custom_bo
from botorch.models.transforms import Normalize, Standardize
# from src.cholesky import one_step_cholesky
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
# from src.model import DerivativeExactGPSEModel
from botorch.generation import gen_candidates_scipy
import gpytorch
import botorch 
from botorch.models import SingleTaskGP
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils import standardize
import sys
sys.path.append('/fs/ess/PAS2983/jontwt/AdaScale-TuRBO/src')

from optimize import fit_model, initialize_model


dim = 50
fun = Michalewicz(dim=dim, negate=True)
Ninit = 10
lb = 0
ub = 3.14
NUMRESTART = 5
RAW_SAMPLES = 20
bo_iter = 1000
replicate = 10

regret_y = [[] for _ in range(replicate)]
for seed in range(replicate):
    # torch.set_num_threads(2)
    
    train_X = torch.quasirandom.SobolEngine(dimension=dim,  scramble=True, seed=seed).draw(Ninit).to(torch.float64)  
    train_Y = fun(lb+(ub-lb)*train_X).unsqueeze(1)
    
    regret_y[seed].append(float(min(-train_Y)))
    for bo in range(bo_iter):
        print('iter = ', bo)
        train_Y_scaled = (train_Y - train_Y.mean(dim=0)) / train_Y.std(dim=0)
 
        if bo%1==0:
            gp = initialize_model(X_train = train_X, Y_train = train_Y_scaled)
            fit_model(gp)             
        else:
            gp.set_train_data(
                inputs=train_X,
                targets=train_Y_scaled.flatten(),
                strict=False,
            )
            # gp.likelihood.noise_covar.set_training_data(train_Y_scaled, strict=False)
        gp.eval()    
        logEI = LogExpectedImprovement(model=gp, best_f=train_Y_scaled.max())
        options = {
            "raw_samples": RAW_SAMPLES,
            "num_restarts": NUMRESTART,
            "retry_on_optimization_warning": False,
            "options": {
                "nonnegative": False,
                "sample_around_best": True,
                "sample_around_best_sigma": 0.1,
                "maxiter": 300,
                "batch_limit": 64,
            },
        }
        bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
        x_next, acq_value = optimize_acqf(
          logEI, bounds=bounds, q=1, gen_candidates=gen_candidates_scipy, **options
        )
        Y_next = torch.cat([fun(lb + (ub - lb) *x_next)]).unsqueeze(1)   
        regret_y[seed].append(min(-float(Y_next[0]), regret_y[seed][-1]))
        
        train_X = torch.cat((x_next,train_X))
        train_Y = torch.cat((Y_next,train_Y))   
       

    torch.save(torch.tensor(regret_y[0:seed+1]), str(dim)+'D_Michalewicz_linear_BO_regret.pt')
