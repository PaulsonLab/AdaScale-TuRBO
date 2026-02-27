#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 12:56:38 2026

@author: jontwt
"""

from botorch.acquisition import AcquisitionFunction, AnalyticAcquisitionFunction
import torch
from botorch.models.model import Model
from typing import Dict, Optional, Tuple, Union
from torch import Tensor
from botorch.acquisition.objective import PosteriorTransform
# from botorch.utils.transforms import convert_to_target_pre_hook, t_batch_mode_transform
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
import numpy as np
from scipy.spatial.distance import cdist, jensenshannon
from botorch.models.transforms.outcome import Standardize
from botorch.test_functions import Rosenbrock, Ackley, Hartmann, StyblinskiTang, Rastrigin, Michalewicz
import gpytorch
import time
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
import cma

def fun(x):
    # x = x.unsqueeze(0)
    return (418.9829*dim - torch.sum(x*torch.sin(torch.sqrt(torch.abs(x))), dim=1))

dim = 50
replicate = 10
N_init = 1
lb = -500
ub = 500
BO_iter = 500
sigma0 = 0.5
# fun = Michalewicz(dim = dim)
reachability_list = [[] for _ in range(replicate)]

for seed in range(replicate):
    np.random.seed(seed)
    
    train_X = lb + (ub-lb)*torch.tensor(np.random.rand(N_init, dim))
    train_Y = fun(train_X)
    
    reachability_list[seed].append(float(train_Y))
    
    x0 = train_X[torch.argmin(train_Y)].numpy()
    
    es = cma.CMAEvolutionStrategy(x0, sigma0)
    while not es.stop():
        if len(train_Y)>BO_iter-1:
            break
        
        solutions = es.ask()
        es.tell(solutions, [float(fun(x.unsqueeze(0))) for x in torch.clamp(torch.tensor(solutions),lb,ub)])
        new_y = fun(torch.clamp(torch.tensor(solutions),lb,ub))
        train_Y = torch.cat((train_Y, new_y))
        es.logger.add()
        es.disp()
        
       
        reachability_list[seed].extend([min(reachability_list[seed][-1],float(train_Y[-1]))]*new_y.shape[0])
        
        
        
  
torch.save(torch.tensor(reachability_list)[:,0:BO_iter], 'CMAES_'+str(dim)+'D_Schwefel_regret.pt')   

