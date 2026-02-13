#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 12:33:49 2025

@author: tang.1856
"""
import sys
import os
sys.path.append(os.path.abspath('/fs/ess/PAS2983/jontwt/AdaScale-TuRBO/src'))

import math
import warnings
from dataclasses import dataclass
from botorch.test_functions.synthetic import Ackley, Rosenbrock, StyblinskiTang, Griewank, Michalewicz, Rastrigin
import torch
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine
from gpytorch.means import ZeroMean
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
# from Generate_GP_prior_samples import PriorGP, GaussianProcessPriorSampler
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch_modules_new import (
    get_covar_module_with_dim_scaled_prior
)

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
dtype = torch.double

batch_size = 1
max_cholesky_size = float("inf")  # Always use Cholesky



@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def get_initial_points(dim, n_pts, seed=0):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init


def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=10,
    raw_samples=512,
    acqf="ts",  # "ei" or "ts"
):
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():  # We don't need gradients when using TS
            X_next = thompson_sampling(X_cand, num_samples=batch_size)

    elif acqf == "ei":
        ei = qLogExpectedImprovement(model, Y.max())
        X_next, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            # options={"sample_around_best": True}
        )

    return X_next

dim = 100
Ninit = 10
lb = -5.12
ub = 5.12
NUM_RESTARTS = 5 
RAW_SAMPLES = 20
N_CANDIDATES = min(5000, max(2000, 200 * dim))
fun = Rastrigin(dim=dim, negate=True)

replicate = 1
bo_iter = 1000

regret = [[] for _ in range(replicate)]
for seed in range(replicate):
    # torch.set_num_threads(4)
   
    bo = 0
    while bo<bo_iter:
        if bo == 0:
            X_turbo = torch.quasirandom.SobolEngine(dimension=dim,  scramble=True, seed=seed).draw(Ninit).to(device).to(torch.float64)  
        else:
            bo += Ninit
            X_turbo = torch.quasirandom.SobolEngine(dimension=dim,  scramble=True, seed = torch.randint(0,int(1e6),()).item()).draw(Ninit).to(device).to(torch.float64) 
        
        Y_turbo = fun(lb+(ub-lb)*X_turbo).detach().to(torch.float64).to(device).unsqueeze(1)
        state = TurboState(dim, batch_size=batch_size, best_value=max(Y_turbo).item())
        
        
        
        if bo == 0:
            regret[seed].append(float(max(Y_turbo)))
        else:
            regret[seed].append(float(max(max(Y_turbo), regret[seed][-1])))
            
        while not state.restart_triggered:  # Run until TuRBO converges
            # Fit a GP model
            train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
            
        
            # Do the fitting and acquisition function optimization inside the Cholesky context
            with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                if (len(regret[seed])-1)%10==0:
                    
                    covar_module = get_covar_module_with_dim_scaled_prior(
                        ard_num_dims=dim,
                        use_rbf_kernel=False,
                        dim = dim,
                    )
                    
                    model = SingleTaskGP(X_turbo, train_Y, covar_module = covar_module)
                    mll = ExactMarginalLogLikelihood(model.likelihood, model)        
                    
                    try:                      
                        fit_gpytorch_mll(mll)
                    except:
                        print('cant fit GP')
                
                        
                else:           
                    model.set_train_data(inputs=X_turbo, targets=train_Y.flatten(), strict=False)
                
                # Create a batch
                X_next = generate_batch(
                    state=state,
                    model=model,
                    X=X_turbo,
                    Y=train_Y,
                    batch_size=batch_size,
                    n_candidates=N_CANDIDATES,
                    num_restarts=NUM_RESTARTS,
                    raw_samples=RAW_SAMPLES,
                    acqf="ei",
                )  
                
            bo+=1
         
            Y_next = fun(lb+(ub-lb)*X_next).unsqueeze(-1)
        
            # Update state
            state = update_state(state=state, Y_next=Y_next)
        
            # Append data
            X_turbo = torch.cat((X_turbo, X_next), dim=0)
            Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)
            print(f"{len(X_turbo)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}")
            
            regret[seed].append(float(max(max(Y_turbo), regret[seed][-1])))
            if bo>=bo_iter:
                bo = 1e6
                break

    for element in regret[0:seed+1]:
        while len(element)<=bo_iter:
            element.append(element[-1])
        
    torch.save(torch.tensor(regret[0:seed+1]), 'Turbo_lognormal_'+str(dim)+'D_Rastrigin_regret.pt')     
    
    
    
    
    
    













