#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 15:50:53 2025

@author: jontwt
"""

from collections.abc import Sequence
from math import log, sqrt
import torch
import gpytorch
from gpytorch.constraints.constraints import GreaterThan, Interval
from gpytorch.kernels import MaternKernel, LinearKernel, ScaleKernel
from gpytorch.priors.torch_priors import LogNormalPrior

SQRT2 = sqrt(2)
SQRT3 = sqrt(3)

class MixedMaternLinearKernel(gpytorch.kernels.Kernel):
    def __init__(
        self,
        ard_num_dims: int,
        batch_shape: torch.Size | None = None,
        active_dims: Sequence[int] | None = None,
        dim: float = 1.0,
        length: float = 1.0
    ):
        super().__init__(batch_shape=batch_shape, active_dims=active_dims)

        # 1. Setup the Matern Part
        lengthscale_prior = LogNormalPrior(loc=SQRT2 + log(dim**0.5*length), scale=SQRT3)
        self.matern_kernel = MaternKernel(
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=GreaterThan(
                2.5e-2, transform=None, initial_value=lengthscale_prior.mode
            ),
        )

        # 2. Setup the Linear Part
        self.linear_kernel = LinearKernel(
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            # variance_constraint=Interval(1e-5, 10)
        )
        # Fix variance to 1.0 and prevent optimization
        self.linear_kernel.variance = torch.ones(batch_shape if batch_shape else (1,))
        self.linear_kernel.raw_variance.requires_grad_(False)

        # 3. Setup the Alpha mixing parameter (constrained between 0 and 1)
        # We register it as a parameter so GPyTorch/PyTorch can optimize it
        self.register_parameter(
            name="raw_alpha", 
            parameter=torch.nn.Parameter(torch.zeros(*batch_shape if batch_shape else (), 1))
        )
        self.register_constraint("raw_alpha", Interval(0, 1))

    @property
    def alpha(self):
        # This applies the Sigmoid/Interval transformation to the raw parameter
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value):
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        # Calculate individual kernels
        k_matern = self.matern_kernel(x1, x2, diag=diag, **params)
        k_linear = self.linear_kernel(x1, x2, diag=diag, **params)
        
        # Convex combination: alpha * Matern + (1 - alpha) * Linear
        res = (self.alpha * k_matern) + ((1 - self.alpha) * k_linear)
        return res

