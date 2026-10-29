#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jontwt
"""
import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from torch import Tensor


class Schwefel(SyntheticTestFunction):
    r"""Negated Schwefel synthetic test function (to be maximized).

    d-dimensional function (usually evaluated on `[-500, 500]^d`):

        f(x) = -(418.9829 d - sum_{i=1}^{d} x_i sin(sqrt(|x_i|)))

    f has its global maximum at `x ~= (420.9687, ..., 420.9687)` with
    `f(x) ~= 0.0`.
    """

    _optimal_value = 0.0

    def __init__(
        self,
        dim=None,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
            dtype: The dtype that is used for the bounds of the function.
        """
        self.dim = dim
        self.continuous_inds = list(range(dim))
        if bounds is None:
            bounds = [(-500, 500) for _ in range(self.dim)]
        self._optimizers = [tuple(420.9687 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, dtype=dtype)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        return -(418.9829 * self.dim - torch.sum(X * torch.sin(torch.sqrt(torch.abs(X))), dim=-1))
