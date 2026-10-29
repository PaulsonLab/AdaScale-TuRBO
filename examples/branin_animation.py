#!/usr/bin/env python3
"""
Run AdaScale-TuRBO (or standard TuRBO, for comparison) on the 2D Branin function
and animate the optimization: evaluated points, the trust region, and the incumbent
at every iteration.

Usage:
    python examples/branin_animation.py [--method adascale|turbo] [--seed 3] [--n-init 5]
                                        [--n-iter 50] [--tol 1e-3] [--length-init 0.4]
                                        [--fps 2]

Both methods use the same initial points, TR rules, and LogEI acquisition; they differ
only in the GP surrogate:
  adascale  Matern-5/2 with the LogNormal lengthscale prior scaled by L*sqrt(D), fit by MAP
  turbo     Scale(Matern-5/2) with box-constrained lengthscales, fit by MLE
            (the BoTorch TuRBO tutorial setup)

The run stops early once the regret f_best - f* drops below --tol.
"""
import argparse
import math
import os
import sys

import gpytorch
import matplotlib
import numpy as np
import torch
from botorch.acquisition import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions.synthetic import Branin
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from AdaScale_TuRBO_optimize import (  # noqa: E402
    TurboState,
    dtype,
    device,
    get_initial_points,
    max_cholesky_size,
    update_state,
)
from gpytorch_modules_new import get_covar_module_with_dim_scaled_prior  # noqa: E402

BRANIN_OPTIMA = np.array([[-math.pi, 12.275], [math.pi, 2.275], [9.42478, 2.475]])
BRANIN_MIN = 0.397887
METHOD_NAMES = {"adascale": "AdaScale-TuRBO", "turbo": "TuRBO"}


def fit_model(method, X, train_Y, length):
    dim = X.shape[-1]
    if method == "adascale":
        covar_module = get_covar_module_with_dim_scaled_prior(
            ard_num_dims=dim, use_rbf_kernel=False, length=length, dim=dim
        )
        model = SingleTaskGP(X, train_Y, covar_module=covar_module)
    else:
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(MaternKernel(
            nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)
        ))
        model = SingleTaskGP(X, train_Y, covar_module=covar_module, likelihood=likelihood)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


def lengthscale(model):
    kernel = model.covar_module
    return (kernel.base_kernel if isinstance(kernel, ScaleKernel) else kernel).lengthscale


def trust_region_bounds(state, model, X, Y):
    """Same TR construction as `generate_batch` in src/AdaScale_TuRBO_optimize.py."""
    x_center = X[Y.argmax(), :].clone()
    weights = lengthscale(model).squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)
    return tr_lb, tr_ub


def run(method, seed, n_init, n_iter, tol, length_init):
    torch.manual_seed(seed)
    fun = Branin(negate=True).to(dtype=dtype, device=device)  # maximize -Branin
    lb, ub = fun.bounds[0], fun.bounds[1]
    dim = 2

    def evaluate(X):
        return fun(lb + (ub - lb) * X).unsqueeze(-1)

    X = get_initial_points(dim, n_init, seed=seed)
    Y = evaluate(X)
    state = TurboState(dim, batch_size=1, length=length_init, best_value=Y.max().item())

    history = []  # one entry per BO iteration
    restarted = False
    for it in range(n_iter):
        train_Y = (Y - Y.mean()) / Y.std()
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            model = fit_model(method, X, train_Y, state.length)
            tr_lb, tr_ub = trust_region_bounds(state, model, X, train_Y)
            # Same as the acqf="ei" branch of `generate_batch`
            X_next, _ = optimize_acqf(
                qLogExpectedImprovement(model, train_Y.max()),
                bounds=torch.stack([tr_lb, tr_ub]), q=1, num_restarts=5, raw_samples=200,
            )
        Y_next = evaluate(X_next)

        history.append(dict(
            length=state.length,
            restarted=restarted,
            tr_lb=(lb + (ub - lb) * tr_lb).cpu().numpy(),
            tr_ub=(lb + (ub - lb) * tr_ub).cpu().numpy(),
            center=(lb + (ub - lb) * X[train_Y.argmax()]).cpu().numpy(),
            x_next=(lb + (ub - lb) * X_next[0]).cpu().numpy(),
        ))

        state = update_state(state=state, Y_next=Y_next)
        X = torch.cat((X, X_next), dim=0)
        Y = torch.cat((Y, Y_next), dim=0)
        print(f"iter {it + 1:3d}  L = {history[-1]['length']:.4f}  "
              f"f(x) = {-Y_next.item():8.4f}  best = {-Y.max().item():.6f}")

        if -Y.max().item() - BRANIN_MIN < tol:
            print(f"  Regret below {tol:g} after iteration {it + 1}; stopping")
            break

        restarted = state.restart_triggered
        if restarted:  # TR collapsed: restart from L0 around the incumbent
            print(f"  TR side length below L_min after iteration {it + 1}; resetting L to {length_init}")
            state = TurboState(dim, batch_size=1, length=length_init,
                               best_value=Y.max().item())

    X_all = (lb + (ub - lb) * X).cpu().numpy()
    f_all = (-Y.squeeze(-1)).cpu().numpy()  # Branin values (minimization)
    return X_all, f_all, history, lb.cpu().numpy(), ub.cpu().numpy()


def _style_axes(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")


def _branin_grid(xlim, ylim, n):
    g1, g2 = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    grid = torch.tensor(np.stack([g1.ravel(), g2.ravel()], -1))
    return g1, g2, Branin()(grid).numpy().reshape(g1.shape)


def animate(X_all, f_all, history, lb, ub, n_init, tol, out_path, name, fps=2):
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.titlesize": 13,
        "axes.labelsize": 12,
    })

    n_iter = len(history)
    regret = np.minimum.accumulate(f_all) - BRANIN_MIN
    full_lim = ((lb[0], ub[0]), (lb[1], ub[1]))
    g1, g2, F = _branin_grid(*full_lim, n=400)
    log_levels = np.linspace(np.log(F.min()), np.log(F.max()), 40)

    fig = plt.figure(figsize=(11, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.1], left=0.06, right=0.98,
                          top=0.86, bottom=0.2, wspace=0.25)
    ax, ax_r = (fig.add_subplot(gs[i]) for i in range(2))

    # Full domain
    ax.contour(g1, g2, np.log(F), levels=log_levels, cmap="Greens_r", linewidths=0.8)
    ax.scatter(*BRANIN_OPTIMA.T, marker="*", s=260, c="green", edgecolor="white",
               linewidth=0.8, zorder=4)
    _style_axes(ax, *full_lim)
    ax.set_aspect("equal")
    ax.set_title("Search space")

    tr_patch = Rectangle((0, 0), 0, 0, facecolor=(1, 0, 0, 0.06), edgecolor="red",
                         linewidth=2.2, zorder=5)
    ax.add_patch(tr_patch)
    pts = ax.scatter([], [], s=28, c="black", zorder=6)
    pt_new = ax.scatter([], [], s=140, facecolor="none", edgecolor="darkorange",
                        linewidth=2.2, zorder=7)
    pt_best = ax.scatter([], [], s=55, c="red", marker="s", edgecolor="white",
                         linewidth=0.6, zorder=8)

    # Regret curve
    evals = np.arange(1, len(f_all) + 1)
    ax_r.axvspan(0.5, n_init + 0.5, color="0.92", zorder=0)
    ax_r.text(n_init / 2 + 0.5, 0.03, "Sobol init", transform=ax_r.get_xaxis_transform(),
              ha="center", va="bottom", fontsize=10, color="0.35")
    restart_markers = []  # (iteration, artists), revealed once the animation reaches them
    for i, h in enumerate(history):
        if h["restarted"]:
            line = ax_r.axvline(n_init + i + 1, color="red", linestyle=":", linewidth=1.2)
            label = ax_r.text(n_init + i + 1.5, 0.97, "TR restart", ha="left", va="top",
                              transform=ax_r.get_xaxis_transform(), fontsize=10, color="red")
            restart_markers.append((i + 1, (line, label)))
    ax_r.axhline(tol, color="green", linestyle="--", linewidth=1.2)
    ax_r.text(n_init + 1, tol * 1.15, "stopping tolerance", ha="left", va="bottom",
              fontsize=10, color="green")
    (curve,) = ax_r.plot([], [], color="black", linewidth=1.8, drawstyle="steps-post")
    ax_r.set_xlim(0.5, len(f_all) + 0.5)
    ax_r.set_yscale("log")
    ax_r.set_ylim(min(regret.min(), tol) / 3, regret.max() * 2)
    ax_r.set_xlabel("Function evaluations")
    ax_r.set_ylabel(r"Regret  $f_{\mathrm{best}} - f^*$")
    ax_r.set_title("Convergence")
    ax_r.grid(alpha=0.3, which="both")

    title = fig.suptitle("", fontsize=15)
    fig.legend(
        handles=[
            Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="red", linewidth=2),
            Line2D([], [], marker="*", color="green", linestyle="", markersize=15),
            Line2D([], [], marker="o", color="black", linestyle="", markersize=6),
            Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="darkorange",
                   markeredgewidth=2, linestyle="", markersize=11),
            Line2D([], [], marker="s", color="red", linestyle="", markersize=7),
        ],
        labels=["Trust region", "Global optima", "Evaluated points", "Newest point",
                "Best point (TR center)"],
        loc="lower center", ncol=5, frameon=True, fontsize=11, handletextpad=0.5,
        columnspacing=1.8,
    )

    hold = 2 * fps  # pause on the last frame

    def update(frame):
        k = min(frame, n_iter)
        n_seen = n_init + k
        h = history[max(k - 1, 0)]
        i_best = np.argmin(f_all[:n_seen])
        new = h["x_next"][None] if k > 0 else np.empty((0, 2))

        pts.set_offsets(X_all[:n_seen])
        pt_new.set_offsets(new)
        pt_best.set_offsets(X_all[i_best][None])
        tr_lb, tr_ub = h["tr_lb"], h["tr_ub"]
        tr_patch.set_xy(tr_lb)
        tr_patch.set_width(tr_ub[0] - tr_lb[0])
        tr_patch.set_height(tr_ub[1] - tr_lb[1])
        tr_patch.set_visible(k > 0)

        curve.set_data(np.append(evals[:n_seen], n_seen + 1),
                       np.append(regret[:n_seen], regret[n_seen - 1]))
        for it, artists in restart_markers:
            for a in artists:
                a.set_visible(k >= it)

        if k == 0:
            title.set_text(f"{name} on Branin: initial design ({n_init} Sobol points)")
        else:
            note = "  (TR restarted)" if h["restarted"] else ""
            if k == n_iter and regret[n_seen - 1] < tol:
                note = "  (converged)"
            title.set_text(
                f"{name} on Branin: iteration {k},  $L = {h['length']:.3g}$,  "
                f"best $f = {f_all[i_best]:.5f}$  ($f^* = {BRANIN_MIN}$){note}"
            )

    anim = FuncAnimation(fig, update, frames=n_iter + 1 + hold, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=90)
    update(n_iter)
    fig.savefig(os.path.splitext(out_path)[0] + "_final.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--method", choices=list(METHOD_NAMES), default="adascale")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-3, help="stop once f_best - f* < tol")
    parser.add_argument("--length-init", type=float, default=0.4,
                        help="initial TR side length (TuRBO default 0.8)")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--out", help="output GIF (default: examples/branin_<method>.gif)")
    args = parser.parse_args()
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "branin_adascale_turbo.gif" if args.method == "adascale" else "branin_turbo.gif",
    )

    X_all, f_all, history, lb, ub = run(
        args.method, args.seed, args.n_init, args.n_iter, args.tol, args.length_init
    )
    animate(X_all, f_all, history, lb, ub, args.n_init, args.tol, out,
            METHOD_NAMES[args.method], fps=args.fps)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
