#!/usr/bin/env python3
"""
Run AdaScale-TuRBO on the 2D Branin function and animate the optimization:
evaluated points, the trust region, and the incumbent at every iteration.

Usage:
    python examples/branin_animation.py [--seed 3] [--n-init 5] [--n-iter 50] [--tol 1e-3]

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
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.test_functions.synthetic import Branin
from gpytorch.mlls import ExactMarginalLogLikelihood

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from AdaScale_TuRBO_optimize import (  # noqa: E402
    TurboState,
    dtype,
    device,
    generate_batch,
    get_initial_points,
    max_cholesky_size,
    update_state,
)
from gpytorch_modules_new import get_covar_module_with_dim_scaled_prior  # noqa: E402

BRANIN_OPTIMA = np.array([[-math.pi, 12.275], [math.pi, 2.275], [9.42478, 2.475]])
BRANIN_MIN = 0.397887


def trust_region_bounds(state, model, X, Y):
    """Same TR construction as `generate_batch`, returned for plotting."""
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)
    return tr_lb, tr_ub


def run(seed, n_init, n_iter, tol):
    torch.manual_seed(seed)
    fun = Branin(negate=True).to(dtype=dtype, device=device)  # maximize -Branin
    lb, ub = fun.bounds[0], fun.bounds[1]
    dim = 2

    def evaluate(X):
        return fun(lb + (ub - lb) * X).unsqueeze(-1)

    X = get_initial_points(dim, n_init, seed=seed)
    Y = evaluate(X)
    state = TurboState(dim, batch_size=1, best_value=Y.max().item())

    history = []  # one entry per BO iteration
    restarted = False
    for it in range(n_iter):
        train_Y = (Y - Y.mean()) / Y.std()
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            covar_module = get_covar_module_with_dim_scaled_prior(
                ard_num_dims=dim, use_rbf_kernel=False, length=state.length, dim=dim
            )
            model = SingleTaskGP(X, train_Y, covar_module=covar_module)
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))

            tr_lb, tr_ub = trust_region_bounds(state, model, X, train_Y)
            X_next = generate_batch(
                state=state, model=model, X=X, Y=train_Y, batch_size=1,
                num_restarts=5, raw_samples=200, acqf="ei",
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
            print(f"  TR side length below L_min after iteration {it + 1}; resetting L to 0.8")
            state = TurboState(dim, batch_size=1, best_value=Y.max().item())

    X_all = (lb + (ub - lb) * X).cpu().numpy()
    f_all = (-Y.squeeze(-1)).cpu().numpy()  # Branin values (minimization)
    return X_all, f_all, history, lb.cpu().numpy(), ub.cpu().numpy()


def _style_axes(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")


def _branin_grid(xlim, ylim, n=300):
    g1, g2 = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    grid = torch.tensor(np.stack([g1.ravel(), g2.ravel()], -1))
    return g1, g2, Branin()(grid).numpy().reshape(g1.shape)


def animate(X_all, f_all, history, lb, ub, n_init, tol, out_path, fps=3):
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

    fig = plt.figure(figsize=(15, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], left=0.045, right=0.985,
                          top=0.86, bottom=0.2, wspace=0.28)
    ax, ax_z, ax_r = (fig.add_subplot(gs[i]) for i in range(3))

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

    connectors = []
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

        # Zoom on the trust region (the full domain before the first TR exists)
        ax_z.clear()
        if k > 0:
            pad = 0.15 * (tr_ub - tr_lb)
            z_lb, z_ub = np.maximum(tr_lb - pad, lb), np.minimum(tr_ub + pad, ub)
        else:
            z_lb, z_ub = lb, ub
        zg1, zg2, zF = _branin_grid((z_lb[0], z_ub[0]), (z_lb[1], z_ub[1]), n=200)
        ax_z.contour(zg1, zg2, np.log(zF), levels=25, cmap="Greens_r", linewidths=0.8)
        if k > 0:
            ax_z.add_patch(Rectangle(tr_lb, *(tr_ub - tr_lb), facecolor=(1, 0, 0, 0.06),
                                     edgecolor="red", linewidth=2.2, zorder=5))
        ax_z.scatter(*X_all[:n_seen].T, s=28, c="black", zorder=6)
        ax_z.scatter(*new.T, s=140, facecolor="none", edgecolor="darkorange",
                     linewidth=2.2, zorder=7)
        ax_z.scatter(*X_all[i_best], s=55, c="red", marker="s", edgecolor="white",
                     linewidth=0.6, zorder=8)
        ax_z.scatter(*BRANIN_OPTIMA.T, marker="*", s=260, c="green", edgecolor="white",
                     linewidth=0.8, zorder=4)
        _style_axes(ax_z, (z_lb[0], z_ub[0]), (z_lb[1], z_ub[1]))
        ax_z.set_title("Trust region (zoomed)" if k > 0 else "Initial design")
        for side in ax_z.spines.values():
            side.set_edgecolor("red" if k > 0 else "black")
            side.set_linewidth(2.2 if k > 0 else 0.8)

        # Dashed connectors from the TR in the full view to the zoom panel
        for c in connectors:
            c.remove()
        connectors.clear()
        if k > 0:
            for xy_a, xy_b in (((tr_ub[0], tr_ub[1]), (0, 1)), ((tr_ub[0], tr_lb[1]), (0, 0))):
                c = ConnectionPatch(xyA=xy_a, coordsA=ax.transData, xyB=xy_b,
                                    coordsB=ax_z.transAxes, color="red", linestyle="--",
                                    linewidth=1.3)
                fig.add_artist(c)
                connectors.append(c)

        curve.set_data(np.append(evals[:n_seen], n_seen + 1),
                       np.append(regret[:n_seen], regret[n_seen - 1]))
        for it, artists in restart_markers:
            for a in artists:
                a.set_visible(k >= it)

        if k == 0:
            title.set_text(f"AdaScale-TuRBO on Branin: initial design ({n_init} Sobol points)")
        else:
            note = "  (TR restarted)" if h["restarted"] else ""
            if k == n_iter and regret[n_seen - 1] < tol:
                note = "  (converged)"
            title.set_text(
                f"AdaScale-TuRBO on Branin: iteration {k},  $L = {h['length']:.3g}$,  "
                f"best $f = {f_all[i_best]:.5f}$  ($f^* = {BRANIN_MIN}$){note}"
            )

    anim = FuncAnimation(fig, update, frames=n_iter + 1 + hold, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=90)
    update(n_iter)
    fig.savefig(os.path.splitext(out_path)[0] + "_final.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-3, help="stop once f_best - f* < tol")
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument(
        "--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "branin_adascale_turbo.gif"),
    )
    args = parser.parse_args()

    X_all, f_all, history, lb, ub = run(args.seed, args.n_init, args.n_iter, args.tol)
    animate(X_all, f_all, history, lb, ub, args.n_init, args.tol, args.out, fps=args.fps)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
