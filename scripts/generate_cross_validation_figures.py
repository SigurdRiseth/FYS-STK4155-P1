"""
Generate report figures for the cross-validation experiment (part d): k-fold CV
test MSE vs. polynomial degree for OLS (compared against the bootstrap estimate
from part c), and vs. degree x lambda for Ridge.

Usage:
    uv run python scripts/generate_cross_validation_figures.py [--degree-max 15]
        [--n 100] [--noise 0.1] [--seed 42] [--n-bootstraps 100]
        [--k-values 5 10] [--lambda-min -8] [--lambda-max -1] [--n-lambdas 15]
        [--out-subdir cross_validation]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from utils.plotting import FIGSIZE_WIDE, save_figure, set_style

from fys_stk4155_p1.data.runge import generate_runge_data
from fys_stk4155_p1.regression.ordinary_least_squares import OLS
from fys_stk4155_p1.resampling.bootstrap import bootstrap_bias_variance_sweep
from fys_stk4155_p1.resampling.cross_validation import kfold_mse_degree_sweep, kfold_mse_ridge_grid

# One color per k, shared between the OLS comparison plot and any future per-k
# annotation, so k=5 and k=10 read consistently across figures.
_K_COLORS = {5: "tab:blue", 10: "tab:orange"}


def plot_ols_cv_vs_bootstrap(
    bv_results: dict,
    cv_results: dict[int, dict],
    k_values: list[int],
    noise_std: float,
) -> Figure:
    fig, ax = plt.subplots()

    ax.plot(
        bv_results["degrees"],
        bv_results["mse_test"],
        "o-",
        markersize=3,
        color="black",
        label="bootstrap test error",
    )
    bv_best_degree = bv_results["degrees"][np.argmin(bv_results["mse_test"])]
    ax.axvline(bv_best_degree, color="black", linestyle="--", linewidth=1, alpha=0.6)

    for k in k_values:
        res = cv_results[k]
        color = _K_COLORS.get(k, None)
        ax.errorbar(
            res["degrees"],
            res["mse_mean"],
            yerr=res["mse_std"],
            fmt="s-",
            markersize=3,
            color=color,
            capsize=2,
            label=f"{k}-fold CV",
        )
        best_k = res["degrees"][np.argmin(res["mse_mean"])]
        ax.axvline(best_k, color=color, linestyle="--", linewidth=1, alpha=0.6)

    ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("MSE")
    ax.set_title(rf"OLS: bootstrap vs. $k$-fold CV test error ($\sigma={noise_std}$)")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_ridge_cv_heatmap(
    cv_results: dict[int, dict], k_values: list[int], noise_std: float
) -> Figure:
    vmin = min(cv_results[k]["mse_mean"].min() for k in k_values)
    vmax = max(cv_results[k]["mse_mean"].max() for k in k_values)

    fig, axes = plt.subplots(1, len(k_values), figsize=(8.0, 3.2), sharey=True)
    meshes = []
    for ax, k in zip(axes, k_values, strict=True):
        res = cv_results[k]
        mesh = ax.pcolormesh(
            res["degrees"],
            res["lambdas"],
            res["mse_mean"].T,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            shading="nearest",
            cmap="viridis_r",
        )
        meshes.append(mesh)
        best_i, best_j = np.unravel_index(np.argmin(res["mse_mean"]), res["mse_mean"].shape)
        ax.scatter(
            res["degrees"][best_i],
            res["lambdas"][best_j],
            marker="*",
            s=150,
            color="red",
            edgecolor="black",
            linewidth=0.5,
            label=f"best: degree={res['degrees'][best_i]}, $\\lambda$={res['lambdas'][best_j]:.1e}",
        )
        ax.set_yscale("log")
        ax.set_xlabel("Polynomial degree")
        ax.set_title(f"{k}-fold CV")
        ax.legend(frameon=False, loc="upper left", fontsize=7)

    axes[0].set_ylabel(r"$\lambda$")
    fig.colorbar(meshes[-1], ax=axes, label="mean CV MSE", pad=0.02)
    fig.suptitle(rf"Ridge: CV MSE over degree $\times$ $\lambda$ ($\sigma={noise_std}$)")
    return fig


def plot_ridge_cv_lines(
    cv_results: dict[int, dict], lambdas: np.ndarray, k_values: list[int], noise_std: float
) -> Figure:
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(lambdas)))

    fig, axes = plt.subplots(1, len(k_values), figsize=FIGSIZE_WIDE, sharey=True)
    for ax, k in zip(axes, k_values, strict=True):
        res = cv_results[k]
        for j, (lam, color) in enumerate(zip(res["lambdas"], colors, strict=True)):
            ax.plot(
                res["degrees"],
                res["mse_mean"][:, j],
                marker="o",
                markersize=2,
                color=color,
                label=rf"$\lambda={lam:.0e}$" if j % 3 == 0 else None,
            )
        ax.set_yscale("log")
        ax.set_xlabel("Polynomial degree")
        ax.set_title(f"{k}-fold CV")

    axes[0].set_ylabel("mean CV MSE")
    axes[-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7)
    fig.suptitle(rf"Ridge: CV MSE vs. degree, by $\lambda$ ($\sigma={noise_std}$)")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree-max", type=int, default=15, help="highest polynomial degree")
    parser.add_argument("--n", type=int, default=100, help="number of sample points")
    parser.add_argument("--noise", type=float, default=0.1, help="Gaussian noise std")
    parser.add_argument("--seed", type=int, default=42, help="RNG / fold-shuffling seed")
    parser.add_argument("--test-size", type=float, default=0.2, help="held-out test fraction")
    parser.add_argument(
        "--n-bootstraps", type=int, default=100, help="bootstrap resamples per degree"
    )
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=[5, 10], help="fold counts to sweep"
    )
    parser.add_argument("--lambda-min", type=float, default=-8, help="log10 of the smallest lambda")
    parser.add_argument("--lambda-max", type=float, default=-1, help="log10 of the largest lambda")
    parser.add_argument("--n-lambdas", type=int, default=15, help="number of lambda grid points")
    parser.add_argument(
        "--out-subdir", default="cross_validation", help="subdirectory under docs/figures/"
    )
    args = parser.parse_args()

    set_style()

    degrees = range(0, args.degree_max + 1)
    x, y = generate_runge_data(n=args.n, noise_std=args.noise, seed=args.seed)

    bv_results = bootstrap_bias_variance_sweep(
        x, y, degrees, OLS, n_bootstraps=args.n_bootstraps, test_size=args.test_size, seed=args.seed
    )
    cv_ols_results = {
        k: kfold_mse_degree_sweep(x, y, degrees, OLS, k=k, seed=args.seed) for k in args.k_values
    }
    ols_fig = plot_ols_cv_vs_bootstrap(bv_results, cv_ols_results, args.k_values, args.noise)
    print(f"Wrote {save_figure(ols_fig, 'ols_cv_vs_bootstrap', args.out_subdir)}")

    lambdas = np.logspace(args.lambda_min, args.lambda_max, args.n_lambdas)
    cv_ridge_results = {
        k: kfold_mse_ridge_grid(x, y, degrees, lambdas, k=k, seed=args.seed) for k in args.k_values
    }

    heatmap_fig = plot_ridge_cv_heatmap(cv_ridge_results, args.k_values, args.noise)
    print(f"Wrote {save_figure(heatmap_fig, 'ridge_cv_heatmap', args.out_subdir)}")

    lines_fig = plot_ridge_cv_lines(cv_ridge_results, lambdas, args.k_values, args.noise)
    print(f"Wrote {save_figure(lines_fig, 'ridge_cv_by_lambda', args.out_subdir)}")


if __name__ == "__main__":
    main()
