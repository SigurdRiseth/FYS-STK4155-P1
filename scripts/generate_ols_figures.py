"""
Generate report figures for the OLS experiment (part a): MSE/R2 vs. polynomial
degree, the resulting coefficient trace, and the effect of dataset size on the
degree sweep.

Usage:
    uv run python scripts/generate_ols_figures.py [--degree-max 15] [--n 100]
        [--noise 0.1] [--seed 42] [--n-values 50 100 300 500 1000]
        [--out-subdir ols]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from utils.plotting import FIGSIZE_WIDE, save_figure, set_style

from fys_stk4155_p1.data.runge import generate_runge_data
from fys_stk4155_p1.regression.degree_sweep import fit_polynomial_degree_sweep
from fys_stk4155_p1.regression.ordinary_least_squares import OLS


def plot_mse_r2_vs_degree(results: dict) -> Figure:
    degrees = results["degrees"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(degrees, results["mse_train"], marker="o", markersize=3, label="train")
    axes[0].plot(degrees, results["mse_test"], marker="o", markersize=3, label="test")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Polynomial degree")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("MSE vs. polynomial degree")
    axes[0].legend()

    axes[1].plot(degrees, results["r2_train"], marker="o", markersize=3, label="train")
    axes[1].plot(degrees, results["r2_test"], marker="o", markersize=3, label="test")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_xlabel("Polynomial degree")
    axes[1].set_ylabel(r"$R^2$")
    axes[1].set_title(r"$R^2$ vs. polynomial degree")
    axes[1].legend()

    fig.tight_layout()
    return fig


def plot_coefficients_vs_degree(results: dict) -> Figure:
    degrees = results["degrees"]
    max_degree = int(degrees.max())
    weights = results["weights"]

    theta_matrix = np.full((len(degrees), max_degree), np.nan)
    for row, theta in enumerate(weights):
        theta_matrix[row, : len(theta)] = theta

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)  # extra width for the outside legend
    ax.axhline(
        results["intercept"], color="black", ls="--", lw=1, label=r"intercept ($\bar{y}_{train}$)"
    )
    for coef_idx in range(max_degree):
        ax.plot(
            degrees,
            theta_matrix[:, coef_idx],
            marker="o",
            markersize=3,
            label=rf"$\theta_{{{coef_idx + 1}}}$",
        )

    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel(r"Coefficient value $\theta_i$")
    ax.set_title(r"OLS coefficients $\theta_i$ vs. polynomial degree")
    ax.legend(ncol=2, fontsize="small", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    return fig


def plot_test_mse_r2_by_n(n_sweep_results: dict[int, dict], n_values: list[int]) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for n in n_values:
        res = n_sweep_results[n]
        axes[0].plot(res["degrees"], res["mse_test"], marker="o", markersize=3, label=f"n={n}")
        axes[1].plot(res["degrees"], res["r2_test"], marker="o", markersize=3, label=f"n={n}")

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Polynomial degree")
    axes[0].set_ylabel("Test MSE")
    axes[0].set_title("Test MSE vs. degree, by dataset size")
    axes[0].legend(fontsize="small")

    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_xlabel("Polynomial degree")
    axes[1].set_ylabel(r"Test $R^2$")
    axes[1].set_title(r"Test $R^2$ vs. degree, by dataset size")
    axes[1].legend(fontsize="small")

    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree-max", type=int, default=15, help="highest polynomial degree")
    parser.add_argument("--n", type=int, default=100, help="number of sample points")
    parser.add_argument("--noise", type=float, default=0.1, help="Gaussian noise std")
    parser.add_argument("--seed", type=int, default=42, help="RNG / train-test-split seed")
    parser.add_argument("--test-size", type=float, default=0.2, help="held-out test fraction")
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[50, 100, 300, 500, 1000],
        help="dataset sizes for the n-sweep figure",
    )
    parser.add_argument("--out-subdir", default="ols", help="subdirectory under docs/figures/")
    args = parser.parse_args()

    set_style()

    degrees = range(1, args.degree_max + 1)

    x, y = generate_runge_data(n=args.n, noise_std=args.noise, seed=args.seed)
    results = fit_polynomial_degree_sweep(
        x, y, degrees, OLS, test_size=args.test_size, seed=args.seed
    )

    mse_r2_fig = plot_mse_r2_vs_degree(results)
    print(f"Wrote {save_figure(mse_r2_fig, 'ols_mse_r2_vs_degree', args.out_subdir)}")

    coef_fig = plot_coefficients_vs_degree(results)
    print(f"Wrote {save_figure(coef_fig, 'ols_coefficients_vs_degree', args.out_subdir)}")

    n_sweep_results = {
        n: fit_polynomial_degree_sweep(
            *generate_runge_data(n=n, noise_std=args.noise, seed=args.seed),
            degrees,
            OLS,
            test_size=args.test_size,
            seed=args.seed,
        )
        for n in args.n_values
    }
    n_sweep_fig = plot_test_mse_r2_by_n(n_sweep_results, args.n_values)
    print(f"Wrote {save_figure(n_sweep_fig, 'ols_test_mse_r2_by_n', args.out_subdir)}")


if __name__ == "__main__":
    main()
