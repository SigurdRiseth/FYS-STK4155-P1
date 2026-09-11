"""
Generate report figures for the bias-variance bootstrap experiment (part c):
the bootstrap bias^2/variance decomposition of test MSE vs. polynomial degree,
and how that decomposition shifts with dataset size and noise level.

Usage:
    uv run python scripts/generate_bias_variance_figures.py [--degree-max 13]
        [--n 100] [--noise 0.1] [--seed 42] [--n-bootstraps 100]
        [--n-values 50 100 200 400] [--noise-values 0.1 0.2 0.4 0.8]
        [--out-subdir bias_variance]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from utils.plotting import save_figure, set_style

from fys_stk4155_p1.data.runge import generate_runge_data
from fys_stk4155_p1.regression.ordinary_least_squares import OLS
from fys_stk4155_p1.resampling.bootstrap import bootstrap_bias_variance_sweep


def _plot_decomposition(ax: Axes, results: dict, noise_std: float) -> None:
    """Plot test error/bias^2/variance vs. degree, with a sigma^2 floor and best-degree marker."""
    degrees = results["degrees"]
    best_degree = degrees[np.argmin(results["mse_test"])]

    ax.plot(degrees, results["mse_test"], "o-", markersize=3, label="test error")
    ax.plot(degrees, results["bias2"], "s-", markersize=3, label=r"bias$^2$ (+ $\sigma^2$)")
    ax.plot(degrees, results["variance"], "d-", markersize=3, label="variance")
    ax.axhline(noise_std**2, color="gray", linestyle=":", linewidth=1.5, label=r"$\sigma^2$")
    ax.axvline(best_degree, color="black", linestyle="--", linewidth=1)
    ax.annotate(
        f"best={best_degree}",
        xy=(best_degree, 1.0),
        xycoords=ax.get_xaxis_transform(),
        xytext=(3, -3),
        textcoords="offset points",
        fontsize=8,
        va="top",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree")


def plot_bias_variance_tradeoff(results: dict, noise_std: float) -> Figure:
    fig, ax = plt.subplots()
    _plot_decomposition(ax, results, noise_std)
    ax.set_ylabel("MSE decomposition")
    ax.set_title(rf"Bias-variance tradeoff via bootstrap ($\sigma={noise_std}$)")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _grid_subplots(n_panels: int, n_cols: int = 2) -> tuple[Figure, np.ndarray]:
    """A grid of `n_panels` panels, `n_cols` wide, with shared x/y axes."""
    n_rows = -(-n_panels // n_cols)  # ceil division
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 3.75 * n_rows), sharex=True, sharey=True
    )
    return fig, np.atleast_1d(axes).ravel()


def plot_bias_variance_by_n(
    n_sweep_results: dict[int, dict], n_values: list[int], noise_std: float
) -> Figure:
    fig, axes = _grid_subplots(len(n_values))
    for ax, n in zip(axes, n_values, strict=True):
        _plot_decomposition(ax, n_sweep_results[n], noise_std)
        ax.set_ylabel("MSE decomposition")
        ax.set_title(f"n = {n}")
        ax.label_outer()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(rf"Effect of sample size on the bias-variance tradeoff ($\sigma={noise_std}$)")
    fig.tight_layout()
    return fig


def plot_bias_variance_by_noise(
    noise_sweep_results: dict[float, dict], noise_values: list[float], n: int
) -> Figure:
    fig, axes = _grid_subplots(len(noise_values))
    for ax, noise in zip(axes, noise_values, strict=True):
        _plot_decomposition(ax, noise_sweep_results[noise], noise)
        ax.set_ylabel("MSE decomposition")
        ax.set_title(rf"$\sigma$ = {noise}")
        ax.label_outer()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(f"Effect of noise level on the bias-variance tradeoff (n={n})")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree-max", type=int, default=13, help="highest polynomial degree")
    parser.add_argument("--n", type=int, default=100, help="number of sample points")
    parser.add_argument("--noise", type=float, default=0.1, help="Gaussian noise std")
    parser.add_argument("--seed", type=int, default=42, help="RNG / train-test-split seed")
    parser.add_argument("--test-size", type=float, default=0.2, help="held-out test fraction")
    parser.add_argument(
        "--n-bootstraps", type=int, default=100, help="bootstrap resamples per degree"
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[50, 100, 200, 400],
        help="dataset sizes for the sample-size sweep",
    )
    parser.add_argument(
        "--noise-values",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.4, 0.8],
        help="noise levels for the noise sweep",
    )
    parser.add_argument(
        "--out-subdir", default="bias_variance", help="subdirectory under docs/figures/"
    )
    args = parser.parse_args()

    set_style()

    degrees = range(0, args.degree_max + 1)

    x, y = generate_runge_data(n=args.n, noise_std=args.noise, seed=args.seed)
    results = bootstrap_bias_variance_sweep(
        x, y, degrees, OLS, n_bootstraps=args.n_bootstraps, test_size=args.test_size, seed=args.seed
    )
    tradeoff_fig = plot_bias_variance_tradeoff(results, args.noise)
    print(f"Wrote {save_figure(tradeoff_fig, 'bias_variance_tradeoff_bootstrap', args.out_subdir)}")

    n_sweep_results = {
        n: bootstrap_bias_variance_sweep(
            *generate_runge_data(n=n, noise_std=args.noise, seed=args.seed),
            degrees,
            OLS,
            n_bootstraps=args.n_bootstraps,
            test_size=args.test_size,
            seed=args.seed,
        )
        for n in args.n_values
    }
    n_sweep_fig = plot_bias_variance_by_n(n_sweep_results, args.n_values, args.noise)
    print(f"Wrote {save_figure(n_sweep_fig, 'bias_variance_tradeoff_by_n', args.out_subdir)}")

    noise_sweep_results = {
        noise: bootstrap_bias_variance_sweep(
            *generate_runge_data(n=args.n, noise_std=noise, seed=args.seed),
            degrees,
            OLS,
            n_bootstraps=args.n_bootstraps,
            test_size=args.test_size,
            seed=args.seed,
        )
        for noise in args.noise_values
    }
    noise_sweep_fig = plot_bias_variance_by_noise(noise_sweep_results, args.noise_values, args.n)
    print(
        f"Wrote {save_figure(noise_sweep_fig, 'bias_variance_tradeoff_by_noise', args.out_subdir)}"
    )


if __name__ == "__main__":
    main()
