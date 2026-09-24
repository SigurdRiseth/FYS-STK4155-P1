"""
Report figures for part b) (Ridge): test MSE vs. degree for several lambda,
the ridge trace and the shrinkage of the SVD modes at SHOWCASE_DEGREE.
All settings from utils/config.py.

Usage: uv run python scripts/generate_ridge_figures.py

LLM-assisted: Claude (claude-opus-5-5, Claude Cowork desktop app, September 2026)
wrote/rewrote this file (level 4) to use the shared settings in utils/config.py.
TODO(author): describe your review/changes.
"""

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from numpy.typing import NDArray
from utils import config as C
from utils.common import data, design
from utils.plotting import COLUMN, save_figure, set_style

from fys_stk4155_p1.data.runge import generate_runge_data
from fys_stk4155_p1.regression.degree_sweep import fit_polynomial_degree_sweep
from fys_stk4155_p1.regression.ordinary_least_squares import OLS
from fys_stk4155_p1.regression.ridge import Ridge
from fys_stk4155_p1.regression.shrinkage import ridge_coefficient_path, singular_value_shrinkage

MSE_LAMBDAS = [1e-8, 1e-6, 1e-4, 1e-2]


def plot_test_mse_by_lambda() -> Figure:
    x, y = generate_runge_data(n=C.N, noise_std=C.NOISE, seed=C.SEED)
    degrees = [d for d in C.DEGREES if d >= 1]
    fig, ax = plt.subplots(figsize=COLUMN)
    ols = fit_polynomial_degree_sweep(x, y, degrees, OLS, test_size=C.TEST_SIZE, seed=C.SEED)
    ax.plot(degrees, ols["mse_test"], "k.-", label="OLS ($\\lambda=0$)")
    cmap = plt.get_cmap("plasma")
    for i, lam in enumerate(MSE_LAMBDAS):
        res = fit_polynomial_degree_sweep(
            x, y, degrees, partial(Ridge, lam=lam), test_size=C.TEST_SIZE, seed=C.SEED
        )
        ax.plot(
            degrees,
            res["mse_test"],
            "o-",
            color=cmap(0.1 + 0.75 * i / 3),
            label=rf"$\lambda=10^{{{int(np.log10(lam))}}}$",
        )
    ax.axhline(C.NOISE**2, color="black", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree $p$")
    ax.set_ylabel("Test MSE (single split)")
    ax.set_xticks(range(0, C.DEGREE_MAX + 1, 2))
    ax.legend(ncol=2)
    return fig


def plot_ridge_trace(X: NDArray, y: NDArray) -> Figure:
    lambdas = np.logspace(-10, 1, 111)
    path = ridge_coefficient_path(X, y, lambdas, fit_intercept_column=True)
    p = X.shape[1] - 1
    fig, ax = plt.subplots(figsize=COLUMN)
    cmap = plt.get_cmap("viridis")
    for j in range(1, p + 1):
        ax.plot(lambdas, path[:, j], color=cmap((j - 1) / (p - 1)), lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\hat\theta_j(\lambda)$ (symlog)")
    fig.colorbar(ScalarMappable(Normalize(1, p), cmap), ax=ax, label="index $j$ of $x^j$")
    return fig


def plot_shrinkage(X: NDArray) -> Figure:
    lambdas = np.logspace(-10, 0, 6)
    s, shrink = singular_value_shrinkage(X[:, 1:], lambdas)
    fig, ax = plt.subplots(figsize=COLUMN)
    cmap = plt.get_cmap("plasma")
    idx = np.arange(1, len(s) + 1)
    for i, (lam, row) in enumerate(zip(lambdas, shrink, strict=True)):
        ax.plot(
            idx,
            row,
            "o-",
            color=cmap(0.1 + 0.75 * i / (len(lambdas) - 1)),
            label=rf"$10^{{{int(np.log10(lam))}}}$",
        )
    ax2 = ax.twinx()
    ax2.semilogy(idx, s**2 / X.shape[0], "k:", lw=0.8)
    ax2.set_ylabel(r"$\sigma_i^2/n$ (dotted)")
    ax2.grid(False)
    ax.set_xlabel(r"Singular-value index $i$ (decreasing $\sigma_i$)")
    ax.set_ylabel(r"$\sigma_i^2/(\sigma_i^2+n\lambda)$")
    fig.legend(
        *ax.get_legend_handles_labels(), title=r"$\lambda$", ncol=6, loc="outside lower center"
    )
    return fig


def main() -> None:
    set_style()
    print(save_figure(plot_test_mse_by_lambda(), "ridge_test_mse_by_lambda", "ridge"))
    _, _, x_tr, _, y_tr, _ = data()
    (X,) = design(x_tr, C.SHOWCASE_DEGREE)
    print(save_figure(plot_ridge_trace(X, y_tr), f"ridge_trace_degree{C.SHOWCASE_DEGREE}", "ridge"))
    print(save_figure(plot_shrinkage(X), f"ridge_shrinkage_degree{C.SHOWCASE_DEGREE}", "ridge"))


if __name__ == "__main__":
    main()
