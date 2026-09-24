"""
Report figures for part a) (OLS): train/test MSE and R^2 vs. degree, the
fitted coefficients vs. degree, and the dependence on the number of points.
All settings from utils/config.py.

Usage: uv run python scripts/generate_ols_figures.py

LLM-assisted: Claude (claude-opus-5-5, Claude Cowork desktop app, September 2026)
wrote/rewrote this file (level 4) to use the shared settings in utils/config.py.
TODO(author): describe your review/changes.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm
from matplotlib.figure import Figure
from utils import config as C
from utils.plotting import COLORS, COLUMN, COLUMN_TALL, save_figure, set_style

from fys_stk4155_p1.data.runge import generate_runge_data
from fys_stk4155_p1.regression.degree_sweep import fit_polynomial_degree_sweep
from fys_stk4155_p1.regression.ordinary_least_squares import OLS

DEGREES = [d for d in C.DEGREES if d >= 1]


def plot_mse_r2_vs_degree(res: dict) -> Figure:
    fig, (ax_mse, ax_r2) = plt.subplots(2, 1, figsize=COLUMN_TALL, sharex=True)
    d = res["degrees"]
    ax_mse.plot(d, res["mse_train"], "o-", color=COLORS["train"], label="train")
    ax_mse.plot(d, res["mse_test"], "s-", color=COLORS["test"], label="test")
    ax_mse.axhline(C.NOISE**2, color="black", ls=":", lw=0.8, label=r"$\sigma^2$")
    best = d[np.argmin(res["mse_test"])]
    ax_mse.axvline(best, color=COLORS["test"], ls="--", lw=0.8)
    ax_mse.set_yscale("log")
    ax_mse.set_ylabel("MSE")
    ax_mse.legend(loc="upper right", ncol=3)
    ax_r2.plot(d, res["r2_train"], "o-", color=COLORS["train"], label="train")
    ax_r2.plot(d, res["r2_test"], "s-", color=COLORS["test"], label="test")
    ax_r2.set_ylabel("$R^2$")
    ax_r2.set_xlabel("Polynomial degree $p$")
    ax_r2.set_xticks(range(0, C.DEGREE_MAX + 1, 2))
    return fig


def plot_coefficients_vs_degree(res: dict) -> Figure:
    """|theta_j| (standardized features) for every degree, as an image."""
    p_max = int(res["degrees"].max())
    theta = np.full((len(res["degrees"]), p_max), np.nan)
    for row, w in enumerate(res["weights"]):
        theta[row, : len(w)] = w
    fig, ax = plt.subplots(figsize=COLUMN)
    lim = np.nanmax(np.abs(theta))
    mesh = ax.pcolormesh(
        np.arange(1, p_max + 1),
        res["degrees"],
        theta,
        cmap="RdBu_r",
        norm=SymLogNorm(linthresh=1.0, vmin=-lim, vmax=lim),
        shading="nearest",
    )
    ax.set_xlabel(r"Coefficient index $j$ (feature $x^j$, standardized)")
    ax.set_ylabel("Polynomial degree $p$")
    ax.set_xticks(range(1, p_max + 1, 2))
    ax.set_yticks(range(1, p_max + 1, 2))
    ax.grid(False)
    fig.colorbar(mesh, ax=ax, label=r"$\hat\theta_j$ (symlog)")
    return fig


def plot_test_mse_by_n(results: dict[int, dict]) -> Figure:
    fig, ax = plt.subplots(figsize=COLUMN)
    cmap = plt.get_cmap("viridis")
    for i, (n, res) in enumerate(results.items()):
        ax.plot(
            res["degrees"],
            res["mse_test"],
            "o-",
            color=cmap(i / (len(results) - 1)),
            label=f"$n={n}$",
        )
    ax.axhline(C.NOISE**2, color="black", ls=":", lw=0.8, label=r"$\sigma^2$")
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree $p$")
    ax.set_ylabel("Test MSE (single split)")
    ax.set_xticks(range(0, C.DEGREE_MAX + 1, 2))
    ax.legend(ncol=2, loc="upper left")
    return fig


def main() -> None:
    set_style()
    x, y = generate_runge_data(n=C.N, noise_std=C.NOISE, seed=C.SEED)
    res = fit_polynomial_degree_sweep(x, y, DEGREES, OLS, test_size=C.TEST_SIZE, seed=C.SEED)
    print(save_figure(plot_mse_r2_vs_degree(res), "ols_mse_r2_vs_degree", "ols"))
    print(save_figure(plot_coefficients_vs_degree(res), "ols_coefficients_vs_degree", "ols"))
    by_n = {
        n: fit_polynomial_degree_sweep(
            *generate_runge_data(n=n, noise_std=C.NOISE, seed=C.SEED),
            DEGREES,
            OLS,
            test_size=C.TEST_SIZE,
            seed=C.SEED,
        )
        for n in C.N_VALUES
    }
    print(save_figure(plot_test_mse_by_n(by_n), "ols_test_mse_by_n", "ols"))


if __name__ == "__main__":
    main()
