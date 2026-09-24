"""
Report figures for part c) (bias-variance trade-off via the bootstrap, OLS):
the decomposition vs. degree at the reference settings, and its dependence
on the number of points and on the noise level. All settings from
utils/config.py.

The squared bias is shown twice: measured against the noisy test targets y
(what is available in practice; it absorbs sigma^2) and against the known
f(x) (possible only because the data are synthetic).

Usage: uv run python scripts/generate_bias_variance_figures.py

LLM-assisted: Claude (claude-opus-5-5, Claude Cowork desktop app, September 2026)
wrote/rewrote this file (level 4) to use the shared settings in utils/config.py.
TODO(author): describe your review/changes.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from utils import config as C
from utils.plotting import COLORS, COLUMN, WIDE, save_figure, set_style

from fys_stk4155_p1.data.runge import generate_runge_data, runge_function
from fys_stk4155_p1.regression.ordinary_least_squares import OLS
from fys_stk4155_p1.resampling.bootstrap import bootstrap_bias_variance_sweep

YLIM = (1e-4, 3.0)


def _sweep(n: int, noise: float) -> dict:
    x, y = generate_runge_data(n=n, noise_std=noise, seed=C.SEED)
    return bootstrap_bias_variance_sweep(
        x,
        y,
        C.DEGREES,
        OLS,
        n_bootstraps=C.N_BOOTSTRAPS,
        test_size=C.TEST_SIZE,
        seed=C.SEED,
        f_true=runge_function,
    )


def _panel(ax: Axes, res: dict, noise: float, legend: bool) -> None:
    d = res["degrees"]
    ax.plot(d, res["mse_test"], "o-", color="black", label="test MSE")
    ax.plot(d, res["bias2"], "s-", color=COLORS["ridge"], ms=2, label=r"bias$^2$ vs. $y$")
    ax.plot(d, res["bias2_f"], "^--", color=COLORS["lasso"], ms=2, label=r"bias$^2$ vs. $f$")
    ax.plot(d, res["variance"], "d-", color=COLORS["ols"], ms=2, label="variance")
    ax.axhline(noise**2, color="gray", ls=":", lw=0.9, label=r"$\sigma^2$")
    best = d[np.argmin(res["mse_test"])]
    ax.axvline(best, color="black", ls="--", lw=0.6)
    ax.set_yscale("log")
    ax.set_ylim(*YLIM)
    ax.set_xticks(range(0, C.DEGREE_MAX + 1, 4))
    ax.set_xlabel("Polynomial degree $p$")
    if legend:
        ax.legend(loc="upper left", ncol=2)


def plot_tradeoff() -> Figure:
    fig, ax = plt.subplots(figsize=COLUMN)
    _panel(ax, _sweep(C.N, C.NOISE), C.NOISE, legend=True)
    ax.set_ylabel("Error (bootstrap)")
    return fig


def plot_sweep(key: str) -> Figure:
    values = C.N_VALUES if key == "n" else C.NOISE_VALUES
    fig, axes = plt.subplots(1, len(values), figsize=WIDE, sharey=True)
    for i, (ax, v) in enumerate(zip(axes, values, strict=True)):
        noise = C.NOISE if key == "n" else float(v)
        res = _sweep(int(v) if key == "n" else C.N, noise)
        _panel(ax, res, noise, legend=False)
        ax.set_title(f"$n={v}$" if key == "n" else rf"$\sigma={v}$")
        if i:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel("Error (bootstrap)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=5)
    return fig


def main() -> None:
    set_style()
    print(save_figure(plot_tradeoff(), "bias_variance_tradeoff_bootstrap", "bias_variance"))
    print(save_figure(plot_sweep("n"), "bias_variance_tradeoff_by_n", "bias_variance"))
    print(save_figure(plot_sweep("noise"), "bias_variance_tradeoff_by_noise", "bias_variance"))


if __name__ == "__main__":
    main()
