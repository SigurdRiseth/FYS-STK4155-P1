"""
Report figures for part g) (Lasso by subgradient descent), on the reference
training set (from e3):

* convergence: relative cost gap vs. iteration for subgradient descent with
  each update rule, at the (degree, lambda) selected by 5-fold CV and at a high
  degree (SHOWCASE_DEGREE, lambda = 1e-4); the optimum C* is scikit-learn's;
* coefficients of our solver (Adam) vs. scikit-learn's at the high degree.

Needs data/results/e3_lasso.json (`make experiments`).

Usage: uv run python scripts/generate_lasso_figures.py

LLM-assisted: Claude (claude-opus-5-5, Claude Cowork desktop app, September 2026)
wrote/rewrote this file (level 4) to use the shared settings in utils/config.py.
TODO(author): describe your review/changes.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from utils.common import load_result
from utils.plotting import COLORS, COLUMN, LABELS, WIDE, save_figure, set_style

OPTIMIZERS = ["plain", "momentum", "adagrad", "rmsprop", "adam"]


def _lam_tex(lam: float) -> str:
    e = np.log10(lam)
    return rf"10^{{{e:.1f}}}".replace(".0}", "}")


def plot_convergence(e3: dict) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=WIDE, sharey=True)
    for ax, (key, tag) in zip(
        axes, (("selected", "(a) CV-selected"), ("high_degree", "(b)")), strict=True
    ):
        cfg = e3["configs"][key]
        for opt in OPTIMIZERS:
            t = cfg["traces"][opt]
            ax.plot(
                t["iter"],
                t["value"],
                color=COLORS[opt],
                label=rf"{LABELS[opt]} ($\gamma={t['learning_rate']:.2g}$)",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-7, 1e2)
        ax.set_xlabel("Iteration $k$")
        ax.set_title(rf"{tag} $p={cfg['degree']}$, $\lambda={_lam_tex(cfg['lambda'])}$", loc="left")
        ax.legend(loc="lower left")
    axes[0].set_ylabel(r"$(C(\theta_k) - C^*)/C^*$")
    return fig


def plot_coefficients(e3: dict) -> Figure:
    cfg = e3["configs"]["high_degree"]
    f = cfg["finals"]
    sk = np.asarray(f["sklearn"]["coef"])[1:]
    j = np.arange(1, len(sk) + 1)
    fig, ax = plt.subplots(figsize=COLUMN)
    ax.axhline(0, color="black", lw=0.5)
    ax.plot(j, sk, "o", ms=5, mfc="none", color=COLORS["sklearn"], label="scikit-learn")
    for key, marker, lab in (
        ("adam_default", "x", "ours, Adam, 5000 it."),
        ("adam_long", "+", r"ours, Adam, $5\times10^4$ it."),
    ):
        ax.plot(
            j,
            np.asarray(f[key]["coef"])[1:],
            marker,
            ms=5,
            color=COLORS["adam"] if key == "adam_long" else COLORS["ridge"],
            label=lab,
        )
    zero = sk == 0
    ax.plot(
        j[zero],
        sk[zero],
        "s",
        ms=6,
        mfc="none",
        color="gray",
        lw=0,
        label="exact zero (scikit-learn)",
    )
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xticks(j)
    ax.set_xlabel(r"Coefficient index $j$ (feature $x^j$)")
    ax.set_ylabel(r"$\hat\theta_j$ (symlog)")
    ax.set_title(rf"$p={cfg['degree']}$, $\lambda={_lam_tex(cfg['lambda'])}$", loc="left")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=2)
    return fig


def main() -> None:
    set_style()
    e3 = load_result("e3_lasso")
    print(save_figure(plot_convergence(e3), "lasso_convergence", "lasso"))
    print(save_figure(plot_coefficients(e3), "lasso_coefficients", "lasso"))


if __name__ == "__main__":
    main()
