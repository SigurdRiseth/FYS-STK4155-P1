"""
Report figures for parts d) and i): model selection by resampling.

* estimator comparison (from e1): test-error estimates vs. degree for the
  bootstrap, k-fold CV, a single split and the true error at the reference
  seed, and the degree each one selects over all data seeds;
* why the estimates blow up: one CV fold at high degree, where the held-out
  point lies outside the training fold's x-range;
* CV heat maps over (degree, lambda) for Ridge (5- and 10-fold) and Lasso
  (from e2);
* final comparison (from e2): OLS vs. Ridge vs. Lasso over all data seeds.

Needs data/results/e1_ols_degree.json and e2_model_selection.json
(`make experiments`). All settings from utils/config.py.

Usage: uv run python scripts/generate_cross_validation_figures.py

LLM-assisted: Claude (claude-opus-5-5, Claude Cowork desktop app, September 2026)
wrote/rewrote this file (level 4) to use the shared settings in utils/config.py.
TODO(author): describe your review/changes.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from sklearn.model_selection import KFold
from utils import config as C
from utils.common import data, design, load_result
from utils.plotting import COLORS, COLUMN, LABELS, WIDE, save_figure, set_style

from fys_stk4155_p1.data.runge import runge_function
from fys_stk4155_p1.regression.ordinary_least_squares import OLS

ESTIMATORS = ["bootstrap", "cv5", "cv10", "single_split", "oracle"]
MARKERS = {"bootstrap": "o", "cv5": "s", "cv10": "D", "single_split": "v", "oracle": "*"}


def plot_estimators(e1: dict) -> Figure:
    ref = next(r for r in e1["per_seed"] if r["seed"] == C.SEED)
    fig, (ax, ax_h) = plt.subplots(1, 2, figsize=WIDE, width_ratios=[1.25, 1])
    for name in ESTIMATORS:
        curve = np.asarray(ref["curves"][name], dtype=float)
        ax.plot(
            C.DEGREES,
            curve,
            marker=MARKERS[name],
            ms=2.5,
            color=COLORS[name],
            lw=1.4 if name == "oracle" else 0.9,
            label=LABELS[name],
        )
        ax.plot(
            ref["best_degree"][name],
            curve.min(),
            marker=MARKERS[name],
            ms=7,
            mfc="none",
            mec=COLORS[name],
            mew=1.0,
        )
    ax.axhline(C.NOISE**2, color="black", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_ylim(5e-3, 1e2)
    ax.set_xticks(range(0, C.DEGREE_MAX + 1, 2))
    ax.set_xlabel("Polynomial degree $p$")
    ax.set_ylabel("Estimated test MSE")
    ax.set_title(f"(a) reference data set (seed {C.SEED})", loc="left")
    ax.legend(loc="upper left", ncol=2)

    width = 0.16
    for i, name in enumerate(ESTIMATORS):
        counts = np.zeros(len(C.DEGREES))
        for r in e1["per_seed"]:
            counts[r["best_degree"][name]] += 1
        ax_h.bar(
            np.array(C.DEGREES) + (i - 2) * width,
            counts,
            width=width,
            color=COLORS[name],
            label=LABELS[name],
        )
    ax_h.set_xlim(2.5, C.DEGREE_MAX + 0.5)
    ax_h.set_xticks(range(3, C.DEGREE_MAX + 1, 2))
    ax_h.set_xlabel("Selected degree $\\hat p$")
    ax_h.set_ylabel(f"Count over {len(e1['per_seed'])} data sets")
    ax_h.set_title("(b) selected degree", loc="left")
    return fig


def plot_extrapolation(degree: int = 15, k: int = 5) -> Figure:
    """The CV fold whose held-out set contains the training split's extreme point."""
    _, _, x_tr, _, y_tr, _ = data()
    folds = list(KFold(k, shuffle=True, random_state=C.SEED).split(x_tr))

    def fold_mse(fold: tuple) -> float:
        a, b = fold
        X_a, X_b = design(x_tr[a], degree, x_tr[b])
        return float(np.mean((y_tr[b] - OLS().fit(X_a, y_tr[a]).predict(X_b)) ** 2))

    tr, te = max(folds, key=fold_mse)  # the fold that dominates the CV estimate
    x_grid = C.X_GRID
    fig, ax = plt.subplots(figsize=COLUMN)
    ax.plot(x_grid, runge_function(x_grid), color="black", lw=1.0, label="$f(x)$")
    for idx, deg, color, lab in (
        (tr, degree, COLORS["ridge"], f"fold fit, $p={degree}$"),
        (np.arange(len(x_tr)), degree, COLORS["ols"], f"all 80 points, $p={degree}$"),
    ):
        X_fit, X_grid = design(x_tr[idx], deg, x_grid)
        pred = OLS().fit(X_fit, y_tr[idx]).predict(X_grid)
        ax.plot(x_grid, pred, color=color, lw=1.0, label=lab)
    ax.scatter(x_tr[tr], y_tr[tr], s=5, color="gray", lw=0, label="training fold")
    ax.scatter(
        x_tr[te], y_tr[te], s=14, marker="x", color=COLORS["oracle"], lw=0.8, label="held-out fold"
    )
    lo, hi = x_tr[tr].min(), x_tr[tr].max()
    ax.axvspan(-1, lo, color="gray", alpha=0.15, lw=0)
    ax.axvspan(hi, 1, color="gray", alpha=0.15, lw=0)
    X_fit, X_te = design(x_tr[tr], degree, x_tr[te])
    pred_te = OLS().fit(X_fit, y_tr[tr]).predict(X_te)
    j = int(np.argmax((y_tr[te] - pred_te) ** 2))
    ax.annotate(
        rf"$\hat f({x_tr[te][j]:.3f}) = {pred_te[j]:.1f}$",
        xy=(x_tr[te][j], float(np.clip(pred_te[j], -0.58, 1.58))),
        xytext=(0.3, -0.45),
        fontsize=7,
        arrowprops={"arrowstyle": "->", "lw": 0.6},
    )
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(loc="upper left", ncol=2)
    return fig


def plot_cv_heatmaps(e2: dict) -> Figure:
    ref = next(r for r in e2["per_seed"] if r["seed"] == C.SEED)
    panels = [
        ("Ridge, 5-fold", ref["ridge_grid_mean"], C.RIDGE_LAMBDAS),
        ("Ridge, 10-fold", ref["ridge_grid_mean_cv10"], C.RIDGE_LAMBDAS),
        ("Lasso, 5-fold", ref["lasso_grid_mean"], C.LASSO_LAMBDAS),
    ]
    fig, axes = plt.subplots(1, 3, figsize=WIDE, sharex=True)
    norm = LogNorm(vmin=5e-3, vmax=1.0)
    mesh = None
    for i, (ax, (title, grid, lambdas)) in enumerate(zip(axes, panels, strict=True)):
        grid = np.asarray(grid, dtype=float)
        mesh = ax.pcolormesh(
            C.DEGREES,
            lambdas,
            np.clip(grid.T, None, 1.0),
            norm=norm,
            cmap="viridis_r",
            shading="nearest",
        )
        a, b = np.unravel_index(np.argmin(grid), grid.shape)
        ax.plot(C.DEGREES[a], lambdas[b], marker="*", ms=8, color="red", mec="white", mew=0.5)
        ax.set_yscale("log")
        ax.grid(False)
        ax.set_xlabel("Polynomial degree $p$")
        ax.set_title(
            f"({'abc'[i]}) {title}: $p={C.DEGREES[a]}$, "
            rf"$\lambda=10^{{{np.log10(lambdas[b]):.1f}}}$".replace(".0}", "}"),
            loc="left",
        )
    axes[0].set_ylabel(r"$\lambda$")
    assert mesh is not None
    fig.colorbar(mesh, ax=axes, label="CV MSE (clipped at 1)", pad=0.01)
    return fig


def plot_final_comparison(e2: dict) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=WIDE, sharey=True)
    methods = ["ols", "ridge", "lasso"]
    names = {"ols": "OLS", "ridge": "Ridge", "lasso": "Lasso"}
    for ax, metric, title in zip(
        axes,
        ("test_mse", "oracle_mse"),
        ("(a) held-out test set (20 points)", "(b) true error (known $f$)"),
        strict=True,
    ):
        vals = {m: np.array([r["selected"][m][metric] for r in e2["per_seed"]]) for m in methods}
        for i, m in enumerate(methods):
            v = vals[m]
            jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(v))
            ax.scatter(i + jitter, v, s=6, color=COLORS[m], alpha=0.8, lw=0)
            ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="black", lw=1.2)
        for j in range(len(vals["ols"])):
            ax.plot(range(3), [vals[m][j] for m in methods], color="gray", lw=0.3, alpha=0.5)
        ax.axhline(C.NOISE**2, color="black", ls=":", lw=0.8)
        ax.set_yscale("log")
        ax.set_xticks(range(3), [names[m] for m in methods])
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("MSE of the CV-selected model")
    return fig


def main() -> None:
    set_style()
    e1 = load_result("e1_ols_degree")
    print(save_figure(plot_estimators(e1), "ols_estimators_vs_degree", "cross_validation"))
    print(save_figure(plot_extrapolation(), "cv_extrapolation_fold", "cross_validation"))
    e2 = load_result("e2_model_selection")
    print(save_figure(plot_cv_heatmaps(e2), "cv_heatmaps", "cross_validation"))
    print(save_figure(plot_final_comparison(e2), "final_comparison", "cross_validation"))


if __name__ == "__main__":
    main()
