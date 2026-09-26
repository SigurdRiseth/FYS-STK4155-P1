"""
Report figures and table for parts e) and f) (gradient descent):

* plain GD with analytical and automatic-differentiation gradients converging
  to the closed-form OLS/Ridge solutions (GD_DEGREE, step 1/L);
* stability: cost after a fixed number of steps vs. learning rate, relative
  to the bound 2/lambda_max(H);
* learning-rate sensitivity of all optimizers and the iterations-to-tolerance
  table (from e4).

All settings from utils/config.py; the e4 parts need
data/results/e4_optimizers.json (`make experiments`).

Usage: uv run python scripts/generate_gradient_descent_figures.py

LLM-assisted: Claude (claude-opus-5-5, Claude Cowork desktop app, September 2026)
wrote/rewrote this file (level 4) to use the shared settings in utils/config.py.
TODO(author): describe your review/changes.
"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from utils import config as C
from utils.common import data, design, load_result, write_table
from utils.plotting import COLORS, COLUMN, LABELS, WIDE, save_figure, set_style

from fys_stk4155_p1.regression.cost import cost, hessian_max_eigenvalue
from fys_stk4155_p1.regression.gradient_descent import GradientDescent
from fys_stk4155_p1.regression.ridge import Ridge

OPTIMIZERS = ["plain", "momentum", "adagrad", "rmsprop", "adam"]


def _problem() -> tuple[NDArray, NDArray]:
    _, _, x_tr, _, y_tr, _ = data()
    (X,) = design(x_tr, C.GD_DEGREE)
    # cost()/hessian_max_eigenvalue() never center y themselves; GradientDescent
    # and Ridge do it internally (see regression.base.LinearModel), so pre-centering
    # here keeps the two consistent.
    return X, y_tr - y_tr.mean()


def plot_convergence(X: NDArray, y: NDArray, max_iter: int = 6000) -> Figure:
    fig, ax = plt.subplots(figsize=COLUMN)
    methods: list[Literal["analytical", "autodiff"]] = ["analytical", "autodiff"]
    for lam, color, name in ((0.0, COLORS["ols"], "OLS"), (C.GD_LAMBDA, COLORS["ridge"], "Ridge")):
        L = float(hessian_max_eigenvalue(X, lam=lam))
        theta_star = Ridge(lam=lam).fit(X, y).coef_
        J_star = float(cost(X, y, theta_star, lam))
        for method, ls, lw in zip(methods, ("-", "--"), (2.2, 0.9), strict=True):
            gd = GradientDescent(
                learning_rate=1 / L,
                lam=lam,
                max_iter=max_iter,
                tol=0.0,
                gradient_method=method,
            ).fit(X, y)
            gap = (gd.cost_history_ - J_star) / J_star
            lab = f"{name}, {method}" + (
                rf" ($\lambda=10^{{{int(np.log10(lam))}}}$)" if lam else ""
            )
            ax.plot(
                np.arange(1, gd.n_iter_ + 1),
                np.maximum(gap, 1e-16),
                ls=ls,
                lw=lw,
                color=color,
                alpha=0.6 if ls == "-" else 1.0,
                label=lab,
            )
    ax.set_yscale("log")
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"$(C(\theta_k)-C(\hat\theta))/C(\hat\theta)$")
    ax.legend(loc="upper right")
    return fig


def plot_stability(X: NDArray, y: NDArray, n_steps: int = 2000) -> Figure:
    ratios = np.linspace(0.05, 1.3, 26)
    fig, ax = plt.subplots(figsize=COLUMN)
    for lam, color, name in ((0.0, COLORS["ols"], "OLS"), (C.GD_LAMBDA, COLORS["ridge"], "Ridge")):
        gamma_max = 2 / float(hessian_max_eigenvalue(X, lam=lam))
        theta_star = Ridge(lam=lam).fit(X, y).coef_
        J_star = float(cost(X, y, theta_star, lam))
        gaps = []
        with np.errstate(over="ignore", invalid="ignore"):
            for r in ratios:
                gd = GradientDescent(
                    learning_rate=r * gamma_max,
                    lam=lam,
                    max_iter=n_steps,
                    tol=0.0,
                ).fit(X, y)
                gaps.append((gd.cost_history_[-1] - J_star) / J_star)
        gaps_arr = np.clip(np.nan_to_num(np.asarray(gaps), nan=1e12, posinf=1e12), 1e-16, 1e12)
        ax.plot(ratios, gaps_arr, "o-", color=color, label=name)
    ax.axvline(1.0, color="black", ls=":", lw=0.8)
    ax.text(0.97, 1e8, r"$\gamma = 2/\lambda_{\max}(\mathbf{H})$", fontsize=7, ha="right")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\gamma\,/\,(2/\lambda_{\max}(\mathbf{H}))$")
    ax.set_ylabel(f"Relative cost gap after {n_steps} steps")
    ax.legend(loc="lower left")
    return fig


def plot_lr_sensitivity(e4: dict) -> Figure:
    keys = [
        ("deg5_lam0", "(a) OLS, $p=5$"),
        (
            f"deg10_lam{C.GD_LAMBDA:g}",
            rf"(b) Ridge, $p=10$, $\lambda=10^{{{int(np.log10(C.GD_LAMBDA))}}}$",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=WIDE, sharey=True)
    for ax, (key, title) in zip(axes, keys, strict=True):
        s = e4["summary"][key]
        for opt in OPTIMIZERS:
            rs = sorted(
                (
                    r
                    for r in e4["runs"]
                    if r["degree"] == s["degree"]
                    and r["lambda"] == s["lambda"]
                    and r["optimizer"] == opt
                ),
                key=lambda r: r["learning_rate"],
            )
            lr = np.array([r["learning_rate"] for r in rs])
            it = np.array(
                [r["iters_param"] if r["iters_param"] is not None else np.nan for r in rs],
                dtype=float,
            )
            ax.plot(lr, it, "o-", color=COLORS[opt], label=LABELS[opt])
        ax.axvline(s["hessian"]["2/L"], color="black", ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Learning rate $\gamma$")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel(rf"Iterations to $\|\theta-\hat\theta\|/\|\hat\theta\|<{C.GD_PARAM_TOL:g}$")
    axes[0].legend(loc="lower left", ncol=2)
    return fig


def _sci(v: float) -> str:
    mantissa, exponent = f"{v:.1e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def optimizer_table(e4: dict) -> str:
    rows = []
    for d, lam in C.GD_BENCH_PROBLEMS:
        s = e4["summary"][f"deg{d}_lam{lam:g}"]
        best = min((s[o]["iters"] for o in OPTIMIZERS if s[o]["iters"] is not None), default=None)
        cells = []
        for o in OPTIMIZERS:
            v = s[o]
            if v["iters"] is None:
                cells.append(rf"--\,\scriptsize{{[{v['final_param_err']:.2f}]}}")
                continue
            c = str(v["iters"])
            if v["iters"] == best:
                c = rf"\textbf{{{c}}}"
            cells.append(c + rf"\,\scriptsize{{({v['n_lr_converged']})}}")
        model = "OLS" if lam == 0 else r"Ridge"
        rows.append(
            f"{d} & {model} & {_sci(s['hessian']['kappa'])} & " + " & ".join(cells) + r" \\"
        )
    header = " & ".join(["$p$", "model", r"$\kappa(\bm{H})$"] + [LABELS[o] for o in OPTIMIZERS])
    return "\n".join(
        [
            r"\begin{tabular}{@{}llr" + "r" * len(OPTIMIZERS) + "@{}}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )


def main() -> None:
    set_style()
    X, y = _problem()
    print(save_figure(plot_convergence(X, y), "gd_convergence", "gradient_descent"))
    print(save_figure(plot_stability(X, y), "gd_stability_sweep", "gradient_descent"))
    e4 = load_result("e4_optimizers")
    print(save_figure(plot_lr_sensitivity(e4), "optimizer_lr_sensitivity", "gradient_descent"))
    print(write_table("optimizers", optimizer_table(e4)))


if __name__ == "__main__":
    main()
