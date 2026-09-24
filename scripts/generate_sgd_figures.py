"""
Report figures for part h) (stochastic gradient descent), OLS and Ridge at
GD_DEGREE on the reference training split:

* relative cost gap vs. computational cost (FLOPs of the gradient
  evaluations) for several mini-batch sizes, plain SGD and Adam;
* a table of final accuracy, FLOPs and wall-clock time with and without
  mini-batches.

Learning rates: 1/L for plain (S)GD and 0.03 for Adam (the best fixed rate in
the e4 benchmark), both with the time-based decay eta_t = eta_0/(1 + 1e-3 t)
in SGD mode. All other settings from utils/config.py.

Usage: uv run python scripts/generate_sgd_figures.py

LLM-assisted: Claude (claude-opus-5-5, Claude Cowork desktop app, September 2026)
wrote/rewrote this file (level 4) to use the shared settings in utils/config.py.
TODO(author): describe your review/changes.
"""

import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from utils import config as C
from utils.common import data, design, write_table
from utils.plotting import WIDE, save_figure, set_style

from fys_stk4155_p1.regression.cost import cost, hessian_max_eigenvalue
from fys_stk4155_p1.regression.gradient_descent import GradientDescent
from fys_stk4155_p1.regression.ridge import Ridge

ADAM_LR = 0.03
LR_DECAY = 1e-3


def _problem() -> tuple[NDArray, NDArray]:
    _, _, x_tr, _, y_tr, _ = data()
    (X,) = design(x_tr, C.GD_DEGREE)
    return X, y_tr


def _lr(X: NDArray, optimizer: str, lam: float) -> float:
    if optimizer == "adam":
        return ADAM_LR
    return 1.0 / float(hessian_max_eigenvalue(X, lam=lam, fit_intercept_column=True))


def _gap(X: NDArray, y: NDArray, lam: float) -> Any:
    theta_star = Ridge(lam=lam, fit_intercept_column=True).fit(X, y).coef_
    J_star = float(cost(X, y, theta_star, lam, fit_intercept_column=True))
    return lambda history: (np.asarray(history) - J_star) / J_star


def plot_batch_size_sweep(X: NDArray, y: NDArray) -> Figure:
    gap = _gap(X, y, 0.0)
    fig, axes = plt.subplots(1, 2, figsize=WIDE, sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, opt, title in zip(axes, ("plain", "adam"), ("(a) plain SGD", "(b) Adam"), strict=True):
        for i, b in enumerate(C.SGD_BATCH_SIZES):
            full = b >= X.shape[0]
            kwargs: dict[str, Any] = (
                {"max_iter": C.SGD_EPOCHS, "tol": 0.0}
                if full
                else {
                    "batch_size": b,
                    "n_epochs": C.SGD_EPOCHS,
                    "random_state": 0,
                    "learning_rate_schedule": "time_based",
                    "lr_decay": LR_DECAY,
                }
            )
            gd = GradientDescent(
                learning_rate=_lr(X, opt, 0.0), optimizer=opt, fit_intercept_column=True, **kwargs
            ).fit(X, y)
            ax.plot(
                gd.cost_flops_,
                np.maximum(gap(gd.cost_history_), 1e-16),
                color=cmap(i / (len(C.SGD_BATCH_SIZES) - 1)),
                label=f"full batch ({b})" if full else f"batch {b}",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Cumulative FLOPs of gradient evaluations")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Relative cost gap (OLS)")
    axes[1].legend(loc="lower left")
    return fig


def _sci(v: float) -> str:
    mantissa, exponent = f"{v:.1e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def sgd_table(X: NDArray, y: NDArray, batch: int = 16, repeats: int = 3) -> str:
    """Final relative cost gap, FLOPs and wall-clock time after SGD_EPOCHS
    epochs, with and without mini-batches (median time over `repeats` runs)."""
    rows = []
    for name, lam in (("OLS", 0.0), ("Ridge", C.GD_LAMBDA)):
        gap = _gap(X, y, lam)
        for opt, label in (("plain", "plain"), ("adam", "Adam")):
            for b in (None, batch):
                kwargs: dict[str, Any] = (
                    {"max_iter": C.SGD_EPOCHS, "tol": 0.0}
                    if b is None
                    else {
                        "batch_size": b,
                        "n_epochs": C.SGD_EPOCHS,
                        "random_state": 0,
                        "learning_rate_schedule": "time_based",
                        "lr_decay": LR_DECAY,
                    }
                )
                times = []
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    gd = GradientDescent(
                        learning_rate=_lr(X, opt, lam),
                        lam=lam,
                        optimizer=opt,
                        fit_intercept_column=True,
                        **kwargs,
                    ).fit(X, y)
                    times.append((time.perf_counter() - t0) * 1e3)
                g = float(gap(gd.cost_history_)[-1])
                g_txt = (
                    r"$<10^{-15}$"
                    if g < 1e-15
                    else f"${g:.1e}$".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-")
                )
                if "times10" in g_txt:
                    g_txt = g_txt[:-1] + "}$"
                rows.append(
                    f"{name} & {label} & {'full' if b is None else b} & {gd.n_updates_} & "
                    f"{_sci(gd.cost_flops_[-1])} & {g_txt} & {np.median(times):.0f}" + r" \\"
                )
    return "\n".join(
        [
            r"\begin{tabular}{@{}lllrrrr@{}}",
            r"\toprule",
            r"model & optimizer & batch & updates & FLOPs & rel.\ gap & ms \\",
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
    print(save_figure(plot_batch_size_sweep(X, y), "sgd_batch_size_sweep", "sgd"))
    print(write_table("sgd", sgd_table(X, y)))


if __name__ == "__main__":
    main()
