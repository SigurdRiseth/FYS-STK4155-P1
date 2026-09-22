"""
Generate report figures for the stochastic gradient descent experiment (part
H): how mini-batch size trades off against computational cost (Section 4.7 of
the lecture notes), and a with-vs-without-SGD head-to-head on final accuracy
(relative to the OLS/Ridge closed-form solutions of parts A/B) and
computational cost (FLOPs and wall-clock time).

Usage:
    uv run python scripts/generate_sgd_figures.py [--degree 8] [--n 200]
        [--noise 0.1] [--seed 42] [--lam 0.01] [--n-epochs 500]
        [--out-subdir sgd]
"""

import argparse
import time
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.plotting import FIGSIZE_WIDE, save_figure, set_style

from fys_stk4155_p1.data.design_matrix import univariate_polynomial_design_matrix
from fys_stk4155_p1.data.runge import generate_runge_data
from fys_stk4155_p1.regression.cost import cost, hessian_max_eigenvalue
from fys_stk4155_p1.regression.gradient_descent import GradientDescent
from fys_stk4155_p1.regression.ridge import Ridge


def _standardize(
    X_train: NDArray[np.float64], X_test: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Scale non-intercept columns (fit on train only); leave column 0 untouched."""
    scaler = StandardScaler()
    X_train_s = np.column_stack([X_train[:, :1], scaler.fit_transform(X_train[:, 1:])])
    X_test_s = np.column_stack([X_test[:, :1], scaler.transform(X_test[:, 1:])])
    return X_train_s, X_test_s


def plot_batch_size_sweep(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    batch_sizes: list[int],
    n_epochs: int,
) -> Figure:
    """Cost vs. cumulative FLOPs, one line per mini-batch size, for plain and Adam.

    Epoch count alone isn't a fair x-axis across batch sizes (a smaller batch
    means more, cheaper updates per epoch); cumulative FLOPs
    (`GradientDescent.cost_flops_`) is, regardless of batch size.
    """
    gamma_max = 2 / hessian_max_eigenvalue(X_train, lam=0.0, fit_intercept_column=True)
    closed_form = Ridge(lam=0.0, fit_intercept_column=True).fit(X_train, y_train)
    closed_form_cost = cost(X_train, y_train, closed_form.coef_, lam=0.0, fit_intercept_column=True)

    optimizers: list[Literal["plain", "adam"]] = ["plain", "adam"]
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, sharey=True)
    for ax, optimizer in zip(axes, optimizers, strict=True):
        for batch_size in batch_sizes:
            gd = GradientDescent(
                learning_rate=0.9 * gamma_max,
                lam=0.0,
                optimizer=optimizer,
                batch_size=batch_size,
                n_epochs=n_epochs,
                learning_rate_schedule="time_based",
                lr_decay=0.02,
                fit_intercept_column=True,
                random_state=0,
            ).fit(X_train, y_train)
            ax.plot(gd.cost_flops_, gd.cost_history_, label=f"batch={batch_size}")

        ax.axhline(closed_form_cost, color="black", ls=":", lw=1, label="closed-form")
        ax.set_yscale("log")
        ax.set_xlabel("cumulative FLOPs")
        ax.set_title(optimizer)
        ax.legend(fontsize="small")

    axes[0].set_ylabel("Cost")
    fig.suptitle("SGD: cost vs. computational cost, by mini-batch size (OLS, degree 8)")
    fig.tight_layout()
    return fig


def plot_with_vs_without_sgd(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    lambdas: dict[str, float],
    n_epochs: int,
) -> Figure:
    """Final cost (relative to closed-form) vs. wall-clock time, full-batch vs. SGD."""
    configs: list[tuple[str, dict[str, Any]]] = [
        ("full-batch, plain", {"optimizer": "plain", "max_iter": n_epochs, "tol": 0.0}),
        ("full-batch, adam", {"optimizer": "adam", "max_iter": n_epochs, "tol": 0.0}),
        (
            "SGD, adam, batch=8",
            {
                "optimizer": "adam",
                "batch_size": 8,
                "n_epochs": n_epochs,
                "learning_rate_schedule": "time_based",
                "lr_decay": 0.02,
                "random_state": 0,
            },
        ),
    ]

    fig, axes = plt.subplots(1, len(lambdas), figsize=FIGSIZE_WIDE, sharey=True)
    for ax, (label, lam) in zip(axes, lambdas.items(), strict=True):
        gamma_max = 2 / hessian_max_eigenvalue(X_train, lam=lam, fit_intercept_column=True)
        closed_form = Ridge(lam=lam, fit_intercept_column=True).fit(X_train, y_train)
        closed_form_cost = cost(X_train, y_train, closed_form.coef_, lam, fit_intercept_column=True)

        for name, kwargs in configs:
            start = time.perf_counter()
            gd = GradientDescent(
                learning_rate=0.9 * gamma_max, lam=lam, fit_intercept_column=True, **kwargs
            ).fit(X_train, y_train)
            elapsed_ms = (time.perf_counter() - start) * 1000
            ratio = gd.cost_history_[-1] / closed_form_cost
            ax.scatter([elapsed_ms], [ratio], label=name, s=40)

        ax.axhline(1.0, color="black", ls=":", lw=1)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("wall-clock time (ms)")
        ax.set_title(label)
        ax.legend(fontsize="small")

    axes[0].set_ylabel("final cost / closed-form")
    fig.suptitle("With vs. without SGD: accuracy vs. wall-clock cost")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=8, help="polynomial degree")
    parser.add_argument("--n", type=int, default=200, help="number of sample points")
    parser.add_argument("--noise", type=float, default=0.1, help="Gaussian noise std")
    parser.add_argument("--seed", type=int, default=42, help="RNG / train-test-split seed")
    parser.add_argument("--test-size", type=float, default=0.2, help="held-out test fraction")
    parser.add_argument("--lam", type=float, default=0.01, help="Ridge penalty strength")
    parser.add_argument(
        "--n-epochs", type=int, default=500, help="epochs / full-batch iterations per fit"
    )
    parser.add_argument("--out-subdir", default="sgd", help="subdirectory under docs/figures/")
    args = parser.parse_args()

    set_style()

    x, y = generate_runge_data(n=args.n, noise_std=args.noise, seed=args.seed)
    X_full = univariate_polynomial_design_matrix(x=x, degree=args.degree)
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=args.test_size, random_state=args.seed
    )
    X_train, _ = _standardize(X_train, X_test)

    lambdas = {"OLS": 0.0, "Ridge": args.lam}

    batch_sizes = [8, 32, 128, X_train.shape[0]]
    batch_fig = plot_batch_size_sweep(X_train, y_train, batch_sizes, n_epochs=args.n_epochs)
    print(f"Wrote {save_figure(batch_fig, 'sgd_batch_size_sweep', args.out_subdir)}")

    cost_fig = plot_with_vs_without_sgd(X_train, y_train, lambdas, n_epochs=args.n_epochs)
    print(f"Wrote {save_figure(cost_fig, 'sgd_with_vs_without', args.out_subdir)}")


if __name__ == "__main__":
    main()
