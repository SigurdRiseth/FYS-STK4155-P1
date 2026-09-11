"""
Generate report figures for the Ridge experiment (part b): the ridge trace
(coefficients vs. lambda) and the shrinkage of SVD modes vs. lambda, both at a
fixed, deliberately over-parameterized polynomial degree.

Usage:
    uv run python scripts/generate_ridge_figures.py [--degree 15] [--n 100]
        [--noise 0.1] [--seed 42] [--out-subdir ridge]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.plotting import FIGSIZE_WIDE, save_figure, set_style

from fys_stk4155_p1.data.design_matrix import univariate_polynomial_design_matrix
from fys_stk4155_p1.data.runge import generate_runge_data
from fys_stk4155_p1.regression.shrinkage import ridge_coefficient_path, singular_value_shrinkage


def _standardize(
    X_train: NDArray[np.float64], X_test: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Scale non-intercept columns (fit on train only); leave column 0 untouched."""
    scaler = StandardScaler()
    X_train_s = np.column_stack([X_train[:, :1], scaler.fit_transform(X_train[:, 1:])])
    X_test_s = np.column_stack([X_test[:, :1], scaler.transform(X_test[:, 1:])])
    return X_train_s, X_test_s


def plot_ridge_trace(
    X_train: NDArray[np.float64], y_train: NDArray[np.float64], degree: int, lambdas: NDArray
) -> Figure:
    theta_path = ridge_coefficient_path(X_train, y_train, lambdas, fit_intercept_column=True)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)  # extra width for the outside legend
    for coef_idx in range(1, degree + 1):  # skip the unpenalized intercept (index 0)
        ax.plot(lambdas, theta_path[:, coef_idx], lw=1, label=rf"$\theta_{{{coef_idx}}}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Coefficient value $\theta_i$")
    ax.set_title(rf"Ridge trace: coefficients vs. $\lambda$ (degree {degree})")
    ax.legend(ncol=2, fontsize="small", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    return fig


def plot_shrinkage(X_train: NDArray[np.float64], degree: int, lambdas: NDArray) -> Figure:
    singular_values, shrinkage = singular_value_shrinkage(X_train[:, 1:], lambdas)

    fig, ax = plt.subplots()
    mode_idx = np.arange(1, len(singular_values) + 1)
    for lam, row in zip(lambdas, shrinkage, strict=True):
        ax.plot(mode_idx, row, marker="o", markersize=3, label=rf"$\lambda={lam:g}$")
    ax.set_xlabel(r"Singular-value mode index (decreasing $s_i$)")
    ax.set_ylabel(r"Shrinkage factor $f_i(\lambda) = s_i^2/(s_i^2+n\lambda)$")
    ax.set_title(rf"Shrinkage of SVD modes vs. $\lambda$ (degree {degree})")
    ax.legend(fontsize="small")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=15, help="polynomial degree")
    parser.add_argument("--n", type=int, default=100, help="number of sample points")
    parser.add_argument("--noise", type=float, default=0.1, help="Gaussian noise std")
    parser.add_argument("--seed", type=int, default=42, help="RNG / train-test-split seed")
    parser.add_argument("--test-size", type=float, default=0.2, help="held-out test fraction")
    parser.add_argument("--out-subdir", default="ridge", help="subdirectory under docs/figures/")
    args = parser.parse_args()

    set_style()

    x, y = generate_runge_data(n=args.n, noise_std=args.noise, seed=args.seed)
    X_full = univariate_polynomial_design_matrix(x=x, degree=args.degree)
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=args.test_size, random_state=args.seed
    )
    X_train_s, _ = _standardize(X_train, X_test)

    trace_lambdas = np.logspace(-8, -2, 60)
    trace_fig = plot_ridge_trace(X_train_s, y_train, args.degree, trace_lambdas)
    print(f"Wrote {save_figure(trace_fig, f'ridge_trace_degree{args.degree}', args.out_subdir)}")

    shrinkage_lambdas = np.array([0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    shrinkage_fig = plot_shrinkage(X_train_s, args.degree, shrinkage_lambdas)
    shrinkage_path = save_figure(
        shrinkage_fig, f"ridge_shrinkage_degree{args.degree}", args.out_subdir
    )
    print(f"Wrote {shrinkage_path}")


if __name__ == "__main__":
    main()
