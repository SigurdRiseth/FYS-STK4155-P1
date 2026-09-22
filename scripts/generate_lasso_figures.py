"""
Generate report figures for the Lasso experiment (part G): a sparsity path
showing coefficients shrinking to zero as the L1 penalty grows, and an
optimizer convergence comparison showing how Lasso-via-gradient-descent
compares to OLS's degree-8 conditioning problem (the L1 penalty adds no
Hessian curvature, unlike Ridge's L2 term).

Like parts A-D (`regression.degree_sweep.fit_polynomial_degree_sweep`), Lasso
here is fit without a bias term: the design matrix has no intercept column,
features are standardized, and the target is centered, with the intercept
recovered separately as `y_train.mean()` rather than fit as an extra
(unpenalized) coefficient.

Usage:
    uv run python scripts/generate_lasso_figures.py [--n 200] [--noise 0.1]
        [--seed 42] [--path-degree 4] [--conditioning-degree 8]
        [--out-subdir lasso]
"""

import argparse
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.plotting import FIGSIZE_WIDE, save_figure, set_style

from fys_stk4155_p1.data.design_matrix import univariate_polynomial_design_matrix
from fys_stk4155_p1.data.runge import generate_runge_data
from fys_stk4155_p1.regression.cost import hessian_max_eigenvalue
from fys_stk4155_p1.regression.gradient_descent import GradientDescent
from fys_stk4155_p1.regression.shrinkage import lasso_coefficient_path


def _standardize(
    X_train: NDArray[np.float64], X_test: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Standardize every column (fit on train only); no intercept column to skip."""
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)


def plot_sparsity_path(
    X_train: NDArray[np.float64],
    y_train_centered: NDArray[np.float64],
    lambdas: NDArray[np.float64],
    learning_rate: float,
    max_iter: int,
) -> Figure:
    """Coefficients vs. lambda: each line is one polynomial coefficient's path."""
    path = lasso_coefficient_path(
        X_train,
        y_train_centered,
        lambdas,
        learning_rate=learning_rate,
        optimizer="adam",
        max_iter=max_iter,
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for j in range(path.shape[1]):
        ax.plot(lambdas, path[:, j], marker="o", markersize=3, label=rf"$\theta_{{{j + 1}}}$")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\theta_j$")
    ax.set_title("Lasso coefficient path")
    ax.legend(fontsize="small", ncol=2)
    fig.tight_layout()
    return fig


def plot_optimizer_convergence(
    X_train: NDArray[np.float64],
    y_train_centered: NDArray[np.float64],
    lam: float,
    max_iter: int,
    safety_factor: float,
) -> Figure:
    """Cost vs. iteration per optimizer, at a learning rate set from the OLS
    Hessian (lam=0): the L1 penalty adds no curvature, so unlike Ridge that is
    the right reference scale for Lasso too (see cost.hessian_max_eigenvalue).
    """
    gamma_max = 2 / hessian_max_eigenvalue(X_train, lam=0.0)
    learning_rate = safety_factor * gamma_max

    optimizers: list[Literal["plain", "momentum", "adagrad", "rmsprop", "adam"]] = [
        "plain",
        "momentum",
        "adagrad",
        "rmsprop",
        "adam",
    ]
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for optimizer in optimizers:
        gd = GradientDescent(
            learning_rate=learning_rate,
            lam=lam,
            max_iter=max_iter,
            optimizer=optimizer,
            penalty="l1",
        ).fit(X_train, y_train_centered)
        ax.plot(
            np.arange(1, gd.n_iter_ + 1),
            gd.cost_history_,
            label=f"{optimizer} ({gd.n_iter_} iters)",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set_title(
        rf"Lasso optimizer comparison ($\lambda={lam:g}$, "
        rf"$\gamma={safety_factor:g}\times 2/\lambda_{{max}}(H_{{OLS}})$)"
    )
    ax.legend(fontsize="small")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="number of sample points")
    parser.add_argument("--noise", type=float, default=0.1, help="Gaussian noise std")
    parser.add_argument("--seed", type=int, default=42, help="RNG / train-test-split seed")
    parser.add_argument("--test-size", type=float, default=0.2, help="held-out test fraction")
    parser.add_argument(
        "--path-degree", type=int, default=4, help="polynomial degree for the sparsity path"
    )
    parser.add_argument(
        "--conditioning-degree",
        type=int,
        default=8,
        help="polynomial degree for the optimizer comparison (matches parts E/F)",
    )
    parser.add_argument("--out-subdir", default="lasso", help="subdirectory under docs/figures/")
    args = parser.parse_args()

    set_style()

    x, y = generate_runge_data(n=args.n, noise_std=args.noise, seed=args.seed)

    X_path_full = univariate_polynomial_design_matrix(x=x, degree=args.path_degree, intercept=False)
    X_path_train, X_path_test, y_path_train, _ = train_test_split(
        X_path_full, y, test_size=args.test_size, random_state=args.seed
    )
    X_path_train, _ = _standardize(X_path_train, X_path_test)
    y_path_train_centered = y_path_train - y_path_train.mean()

    lambdas = np.logspace(-3, 1, 25)
    sparsity_fig = plot_sparsity_path(
        X_path_train, y_path_train_centered, lambdas, learning_rate=0.05, max_iter=5000
    )
    print(f"Wrote {save_figure(sparsity_fig, 'lasso_sparsity_path', args.out_subdir)}")

    X_cond_full = univariate_polynomial_design_matrix(
        x=x, degree=args.conditioning_degree, intercept=False
    )
    X_cond_train, X_cond_test, y_cond_train, _ = train_test_split(
        X_cond_full, y, test_size=args.test_size, random_state=args.seed
    )
    X_cond_train, _ = _standardize(X_cond_train, X_cond_test)
    y_cond_train_centered = y_cond_train - y_cond_train.mean()

    convergence_fig = plot_optimizer_convergence(
        X_cond_train, y_cond_train_centered, lam=0.05, max_iter=3000, safety_factor=0.9
    )
    print(f"Wrote {save_figure(convergence_fig, 'lasso_optimizer_convergence', args.out_subdir)}")


if __name__ == "__main__":
    main()
