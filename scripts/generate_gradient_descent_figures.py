"""
Generate report figures for the gradient descent experiment (part E): does
plain, fixed-learning-rate gradient descent converge to the closed-form
OLS/Ridge solution, how many iterations does it take, and how does the
largest stable learning rate relate to the cost Hessian's largest
eigenvalue (Section 4.5 of the lecture notes)?

Usage:
    uv run python scripts/generate_gradient_descent_figures.py [--degree 8]
        [--n 200] [--noise 0.1] [--seed 42] [--lam 0.01] [--max-iter 2000]
        [--out-subdir gradient_descent]
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


def plot_convergence(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    lambdas: dict[str, float],
    max_iter: int,
    safety_factor: float,
) -> Figure:
    """Cost vs. iteration for plain GD, analytical and autodiff gradients overlaid.

    Each panel's learning rate is `safety_factor * 2 / hessian_max_eigenvalue(...)`
    (Section 4.5), i.e. a fixed fraction of the largest stable rate for that
    panel's own cost surface, so both panels converge in a comparable number
    of steps despite Ridge's Hessian being better conditioned than OLS's.
    """
    # Independent y-scales: OLS's Hessian is far worse-conditioned than
    # Ridge's, so its cost sits on a different scale and a shared axis would
    # flatten out one panel's own convergence shape.
    fig, axes = plt.subplots(1, len(lambdas), figsize=FIGSIZE_WIDE, sharey=False)

    for ax, (label, lam) in zip(axes, lambdas.items(), strict=True):
        gamma_max = 2 / hessian_max_eigenvalue(X_train, lam=lam, fit_intercept_column=True)
        learning_rate = safety_factor * gamma_max

        closed_form = Ridge(lam=lam, fit_intercept_column=True).fit(X_train, y_train)
        closed_form_cost = cost(X_train, y_train, closed_form.coef_, lam, fit_intercept_column=True)

        # analytical and autodiff agree to ~1e-10 (test_autodiff.py), so their
        # cost histories coincide almost exactly; a thick solid line under a
        # thin dashed one shows both are actually drawn, rather than one
        # silently hiding behind the other.
        gradient_method_styles: dict[Literal["analytical", "autodiff"], dict[str, object]] = {
            "analytical": {"lw": 2.5, "ls": "-"},
            "autodiff": {"lw": 1.25, "ls": "--"},
        }
        for gradient_method, style in gradient_method_styles.items():
            gd = GradientDescent(
                learning_rate=learning_rate,
                lam=lam,
                max_iter=max_iter,
                gradient_method=gradient_method,
                fit_intercept_column=True,
            ).fit(X_train, y_train)
            ax.plot(
                np.arange(1, gd.n_iter_ + 1),
                gd.cost_history_,
                label=f"{gradient_method} ({gd.n_iter_} iters)",
                **style,
            )

        ax.axhline(closed_form_cost, color="black", ls="--", lw=1, label="closed-form")
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")
        ax.set_title(rf"{label} ($\gamma={safety_factor:g}\times 2/\lambda_{{max}}(H)$)")
        ax.legend(fontsize="small")

    fig.suptitle("Plain gradient descent: convergence to the closed-form solution")
    fig.tight_layout()
    return fig


def plot_stability_sweep(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    lambdas: dict[str, float],
    max_iter: int,
    ratios: NDArray[np.float64],
) -> Figure:
    """Final cost after max_iter plain-GD steps vs. learning rate, as a fraction of
    the theoretical stability boundary gamma_max = 2 / hessian_max_eigenvalue(...).
    """
    fig, axes = plt.subplots(1, len(lambdas), figsize=FIGSIZE_WIDE, sharey=True)

    for ax, (label, lam) in zip(axes, lambdas.items(), strict=True):
        gamma_max = 2 / hessian_max_eigenvalue(X_train, lam=lam, fit_intercept_column=True)

        closed_form = Ridge(lam=lam, fit_intercept_column=True).fit(X_train, y_train)
        closed_form_cost = cost(X_train, y_train, closed_form.coef_, lam, fit_intercept_column=True)

        final_costs = []
        # Beyond gamma_max the iterates blow up geometrically, reaching values from
        # merely huge to inf/nan within max_iter steps depending on how far past
        # the boundary ratio is; that's the point being demonstrated, but
        # matplotlib's log-scale tick formatter can't handle inf, so clip
        # everything to a fixed ceiling for display (errstate silences the
        # resulting overflow warnings).
        with np.errstate(over="ignore", invalid="ignore"):
            for ratio in ratios:
                gd = GradientDescent(
                    learning_rate=ratio * gamma_max,
                    lam=lam,
                    max_iter=max_iter,
                    tol=0.0,  # never stop early: compare the same fixed step budget
                    fit_intercept_column=True,
                ).fit(X_train, y_train)
                final_costs.append(gd.cost_history_[-1])
        cost_ceiling = max(closed_form_cost * 1e10, 1e10)
        final_costs = np.nan_to_num(final_costs, nan=cost_ceiling, posinf=cost_ceiling)
        final_costs = np.clip(final_costs, None, cost_ceiling)

        ax.plot(ratios, final_costs, marker="o", markersize=3)
        ax.axhline(closed_form_cost, color="black", ls="--", lw=1, label="closed-form")
        ax.axvline(1.0, color="tab:red", ls=":", lw=1, label=r"$\gamma = 2/\lambda_{max}(H)$")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\gamma \,/\, \gamma_{max}$")
        ax.set_title(label)
        ax.legend(fontsize="small")

    axes[0].set_ylabel(f"Cost after {max_iter} steps")
    fig.suptitle("Gradient descent stability vs. learning rate")
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
    parser.add_argument("--max-iter", type=int, default=2000, help="gradient steps per fit")
    parser.add_argument(
        "--out-subdir", default="gradient_descent", help="subdirectory under docs/figures/"
    )
    args = parser.parse_args()

    set_style()

    x, y = generate_runge_data(n=args.n, noise_std=args.noise, seed=args.seed)
    X_full = univariate_polynomial_design_matrix(x=x, degree=args.degree)
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=args.test_size, random_state=args.seed
    )
    X_train_s, _ = _standardize(X_train, X_test)

    lambdas = {"OLS": 0.0, "Ridge": args.lam}

    convergence_fig = plot_convergence(
        X_train_s, y_train, lambdas, max_iter=args.max_iter, safety_factor=0.9
    )
    print(f"Wrote {save_figure(convergence_fig, 'gd_convergence', args.out_subdir)}")

    ratios = np.linspace(0.1, 1.5, 25)
    stability_fig = plot_stability_sweep(
        X_train_s, y_train, lambdas, max_iter=args.max_iter, ratios=ratios
    )
    print(f"Wrote {save_figure(stability_fig, 'gd_stability_sweep', args.out_subdir)}")


if __name__ == "__main__":
    main()
