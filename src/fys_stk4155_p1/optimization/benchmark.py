"""Iterations-to-tolerance benchmark for the optimizers against a known minimizer."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from fys_stk4155_p1.optimization.optimizers import OPTIMIZER_REGISTRY
from fys_stk4155_p1.regression.cost import analytical_gradient, cost


def iterations_to_tolerance(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    theta_star: NDArray[np.float64],
    optimizer: str,
    learning_rate: float,
    lam: float = 0.0,
    fit_intercept_column: bool = False,
    param_tol: float = 1e-3,
    cost_tol: float = 1e-6,
    max_iter: int = 100_000,
    divergence_factor: float = 1e6,
    **optimizer_kwargs: Any,
) -> dict[str, Any]:
    """Run one optimizer from theta = 0 on the L2 cost and record when it gets close
    to a known minimizer `theta_star` (e.g. the closed-form OLS/Ridge solution).

    Two stopping criteria are tracked independently, since on an ill-conditioned
    problem the cost can be near-optimal long before the parameters are:

    * parameter error: ``||theta_k - theta_star|| / ||theta_star|| < param_tol``;
    * cost gap: ``(J(theta_k) - J(theta_star)) / J(theta_star) < cost_tol``.

    The run stops once both are met, after `max_iter` steps, or when the
    parameter error exceeds `divergence_factor` or becomes non-finite.

    Args:
        X: Design matrix, shape (n_samples, n_features).
        y: Targets, shape (n_samples,).
        theta_star: Known minimizer of `cost(X, y, ., lam, fit_intercept_column)`.
        optimizer: Key of `OPTIMIZER_REGISTRY`.
        learning_rate: Optimizer step size.
        lam: L2 penalty strength (0 for OLS).
        fit_intercept_column: Exclude column 0 from the penalty.
        param_tol: Relative parameter-error tolerance.
        cost_tol: Relative cost-gap tolerance.
        max_iter: Maximum number of steps.
        divergence_factor: Relative parameter error treated as divergence.
        **optimizer_kwargs: Passed to the optimizer constructor (e.g. `beta`).

    Returns:
        Dict with "iters_param" and "iters_cost" (first step count at which each
        criterion held, or None), "final_param_err", "final_cost_gap", "n_steps",
        and "diverged".

    LLM-assisted
    ------------
    Tool: Claude (claude-opus-5-5, Claude Cowork desktop app, September 2026)
    Role: Wrote the function.
    Verification: tests/optimization/test_benchmark.py.
    Modifications: TODO(author): describe your review/changes.
    """
    opt = OPTIMIZER_REGISTRY[optimizer](learning_rate=learning_rate, **optimizer_kwargs)
    n_features = X.shape[1]
    opt.reset(n_features)
    theta = np.zeros(n_features, dtype=np.float64)
    theta_norm = float(np.linalg.norm(theta_star))
    cost_star = float(cost(X, y, theta_star, lam, fit_intercept_column))

    iters_param: int | None = None
    iters_cost: int | None = None
    diverged = False
    param_err = np.inf
    cost_gap = np.inf
    k = 0
    for k in range(1, max_iter + 1):
        grad = analytical_gradient(X, y, theta, lam, fit_intercept_column)
        theta = opt.step(theta, grad)
        param_err = float(np.linalg.norm(theta - theta_star)) / theta_norm
        if not np.isfinite(param_err) or param_err > divergence_factor:
            diverged = True
            break
        if iters_param is None and param_err < param_tol:
            iters_param = k
        if iters_cost is None:
            cost_gap = (float(cost(X, y, theta, lam, fit_intercept_column)) - cost_star) / cost_star
            if cost_gap < cost_tol:
                iters_cost = k
        if iters_param is not None and iters_cost is not None:
            break

    if not diverged:
        cost_gap = (float(cost(X, y, theta, lam, fit_intercept_column)) - cost_star) / cost_star
    return {
        "iters_param": iters_param,
        "iters_cost": iters_cost,
        "final_param_err": param_err,
        "final_cost_gap": cost_gap,
        "n_steps": k,
        "diverged": diverged,
    }
