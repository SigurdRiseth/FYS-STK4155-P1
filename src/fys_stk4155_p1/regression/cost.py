"""Cost, gradient, and Hessian eigenvalue for OLS/Ridge (L2) and Lasso (L1), used by
gradient descent."""

import numpy as np
from numpy.typing import NDArray

from fys_stk4155_p1.metrics import mean_squared_error


def cost(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    theta: NDArray[np.float64],
    lam: float = 0.0,
    fit_intercept_column: bool = False,
) -> np.float64:
    """Compute the mean squared error with L2 regularization.

    The regularization penalty is excluded from the intercept term when
    ``fit_intercept_column`` is True.

    This matches Eq. (3.43) in Hjorth-Jensen (2026).

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        y: True target values, shape (n_samples,).
        theta: Model parameters, shape (n_features,).
        lam: L2 regularization strength. Defaults to 0.0.
        fit_intercept_column: Whether the first column of X/theta is an
            intercept and should therefore be excluded from regularization.
            Defaults to False.

    Returns:
        The mean squared error plus the L2 regularization penalty.
    """
    y_pred = X @ theta

    penalty_theta = theta[1:] if fit_intercept_column else theta

    return mean_squared_error(y, y_pred) + lam * np.sum(penalty_theta**2)


def analytical_gradient(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    theta: NDArray[np.float64],
    lam: float = 0.0,
    fit_intercept_column: bool = False,
) -> NDArray[np.float64]:
    """Gradient of ``cost`` with respect to theta.

    (2/n) X^T (X theta - y) + 2*lam*theta, with theta[0]'s penalty
    contribution zeroed out (not dropped, so the return shape still matches
    theta) when fit_intercept_column is True.

    This matches Eq. (4.17) from Hjorth-Jensen.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        y: True target values, shape (n_samples,).
        theta: Model parameters, shape (n_features,).
        lam: L2 regularization strength. Defaults to 0.0.
        fit_intercept_column: Whether the first column of X/theta is an
            intercept and should therefore be excluded from regularization.
            Defaults to False.

    Returns:
        The gradient of the cost with respect to theta, shape (n_features,).
    """
    n = X.shape[0]

    penalty_theta = np.copy(theta)
    if fit_intercept_column:
        penalty_theta[0] = 0.0

    return 2 / n * X.T @ (X @ theta - y) + 2 * lam * penalty_theta


def lasso_cost(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    theta: NDArray[np.float64],
    lam: float = 0.0,
    fit_intercept_column: bool = False,
) -> np.float64:
    """Compute the mean squared error with L1 (Lasso) regularization.

    Same shape as `cost`, but penalizes ``sum(abs(theta))`` instead of
    ``sum(theta**2)``. The regularization penalty is excluded from the
    intercept term when ``fit_intercept_column`` is True.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        y: True target values, shape (n_samples,).
        theta: Model parameters, shape (n_features,).
        lam: L1 regularization strength. Defaults to 0.0.
        fit_intercept_column: Whether the first column of X/theta is an
            intercept and should therefore be excluded from regularization.
            Defaults to False.

    Returns:
        The mean squared error plus the L1 regularization penalty.
    """
    y_pred = X @ theta

    penalty_theta = theta[1:] if fit_intercept_column else theta

    return mean_squared_error(y, y_pred) + lam * np.sum(np.abs(penalty_theta))


def lasso_subgradient(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    theta: NDArray[np.float64],
    lam: float = 0.0,
    fit_intercept_column: bool = False,
) -> NDArray[np.float64]:
    """A subgradient of `lasso_cost` with respect to theta.

    (2/n) X^T (X theta - y) + lam*sign(theta), with theta[0]'s penalty
    contribution zeroed out (not dropped, so the return shape still matches
    theta) when fit_intercept_column is True.

    ``|theta_j|`` is not differentiable at ``theta_j = 0``; its subdifferential
    there is the interval [-1, 1]. This uses ``np.sign(0) == 0``, the
    zero-valued member of that interval. JAX's autodiff instead returns
    ``+1.0`` at exactly 0 (see `regression.autodiff.lasso_autodiff_gradient`)
    — a different, equally valid subgradient. The two therefore agree
    everywhere except exactly at theta_j = 0 (see
    `tests/regression/test_autodiff.py`).

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        y: True target values, shape (n_samples,).
        theta: Model parameters, shape (n_features,).
        lam: L1 regularization strength. Defaults to 0.0.
        fit_intercept_column: Whether the first column of X/theta is an
            intercept and should therefore be excluded from regularization.
            Defaults to False.

    Returns:
        A subgradient of the L1-penalized cost with respect to theta, shape
        (n_features,).
    """
    n = X.shape[0]

    penalty_theta = np.copy(theta)
    if fit_intercept_column:
        penalty_theta[0] = 0.0

    return 2 / n * X.T @ (X @ theta - y) + lam * np.sign(penalty_theta)


def sklearn_alpha_to_lam(alpha: float) -> float:
    """Convert scikit-learn's Lasso `alpha` to this package's `lam`.

    `lasso_cost` uses the full mean squared error,
    ``(1/n)||y - X theta||^2 + lam*||theta||_1``, while scikit-learn's
    ``Lasso`` objective halves it, ``(1/(2n))||y - Xw||^2 + alpha*||w||_1``.
    The two objectives are proportional (share the same minimizer) exactly
    when ``lam = 2*alpha``, since halving `lasso_cost` recovers scikit-learn's
    form with ``alpha = lam/2``.

    Args:
        alpha: scikit-learn's Lasso penalty strength.

    Returns:
        The equivalent `lam` for `lasso_cost`/`lasso_subgradient`.
    """
    return 2.0 * alpha


def hessian_max_eigenvalue(
    X: NDArray[np.float64], lam: float = 0.0, fit_intercept_column: bool = False
) -> np.float64:
    """Largest eigenvalue of (2/n) X^T X (+ 2*lam*I), via np.linalg.eigvalsh.

    The intercept row/column of the penalty term is zeroed out when
    fit_intercept_column is True, mirroring ridge.Ridge.fit's penalty
    matrix.

    This matches Eq. (4.17) from Hjorth-Jensen (2026), which is also
    analytical_gradient's linear (in theta) coefficient: differentiating
    (2/n) X^T (X theta - y) + 2*lam*theta once more with respect to theta
    gives this Hessian.

    Plain GD is stable for learning_rate < 2 / hessian_max_eigenvalue(...);
    used by the figure script to annotate the stability sweep for part E.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        lam: L2 regularization strength. Defaults to 0.0.
        fit_intercept_column: Whether the first column of X/theta is an
            intercept and should therefore be excluded from regularization.
            Defaults to False.

    Returns:
        The largest eigenvalue of the cost's Hessian with respect to theta.
    """
    n_samples, n_features = X.shape

    penalty = np.eye(n_features)
    if fit_intercept_column:
        penalty[0, 0] = 0.0

    hessian = 2 / n_samples * X.T @ X + 2 * lam * penalty

    # eigvalsh returns eigenvalues in ascending order for symmetric input.
    return np.linalg.eigvalsh(hessian)[-1]
