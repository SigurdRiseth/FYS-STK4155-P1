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

def analytical_gradient(X, y, theta, lam=0.0, fit_intercept_column=False) -> NDArray[np.float64]:
    """(2/n) X^T (X theta - y) + 2*lam*theta, theta[0] excluded from the
    penalty term when fit_intercept_column."""
    pass

def hessian_max_eigenvalue(X, lam=0.0, fit_intercept_column=False) -> np.float64:
    """Largest eigenvalue of (2/n) X^T X (+ 2*lam*I), via np.linalg.eigvalsh.
    Plain GD is stable for learning_rate < 2 / hessian_max_eigenvalue(...);
    used by the figure script to annotate the stability sweep for part E."""
    pass
