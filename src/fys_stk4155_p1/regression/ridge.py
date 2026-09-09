"""Ridge regression."""

import numpy as np
from numpy.typing import NDArray

from fys_stk4155_p1.regression.base import LinearModel


class Ridge(LinearModel):
    """Ridge regression via the normal equations.

    Solves (X^T X + lam * I) theta = X^T y for theta. The ridge term makes
    the system well-conditioned, so np.linalg.solve is used rather than the
    pseudoinverse.

    Args:
        lam: Non-negative regularization strength. lam = 0 recovers OLS.
        fit_intercept_column: If True, the first column of X is treated as an
            all-ones intercept term and is excluded from the penalty. Leave
            False when features are standardized and y is centered.
    """

    def __init__(self, lam: float, fit_intercept_column: bool = False) -> None:
        # TODO: discover whether it should be n*lam for cross-validation as it
        # should be a penalty *per observation* (ref. Section 3.10 of the lecture notes)
        self.lam = lam
        self.fit_intercept_column = fit_intercept_column

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "Ridge":
        """Fit theta by penalized least squares.

        Args:
            X: Design matrix, shape (n_samples, n_features).
            y: Target values, shape (n_samples,).

        Returns:
            self, with coef_ set to the fitted coefficients, shape (n_features,).
        """
        X, y = self._validate_inputs(X, y)
        if self.lam < 0:
            raise ValueError(f"lam must be non-negative, got {self.lam}.")

        n_features = X.shape[1]
        penalty = np.eye(n_features)
        if self.fit_intercept_column:
            penalty[0, 0] = 0.0  # don't shrink the intercept

        self.coef_ = np.linalg.solve(X.T @ X + self.lam * penalty, X.T @ y)
        return self
