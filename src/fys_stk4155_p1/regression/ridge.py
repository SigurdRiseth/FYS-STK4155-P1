"""Ridge regression."""

import numpy as np
from numpy.typing import NDArray

from fys_stk4155_p1.regression.base import LinearModel


class Ridge(LinearModel):
    """Ridge regression via the normal equations.

    Minimizes (1/n) ||y - X theta||^2 + lam * ||theta||^2. Its normal equations are
    (X^T X + n * lam * I) theta = X^T y; the ridge term makes the system
    well-conditioned, so np.linalg.solve is used rather than the
    pseudoinverse. Fits no intercept column: `LinearModel.fit` centers `y`
    before calling `_fit_centered`, so every coefficient here is penalized --
    there is no separate, unpenalized intercept coefficient to exclude.
    `intercept_` (the training mean) is added back at `predict`.

    Args:
        lam: Non-negative regularization strength. lam = 0 recovers OLS.
    """

    def __init__(self, lam: float) -> None:
        self.lam = lam

    def _fit_centered(self, X: NDArray[np.float64], y_centered: NDArray[np.float64]) -> None:
        """Fit theta by penalized least squares against the centered target.

        Args:
            X: Design matrix, shape (n_samples, n_features).
            y_centered: Mean-centered targets, shape (n_samples,).

        Sets:
            coef_: The fitted coefficients, shape (n_features,).

        Raises:
            ValueError: If `lam` is negative.
        """
        if self.lam < 0:
            raise ValueError(f"lam must be non-negative, got {self.lam}.")

        n_samples, n_features = X.shape
        penalty = np.eye(n_features)

        self.coef_ = np.linalg.solve(X.T @ X + n_samples * self.lam * penalty, X.T @ y_centered)
