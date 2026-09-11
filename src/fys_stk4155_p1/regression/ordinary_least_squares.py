"""Ordinary least squares regression."""

import numpy as np
from numpy.typing import NDArray

from fys_stk4155_p1.regression.base import LinearModel


class OLS(LinearModel):
    """Ordinary least squares via the SVD-based pseudoinverse.

    Solves min_theta ||X @ theta - y||_2^2 using the SVD-based pseudoinverse,
    which is stable for rank-deficient or ill-conditioned X.

    Args:
        rcond: Singular values below rcond * max(s) are treated as zero.
            None uses a machine-precision default scaled by max(X.shape).

    LLM-assisted
    ------------
    Tool: Claude Opus 4.8 (September 2026)
    Role: Suggested the rcond parameter and singular-value cutoff logic for the SVD-based
        pseudoinverse.
    Modifications: Integrated the suggestion into the existing OLS implementation and verified the
        resulting behavior.
    """

    def __init__(self, rcond: float | None = None) -> None:
        self.rcond = rcond

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "OLS":
        """Fit theta by least squares.

        Args:
            X: Design matrix, shape (n_samples, n_features).
            y: Targets, shape (n_samples,).

        Returns:
            self, with coef_ set to the fitted coefficients, shape (n_features,).
        """
        X, y = self._validate_inputs(X, y)

        U, s, Vt = np.linalg.svd(X, full_matrices=False)

        rcond = self.rcond
        if rcond is None:
            rcond = np.finfo(np.float64).eps * max(X.shape)

        # LLM-assisted: Claude Opus 4.8 (September 2026) suggested the
        # singular-value cutoff and pseudoinverse handling below.
        cutoff = rcond * s[0] if s.size else 0.0
        s_inv = np.where(s > cutoff, 1.0 / s, 0.0)

        self.coef_ = Vt.T @ (s_inv * (U.T @ y))
        return self
