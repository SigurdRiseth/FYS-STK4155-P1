"""Shared estimator interface for the linear regression models."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class LinearModel(ABC):
    """Base class for linear regression estimators.

    Follows the scikit-learn estimator convention (fit/predict, hyperparameters
    set in __init__) so subclasses can be dropped into scikit-learn tools such
    as KFold and cross_validate. Subclasses implement fit() to compute coef_
    and store it on self; this base class does not add an intercept column, so
    prepend a column of ones to X beforehand if a bias term is needed.
    """

    coef_: NDArray[np.float64]

    @abstractmethod
    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "LinearModel":
        """Fit the model to (X, y) and return self."""

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict targets for X using the fitted coefficients.

        Args:
            X: Design matrix, shape (n_samples, n_features).

        Returns:
            Predicted targets, shape (n_samples,).
        """
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_

    @staticmethod
    def _validate_inputs(
        X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y disagree on n_samples: {X.shape[0]} vs {y.shape[0]}.")

        return X, y
