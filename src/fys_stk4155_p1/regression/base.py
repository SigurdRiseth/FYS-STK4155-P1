"""Shared estimator interface for the linear regression models."""

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin


class LinearModel(BaseEstimator, RegressorMixin, ABC):
    """Base class for linear regression estimators.

    Follows the scikit-learn estimator convention (fit/predict, hyperparameters
    set in __init__) so subclasses can be dropped into scikit-learn tools such
    as KFold and cross_validate. Inheriting BaseEstimator/RegressorMixin
    supplies get_params/set_params (needed by sklearn.base.clone, which
    cross_val_score/cross_validate call internally) and a default R^2 score().

    No subclass fits an intercept column. `fit` centers `y` on its training
    mean (stored as `intercept_`) and delegates to the abstract
    `_fit_centered`, so every subclass only ever sees a mean-zero target and
    never special-cases a bias term or excludes it from a penalty; `predict`
    adds `intercept_` back. This is exact whenever the columns of `X` are
    themselves mean-zero (e.g. after `StandardScaler`), since the penalized
    coefficients and the intercept then decouple completely: for any penalty
    on the non-intercept coefficients alone, the optimal intercept is exactly
    `y.mean()` regardless of their value.
    """

    coef_: NDArray[np.float64]
    intercept_: np.float64

    @abstractmethod
    def _fit_centered(self, X: NDArray[np.float64], y_centered: NDArray[np.float64]) -> None:
        """Fit `coef_` on `X` against the already mean-centered target `y_centered`."""

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> Self:
        """Fit the model to (X, y) and return self.

        Centers `y` on its training mean (stored as `intercept_`) before
        delegating to `_fit_centered`. `X` is never centered here; standardize
        it beforehand if desired (its columns must be mean-zero for
        `intercept_` to be the unpenalized optimum -- see the class
        docstring).

        Args:
            X: Design matrix, shape (n_samples, n_features).
            y: Target values, shape (n_samples,).

        Returns:
            self, with `coef_` and `intercept_` set.
        """
        X, y = self._validate_inputs(X, y)
        self.intercept_ = y.mean()
        self._fit_centered(X, y - self.intercept_)
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict targets for X using the fitted coefficients and intercept.

        Args:
            X: Design matrix, shape (n_samples, n_features).

        Returns:
            Predicted targets, shape (n_samples,).
        """
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_

    @staticmethod
    def _validate_inputs(
        X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Validate and convert a design matrix and target vector.

        Args:
            X: Design matrix with shape ``(n_samples, n_features)``.
            y: Target vector with shape ``(n_samples,)``.

        Returns:
            The inputs converted to ``float64`` NumPy arrays.

        Raises:
            ValueError: If either input has an invalid dimension or their
                numbers of samples do not match.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y disagree on n_samples: {X.shape[0]} vs {y.shape[0]}.")

        return X, y
