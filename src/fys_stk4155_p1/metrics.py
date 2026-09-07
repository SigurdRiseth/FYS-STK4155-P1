"""Regression metrics."""

import numpy as np
from numpy.typing import NDArray


def mean_squared_error(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> np.float64:
    """Mean squared error between true and predicted targets.

    Args:
        y_true: True targets, shape (n_samples,).
        y_pred: Predicted targets, shape (n_samples,).

    Returns:
        The mean squared error.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> np.float64:
    """R^2 (coefficient of determination) score.

    Args:
        y_true: True targets, shape (n_samples,).
        y_pred: Predicted targets, shape (n_samples,).

    Returns:
        The R^2 score. 1.0 is a perfect prediction; a constant model that
        always predicts mean(y_true) scores 0.0.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual_sum_squares = np.sum((y_true - y_pred) ** 2)
    total_sum_squares = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - residual_sum_squares / total_sum_squares
