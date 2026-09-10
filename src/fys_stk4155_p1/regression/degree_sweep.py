"""Model-agnostic polynomial-degree sweep, shared by the OLS/Ridge experiments."""

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fys_stk4155_p1.data.design_matrix import univariate_polynomial_design_matrix
from fys_stk4155_p1.metrics import mean_squared_error, r2_score
from fys_stk4155_p1.regression.base import LinearModel


def fit_polynomial_degree_sweep(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    degrees: Iterable[int],
    model_factory: Callable[[], LinearModel],
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit a linear model on polynomial features of x, for each degree in `degrees`.

    Builds the design matrix [x, x^2, ..., x^max(degrees)] once (no intercept
    column), then for each degree slices out the first `degree` columns
    (x^1..x^degree), standardizes them with a scaler fit on the training split
    only, and fits `model_factory()` on the standardized features against the
    centered target `y_train - y_mean`. The intercept isn't a fitted column:
    since standardized features have zero mean, it's exactly `y_mean`, added
    back at prediction time.

    Model-agnostic so the same sweep drives both the OLS and Ridge
    polynomial-degree experiments: pass `OLS` itself, or e.g.
    `lambda: Ridge(lam=lam, fit_intercept_column=False)` for a fixed `lam`.

    Args:
        x: x-coordinates, shape (n,).
        y: targets, shape (n,).
        degrees: polynomial degrees to fit, e.g. range(1, 16).
        model_factory: Zero-argument callable returning a fresh, unfitted
            `LinearModel` for each degree.
        test_size: fraction of samples held out for testing.
        seed: seed for the train/test split, for reproducibility.

    Returns:
        Dict with keys "degrees", "intercept" (scalar `y_train.mean()`, identical
        across degrees since it only depends on the split), "weights" (list of
        theta arrays, one per degree; theta[i] is the coefficient of the
        standardized x^(i+1) term), "mse_train", "mse_test", "r2_train", "r2_test"
        (arrays aligned with "degrees").
    """
    degrees_arr = np.array(list(degrees))
    max_degree = int(degrees_arr.max())

    X_full = univariate_polynomial_design_matrix(x=x, degree=max_degree, intercept=False)

    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_full, y, test_size=test_size, random_state=seed
    )

    y_mean = y_train.mean()
    y_train_centered = y_train - y_mean

    weights = []
    mse_train, mse_test = [], []
    r2_train, r2_test = [], []

    for degree in degrees_arr:
        # No intercept column: X_full's columns are x^1..x^max_degree, so the
        # first `degree` columns are x^1..x^degree.
        X_train = X_train_full[:, :degree]
        X_test = X_test_full[:, :degree]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = model_factory().fit(X_train_s, y_train_centered)
        weights.append(model.coef_)

        y_pred_train = model.predict(X_train_s) + y_mean
        y_pred_test = model.predict(X_test_s) + y_mean

        mse_train.append(mean_squared_error(y_train, y_pred_train))
        mse_test.append(mean_squared_error(y_test, y_pred_test))
        r2_train.append(r2_score(y_train, y_pred_train))
        r2_test.append(r2_score(y_test, y_pred_test))

    return {
        "degrees": degrees_arr,
        "intercept": y_mean,
        "weights": weights,
        "mse_train": np.array(mse_train),
        "mse_test": np.array(mse_test),
        "r2_train": np.array(r2_train),
        "r2_test": np.array(r2_test),
    }
