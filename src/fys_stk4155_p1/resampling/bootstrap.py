"""Bootstrap resampling for bias-variance decomposition of the test error."""

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from fys_stk4155_p1.data.design_matrix import univariate_polynomial_design_matrix
from fys_stk4155_p1.regression.base import LinearModel
from fys_stk4155_p1.regression.ordinary_least_squares import OLS


def bootstrap_resample(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Draw one bootstrap sample (rows, with replacement) from (X, y).

    Args:
        X: Design matrix, shape (n_samples, n_features).
        y: Targets, shape (n_samples,).
        rng: Generator to draw resampling indices from; pass a single
            `np.random.default_rng(seed)` across calls for reproducibility.

    Returns:
        (X_resampled, y_resampled), each with `n_samples` rows drawn with
        replacement from the input.
    """
    idx = rng.integers(0, X.shape[0], size=X.shape[0])
    return X[idx], y[idx]


def bootstrap_bias_variance_sweep(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    degrees: Iterable[int],
    model_factory: Callable[[], LinearModel] = OLS,
    n_bootstraps: int = 100,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Bias-variance decomposition of the test MSE via bootstrap, per degree.

    For each degree, the polynomial design matrix (with an intercept column)
    is split once into train/test. `n_bootstraps` bootstrap resamples of the
    training split are then each used to fit a fresh `model_factory()` and
    predict on the fixed test split, giving a matrix of test predictions of
    shape (n_test, n_bootstraps). The test MSE, squared bias, and variance
    are estimated from that matrix following Hastie, Tibshirani & Friedman
    (ESL, eq. 7.9):

        error = E_boot[ (y_test - y_pred)^2 ]
        bias^2 = ( y_test - E_boot[y_pred] )^2   [in expectation, bias^2 + sigma^2]
        variance = Var_boot[y_pred]

    all further averaged over the test samples, so that
    `mse ~= bias2 + variance` up to the irreducible noise variance already
    folded into `bias2` through `y_test`.

    Args:
        x: x-coordinates, shape (n,).
        y: targets, shape (n,).
        degrees: polynomial degrees to sweep, e.g. range(0, 14).
        model_factory: Zero-argument callable returning a fresh, unfitted
            `LinearModel` for each bootstrap fit. Defaults to `OLS`.
        n_bootstraps: number of bootstrap resamples per degree.
        test_size: fraction of samples held out for testing (fixed across
            bootstraps and degrees).
        seed: seed for the train/test split and all bootstrap draws.

    Returns:
        Dict with keys "degrees", "mse_test", "bias2", "variance" (arrays
        aligned with "degrees").
    """
    degrees_arr = np.array(list(degrees))
    rng = np.random.default_rng(seed)

    mse_test = np.empty(degrees_arr.shape)
    bias2 = np.empty(degrees_arr.shape)
    variance = np.empty(degrees_arr.shape)

    for i, degree in enumerate(degrees_arr):
        X = univariate_polynomial_design_matrix(x=x, degree=int(degree), intercept=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )

        y_pred = np.empty((y_test.shape[0], n_bootstraps))
        for b in range(n_bootstraps):
            X_, y_ = bootstrap_resample(X_train, y_train, rng)
            model = model_factory().fit(X_, y_)
            y_pred[:, b] = model.predict(X_test)

        y_test_col = y_test.reshape(-1, 1)
        mse_test[i] = np.mean((y_test_col - y_pred) ** 2)
        bias2[i] = np.mean((y_test_col - np.mean(y_pred, axis=1, keepdims=True)) ** 2)
        variance[i] = np.mean(np.var(y_pred, axis=1, keepdims=True))

    return {
        "degrees": degrees_arr,
        "mse_test": mse_test,
        "bias2": bias2,
        "variance": variance,
    }
