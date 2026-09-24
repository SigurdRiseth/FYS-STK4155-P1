"""K-fold cross-validation sweeps, via scikit-learn's KFold/cross_val_score."""

from collections.abc import Callable, Iterable
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fys_stk4155_p1.data.design_matrix import univariate_polynomial_design_matrix
from fys_stk4155_p1.regression.base import LinearModel
from fys_stk4155_p1.regression.lasso import Lasso
from fys_stk4155_p1.regression.ridge import Ridge


def _make_pipeline(degree: int, model_factory: Callable[[], LinearModel]) -> Pipeline:
    """Build a fit-within-fold scaling pipeline for a given polynomial degree.

    Column 0 (the intercept) is passed through unscaled; columns 1: (x^1..
    x^degree) are standardized by a `StandardScaler` fit inside the pipeline,
    i.e. on the training fold only. Degree 0 has no non-intercept columns to
    scale (StandardScaler rejects a zero-column input), so the scaler is
    omitted for that case.
    """
    steps: list[tuple[str, Any]] = []
    if degree > 0:
        scale = ColumnTransformer(
            [
                ("intercept", "passthrough", [0]),
                ("scale_poly", StandardScaler(), slice(1, None)),
            ]
        )
        steps.append(("scale", scale))
    steps.append(("model", model_factory()))
    return Pipeline(steps)


def kfold_mse_degree_sweep(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    degrees: Iterable[int],
    model_factory: Callable[[], LinearModel],
    k: int,
    seed: int = 42,
) -> dict[str, Any]:
    """K-fold cross-validated test (and train) MSE, per polynomial degree.

    For each degree, builds the polynomial design matrix (with an intercept
    column) and scores `model_factory()` with `sklearn.model_selection.
    cross_validate` over a `KFold(k, shuffle=True)` split. Scaling
    (`StandardScaler` on the non-intercept columns) is done inside a
    `Pipeline`, so it is fit on each training fold only, never on the held-out
    fold or the full dataset.

    Args:
        x: x-coordinates, shape (n,).
        y: targets, shape (n,).
        degrees: polynomial degrees to sweep, e.g. range(0, 14).
        model_factory: Zero-argument callable returning a fresh, unfitted
            `LinearModel` for each fold.
        k: number of folds.
        seed: seed for the fold shuffling.

    Returns:
        Dict with keys "degrees", "mse_mean", "mse_std" (held-out-fold test
        MSE, arrays aligned with "degrees"), "mse_folds" (the k per-fold
        held-out MSEs, shape (n_degrees, k), in `KFold` order, so two sweeps
        with the same `seed` can be compared fold by fold), and
        "mse_train_mean", "mse_train_std" (in-fold training MSE). The "_std"
        entries are the standard deviation of the k per-fold MSEs.
    """
    degrees_arr = np.array(list(degrees))
    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)

    mse_mean = np.empty(degrees_arr.shape)
    mse_std = np.empty(degrees_arr.shape)
    mse_folds = np.empty((degrees_arr.shape[0], k))
    mse_train_mean = np.empty(degrees_arr.shape)
    mse_train_std = np.empty(degrees_arr.shape)

    for i, degree in enumerate(degrees_arr):
        X = univariate_polynomial_design_matrix(x=x, degree=int(degree), intercept=True)
        pipeline = _make_pipeline(int(degree), model_factory)
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=kfold,
            scoring="neg_mean_squared_error",
            return_train_score=True,
        )
        mse_mean[i] = -scores["test_score"].mean()
        mse_std[i] = scores["test_score"].std()
        mse_folds[i] = -scores["test_score"]
        mse_train_mean[i] = -scores["train_score"].mean()
        mse_train_std[i] = scores["train_score"].std()

    return {
        "degrees": degrees_arr,
        "mse_mean": mse_mean,
        "mse_std": mse_std,
        "mse_folds": mse_folds,
        "mse_train_mean": mse_train_mean,
        "mse_train_std": mse_train_std,
    }


def kfold_mse_ridge_grid(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    degrees: Iterable[int],
    lambdas: NDArray[np.float64],
    k: int,
    seed: int = 42,
) -> dict[str, Any]:
    """K-fold cross-validated test MSE for Ridge, over a degree x lambda grid.

    Same fold-only scaling as `kfold_mse_degree_sweep`, repeated for every
    (degree, lambda) pair.

    Args:
        x: x-coordinates, shape (n,).
        y: targets, shape (n,).
        degrees: polynomial degrees to sweep, e.g. range(0, 14).
        lambdas: Ridge penalty strengths to sweep, shape (n_lambdas,).
        k: number of folds.
        seed: seed for the fold shuffling.

    Returns:
        Dict with keys "degrees", "lambdas", and "mse_mean", a
        (n_degrees, n_lambdas) array of mean CV MSE.
    """
    degrees_arr = np.array(list(degrees))
    lambdas_arr = np.asarray(lambdas, dtype=np.float64)
    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)

    mse_mean = np.empty((degrees_arr.shape[0], lambdas_arr.shape[0]))

    for i, degree in enumerate(degrees_arr):
        X = univariate_polynomial_design_matrix(x=x, degree=int(degree), intercept=True)
        for j, lam in enumerate(lambdas_arr):

            def _ridge_factory(lam: float = lam) -> Ridge:
                return Ridge(lam=lam, fit_intercept_column=True)

            pipeline = _make_pipeline(int(degree), _ridge_factory)
            scores = cross_val_score(pipeline, X, y, cv=kfold, scoring="neg_mean_squared_error")
            mse_mean[i, j] = -scores.mean()

    return {"degrees": degrees_arr, "lambdas": lambdas_arr, "mse_mean": mse_mean}


def kfold_mse_lasso_grid(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    degrees: Iterable[int],
    lambdas: NDArray[np.float64],
    k: int,
    seed: int = 42,
    learning_rate: float = 0.05,
    optimizer: Literal["plain", "momentum", "adagrad", "rmsprop", "adam"] = "adam",
    max_iter: int = 3000,
) -> dict[str, Any]:
    """K-fold cross-validated test MSE for Lasso, over a degree x lambda grid.

    Same fold-only scaling as `kfold_mse_degree_sweep`, repeated for every
    (degree, lambda) pair. Lasso has no closed-form solution, so each fit is
    gradient descent; `learning_rate`/`optimizer`/`max_iter` are passed
    straight through to `Lasso`.

    Args:
        x: x-coordinates, shape (n,).
        y: targets, shape (n,).
        degrees: polynomial degrees to sweep, e.g. range(0, 14).
        lambdas: Lasso penalty strengths to sweep, shape (n_lambdas,).
        k: number of folds.
        seed: seed for the fold shuffling.
        learning_rate: step size passed to `Lasso`.
        optimizer: `Lasso` optimizer name.
        max_iter: max gradient steps per fit.

    Returns:
        Dict with keys "degrees", "lambdas", and "mse_mean", a
        (n_degrees, n_lambdas) array of mean CV MSE.
    """
    degrees_arr = np.array(list(degrees))
    lambdas_arr = np.asarray(lambdas, dtype=np.float64)
    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)

    mse_mean = np.empty((degrees_arr.shape[0], lambdas_arr.shape[0]))

    for i, degree in enumerate(degrees_arr):
        X = univariate_polynomial_design_matrix(x=x, degree=int(degree), intercept=True)
        for j, lam in enumerate(lambdas_arr):

            def _lasso_factory(lam: float = lam) -> Lasso:
                return Lasso(
                    learning_rate=learning_rate,
                    lam=lam,
                    max_iter=max_iter,
                    optimizer=optimizer,
                    fit_intercept_column=True,
                )

            pipeline = _make_pipeline(int(degree), _lasso_factory)
            scores = cross_val_score(pipeline, X, y, cv=kfold, scoring="neg_mean_squared_error")
            mse_mean[i, j] = -scores.mean()

    return {"degrees": degrees_arr, "lambdas": lambdas_arr, "mse_mean": mse_mean}
