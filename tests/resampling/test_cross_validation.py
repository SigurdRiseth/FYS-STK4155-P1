import numpy as np

from fys_stk4155_p1.regression.ordinary_least_squares import OLS
from fys_stk4155_p1.resampling.cross_validation import (
    kfold_mse_degree_sweep,
    kfold_mse_lasso_grid,
    kfold_mse_ridge_grid,
)


def _sample_data(n: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = 1.0 + 2.0 * x - 0.5 * x**2 + 0.05 * rng.normal(size=n)
    return x, y


def test_degree_sweep_returns_shapes_aligned_with_degrees() -> None:
    x, y = _sample_data()

    results = kfold_mse_degree_sweep(x, y, range(0, 4), OLS, k=5)

    np.testing.assert_array_equal(results["degrees"], [0, 1, 2, 3])
    assert results["mse_mean"].shape == (4,)
    assert results["mse_std"].shape == (4,)


def test_degree_sweep_is_deterministic() -> None:
    x, y = _sample_data()

    results1 = kfold_mse_degree_sweep(x, y, range(0, 4), OLS, k=5, seed=7)
    results2 = kfold_mse_degree_sweep(x, y, range(0, 4), OLS, k=5, seed=7)

    np.testing.assert_array_equal(results1["mse_mean"], results2["mse_mean"])


def test_degree_sweep_handles_degree_zero() -> None:
    x, y = _sample_data()

    results = kfold_mse_degree_sweep(x, y, range(0, 1), OLS, k=5)

    assert np.isfinite(results["mse_mean"]).all()


def test_higher_degree_fits_better_than_underfit_degree() -> None:
    x, y = _sample_data(n=200)

    results = kfold_mse_degree_sweep(x, y, [1, 2], OLS, k=5)

    assert results["mse_mean"][1] < results["mse_mean"][0]


def test_ridge_grid_returns_shapes_aligned_with_degrees_and_lambdas() -> None:
    x, y = _sample_data()
    lambdas = np.array([0.0, 1e-3, 1e-1])

    results = kfold_mse_ridge_grid(x, y, range(0, 3), lambdas, k=5)

    np.testing.assert_array_equal(results["degrees"], [0, 1, 2])
    np.testing.assert_array_equal(results["lambdas"], lambdas)
    assert results["mse_mean"].shape == (3, 3)
    assert np.isfinite(results["mse_mean"]).all()


def test_ridge_grid_is_deterministic() -> None:
    x, y = _sample_data()
    lambdas = np.array([0.0, 1e-2])

    results1 = kfold_mse_ridge_grid(x, y, range(0, 2), lambdas, k=5, seed=3)
    results2 = kfold_mse_ridge_grid(x, y, range(0, 2), lambdas, k=5, seed=3)

    np.testing.assert_array_equal(results1["mse_mean"], results2["mse_mean"])


def test_lasso_grid_returns_shapes_aligned_with_degrees_and_lambdas() -> None:
    x, y = _sample_data()
    lambdas = np.array([1e-3, 1e-1])

    results = kfold_mse_lasso_grid(x, y, range(0, 2), lambdas, k=5, max_iter=200)

    np.testing.assert_array_equal(results["degrees"], [0, 1])
    np.testing.assert_array_equal(results["lambdas"], lambdas)
    assert results["mse_mean"].shape == (2, 2)
    assert np.isfinite(results["mse_mean"]).all()


def test_lasso_grid_is_deterministic() -> None:
    x, y = _sample_data()
    lambdas = np.array([1e-3, 1e-1])

    results1 = kfold_mse_lasso_grid(x, y, range(0, 2), lambdas, k=5, seed=3, max_iter=200)
    results2 = kfold_mse_lasso_grid(x, y, range(0, 2), lambdas, k=5, seed=3, max_iter=200)

    np.testing.assert_array_equal(results1["mse_mean"], results2["mse_mean"])


def test_degree_sweep_fold_mses_average_to_mean() -> None:
    x, y = _sample_data()

    results = kfold_mse_degree_sweep(x, y, range(0, 4), OLS, k=5)

    assert results["mse_folds"].shape == (4, 5)
    np.testing.assert_allclose(results["mse_folds"].mean(axis=1), results["mse_mean"])
