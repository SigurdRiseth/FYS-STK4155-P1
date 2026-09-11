import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from fys_stk4155_p1.regression.degree_sweep import fit_polynomial_degree_sweep
from fys_stk4155_p1.regression.ordinary_least_squares import OLS
from fys_stk4155_p1.regression.ridge import Ridge


def _sample_data(n: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = 1.0 + 2.0 * x - 0.5 * x**2 + 0.05 * rng.normal(size=n)
    return x, y


def test_returns_shapes_aligned_with_degrees() -> None:
    x, y = _sample_data()
    degrees = range(1, 4)

    results = fit_polynomial_degree_sweep(x, y, degrees, OLS)

    np.testing.assert_array_equal(results["degrees"], [1, 2, 3])
    assert len(results["weights"]) == 3
    assert [len(theta) for theta in results["weights"]] == [1, 2, 3]
    for key in ("mse_train", "mse_test", "r2_train", "r2_test"):
        assert results[key].shape == (3,)


def test_is_deterministic() -> None:
    x, y = _sample_data()

    results1 = fit_polynomial_degree_sweep(x, y, range(1, 5), OLS)
    results2 = fit_polynomial_degree_sweep(x, y, range(1, 5), OLS)

    np.testing.assert_array_equal(results1["mse_test"], results2["mse_test"])
    for theta1, theta2 in zip(results1["weights"], results2["weights"], strict=True):
        np.testing.assert_array_equal(theta1, theta2)


def test_intercept_is_training_target_mean() -> None:
    x, y = _sample_data()
    seed, test_size = 7, 0.25

    results = fit_polynomial_degree_sweep(x, y, range(1, 3), OLS, test_size=test_size, seed=seed)

    _, _, y_train, _ = train_test_split(x, y, test_size=test_size, random_state=seed)
    assert results["intercept"] == pytest.approx(y_train.mean())


def test_model_factory_is_used_per_degree_ridge_matches_ols_at_lambda_zero() -> None:
    x, y = _sample_data()
    degrees = range(1, 5)

    ols_results = fit_polynomial_degree_sweep(x, y, degrees, OLS)
    ridge_results = fit_polynomial_degree_sweep(
        x, y, degrees, lambda: Ridge(lam=0.0, fit_intercept_column=False)
    )

    for theta_ols, theta_ridge in zip(
        ols_results["weights"], ridge_results["weights"], strict=True
    ):
        np.testing.assert_allclose(theta_ridge, theta_ols, atol=1e-8)


def test_higher_lambda_shrinks_coefficients() -> None:
    x, y = _sample_data(n=200)
    degrees = range(5, 6)

    small_lam = fit_polynomial_degree_sweep(
        x, y, degrees, lambda: Ridge(lam=1e-3, fit_intercept_column=False)
    )
    large_lam = fit_polynomial_degree_sweep(
        x, y, degrees, lambda: Ridge(lam=1e3, fit_intercept_column=False)
    )

    assert np.linalg.norm(large_lam["weights"][0]) < np.linalg.norm(small_lam["weights"][0])
