import numpy as np

from fys_stk4155_p1.regression.ordinary_least_squares import OLS
from fys_stk4155_p1.resampling.bootstrap import bootstrap_bias_variance_sweep, bootstrap_resample


def _sample_data(n: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = 1.0 + 2.0 * x - 0.5 * x**2 + 0.05 * rng.normal(size=n)
    return x, y


def test_bootstrap_resample_preserves_shape_and_pairing() -> None:
    rng = np.random.default_rng(0)
    X = np.arange(10).reshape(5, 2).astype(float)
    y = np.arange(5).astype(float)

    X_, y_ = bootstrap_resample(X, y, rng)

    assert X_.shape == X.shape
    assert y_.shape == y.shape
    np.testing.assert_array_equal(X_[:, 1], X_[:, 0] + 1)
    np.testing.assert_array_equal(y_, X_[:, 0] / 2)


def test_returns_shapes_aligned_with_degrees() -> None:
    x, y = _sample_data()
    degrees = range(0, 4)

    results = bootstrap_bias_variance_sweep(x, y, degrees, OLS, n_bootstraps=20)

    np.testing.assert_array_equal(results["degrees"], [0, 1, 2, 3])
    for key in ("mse_test", "bias2", "variance"):
        assert results[key].shape == (4,)


def test_is_deterministic() -> None:
    x, y = _sample_data()

    results1 = bootstrap_bias_variance_sweep(x, y, range(0, 4), OLS, n_bootstraps=20)
    results2 = bootstrap_bias_variance_sweep(x, y, range(0, 4), OLS, n_bootstraps=20)

    for key in ("mse_test", "bias2", "variance"):
        np.testing.assert_array_equal(results1[key], results2[key])


def test_mse_decomposes_into_bias_and_variance() -> None:
    x, y = _sample_data(n=100)

    results = bootstrap_bias_variance_sweep(x, y, range(1, 5), OLS, n_bootstraps=50)

    np.testing.assert_allclose(
        results["mse_test"], results["bias2"] + results["variance"], rtol=1e-10
    )


def test_variance_is_zero_for_a_single_bootstrap() -> None:
    x, y = _sample_data()

    results = bootstrap_bias_variance_sweep(x, y, range(1, 3), OLS, n_bootstraps=1)

    np.testing.assert_allclose(results["variance"], 0.0, atol=1e-12)
    np.testing.assert_allclose(results["mse_test"], results["bias2"], rtol=1e-10)
