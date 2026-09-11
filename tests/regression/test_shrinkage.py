import numpy as np

from fys_stk4155_p1.regression.shrinkage import ridge_coefficient_path, singular_value_shrinkage


def test_ridge_coefficient_path_shape_and_lambda0_matches_ols() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    y = rng.normal(size=50)
    lambdas = np.array([0.0, 1.0, 10.0])

    path = ridge_coefficient_path(X, y, lambdas)

    assert path.shape == (3, 4)
    theta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    np.testing.assert_allclose(path[0], theta_ols, atol=1e-8)


def test_ridge_coefficient_path_shrinks_toward_zero() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 4))
    y = rng.normal(size=50)
    lambdas = np.array([0.0, 1e3])

    path = ridge_coefficient_path(X, y, lambdas)

    assert np.linalg.norm(path[1]) < np.linalg.norm(path[0])


def test_singular_value_shrinkage_bounds_and_monotone_in_lambda() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 5))
    lambdas = np.array([0.0, 1.0, 100.0])

    singular_values, shrinkage = singular_value_shrinkage(X, lambdas)

    assert singular_values.shape == (5,)
    assert shrinkage.shape == (3, 5)
    np.testing.assert_allclose(shrinkage[0], 1.0)  # lambda=0: no shrinkage at all
    assert np.all(shrinkage[1] <= shrinkage[0])
    assert np.all(shrinkage[2] <= shrinkage[1])
    assert np.all((shrinkage >= 0) & (shrinkage <= 1))


def test_singular_value_shrinkage_smaller_modes_shrink_more() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 5))

    _, shrinkage = singular_value_shrinkage(X, np.array([1.0]))

    # np.linalg.svd sorts singular values decreasing, and f_i is increasing in
    # s_i, so the shrinkage factor should be non-decreasing along mode index.
    assert np.all(np.diff(shrinkage[0]) <= 1e-12)
