import numpy as np
import pytest

from fys_stk4155_p1.regression.cost import analytical_gradient, cost, hessian_max_eigenvalue


def test_cost_without_regularization():
    X = np.array(
        [
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
        ]
    )
    y = np.array([2.0, 3.0, 5.0])
    theta = np.array([1.0, 1.0])

    expected = 2.0 / 3.0

    result = cost(X, y, theta)

    assert result == pytest.approx(expected)


def test_cost_with_regularization():
    X = np.array(
        [
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
        ]
    )
    y = np.array([2.0, 3.0, 5.0])
    theta = np.array([1.0, 1.0])

    expected = 8.0 / 3.0

    result = cost(X, y, theta, lam=1.0)

    assert result == pytest.approx(expected)


def test_cost_excludes_intercept_from_regularization():
    X = np.array(
        [
            [1.0, 2.0],
            [1.0, 3.0],
        ]
    )
    y = np.array([2.0, 3.0])
    theta = np.array([10.0, 1.0])

    # Predictions: [12, 13]
    # MSE = ((2-12)^2 + (3-13)^2) / 2 = 100
    #
    # Only theta[1] is penalized:
    # penalty = 1^2 = 1
    #
    # Total = 101
    expected = 101.0

    result = cost(
        X,
        y,
        theta,
        lam=1.0,
        fit_intercept_column=True,
    )

    assert result == pytest.approx(expected)


def test_cost_regularizes_intercept_when_not_fit_intercept_column():
    X = np.array(
        [
            [1.0, 2.0],
            [1.0, 3.0],
        ]
    )
    y = np.array([2.0, 3.0])
    theta = np.array([10.0, 1.0])

    # MSE = 100
    # penalty = 10^2 + 1^2 = 101
    # Total = 201
    expected = 201.0

    result = cost(
        X,
        y,
        theta,
        lam=1.0,
        fit_intercept_column=False,
    )

    assert result == pytest.approx(expected)


def test_cost_zero_regularization():
    X = np.array(
        [
            [1.0, 2.0],
            [1.0, 3.0],
        ]
    )
    y = np.array([2.0, 3.0])
    theta = np.array([1.0, 1.0])

    result = cost(X, y, theta, lam=0.0)

    assert result == pytest.approx(1.0)


def test_analytical_gradient_without_regularization():
    X = np.array(
        [
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
        ]
    )
    y = np.array([2.0, 3.0, 5.0])
    theta = np.array([1.0, 1.0])

    # Same setup as test_cost_without_regularization: predictions [3, 4, 5],
    # residuals [1, 1, 0]. grad = (2/n) X^T @ residual = (2/3) * [2, 5].
    expected = np.array([4.0 / 3.0, 10.0 / 3.0])

    result = analytical_gradient(X, y, theta)

    np.testing.assert_allclose(result, expected)


def test_analytical_gradient_with_regularization():
    X = np.array(
        [
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
        ]
    )
    y = np.array([2.0, 3.0, 5.0])
    theta = np.array([1.0, 1.0])

    # Unregularized gradient is [4/3, 10/3] (see above); + 2*lam*theta = [2, 2].
    expected = np.array([10.0 / 3.0, 16.0 / 3.0])

    result = analytical_gradient(X, y, theta, lam=1.0)

    np.testing.assert_allclose(result, expected)


def test_analytical_gradient_excludes_intercept_from_regularization():
    X = np.array(
        [
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
        ]
    )
    y = np.array([2.0, 3.0, 5.0])
    theta = np.array([1.0, 1.0])

    # Only theta[1] is penalized: + 2*lam*theta = [0, 2].
    expected = np.array([4.0 / 3.0, 16.0 / 3.0])

    result = analytical_gradient(X, y, theta, lam=1.0, fit_intercept_column=True)

    np.testing.assert_allclose(result, expected)


def test_analytical_gradient_shape_matches_theta():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 4))
    y = rng.normal(size=10)
    theta = rng.normal(size=4)

    result = analytical_gradient(X, y, theta, lam=0.3)

    assert result.shape == theta.shape


def test_analytical_gradient_matches_numerical_gradient():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 5))
    y = rng.normal(size=20)
    theta = rng.normal(size=5)
    eps = 1e-6

    for lam, fit_intercept_column in [(0.0, False), (0.7, False), (0.7, True)]:
        numerical = np.zeros_like(theta)
        for i in range(theta.size):
            step = np.zeros_like(theta)
            step[i] = eps
            numerical[i] = (
                cost(X, y, theta + step, lam, fit_intercept_column)
                - cost(X, y, theta - step, lam, fit_intercept_column)
            ) / (2 * eps)

        result = analytical_gradient(X, y, theta, lam, fit_intercept_column)

        np.testing.assert_allclose(result, numerical, atol=1e-5)


def test_hessian_max_eigenvalue_without_regularization():
    X = np.array(
        [
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
        ]
    )

    # Independent recomputation of the Hessian, (2/n) X^T X.
    H = 2.0 / 3.0 * X.T @ X
    expected = np.linalg.eigvalsh(H)[-1]

    result = hessian_max_eigenvalue(X)

    assert result == pytest.approx(expected)


def test_hessian_max_eigenvalue_with_regularization():
    X = np.array(
        [
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
        ]
    )
    lam = 0.5

    H = 2.0 / 3.0 * X.T @ X + 2.0 * lam * np.eye(2)
    expected = np.linalg.eigvalsh(H)[-1]

    result = hessian_max_eigenvalue(X, lam=lam)

    assert result == pytest.approx(expected)


def test_hessian_max_eigenvalue_excludes_intercept_from_regularization():
    X = np.array(
        [
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
        ]
    )
    lam = 0.5

    penalty = np.eye(2)
    penalty[0, 0] = 0.0
    H = 2.0 / 3.0 * X.T @ X + 2.0 * lam * penalty
    expected = np.linalg.eigvalsh(H)[-1]

    result = hessian_max_eigenvalue(X, lam=lam, fit_intercept_column=True)

    assert result == pytest.approx(expected)


def test_hessian_max_eigenvalue_matches_hand_built_hessian():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(15, 4))
    n_samples, n_features = X.shape

    for lam, fit_intercept_column in [(0.0, False), (0.2, False), (0.2, True)]:
        penalty = np.eye(n_features)
        if fit_intercept_column:
            penalty[0, 0] = 0.0
        H = 2.0 / n_samples * X.T @ X + 2.0 * lam * penalty
        expected = np.linalg.eigvalsh(H)[-1]

        result = hessian_max_eigenvalue(X, lam=lam, fit_intercept_column=fit_intercept_column)

        assert result == pytest.approx(expected)


def test_hessian_max_eigenvalue_zero_for_zero_matrix():
    X = np.zeros((5, 3))

    result = hessian_max_eigenvalue(X, lam=0.0)

    assert result == pytest.approx(0.0, abs=1e-12)
