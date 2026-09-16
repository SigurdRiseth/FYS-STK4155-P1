import numpy as np
import pytest

from fys_stk4155_p1.regression.cost import cost


def test_cost_without_regularization():
    X = np.array([
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 4.0],
    ])
    y = np.array([2.0, 3.0, 5.0])
    theta = np.array([1.0, 1.0])

    # Predictions: [3, 4, 5]
    # Residuals:   [-1, -1, 0]
    # MSE = (1 + 1 + 0) / 3 = 2/3
    expected = 2.0 / 3.0

    result = cost(X, y, theta)

    assert result == pytest.approx(expected)

def test_cost_with_regularization():
    X = np.array([
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 4.0],
    ])
    y = np.array([2.0, 3.0, 5.0])
    theta = np.array([1.0, 1.0])

    expected = 8.0 / 3.0

    result = cost(X, y, theta, lam=1.0)

    assert result == pytest.approx(expected)

def test_cost_excludes_intercept_from_regularization():
    X = np.array([
        [1.0, 2.0],
        [1.0, 3.0],
    ])
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
    X = np.array([
        [1.0, 2.0],
        [1.0, 3.0],
    ])
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
    X = np.array([
        [1.0, 2.0],
        [1.0, 3.0],
    ])
    y = np.array([2.0, 3.0])
    theta = np.array([1.0, 1.0])

    result = cost(X, y, theta, lam=0.0)

    assert result == pytest.approx(1.0)
