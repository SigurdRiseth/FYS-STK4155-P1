import numpy as np
import pytest

from fys_stk4155_p1.regression.ordinary_least_squares import OLS


def test_fit_returns_self() -> None:
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0])
    model = OLS()
    assert model.fit(X, y) is model


def test_recovers_known_line() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 50)
    X = np.column_stack([np.ones_like(x), x])
    y = 3.0 + 2.0 * x

    model = OLS().fit(X, y)

    np.testing.assert_allclose(model.coef_, [3.0, 2.0], atol=1e-10)


def test_predict_matches_design_matrix_times_coef() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 3))
    y = rng.normal(size=20)

    model = OLS().fit(X, y)
    y_pred = model.predict(X)

    np.testing.assert_allclose(y_pred, X @ model.coef_)


def test_fit_is_deterministic() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 4))
    y = rng.normal(size=30)

    coef1 = OLS().fit(X, y).coef_
    coef2 = OLS().fit(X, y).coef_

    np.testing.assert_array_equal(coef1, coef2)


def test_rejects_mismatched_n_samples() -> None:
    X = np.zeros((5, 2))
    y = np.zeros(4)
    with pytest.raises(ValueError, match="disagree on n_samples"):
        OLS().fit(X, y)


def test_rejects_non_2d_X() -> None:
    X = np.zeros(5)
    y = np.zeros(5)
    with pytest.raises(ValueError, match="X must be 2-D"):
        OLS().fit(X, y)
