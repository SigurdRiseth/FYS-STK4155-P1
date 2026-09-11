import numpy as np
import pytest

from fys_stk4155_p1.regression.ridge import Ridge


def test_fit_returns_self() -> None:
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0])
    model = Ridge(lam=1.0)
    assert model.fit(X, y) is model


def test_zero_lambda_matches_ols() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 4))
    y = rng.normal(size=30)

    ridge_coef = Ridge(lam=0.0).fit(X, y).coef_
    ols_coef = np.linalg.lstsq(X, y, rcond=None)[0]

    np.testing.assert_allclose(ridge_coef, ols_coef, atol=1e-8)


def test_larger_lambda_shrinks_coefficients() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 4))
    y = rng.normal(size=30)

    small_lam_coef = Ridge(lam=0.1).fit(X, y).coef_
    large_lam_coef = Ridge(lam=100.0).fit(X, y).coef_

    assert np.linalg.norm(large_lam_coef) < np.linalg.norm(small_lam_coef)


def test_fit_intercept_column_is_not_penalized() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=30)
    X = np.column_stack([np.ones_like(x), x])
    y = 5.0 + 2.0 * x

    model = Ridge(lam=1.0, fit_intercept_column=True).fit(X, y)

    # a large penalty should still leave the (unpenalized) intercept close to
    # the true value while shrinking the (penalized) slope towards zero.
    unpenalized = Ridge(lam=1e6, fit_intercept_column=True).fit(X, y)
    assert unpenalized.coef_[0] == pytest.approx(5.0, abs=0.5)
    assert abs(unpenalized.coef_[1]) < abs(model.coef_[1])


def test_predict_matches_design_matrix_times_coef() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 3))
    y = rng.normal(size=20)

    model = Ridge(lam=0.5).fit(X, y)
    y_pred = model.predict(X)

    np.testing.assert_allclose(y_pred, X @ model.coef_)


def test_rejects_negative_lambda() -> None:
    X = np.zeros((5, 2))
    y = np.zeros(5)
    with pytest.raises(ValueError, match="lam must be non-negative"):
        Ridge(lam=-1.0).fit(X, y)


def test_rejects_mismatched_n_samples() -> None:
    X = np.zeros((5, 2))
    y = np.zeros(4)
    with pytest.raises(ValueError, match="disagree on n_samples"):
        Ridge(lam=1.0).fit(X, y)
