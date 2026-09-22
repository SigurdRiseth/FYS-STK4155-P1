import numpy as np

from fys_stk4155_p1.regression.autodiff import autodiff_gradient
from fys_stk4155_p1.regression.cost import analytical_gradient
from fys_stk4155_p1.regression.ordinary_least_squares import OLS


def test_gradients_agree_without_regularization() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    y = rng.normal(size=20)
    theta = rng.normal(size=5)

    np.testing.assert_allclose(
        autodiff_gradient(X, y, theta), analytical_gradient(X, y, theta), atol=1e-10
    )


def test_gradients_agree_with_ridge_regularization() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 5))
    y = rng.normal(size=20)
    theta = rng.normal(size=5)
    lam = 0.7

    np.testing.assert_allclose(
        autodiff_gradient(X, y, theta, lam=lam),
        analytical_gradient(X, y, theta, lam=lam),
        atol=1e-10,
    )


def test_gradients_agree_with_unpenalized_intercept() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=20)
    X = np.column_stack([np.ones_like(x), x, x**2])
    y = rng.normal(size=20)
    theta = rng.normal(size=3)
    lam = 0.5

    np.testing.assert_allclose(
        autodiff_gradient(X, y, theta, lam=lam, fit_intercept_column=True),
        analytical_gradient(X, y, theta, lam=lam, fit_intercept_column=True),
        atol=1e-10,
    )


def test_gradients_agree_at_the_ols_minimum() -> None:
    # At the OLS closed-form solution the analytical gradient is ~0 by
    # construction; checking agreement there as well as away from it (the
    # tests above) guards against the two methods only matching when the
    # gradient itself is large.
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 5))
    y = rng.normal(size=20)
    theta = OLS().fit(X, y).coef_

    analytical = analytical_gradient(X, y, theta)
    autodiff = autodiff_gradient(X, y, theta)

    np.testing.assert_allclose(analytical, np.zeros_like(theta), atol=1e-10)
    np.testing.assert_allclose(autodiff, analytical, atol=1e-10)
