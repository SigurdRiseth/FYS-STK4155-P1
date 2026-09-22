from typing import Any, Literal

import numpy as np
import pytest

from fys_stk4155_p1.regression.gradient_descent import GradientDescent, gradient_descent_sweep
from fys_stk4155_p1.regression.ridge import Ridge


def test_fit_returns_self() -> None:
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0])
    model = GradientDescent(learning_rate=0.1)
    assert model.fit(X, y) is model


@pytest.mark.parametrize("optimizer", ["plain", "momentum", "adagrad", "adam"])
def test_converges_close_to_ridge(
    optimizer: Literal["plain", "momentum", "adagrad", "rmsprop", "adam"],
) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 0.01 * rng.normal(size=100)

    ridge = Ridge(lam=0.1).fit(X, y)
    model = GradientDescent(learning_rate=0.5, lam=0.1, max_iter=5000, optimizer=optimizer).fit(
        X, y
    )

    np.testing.assert_allclose(model.coef_, ridge.coef_, atol=1e-4)


def test_analytical_and_autodiff_gradients_agree() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3))
    y = rng.normal(size=50)

    analytical = GradientDescent(learning_rate=0.1, max_iter=200, gradient_method="analytical").fit(
        X, y
    )
    autodiff = GradientDescent(learning_rate=0.1, max_iter=200, gradient_method="autodiff").fit(
        X, y
    )

    np.testing.assert_allclose(analytical.coef_, autodiff.coef_, atol=1e-10)


def test_stops_early_once_converged() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 3))
    y = X @ np.array([1.0, -1.0, 0.5])

    model = GradientDescent(learning_rate=0.5, max_iter=10000, tol=1e-6).fit(X, y)

    assert model.n_iter_ < 10000
    assert model.cost_history_.shape == (model.n_iter_,)


def test_rejects_negative_lam() -> None:
    X = np.zeros((5, 2))
    y = np.zeros(5)
    with pytest.raises(ValueError, match="lam must be non-negative"):
        GradientDescent(learning_rate=0.1, lam=-1.0).fit(X, y)


def test_rejects_non_positive_max_iter() -> None:
    X = np.zeros((5, 2))
    y = np.zeros(5)
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        GradientDescent(learning_rate=0.1, max_iter=0).fit(X, y)


def test_rejects_unknown_optimizer() -> None:
    X = np.zeros((5, 2))
    y = np.zeros(5)
    with pytest.raises(ValueError, match="optimizer must be one of"):
        GradientDescent(learning_rate=0.1, optimizer="bogus").fit(X, y)  # type: ignore[arg-type]


def test_rejects_unknown_gradient_method() -> None:
    X = np.zeros((5, 2))
    y = np.zeros(5)
    with pytest.raises(ValueError, match="gradient_method must be one of"):
        GradientDescent(learning_rate=0.1, gradient_method="bogus").fit(X, y)  # type: ignore[arg-type]


def test_sweep_returns_one_fitted_model_per_variant() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 2))
    y = rng.normal(size=30)

    variants: dict[str, dict[str, Any]] = {
        "plain": {"learning_rate": 0.1},
        "adam": {"learning_rate": 0.1, "optimizer": "adam"},
    }
    models = gradient_descent_sweep(X, y, variants)

    assert set(models) == {"plain", "adam"}
    assert all(isinstance(m, GradientDescent) for m in models.values())
    assert all(hasattr(m, "coef_") for m in models.values())
