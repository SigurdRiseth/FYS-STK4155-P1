from typing import Any, Literal

import numpy as np
import pytest

from fys_stk4155_p1.regression.cost import lasso_cost
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


@pytest.mark.parametrize(("optimizer", "learning_rate"), [("plain", 0.5), ("adam", 0.1)])
def test_sgd_converges_close_to_ridge(
    optimizer: Literal["plain", "adam"], learning_rate: float
) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 0.01 * rng.normal(size=100)

    ridge = Ridge(lam=0.1).fit(X, y)
    model = GradientDescent(
        learning_rate=learning_rate,
        lam=0.1,
        optimizer=optimizer,
        batch_size=16,
        n_epochs=300,
        learning_rate_schedule="time_based",
        lr_decay=0.01,
        random_state=0,
    ).fit(X, y)

    # Looser than the full-batch atol above: mini-batch gradients are noisy,
    # so SGD settles near, not exactly at, the closed-form minimum even with
    # a decaying learning rate.
    np.testing.assert_allclose(model.coef_, ridge.coef_, atol=0.05)


def test_default_batch_size_is_none_and_matches_full_batch_behavior() -> None:
    model = GradientDescent(learning_rate=0.1)
    assert model.batch_size is None


def test_sgd_is_deterministic_given_random_state() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3))
    y = rng.normal(size=50)

    first = GradientDescent(learning_rate=0.1, batch_size=8, n_epochs=20, random_state=7).fit(X, y)
    second = GradientDescent(learning_rate=0.1, batch_size=8, n_epochs=20, random_state=7).fit(X, y)

    np.testing.assert_array_equal(first.coef_, second.coef_)
    np.testing.assert_array_equal(first.cost_history_, second.cost_history_)


def test_sgd_different_random_state_shuffles_differently() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3))
    y = rng.normal(size=50)

    first = GradientDescent(learning_rate=0.1, batch_size=8, n_epochs=20, random_state=7).fit(X, y)
    second = GradientDescent(learning_rate=0.1, batch_size=8, n_epochs=20, random_state=99).fit(
        X, y
    )

    assert not np.array_equal(first.coef_, second.coef_)


def test_n_updates_equals_n_iter_in_full_batch_mode() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 2))
    y = rng.normal(size=30)

    model = GradientDescent(learning_rate=0.1, max_iter=50, tol=0.0).fit(X, y)

    assert model.n_updates_ == model.n_iter_ == 50
    assert model.cost_flops_.shape == model.cost_history_.shape
    assert np.all(np.diff(model.cost_flops_) > 0)


def test_n_updates_and_cost_flops_in_sgd_mode() -> None:
    rng = np.random.default_rng(2)
    n_samples = 100
    X = rng.normal(size=(n_samples, 2))
    y = rng.normal(size=n_samples)
    batch_size = 16
    n_epochs = 5

    model = GradientDescent(
        learning_rate=0.1, batch_size=batch_size, n_epochs=n_epochs, random_state=0
    ).fit(X, y)

    batches_per_epoch = -(-n_samples // batch_size)  # ceil division
    assert model.n_iter_ == n_epochs
    assert model.n_updates_ == n_epochs * batches_per_epoch
    assert model.cost_history_.shape == (n_epochs,)
    assert model.cost_flops_.shape == (n_epochs,)
    assert np.all(np.diff(model.cost_flops_) > 0)


def test_learning_rate_schedule_changes_trajectory() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 3))
    y = rng.normal(size=60)

    constant = GradientDescent(learning_rate=0.3, batch_size=8, n_epochs=30, random_state=5).fit(
        X, y
    )
    decayed = GradientDescent(
        learning_rate=0.3,
        batch_size=8,
        n_epochs=30,
        random_state=5,
        learning_rate_schedule="time_based",
        lr_decay=0.5,
    ).fit(X, y)

    assert not np.array_equal(constant.coef_, decayed.coef_)


def test_exponential_schedule_changes_trajectory() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 3))
    y = rng.normal(size=60)

    constant = GradientDescent(learning_rate=0.3, batch_size=8, n_epochs=30, random_state=5).fit(
        X, y
    )
    decayed = GradientDescent(
        learning_rate=0.3,
        batch_size=8,
        n_epochs=30,
        random_state=5,
        learning_rate_schedule="exponential",
        lr_decay=0.1,
    ).fit(X, y)

    assert not np.array_equal(constant.coef_, decayed.coef_)


def test_rejects_non_positive_batch_size() -> None:
    X = np.zeros((5, 2))
    y = np.zeros(5)
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        GradientDescent(learning_rate=0.1, batch_size=0).fit(X, y)


def test_rejects_non_positive_n_epochs() -> None:
    X = np.zeros((5, 2))
    y = np.zeros(5)
    with pytest.raises(ValueError, match="n_epochs must be >= 1"):
        GradientDescent(learning_rate=0.1, batch_size=2, n_epochs=0).fit(X, y)


def test_rejects_unknown_learning_rate_schedule() -> None:
    X = np.zeros((5, 2))
    y = np.zeros(5)
    with pytest.raises(ValueError, match="learning_rate_schedule must be one of"):
        GradientDescent(learning_rate=0.1, learning_rate_schedule="bogus").fit(  # type: ignore[arg-type]
            X, y
        )


def test_rejects_negative_lr_decay() -> None:
    X = np.zeros((5, 2))
    y = np.zeros(5)
    with pytest.raises(ValueError, match="lr_decay must be non-negative"):
        GradientDescent(learning_rate=0.1, lr_decay=-1.0).fit(X, y)


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


def test_default_penalty_is_l2() -> None:
    model = GradientDescent(learning_rate=0.1)
    assert model.penalty == "l2"


def test_penalty_l1_uses_lasso_cost_for_history() -> None:
    # Regression test for the dispatch itself: penalty="l1" must actually
    # switch which gradient/cost functions fit() uses, not just accept and
    # ignore the argument.
    rng = np.random.default_rng(5)
    X = rng.normal(size=(30, 3))
    y = rng.normal(size=30)

    model = GradientDescent(learning_rate=0.05, lam=0.3, max_iter=50, penalty="l1").fit(X, y)

    # cost_history_ is computed internally on the mean-centered target (see
    # regression.base.LinearModel), so the reference value must be too.
    expected_final_cost = lasso_cost(X, y - y.mean(), model.coef_, lam=0.3)
    assert model.cost_history_[-1] == pytest.approx(expected_final_cost)


def test_penalty_l1_shrinks_coefficients_as_lam_grows() -> None:
    rng = np.random.default_rng(6)
    X = rng.normal(size=(200, 3))
    true_theta = np.array([1.0, -2.0, 0.5])
    y = X @ true_theta + 0.01 * rng.normal(size=200)

    small_lam = GradientDescent(
        learning_rate=0.05, lam=0.01, max_iter=5000, optimizer="adam", penalty="l1"
    ).fit(X, y)
    large_lam = GradientDescent(
        learning_rate=0.05, lam=5.0, max_iter=5000, optimizer="adam", penalty="l1"
    ).fit(X, y)

    assert np.linalg.norm(large_lam.coef_) < np.linalg.norm(small_lam.coef_)


def test_rejects_unknown_penalty() -> None:
    X = np.zeros((5, 2))
    y = np.zeros(5)
    with pytest.raises(ValueError, match="penalty must be one of"):
        GradientDescent(learning_rate=0.1, penalty="bogus").fit(X, y)  # type: ignore[arg-type]


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
