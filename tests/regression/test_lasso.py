import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import Lasso as SkLasso

from fys_stk4155_p1.regression.cost import sklearn_alpha_to_lam
from fys_stk4155_p1.regression.lasso import Lasso


def test_fit_returns_self() -> None:
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0])
    model = Lasso(learning_rate=0.1)
    assert model.fit(X, y) is model


def test_larger_lambda_increases_sparsity() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    true_theta = np.array([1.0, -2.0, 0.5])
    y = X @ true_theta + 0.01 * rng.normal(size=200)

    small_lam = Lasso(learning_rate=0.05, lam=0.01, max_iter=5000, optimizer="adam").fit(X, y)
    large_lam = Lasso(learning_rate=0.05, lam=5.0, max_iter=5000, optimizer="adam").fit(X, y)

    assert np.linalg.norm(large_lam.coef_) < np.linalg.norm(small_lam.coef_)


def test_fit_intercept_column_is_not_penalized() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=100)
    X = np.column_stack([np.ones_like(x), x])
    y = 5.0 + 2.0 * x

    model = Lasso(
        learning_rate=0.1, lam=0.1, max_iter=5000, optimizer="adam", fit_intercept_column=True
    ).fit(X, y)

    # a large penalty should still leave the (unpenalized) intercept close to
    # the true value while driving the (penalized) slope toward zero.
    unpenalized = Lasso(
        learning_rate=0.1, lam=1e3, max_iter=5000, optimizer="adam", fit_intercept_column=True
    ).fit(X, y)
    assert unpenalized.coef_[0] == pytest.approx(5.0, abs=0.5)
    assert abs(unpenalized.coef_[1]) < abs(model.coef_[1])


def test_predict_matches_design_matrix_times_coef() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 3))
    y = rng.normal(size=20)

    model = Lasso(learning_rate=0.05, lam=0.1, max_iter=200).fit(X, y)
    y_pred = model.predict(X)

    np.testing.assert_allclose(y_pred, X @ model.coef_)


def test_rejects_negative_lambda() -> None:
    X = np.zeros((5, 2))
    y = np.zeros(5)
    with pytest.raises(ValueError, match="lam must be non-negative"):
        Lasso(learning_rate=0.1, lam=-1.0).fit(X, y)


def test_rejects_mismatched_n_samples() -> None:
    X = np.zeros((5, 2))
    y = np.zeros(4)
    with pytest.raises(ValueError, match="disagree on n_samples"):
        Lasso(learning_rate=0.1).fit(X, y)


def test_clone_round_trips_without_penalty_kwarg() -> None:
    # Lasso always fits penalty="l1" internally (see its docstring); confirm
    # that's invisible to sklearn's get_params()/clone() machinery, which
    # only introspects Lasso's own __init__ signature, and that the clone
    # still behaves like a Lasso (not a plain GradientDescent).
    model = Lasso(learning_rate=0.1, lam=0.2, optimizer="adam")

    params = model.get_params()
    assert "penalty" not in params

    cloned = clone(model)
    assert isinstance(cloned, Lasso)
    assert cloned.get_params() == params


def test_matches_sklearn_lasso_with_alpha_conversion() -> None:
    # Careful with conventions (per the issue): our lam and sklearn's alpha
    # relate by lam = 2*alpha (see sklearn_alpha_to_lam's docstring). No bias
    # term on either side: our Lasso fits with fit_intercept_column=False
    # (the default) on a pre-centered y, matching sklearn's fit_intercept=False
    # on that same centered y, rather than either side fitting its own
    # intercept.
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 4))
    true_theta = np.array([1.5, 0.0, -0.8, 0.0])
    y = X @ true_theta + 0.05 * rng.normal(size=300)
    y_centered = y - y.mean()

    alpha = 0.05
    lam = sklearn_alpha_to_lam(alpha)

    ours = Lasso(learning_rate=0.05, lam=lam, max_iter=20000, optimizer="adam").fit(X, y_centered)
    sk = SkLasso(alpha=alpha, fit_intercept=False, max_iter=100000, tol=1e-12).fit(X, y_centered)

    # Subgradient descent doesn't land on exact zeros the way sklearn's
    # coordinate descent does (see lasso_subgradient's docstring), so this
    # is a loose tolerance, not a tight numerical match.
    np.testing.assert_allclose(ours.coef_, sk.coef_, atol=0.05)
