import numpy as np

from fys_stk4155_p1.optimization.benchmark import iterations_to_tolerance
from fys_stk4155_p1.regression.ridge import Ridge


def _problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.normal(size=50)
    y = y - y.mean()  # cost/analytical_gradient never center y themselves
    theta_star = Ridge(lam=0.0).fit(X, y).coef_
    return X, y, theta_star


def test_plain_gd_reaches_closed_form_on_well_conditioned_problem() -> None:
    X, y, theta_star = _problem()

    result = iterations_to_tolerance(X, y, theta_star, "plain", learning_rate=0.1)

    assert not result["diverged"]
    assert result["iters_param"] is not None
    assert result["final_param_err"] < 1e-3


def test_too_large_learning_rate_is_reported_as_divergence() -> None:
    X, y, theta_star = _problem()

    result = iterations_to_tolerance(X, y, theta_star, "plain", learning_rate=10.0)

    assert result["diverged"]
    assert result["iters_param"] is None


def test_is_deterministic() -> None:
    X, y, theta_star = _problem()

    r1 = iterations_to_tolerance(X, y, theta_star, "adam", learning_rate=0.05, max_iter=500)
    r2 = iterations_to_tolerance(X, y, theta_star, "adam", learning_rate=0.05, max_iter=500)

    assert r1 == r2
