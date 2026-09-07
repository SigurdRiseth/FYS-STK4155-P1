import numpy as np

from fys_stk4155_p1.metrics import mean_squared_error, r2_score


def test_mean_squared_error_zero_for_perfect_prediction() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert mean_squared_error(y, y) == 0.0


def test_mean_squared_error_matches_definition() -> None:
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    # mean of [1, 4, 9] = 14 / 3
    assert mean_squared_error(y_true, y_pred) == 14.0 / 3.0


def test_r2_score_one_for_perfect_prediction() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_score(y, y) == 1.0


def test_r2_score_zero_for_mean_prediction() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.full_like(y_true, np.mean(y_true))
    assert r2_score(y_true, y_pred) == 0.0


def test_r2_score_negative_for_worse_than_mean_prediction() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([10.0, -10.0, 10.0, -10.0])
    assert r2_score(y_true, y_pred) < 0.0
