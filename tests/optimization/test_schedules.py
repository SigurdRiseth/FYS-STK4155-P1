import numpy as np
import pytest

from fys_stk4155_p1.optimization.schedules import (
    constant_schedule,
    exponential_decay,
    time_based_decay,
)


def test_constant_schedule_returns_fixed_value() -> None:
    schedule = constant_schedule(0.1)
    assert schedule(0) == pytest.approx(0.1)
    assert schedule(1000) == pytest.approx(0.1)


def test_time_based_decay_formula() -> None:
    schedule = time_based_decay(learning_rate=0.5, decay_rate=0.1)
    for t in [0, 1, 10, 100]:
        assert schedule(t) == pytest.approx(0.5 / (1 + 0.1 * t))


def test_time_based_decay_at_zero_matches_learning_rate() -> None:
    schedule = time_based_decay(learning_rate=0.3, decay_rate=0.2)
    assert schedule(0) == pytest.approx(0.3)


def test_time_based_decay_reduces_to_constant_when_decay_zero() -> None:
    schedule = time_based_decay(learning_rate=0.3, decay_rate=0.0)
    for t in [0, 5, 500]:
        assert schedule(t) == pytest.approx(0.3)


def test_time_based_decay_is_monotonically_non_increasing() -> None:
    schedule = time_based_decay(learning_rate=0.3, decay_rate=0.1)
    values = [schedule(t) for t in range(50)]
    assert np.all(np.diff(values) <= 0)


def test_exponential_decay_formula() -> None:
    schedule = exponential_decay(learning_rate=0.5, decay_rate=0.1)
    for t in [0, 1, 10, 100]:
        assert schedule(t) == pytest.approx(0.5 * np.exp(-0.1 * t))


def test_exponential_decay_at_zero_matches_learning_rate() -> None:
    schedule = exponential_decay(learning_rate=0.3, decay_rate=0.2)
    assert schedule(0) == pytest.approx(0.3)


def test_exponential_decay_reduces_to_constant_when_decay_zero() -> None:
    schedule = exponential_decay(learning_rate=0.3, decay_rate=0.0)
    for t in [0, 5, 500]:
        assert schedule(t) == pytest.approx(0.3)


def test_exponential_decay_is_monotonically_non_increasing() -> None:
    schedule = exponential_decay(learning_rate=0.3, decay_rate=0.1)
    values = [schedule(t) for t in range(50)]
    assert np.all(np.diff(values) <= 0)
