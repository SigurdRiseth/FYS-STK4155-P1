"""Learning-rate schedules for stochastic gradient descent (Section 4.7.1 of
Hjorth-Jensen (2026); Géron, Chapter 11). Each factory returns a callable mapping
a global step count `t` (total mini-batch updates taken so far) to that step's
learning rate; `regression.gradient_descent.GradientDescent` dispatches to these
internally via its `learning_rate_schedule`/`lr_decay` parameters rather than
accepting a callable directly (see that module for why)."""

from collections.abc import Callable

import numpy as np


def constant_schedule(learning_rate: float) -> Callable[[int], float]:
    """A schedule that never decays: every step uses the same learning rate.

    Args:
        learning_rate: The fixed learning rate to return at every step.

    Returns:
        A callable `t -> learning_rate`, ignoring `t`.
    """
    return lambda t: learning_rate


def time_based_decay(learning_rate: float, decay_rate: float) -> Callable[[int], float]:
    """Time-based decay: `learning_rate / (1 + decay_rate * t)`.

    The same family as the classic Bottou/Hjorth-Jensen (Section 4.7.1) schedule
    `t0 / (t + t1)`: writing `learning_rate = t0/t1` and `decay_rate = 1/t1` gives
    `t0/(t+t1) = learning_rate/(1 + decay_rate*t)`, i.e. this is that schedule
    under a reparameterization directly in terms of the initial learning rate.

    Args:
        learning_rate: Learning rate at `t=0`.
        decay_rate: Non-negative decay strength; `decay_rate=0` recovers a
            constant schedule.

    Returns:
        A callable `t -> learning_rate / (1 + decay_rate * t)`.
    """
    return lambda t: learning_rate / (1 + decay_rate * t)


def exponential_decay(learning_rate: float, decay_rate: float) -> Callable[[int], float]:
    """Exponential decay: `learning_rate * exp(-decay_rate * t)` (Géron, Ch. 11).

    Args:
        learning_rate: Learning rate at `t=0`.
        decay_rate: Non-negative decay strength; `decay_rate=0` recovers a
            constant schedule.

    Returns:
        A callable `t -> learning_rate * exp(-decay_rate * t)`.
    """
    return lambda t: float(learning_rate * np.exp(-decay_rate * t))
