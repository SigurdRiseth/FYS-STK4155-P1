"""Optimizers."""

# TODO: Consider adding a `decay` parameter for learning rate scheduling
# (ref. Géron Chapter 11)

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Optimizer(ABC):
    """Base class for gradient-based optimizers.

    Subclasses implement `step` to compute a parameter update from a
    gradient, and `reset` to clear any internal state (e.g. momentum
    buffers) between independent optimization runs.

    Args:
        learning_rate: Step size scaling the gradient in each update.

    Raises:
        ValueError: If `learning_rate` is not strictly positive.
    """

    def __init__(self, learning_rate: float) -> None:
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}.")

        self.learning_rate = learning_rate

    @abstractmethod
    def reset(self, n_params: int) -> None:
        """Reset internal state ahead of a new optimization run.

        Args:
            n_params: Number of parameters being optimized, used to size
                any internal state buffers.
        """

    @abstractmethod
    def step(self, theta: NDArray[np.float64], grad: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the updated parameters for one optimization step.

        Args:
            theta: Current parameter values.
            grad: Gradient of the cost function at `theta`.

        Returns:
            Updated parameter values.
        """


class Plain(Optimizer):
    """Plain gradient descent.

    Uses only the gradient and a constant learning rate to determine the
    step size, following the iterative approach given by Eq. (4.10) from
    Hjorth-Jensen (2026).
    """

    def reset(self, n_params: int) -> None:  # noqa: ARG002
        """No-op: plain gradient descent holds no internal state.

        Args:
            n_params: Number of parameters being optimized (unused).
        """

    def step(self, theta: NDArray[np.float64], grad: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the updated parameters for one optimization step.

        Args:
            theta: Current parameter values.
            grad: Gradient of the cost function at `theta`.

        Returns:
            Updated parameter values.
        """
        return theta - self.learning_rate * grad


class Momentum(Optimizer):
    """Gradient descent with momentum.

    Accumulates an exponentially weighted moving average of past gradients
    in a velocity buffer and steps along that instead of the raw gradient,
    following Eq. (4.28) in Hjorth-Jensen (2026):

        v_t = beta * v_{t-1} + learning_rate * grad
        theta_t = theta_{t-1} - v_t

    The averaging damps oscillations across steep directions of the cost
    surface while letting consistent gradient directions build up speed,
    so ill-conditioned problems converge in fewer iterations than plain
    gradient descent. `beta` sets the memory of the average: 0 recovers
    `Plain`, and 1 / (1 - beta) is roughly the number of past gradients
    that contribute meaningfully (beta = 0.9 averages about 10).

    Args:
        learning_rate: Step size scaling the gradient in each update.
        beta: Momentum coefficient in [0, 1), the decay applied to the
            velocity from the previous step. Defaults to 0.9.

    Raises:
        ValueError: If `learning_rate` is not strictly positive, or if
            `beta` lies outside [0, 1).
    """

    v_: NDArray[np.float64]

    def __init__(self, learning_rate: float, beta: float = 0.9) -> None:
        super().__init__(learning_rate)

        # beta >= 1 never decays the velocity, so the updates diverge.
        if not 0 <= beta < 1:
            raise ValueError(f"beta must be in [0, 1), got {beta}.")

        self.beta = beta
        self.v_ = np.zeros(0, dtype=np.float64)

    def reset(self, n_params: int) -> None:
        """Zero the velocity buffer ahead of a new optimization run.

        Args:
            n_params: Number of parameters being optimized, used to size
                the velocity buffer.
        """
        self.v_ = np.zeros(n_params, dtype=np.float64)

    def step(self, theta: NDArray[np.float64], grad: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the updated parameters for one momentum step.

        The velocity buffer is allocated on first use, so calling `reset`
        beforehand is only needed to discard velocity carried over from a
        previous run.

        Args:
            theta: Current parameter values.
            grad: Gradient of the cost function at `theta`.

        Returns:
            Updated parameter values.
        """
        if self.v_.shape != grad.shape:
            self.v_ = np.zeros_like(grad, dtype=np.float64)

        self.v_ = self.beta * self.v_ + self.learning_rate * grad

        return theta - self.v_


# TODO: Add AdaGrad
# TODO: Add RMSProp
# TODO: Add Adam

OPTIMIZER_REGISTRY: dict[str, type[Optimizer]] = {
    "plain": Plain,
    "momentum": Momentum,
}
