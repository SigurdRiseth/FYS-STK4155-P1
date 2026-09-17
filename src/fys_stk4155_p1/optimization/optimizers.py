"""Optimizers."""

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
    """

    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    @abstractmethod
    def reset(self, n_params: int) -> None:
        """Reset internal state ahead of a new optimization run.

        Args:
            n_params: Number of parameters being optimized, used to size
                any internal state buffers.
        """

    @abstractmethod
    def step(
        self, theta: NDArray[np.float64], grad: NDArray[np.float64]
    ) -> NDArray[np.float64]:
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

    def reset(self, n_params: int) -> None:
        """No-op: plain gradient descent holds no internal state.

        Args:
            n_params: Number of parameters being optimized (unused).
        """

    def step(
        self, theta: NDArray[np.float64], grad: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute the updated parameters for one optimization step.

        Args:
            theta: Current parameter values.
            grad: Gradient of the cost function at `theta`.

        Returns:
            Updated parameter values.
        """
        return theta - self.learning_rate * grad


OPTIMIZER_REGISTRY: dict[str, type[Optimizer]] = {
    "plain": Plain,
}
