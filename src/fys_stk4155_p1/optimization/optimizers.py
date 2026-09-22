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

    @staticmethod
    def _check_shapes(theta: NDArray[np.float64], grad: NDArray[np.float64]) -> None:
        """Validate that `theta` and `grad` have matching shapes.

        Args:
            theta: Current parameter values.
            grad: Gradient of the cost function at `theta`.

        Raises:
            ValueError: If `theta.shape != grad.shape`.
        """
        if theta.shape != grad.shape:
            raise ValueError(f"shape mismatch: theta {theta.shape}, grad {grad.shape}")

    @staticmethod
    def _init_or_check_state(
        state: NDArray[np.float64] | None, grad: NDArray[np.float64], name: str
    ) -> NDArray[np.float64]:
        """Lazily allocate a per-parameter state buffer, or validate its shape.

        Args:
            state: Existing state buffer, or `None` if not yet allocated.
            grad: Gradient of the cost function, used to size/validate `state`.
            name: State attribute name, used in the error message.

        Returns:
            `state` if already allocated and matching `grad.shape`, otherwise
            a new zero-filled buffer shaped like `grad`.

        Raises:
            ValueError: If `state` is allocated but its shape differs from
                `grad.shape`.
        """
        if state is None:
            return np.zeros_like(grad, dtype=np.float64)
        if state.shape != grad.shape:
            raise ValueError(
                f"grad shape {grad.shape} differs from {name} shape {state.shape}; call reset()"
            )
        return state


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
        self._check_shapes(theta, grad)
        return theta - self.learning_rate * grad


class Momentum(Optimizer):  # TODO: Should we add a Nesterov flag?
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

    v_: NDArray[np.float64] | None

    def __init__(self, learning_rate: float, beta: float = 0.9) -> None:
        super().__init__(learning_rate)

        # beta >= 1 never decays the velocity, so the updates diverge.
        if not 0 <= beta < 1:
            raise ValueError(f"beta must be in [0, 1), got {beta}.")

        self.beta = beta
        self.v_ = None

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
        previous run. If `grad`'s shape differs from an already-allocated
        velocity buffer, call `reset` first.

        Args:
            theta: Current parameter values.
            grad: Gradient of the cost function at `theta`.

        Returns:
            Updated parameter values.

        Raises:
            ValueError: If `theta` and `grad` shapes differ, or if `grad`'s
                shape differs from the existing velocity buffer's shape.
        """
        self._check_shapes(theta, grad)
        self.v_ = self._init_or_check_state(self.v_, grad, "v_")

        self.v_ = self.beta * self.v_ + self.learning_rate * grad

        return theta - self.v_


class AdaGrad(Optimizer):
    """AdaGrad gradient descent.

    Maintains a running sum of squared gradients for each coordinate,
    following Eq. (4.45) in Hjorth-Jensen (2026):

        r_t     = r_{t-1} + grad * grad
        theta_t = theta_{t-1} - learning_rate / (sqrt(r_t) + eps) * grad

    Each parameter thus has its own effective step size. `r` is
    initialized to zero on the first call to `step` (or by `reset`).

    Args:
        learning_rate: Step size scaling the gradient in each update.
        eps: Small positive constant for numerical stability.

    Raises:
        ValueError: If `learning_rate` or `eps` is not strictly positive.
    """

    r_: NDArray[np.float64] | None
    eps: float

    def __init__(self, learning_rate: float, eps: float = 1e-8) -> None:
        super().__init__(learning_rate)

        if eps <= 0:
            raise ValueError(f"eps must be strictly positive, got {eps}")
        self.eps = eps
        self.r_ = None

    def reset(self, n_params: int) -> None:
        """Discard accumulated squared gradients."""
        self.r_ = None

    def step(self, theta: NDArray[np.float64], grad: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the updated parameters for one AdaGrad step.

        Args:
            theta: Current parameter values.
            grad: Gradient of the cost function at `theta`.

        Returns:
            Updated parameter values.

        Raises:
            ValueError: If `theta` and `grad` shapes differ, or if `grad`'s
                shape differs from the existing accumulator's shape.
        """
        self._check_shapes(theta, grad)
        self.r_ = self._init_or_check_state(self.r_, grad, "r_")

        self.r_ += grad * grad
        return theta - self.learning_rate * grad / (np.sqrt(self.r_) + self.eps)


class RMSProp(Optimizer):
    """RMSProp gradient descent.

    Maintains an exponentially decaying average of squared gradients:

        v_t     = rho * v_{t-1} + (1 - rho) * grad * grad
        theta_t = theta_{t-1} - learning_rate / (sqrt(v_t) + eps) * grad

    `v` is initialized to zero on the first call to `step`.

    Args:
        learning_rate: Step size scaling the gradient in each update.
        rho: Decay rate of the squared-gradient average, in [0, 1).
        eps: Small positive constant for numerical stability.

    Raises:
        ValueError: If `learning_rate` or `eps` is not strictly positive,
            or `rho` is not in [0, 1).
    """

    rho: float
    eps: float
    v_: NDArray[np.float64] | None

    def __init__(self, learning_rate: float, rho: float = 0.9, eps: float = 1e-8) -> None:
        super().__init__(learning_rate)
        if not 0 <= rho < 1:
            raise ValueError(f"rho must be in [0, 1), got {rho}")
        if eps <= 0:
            raise ValueError(f"eps must be strictly positive, got {eps}")
        self.eps = eps
        self.rho = rho
        self.v_ = None

    def reset(self, n_params: int) -> None:
        """Discard accumulated state."""
        self.v_ = None

    def step(self, theta: NDArray[np.float64], grad: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the updated parameters for one RMSProp step.

        Args:
            theta: Current parameter values.
            grad: Gradient of the cost function at `theta`.

        Returns:
            Updated parameter values.

        Raises:
            ValueError: If `theta` and `grad` shapes differ, or if `grad`'s
                shape differs from the existing accumulator's shape.
        """
        self._check_shapes(theta, grad)
        self.v_ = self._init_or_check_state(self.v_, grad, "v_")

        self.v_ = self.rho * self.v_ + (1 - self.rho) * grad * grad
        return theta - self.learning_rate / (np.sqrt(self.v_) + self.eps) * grad


# TODO: Implement Adam
# class Adam(Optimizer):
#    pass


OPTIMIZER_REGISTRY: dict[str, type[Optimizer]] = {
    "plain": Plain,
    "momentum": Momentum,
    "adagrad": AdaGrad,
    "rmsprop": RMSProp,
    #    "adam": Adam,
}
