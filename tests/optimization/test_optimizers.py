import numpy as np
import pytest

from fys_stk4155_p1.optimization.optimizers import (
    OPTIMIZER_REGISTRY,
    AdaGrad,
    Adam,
    Momentum,
    Optimizer,
    Plain,
    RMSProp,
)


@pytest.mark.parametrize("cls", OPTIMIZER_REGISTRY.values(), ids=OPTIMIZER_REGISTRY.keys())
class TestAllOptimizers:
    """Behavior every `Optimizer` subclass must satisfy, regardless of update rule."""

    @staticmethod
    def quadratic_grad(theta: np.ndarray) -> np.ndarray:
        """Gradient of f(theta) = (theta - 3)**2, minimized at theta = 3."""
        return 2.0 * (theta - 3.0)

    def test_converges_on_quadratic(self, cls: type[Optimizer]) -> None:
        optimizer = cls(learning_rate=0.1)
        theta = np.array([0.0])
        for _ in range(2000):
            theta = optimizer.step(theta, self.quadratic_grad(theta))
        assert theta == pytest.approx(3.0, abs=1e-2)

    def test_rejects_shape_mismatch(self, cls: type[Optimizer]) -> None:
        optimizer = cls(learning_rate=0.1)
        theta = np.zeros(3)
        grad = np.zeros(2)
        with pytest.raises(ValueError, match="shape mismatch"):
            optimizer.step(theta, grad)

    def test_rejects_non_positive_learning_rate(self, cls: type[Optimizer]) -> None:
        with pytest.raises(ValueError, match="learning_rate"):
            cls(learning_rate=0.0)

    def test_reset_makes_step_independent_of_history(self, cls: type[Optimizer]) -> None:
        optimizer = cls(learning_rate=0.1)
        theta = np.array([0.0])
        for _ in range(5):
            theta = optimizer.step(theta, self.quadratic_grad(theta))

        optimizer.reset(1)
        from_reset = optimizer.step(np.array([0.0]), self.quadratic_grad(np.array([0.0])))

        fresh = cls(learning_rate=0.1)
        from_fresh = fresh.step(np.array([0.0]), self.quadratic_grad(np.array([0.0])))

        np.testing.assert_allclose(from_reset, from_fresh)


def test_plain_step_matches_formula() -> None:
    theta = np.array([1.0, 2.0])
    grad = np.array([0.5, -1.0])
    result = Plain(learning_rate=0.1).step(theta, grad)
    np.testing.assert_allclose(result, theta - 0.1 * grad)


def test_momentum_accumulates_velocity_across_steps() -> None:
    theta = np.array([1.0])
    grad = np.array([1.0])
    optimizer = Momentum(learning_rate=0.1, beta=0.9)

    theta = optimizer.step(theta, grad)  # v = 0.1
    np.testing.assert_allclose(theta, [0.9])

    theta = optimizer.step(theta, grad)  # v = 0.9 * 0.1 + 0.1 = 0.19
    np.testing.assert_allclose(theta, [0.71])


def test_momentum_rejects_invalid_beta() -> None:
    with pytest.raises(ValueError, match="beta"):
        Momentum(learning_rate=0.1, beta=1.0)


def test_adagrad_step_matches_formula() -> None:
    theta = np.array([1.0])
    grad = np.array([2.0])
    optimizer = AdaGrad(learning_rate=0.1, eps=1e-8)

    result = optimizer.step(theta, grad)
    expected = theta - 0.1 * grad / (np.sqrt(grad**2) + 1e-8)
    np.testing.assert_allclose(result, expected)


def test_rmsprop_step_matches_formula() -> None:
    theta = np.array([1.0])
    grad = np.array([2.0])
    optimizer = RMSProp(learning_rate=0.1, rho=0.9, eps=1e-8)

    result = optimizer.step(theta, grad)
    v = (1 - 0.9) * grad**2
    expected = theta - 0.1 * grad / (np.sqrt(v) + 1e-8)
    np.testing.assert_allclose(result, expected)


def test_rmsprop_rejects_invalid_rho() -> None:
    with pytest.raises(ValueError, match="rho"):
        RMSProp(learning_rate=0.1, rho=1.0)


def test_adam_bias_correction_cancels_on_first_step() -> None:
    # With m_ and v_ initialized to zero, the bias correction on the very
    # first step exactly cancels the (1 - beta) decay factors, so the
    # update reduces to theta - lr * grad / (|grad| + eps).
    theta = np.array([1.0])
    grad = np.array([2.0])
    optimizer = Adam(learning_rate=0.1, eps=1e-8)

    result = optimizer.step(theta, grad)
    expected = theta - 0.1 * grad / (np.abs(grad) + 1e-8)
    np.testing.assert_allclose(result, expected)


def test_adam_rejects_invalid_betas() -> None:
    with pytest.raises(ValueError, match="beta1"):
        Adam(learning_rate=0.1, beta1=1.0)
    with pytest.raises(ValueError, match="beta2"):
        Adam(learning_rate=0.1, beta2=1.0)
