"""Lasso regression."""

from typing import Literal

from fys_stk4155_p1.regression.gradient_descent import GradientDescent


class Lasso(GradientDescent):
    """Lasso regression, fit by gradient descent or, if `batch_size` is set,
    mini-batch stochastic gradient descent (SGD).

    Unlike OLS/Ridge, the L1-penalized cost has no closed-form solution, so
    this is a thin convenience subclass of `GradientDescent` that always
    fits with `penalty="l1"` — see that class's docstring for the shared
    full-batch/SGD fit loops, the `Optimizer` registry, the learning-rate
    schedules, and the analytical/autodiff subgradient discussion. Exposes
    the same knobs as `GradientDescent` minus `penalty` itself, so
    `Lasso(learning_rate=..., lam=...)` reads symmetric to `Ridge(lam=...)`.

    Args:
        learning_rate: Step size passed to the optimizer (its initial value,
            if `learning_rate_schedule` is not `"constant"`).
        lam: L1 regularization strength. Defaults to 0.0 (plain OLS cost).
        max_iter: Maximum number of gradient steps to take. Only used when
            `batch_size` is `None` (full-batch mode).
        tol: Convergence tolerance: full-batch iteration stops early once
            `max(abs(grad)) < tol`. Only used when `batch_size` is `None`.
        gradient_method: How to compute the gradient at each step.
        optimizer: Which `Optimizer` subclass to fit with.
        momentum: Velocity decay, used only when `optimizer="momentum"`
            (passed as that optimizer's `beta`).
        beta1: First-moment decay, used only when `optimizer="adam"`.
        beta2: Second-moment decay, used only when `optimizer="adam"`.
        rho: Squared-gradient decay, used only when `optimizer="rmsprop"`.
        eps: Numerical-stability constant, used by every optimizer except
            "plain" and "momentum".
        batch_size: Mini-batch size for SGD. `None` (the default) fits by
            full-batch gradient descent instead.
        n_epochs: Number of passes over the training set. Only used when
            `batch_size` is not `None` (SGD mode).
        learning_rate_schedule: How the learning rate evolves over SGD
            updates: `"constant"`, `"time_based"`, or `"exponential"` (see
            `GradientDescent`). Only used in SGD mode.
        lr_decay: Decay strength for a non-`"constant"` schedule. Only used
            in SGD mode.
        random_state: Seeds the per-epoch shuffling in SGD mode.
    """

    def __init__(
        self,
        learning_rate: float,
        lam: float = 0.0,
        max_iter: int = 1000,
        tol: float = 1e-8,
        gradient_method: Literal["analytical", "autodiff"] = "analytical",
        optimizer: Literal["plain", "momentum", "adagrad", "rmsprop", "adam"] = "plain",
        momentum: float = 0.9,
        beta1: float = 0.9,
        beta2: float = 0.999,
        rho: float = 0.9,
        eps: float = 1e-8,
        batch_size: int | None = None,
        n_epochs: int = 100,
        learning_rate_schedule: Literal["constant", "time_based", "exponential"] = "constant",
        lr_decay: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            lam=lam,
            max_iter=max_iter,
            tol=tol,
            gradient_method=gradient_method,
            optimizer=optimizer,
            momentum=momentum,
            beta1=beta1,
            beta2=beta2,
            rho=rho,
            eps=eps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate_schedule=learning_rate_schedule,
            lr_decay=lr_decay,
            random_state=random_state,
            penalty="l1",
        )
