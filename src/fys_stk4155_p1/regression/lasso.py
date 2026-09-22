"""Lasso regression."""

from typing import Literal

from fys_stk4155_p1.regression.gradient_descent import GradientDescent


class Lasso(GradientDescent):
    """Lasso regression, fit by gradient descent.

    Unlike OLS/Ridge, the L1-penalized cost has no closed-form solution, so
    this is a thin convenience subclass of `GradientDescent` that always
    fits with `penalty="l1"` — see that class's docstring for the shared
    fit loop, the `Optimizer` registry, and the analytical/autodiff
    subgradient discussion. Exposes the same knobs as `GradientDescent`
    minus `penalty` itself, so `Lasso(learning_rate=..., lam=...)` reads
    symmetric to `Ridge(lam=...)`.

    Args:
        learning_rate: Step size passed to the optimizer.
        lam: L1 regularization strength. Defaults to 0.0 (plain OLS cost).
        max_iter: Maximum number of gradient steps to take.
        tol: Convergence tolerance: iteration stops early once
            `max(abs(grad)) < tol`.
        gradient_method: How to compute the gradient at each step.
        optimizer: Which `Optimizer` subclass to fit with.
        momentum: Velocity decay, used only when `optimizer="momentum"`
            (passed as that optimizer's `beta`).
        beta1: First-moment decay, used only when `optimizer="adam"`.
        beta2: Second-moment decay, used only when `optimizer="adam"`.
        rho: Squared-gradient decay, used only when `optimizer="rmsprop"`.
        eps: Numerical-stability constant, used by every optimizer except
            "plain" and "momentum".
        fit_intercept_column: If True, the first column of X is treated as
            an all-ones intercept term and is excluded from the penalty.
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
        fit_intercept_column: bool = False,
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
            fit_intercept_column=fit_intercept_column,
            penalty="l1",
        )
