"""Gradient-descent regression: pairs a gradient (analytical or autodiff) with
any `Optimizer` from `optimization.optimizers` to fit OLS/Ridge by iteration
instead of the normal equations."""

from collections.abc import Callable
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import NDArray

from fys_stk4155_p1.optimization.optimizers import OPTIMIZER_REGISTRY
from fys_stk4155_p1.regression.autodiff import autodiff_gradient
from fys_stk4155_p1.regression.base import LinearModel
from fys_stk4155_p1.regression.cost import analytical_gradient, cost

_GradientFn = Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float, bool],
    NDArray[np.float64],
]

_GRADIENT_FNS: dict[str, _GradientFn] = {
    "analytical": analytical_gradient,
    "autodiff": autodiff_gradient,
}

# Per-optimizer constructor kwargs, keyed by OPTIMIZER_REGISTRY's names; built
# from GradientDescent's own hyperparameters in fit(). Plain takes only
# learning_rate (added separately below), so it needs no entry here.
_OPTIMIZER_KWARGS: dict[str, tuple[str, ...]] = {
    "momentum": ("momentum",),
    "adagrad": ("eps",),
    "rmsprop": ("rho", "eps"),
    "adam": ("beta1", "beta2", "eps"),
}

# Maps a GradientDescent attribute name to the constructor keyword the target
# Optimizer expects it under (only "momentum" -> "beta" differs).
_OPTIMIZER_KWARG_ALIASES: dict[str, str] = {"momentum": "beta"}


class GradientDescent(LinearModel):
    """Ridge/OLS regression fit by iterative gradient descent.

    Minimizes the same penalized cost as `Ridge` (`lam=0` recovers OLS), but
    via repeated `optimizer.step` calls on `theta` instead of solving the
    normal equations directly, following the general form of Eq. (4.10) in
    Hjorth-Jensen (2026). The gradient at each step comes from either the
    closed form (`regression.cost.analytical_gradient`) or JAX autodiff
    (`regression.autodiff.autodiff_gradient`); both compute the same
    quantity (see `tests/regression/test_autodiff.py`), so `gradient_method`
    only affects how it's computed, not the fitted result. `optimizer`
    selects which `Optimizer` subclass drives the updates (see
    `optimization.optimizers`); only the hyperparameters relevant to the
    chosen optimizer are used; the rest are ignored.

    Args:
        learning_rate: Step size passed to the optimizer.
        lam: L2 regularization strength. Defaults to 0.0 (plain OLS cost).
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

    Attributes:
        coef_: Fitted coefficients, set by `fit`.
        cost_history_: Cost at the end of each step actually taken, shape
            (n_iter_,).
        n_iter_: Number of gradient steps actually taken (<= max_iter).
    """

    cost_history_: NDArray[np.float64]
    n_iter_: int

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
        # flat verbatim assignment only — sklearn clonability
        self.learning_rate = learning_rate
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.gradient_method = gradient_method
        self.optimizer = optimizer
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.rho = rho
        self.eps = eps
        self.fit_intercept_column = fit_intercept_column

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> Self:
        """Fit theta by gradient descent.

        Args:
            X: Design matrix, shape (n_samples, n_features).
            y: Target values, shape (n_samples,).

        Returns:
            self, with coef_, cost_history_, and n_iter_ set.

        Raises:
            ValueError: If `lam` is negative, `max_iter` is not strictly
                positive, or `gradient_method`/`optimizer` is not one of
                the supported names.
        """
        X, y = self._validate_inputs(X, y)
        if self.lam < 0:
            raise ValueError(f"lam must be non-negative, got {self.lam}.")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}.")
        if self.gradient_method not in _GRADIENT_FNS:
            raise ValueError(
                f"gradient_method must be one of {sorted(_GRADIENT_FNS)}, "
                f"got {self.gradient_method!r}."
            )
        if self.optimizer not in OPTIMIZER_REGISTRY:
            raise ValueError(
                f"optimizer must be one of {sorted(OPTIMIZER_REGISTRY)}, got {self.optimizer!r}."
            )

        grad_fn = _GRADIENT_FNS[self.gradient_method]

        kwargs: dict[str, Any] = {}
        for attr in _OPTIMIZER_KWARGS.get(self.optimizer, ()):
            kwargs[_OPTIMIZER_KWARG_ALIASES.get(attr, attr)] = getattr(self, attr)
        opt = OPTIMIZER_REGISTRY[self.optimizer](learning_rate=self.learning_rate, **kwargs)

        n_features = X.shape[1]
        opt.reset(n_features)
        theta = np.zeros(n_features, dtype=np.float64)

        cost_history = []
        for _ in range(self.max_iter):
            grad = grad_fn(X, y, theta, self.lam, self.fit_intercept_column)
            theta = opt.step(theta, grad)
            cost_history.append(cost(X, y, theta, self.lam, self.fit_intercept_column))
            if np.max(np.abs(grad)) < self.tol:
                break

        self.coef_ = theta
        self.cost_history_ = np.array(cost_history, dtype=np.float64)
        self.n_iter_ = self.cost_history_.shape[0]
        return self


def gradient_descent_sweep(
    X: NDArray[np.float64], y: NDArray[np.float64], variants: dict[str, dict[str, Any]]
) -> dict[str, GradientDescent]:
    """Fit one GradientDescent(**kwargs) per named variant.

    Mirrors `shrinkage.ridge_coefficient_path`'s one-model-per-setting
    pattern, generalized to arbitrary keyword combinations (e.g. comparing
    optimizers or learning rates) rather than a single swept parameter.

    Args:
        X: Design matrix, shape (n_samples, n_features).
        y: Target values, shape (n_samples,).
        variants: Maps a variant name to the keyword arguments passed to
            `GradientDescent`.

    Returns:
        Fitted models keyed by variant name.
    """
    return {name: GradientDescent(**kwargs).fit(X, y) for name, kwargs in variants.items()}
