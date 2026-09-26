"""Gradient-descent regression: pairs a gradient (analytical or autodiff) with
any `Optimizer` from `optimization.optimizers` to fit OLS/Ridge (L2 penalty) or
Lasso (L1 penalty) by iteration instead of a closed-form solution. Supports both
full-batch gradient descent (`batch_size=None`, the default) and mini-batch
stochastic gradient descent (`batch_size` set), the latter looping over epochs
with a freshly shuffled ordering each epoch and an optional learning-rate
schedule (Section 4.7.1 of Hjorth-Jensen (2026))."""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from fys_stk4155_p1.optimization.optimizers import OPTIMIZER_REGISTRY, Optimizer
from fys_stk4155_p1.optimization.schedules import (
    constant_schedule,
    exponential_decay,
    time_based_decay,
)
from fys_stk4155_p1.regression.autodiff import autodiff_gradient, lasso_autodiff_gradient
from fys_stk4155_p1.regression.base import LinearModel
from fys_stk4155_p1.regression.cost import (
    analytical_gradient,
    cost,
    gradient_flops,
    lasso_cost,
    lasso_subgradient,
)

_GradientFn = Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float],
    NDArray[np.float64],
]
_CostFn = Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float],
    np.float64,
]
# theta, cost_history, cost_flops, n_updates
_FitResult = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int]

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

_PENALTY_GRADIENT_FNS: dict[str, dict[str, _GradientFn]] = {
    "l2": _GRADIENT_FNS,  # existing dict, unchanged
    "l1": {"analytical": lasso_subgradient, "autodiff": lasso_autodiff_gradient},
}
_COST_FNS: dict[str, _CostFn] = {"l2": cost, "l1": lasso_cost}

_SCHEDULES: tuple[str, ...] = ("constant", "time_based", "exponential")


class GradientDescent(LinearModel):
    """Ridge/OLS/Lasso regression fit by full-batch gradient descent or
    mini-batch stochastic gradient descent (SGD).

    Minimizes the same penalized cost as `Ridge` when `penalty="l2"`
    (`lam=0` recovers OLS), or a Lasso cost when `penalty="l1"` (no closed
    form exists for that case, unlike OLS/Ridge), via repeated
    `optimizer.step` calls on `theta` instead of solving the normal
    equations directly, following the general form of Eq. (4.10) in
    Hjorth-Jensen (2026). The gradient at each step comes from either the
    closed form (`regression.cost.analytical_gradient`/`lasso_subgradient`)
    or JAX autodiff (`regression.autodiff.autodiff_gradient`/
    `lasso_autodiff_gradient`); the L2 pair agrees to machine precision
    everywhere (see `tests/regression/test_autodiff.py`), while the L1 pair
    agrees everywhere except exactly at a zero coefficient, where `|theta_j|`
    is non-differentiable (see `lasso_subgradient`'s docstring). `optimizer`
    selects which `Optimizer` subclass drives the updates (see
    `optimization.optimizers`); only the hyperparameters relevant to the
    chosen optimizer are used; the rest are ignored. See also `Lasso`, a
    thin subclass that fixes `penalty="l1"`.

    By default (`batch_size=None`) this fits by full-batch gradient descent:
    one `optimizer.step` per iteration, on the gradient of the whole
    training set, for up to `max_iter` iterations, stopping early once
    `max(abs(grad)) < tol` (Section 4.5). Setting `batch_size` switches to
    mini-batch SGD (Section 4.7): for `n_epochs` epochs, the training set is
    shuffled and split into `batch_size`-sized mini-batches (the last one
    shorter if `batch_size` doesn't divide the training set evenly), and one
    `optimizer.step` is taken per mini-batch, on that mini-batch's gradient
    alone. `tol`/`max_iter` are not used in this mode -- a single
    mini-batch's gradient norm is too noisy to be a meaningful early-stopping
    signal, so SGD mode always runs exactly `n_epochs` epochs -- and
    `cost_history_` instead records one entry per epoch, evaluated on the
    full training set, so it stays comparable to full-batch mode's history.
    `learning_rate_schedule`/`lr_decay` (see `optimization.schedules`) let
    the learning rate decay over the run rather than stay fixed, keyed on
    the total number of mini-batch updates taken so far, matching Section
    4.7.1's/Géron Ch. 11's own convention.

    Args:
        learning_rate: Step size passed to the optimizer (its initial value,
            if `learning_rate_schedule` is not `"constant"`).
        penalty: Which penalty to fit: `"l2"` (Ridge/OLS) or `"l1"` (Lasso).
            Defaults to `"l2"`.
        lam: Regularization strength (L2 or L1, per `penalty`). Defaults to
            0.0 (plain OLS cost).
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
            updates: `"constant"` (unchanged throughout), `"time_based"`
            (`learning_rate / (1 + lr_decay * t)`), or `"exponential"`
            (`learning_rate * exp(-lr_decay * t)`), where `t` is the number
            of mini-batch updates taken so far. Only used in SGD mode.
        lr_decay: Decay strength for a non-`"constant"` schedule. Only used
            in SGD mode.
        random_state: Seeds the per-epoch shuffling in SGD mode.

    Attributes:
        coef_: Fitted coefficients, set by `fit`.
        cost_history_: Cost at the end of each step actually taken
            (full-batch mode) or each epoch (SGD mode), shape (n_iter_,).
        cost_flops_: Cumulative approximate FLOPs (`regression.cost.
            gradient_flops`) spent on gradient evaluations up to and
            including each `cost_history_` entry, shape (n_iter_,) --
            a batch-size-independent x-axis for comparing computational
            cost against `cost_history_`.
        n_iter_: Number of gradient steps actually taken (full-batch mode,
            `<= max_iter`) or number of epochs run (SGD mode, `==
            n_epochs`); always equal to `len(cost_history_)`.
        n_updates_: Total number of `optimizer.step` calls performed. Equal
            to `n_iter_` in full-batch mode; in SGD mode, `n_epochs` times
            the number of mini-batches per epoch (`>= n_iter_`).
    """

    cost_history_: NDArray[np.float64]
    cost_flops_: NDArray[np.float64]
    n_iter_: int
    n_updates_: int

    def __init__(
        self,
        learning_rate: float,
        penalty: Literal["l2", "l1"] = "l2",
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
        # flat verbatim assignment only — sklearn clonability
        self.learning_rate = learning_rate
        self.penalty = penalty
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
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate_schedule = learning_rate_schedule
        self.lr_decay = lr_decay
        self.random_state = random_state

    def _build_schedule(self) -> Callable[[int], float]:
        """Dispatch `learning_rate_schedule`/`lr_decay` to a `t -> learning_rate`
        callable via `optimization.schedules` (see that module for why this
        isn't exposed as a constructor param directly)."""
        if self.learning_rate_schedule == "constant":
            return constant_schedule(self.learning_rate)
        if self.learning_rate_schedule == "time_based":
            return time_based_decay(self.learning_rate, self.lr_decay)
        return exponential_decay(self.learning_rate, self.lr_decay)

    def _fit_full_batch(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        grad_fn: _GradientFn,
        cost_fn: _CostFn,
        opt: Optimizer,
    ) -> _FitResult:
        """One `optimizer.step` per iteration on the full training set, up to
        `max_iter` iterations, stopping early once `max(abs(grad)) < tol`.
        With zero features (e.g. a degree-0 polynomial, no slope terms left
        once the intercept is centered out) there is nothing to iterate on:
        the only step is the trivial empty coefficient vector."""
        n_samples, n_features = X.shape
        opt.reset(n_features)
        theta = np.zeros(n_features, dtype=np.float64)

        cost_history = []
        cost_flops = []
        flops = 0
        for _ in range(self.max_iter):
            grad = grad_fn(X, y, theta, self.lam)
            theta = opt.step(theta, grad)
            flops += gradient_flops(n_samples, n_features)
            cost_history.append(cost_fn(X, y, theta, self.lam))
            cost_flops.append(flops)
            if grad.size == 0 or np.max(np.abs(grad)) < self.tol:
                break

        return (
            theta,
            np.array(cost_history, dtype=np.float64),
            np.array(cost_flops, dtype=np.float64),
            len(cost_history),
        )

    def _fit_minibatch(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        grad_fn: _GradientFn,
        cost_fn: _CostFn,
        opt: Optimizer,
        batch_size: int,
    ) -> _FitResult:
        """One `optimizer.step` per mini-batch, looping over `n_epochs` epochs;
        the training set is freshly shuffled each epoch. `optimizer.reset` is
        called once, before the first epoch, so momentum/Adam-style state
        persists across the whole run rather than being cleared per epoch."""
        n_samples, n_features = X.shape
        opt.reset(n_features)
        theta = np.zeros(n_features, dtype=np.float64)
        schedule = self._build_schedule()
        rng = np.random.default_rng(self.random_state)

        cost_history = []
        cost_flops = []
        flops = 0
        t = 0
        for _ in range(self.n_epochs):
            indices = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start : start + batch_size]
                X_batch, y_batch = X[batch_idx], y[batch_idx]

                opt.learning_rate = schedule(t)
                grad = grad_fn(X_batch, y_batch, theta, self.lam)
                theta = opt.step(theta, grad)
                flops += gradient_flops(batch_idx.shape[0], n_features)
                t += 1

            cost_history.append(cost_fn(X, y, theta, self.lam))
            cost_flops.append(flops)

        return (
            theta,
            np.array(cost_history, dtype=np.float64),
            np.array(cost_flops, dtype=np.float64),
            t,
        )

    def _fit_centered(self, X: NDArray[np.float64], y_centered: NDArray[np.float64]) -> None:
        """Fit theta by gradient descent or, if `batch_size` is set, SGD,
        against the already mean-centered target.

        Args:
            X: Design matrix, shape (n_samples, n_features).
            y_centered: Mean-centered targets, shape (n_samples,).

        Sets:
            coef_, cost_history_, cost_flops_, n_iter_, and n_updates_.

        Raises:
            ValueError: If `lam`/`lr_decay` is negative, `max_iter` (in
                full-batch mode) or `batch_size`/`n_epochs` (in SGD mode) is
                not strictly positive, or `penalty`/`gradient_method`/
                `optimizer`/`learning_rate_schedule` is not one of the
                supported names.
        """
        y = y_centered
        if self.lam < 0:
            raise ValueError(f"lam must be non-negative, got {self.lam}.")
        if self.lr_decay < 0:
            raise ValueError(f"lr_decay must be non-negative, got {self.lr_decay}.")
        if self.penalty not in _PENALTY_GRADIENT_FNS:
            raise ValueError(
                f"penalty must be one of {sorted(_PENALTY_GRADIENT_FNS)}, got {self.penalty!r}."
            )
        if self.gradient_method not in _GRADIENT_FNS:
            raise ValueError(
                f"gradient_method must be one of {sorted(_GRADIENT_FNS)}, "
                f"got {self.gradient_method!r}."
            )
        if self.optimizer not in OPTIMIZER_REGISTRY:
            raise ValueError(
                f"optimizer must be one of {sorted(OPTIMIZER_REGISTRY)}, got {self.optimizer!r}."
            )
        if self.learning_rate_schedule not in _SCHEDULES:
            raise ValueError(
                f"learning_rate_schedule must be one of {_SCHEDULES}, "
                f"got {self.learning_rate_schedule!r}."
            )
        if self.batch_size is None:
            if self.max_iter < 1:
                raise ValueError(f"max_iter must be >= 1, got {self.max_iter}.")
        else:
            if self.batch_size < 1:
                raise ValueError(f"batch_size must be >= 1, got {self.batch_size}.")
            if self.n_epochs < 1:
                raise ValueError(f"n_epochs must be >= 1, got {self.n_epochs}.")

        grad_fn = _PENALTY_GRADIENT_FNS[self.penalty][self.gradient_method]
        cost_fn = _COST_FNS[self.penalty]

        kwargs: dict[str, Any] = {}
        for attr in _OPTIMIZER_KWARGS.get(self.optimizer, ()):
            kwargs[_OPTIMIZER_KWARG_ALIASES.get(attr, attr)] = getattr(self, attr)
        opt = OPTIMIZER_REGISTRY[self.optimizer](learning_rate=self.learning_rate, **kwargs)

        if self.batch_size is None:
            theta, cost_history, cost_flops, n_updates = self._fit_full_batch(
                X, y, grad_fn, cost_fn, opt
            )
        else:
            theta, cost_history, cost_flops, n_updates = self._fit_minibatch(
                X, y, grad_fn, cost_fn, opt, self.batch_size
            )

        self.coef_ = theta
        self.cost_history_ = cost_history
        self.cost_flops_ = cost_flops
        self.n_iter_ = cost_history.shape[0]
        self.n_updates_ = n_updates


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
