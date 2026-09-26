"""JAX-autodiff gradients for regression.cost.cost (L2) and regression.cost.lasso_cost
(L1), cross-checked against analytical_gradient/lasso_subgradient."""

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

# Must run before any jax array is created or traced (i.e. before the first
# call to autodiff_gradient) — safe here since jax.jit below only wraps
# _cost_jax, it doesn't trace it yet.
jax.config.update("jax_enable_x64", True)


def _cost_jax(X: jnp.ndarray, y: jnp.ndarray, theta: jnp.ndarray, lam: float) -> jnp.ndarray:
    """Same formula as regression.cost.cost, written in jax.numpy so jax.grad can
    differentiate it."""
    return jnp.sum((X @ theta - y) ** 2) / len(y) + lam * jnp.sum(theta**2)


# Built once at module scope, not inside autodiff_gradient: GradientDescent.fit
# calls the gradient once per iteration (up to max_iter times), so pre-jitting
# here avoids re-tracing the computation graph on every step.
_grad_jax = jax.jit(jax.grad(_cost_jax, argnums=2))


def autodiff_gradient(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    theta: NDArray[np.float64],
    lam: float = 0.0,
) -> NDArray[np.float64]:
    """Gradient of regression.cost.cost with respect to theta, via jax.grad instead of
    the closed-form formula in regression.cost.analytical_gradient. Returns a plain
    NumPy float64 array so callers never need to know JAX was involved.

    This matches Eq. (4.17) from Hjorth-Jensen (2026), computed by automatic
    differentiation rather than analytically — used to verify analytical_gradient
    agrees to machine precision (see tests/regression/test_autodiff.py).

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        y: True target values, shape (n_samples,).
        theta: Model parameters, shape (n_features,).
        lam: L2 regularization strength. Defaults to 0.0.

    Returns:
        The gradient of the cost with respect to theta, shape (n_features,).
    """
    grad_jax = _grad_jax(jnp.asarray(X), jnp.asarray(y), jnp.asarray(theta), lam)
    return np.asarray(grad_jax, dtype=np.float64)


def _lasso_cost_jax(X: jnp.ndarray, y: jnp.ndarray, theta: jnp.ndarray, lam: float) -> jnp.ndarray:
    """Same formula as regression.cost.lasso_cost, written in jax.numpy so jax.grad
    can differentiate it."""
    return jnp.sum((X @ theta - y) ** 2) / len(y) + lam * jnp.sum(jnp.abs(theta))


# Built once at module scope for the same reason as _grad_jax above.
_lasso_grad_jax = jax.jit(jax.grad(_lasso_cost_jax, argnums=2))


def lasso_autodiff_gradient(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    theta: NDArray[np.float64],
    lam: float = 0.0,
) -> NDArray[np.float64]:
    """Gradient of regression.cost.lasso_cost with respect to theta, via jax.grad
    instead of the closed-form subgradient in regression.cost.lasso_subgradient.
    Returns a plain NumPy float64 array so callers never need to know JAX was
    involved.

    ``|theta_j|`` is not differentiable at ``theta_j = 0``. JAX's autodiff resolves
    this by returning exactly ``+1.0`` at ``theta_j = 0`` (verified empirically:
    ``jax.grad(jnp.abs)(0.0) == 1.0``) — a valid subgradient (the subdifferential of
    ``|x|`` at 0 is [-1, 1], and 1 is a member) but an arbitrary choice, different
    from `lasso_subgradient`'s ``np.sign(0) == 0``. The two therefore agree
    everywhere except exactly at theta_j = 0 (see
    `tests/regression/test_autodiff.py`).

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        y: True target values, shape (n_samples,).
        theta: Model parameters, shape (n_features,).
        lam: L1 regularization strength. Defaults to 0.0.

    Returns:
        The gradient of the L1-penalized cost with respect to theta, shape
        (n_features,).
    """
    grad_jax = _lasso_grad_jax(jnp.asarray(X), jnp.asarray(y), jnp.asarray(theta), lam)
    return np.asarray(grad_jax, dtype=np.float64)
