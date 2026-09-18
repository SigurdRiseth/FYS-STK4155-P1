"""JAX-autodiff gradient for regression.cost.cost, cross-checked against analytical_gradient."""

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

# Must run before any jax array is created or traced (i.e. before the first
# call to autodiff_gradient) — safe here since jax.jit below only wraps
# _cost_jax, it doesn't trace it yet.
jax.config.update("jax_enable_x64", True)


def _cost_jax(
    X: jnp.ndarray,
    y: jnp.ndarray,
    theta: jnp.ndarray,
    lam: float,
    fit_intercept_column: bool,
) -> jnp.ndarray:
    """Same formula as regression.cost.cost, written in jax.numpy so jax.grad can
    differentiate it. fit_intercept_column must be passed as a static (non-traced)
    argument, since it gates a Python-level branch rather than an array operation.
    """
    penalty_theta = theta[1:] if fit_intercept_column else theta
    return jnp.sum((X @ theta - y) ** 2) / len(y) + lam * jnp.sum(penalty_theta**2)


# Built once at module scope, not inside autodiff_gradient: GradientDescent.fit
# calls the gradient once per iteration (up to max_iter times), so pre-jitting
# here avoids re-tracing the computation graph on every step.
_grad_jax = jax.jit(
    jax.grad(_cost_jax, argnums=2),
    static_argnames=("fit_intercept_column",),
)


def autodiff_gradient(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    theta: NDArray[np.float64],
    lam: float = 0.0,
    fit_intercept_column: bool = False,
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
        fit_intercept_column: Whether the first column of X/theta is an
            intercept and should therefore be excluded from regularization.
            Defaults to False.

    Returns:
        The gradient of the cost with respect to theta, shape (n_features,).
    """
    grad_jax = _grad_jax(
        jnp.asarray(X), jnp.asarray(y), jnp.asarray(theta), lam, fit_intercept_column
    )
    return np.asarray(grad_jax, dtype=np.float64)
