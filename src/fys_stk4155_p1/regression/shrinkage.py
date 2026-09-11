"""Ridge-specific analysis helpers: coefficient paths and SVD-mode shrinkage."""

import numpy as np
from numpy.typing import NDArray

from fys_stk4155_p1.regression.ridge import Ridge


def ridge_coefficient_path(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    lambdas: NDArray[np.float64],
    fit_intercept_column: bool = False,
) -> NDArray[np.float64]:
    """Fit Ridge once per lambda and stack the resulting coefficients.

    Args:
        X: Design matrix, shape (n_samples, n_features).
        y: Targets, shape (n_samples,).
        lambdas: Penalty strengths to sweep, shape (n_lambdas,).
        fit_intercept_column: Passed through to `Ridge` (see its docstring).

    Returns:
        Coefficient path, shape (n_lambdas, n_features); row k is the fit at
        lambdas[k].
    """
    return np.array(
        [
            Ridge(lam=lam, fit_intercept_column=fit_intercept_column).fit(X, y).coef_
            for lam in lambdas
        ]
    )


def singular_value_shrinkage(
    X: NDArray[np.float64], lambdas: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Ridge's per-mode shrinkage factor f_i(lambda) = s_i^2 / (s_i^2 + n*lambda).

    Writing X = U S V^T, the OLS solution is theta = sum_i (1/s_i) (u_i^T y) v_i
    and the Ridge solution is theta = sum_i f_i(lambda) * (1/s_i) (u_i^T y) v_i:
    Ridge is OLS with every SVD mode of X shrunk by f_i(lambda) (Section 3.8 of
    the lecture notes). The n*lambda scaling matches `Ridge`'s normal equations,
    which solve the per-observation cost (1/n)||y - X theta||^2 + lambda ||theta||^2
    (Section 3.10). Singular values are those of X itself, largest first, as
    returned by `np.linalg.svd`.

    Args:
        X: Matrix whose SVD modes are shrunk (typically the non-intercept,
            standardized columns of a design matrix), shape (n_samples, n_features).
        lambdas: Penalty strengths, shape (n_lambdas,).

    Returns:
        Tuple (singular_values, shrinkage): singular_values has shape
        (n_features,); shrinkage has shape (n_lambdas, n_features), where row k
        is f_i(lambdas[k]) for every mode i.
    """
    n_samples = X.shape[0]
    singular_values = np.linalg.svd(X, compute_uv=False)
    lambdas_arr = np.asarray(lambdas, dtype=np.float64)
    shrinkage = singular_values**2 / (singular_values**2 + n_samples * lambdas_arr[:, None])
    return singular_values, shrinkage
