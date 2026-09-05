"""Polynomial design matrix for 2D inputs, shared by OLS/Ridge/LASSO."""

import numpy as np
from numpy.typing import NDArray


def univariate_polynomial_design_matrix(
    x: NDArray[np.float64],
    degree: int,
) -> NDArray[np.float64]:
    """Build a univariate polynomial design matrix.

    Columns are 1, x, x^2, ..., x^degree. Column 0 is the all-ones
    intercept term.

    Args:
        x: x-coordinates, shape (n,).
        degree: highest degree to include.

    Returns:
        Design matrix of shape (n, degree + 1).
    """
    return np.column_stack([x**i for i in range(degree + 1)])


def bivariate_polynomial_design_matrix(
    x: NDArray[np.float64], y: NDArray[np.float64], degree: int
) -> NDArray[np.float64]:
    """Build a bivariate polynomial design matrix for regression on (x, y).

    Columns are every monomial x^i * y^j with i + j <= degree, ordered by
    increasing total degree: [1, x, y, x^2, xy, y^2, ...]. Column 0 is
    always the all-ones intercept term (i = j = 0).

    Args:
        x: x-coordinates, shape (n,).
        y: y-coordinates, shape (n,).
        degree: highest total degree (i + j) to include.

    Returns:
        Design matrix of shape (n, (degree + 1) * (degree + 2) / 2).
    """
    cols = [
        x**i * y**j for total in range(degree + 1) for i in range(total + 1) for j in [total - i]
    ]
    return np.column_stack(cols)
