import numpy as np

from fys_stk4155_p1.data.design_matrix import (
    bivariate_polynomial_design_matrix,
    univariate_polynomial_design_matrix,
)


def test_univariate_design_matrix_shape() -> None:
    x = np.linspace(-1, 1, 7)
    X = univariate_polynomial_design_matrix(x, degree=3)
    assert X.shape == (7, 4)


def test_univariate_design_matrix_columns_are_powers_of_x() -> None:
    x = np.array([2.0, 3.0])
    X = univariate_polynomial_design_matrix(x, degree=2)
    expected = np.array([[1.0, 2.0, 4.0], [1.0, 3.0, 9.0]])
    np.testing.assert_allclose(X, expected)


def test_bivariate_design_matrix_shape() -> None:
    x = np.linspace(-1, 1, 5)
    y = np.linspace(-1, 1, 5)
    degree = 3
    X = bivariate_polynomial_design_matrix(x, y, degree)
    n_features = (degree + 1) * (degree + 2) // 2
    assert X.shape == (5, n_features)


def test_bivariate_design_matrix_intercept_column_is_ones() -> None:
    x = np.array([1.0, 2.0, -3.0])
    y = np.array([4.0, -5.0, 6.0])
    X = bivariate_polynomial_design_matrix(x, y, degree=2)
    np.testing.assert_array_equal(X[:, 0], np.ones(3))


def test_bivariate_design_matrix_matches_explicit_monomials() -> None:
    x = np.array([2.0, 3.0])
    y = np.array([5.0, 7.0])
    X = bivariate_polynomial_design_matrix(x, y, degree=2)
    # ordered by increasing total degree: [1, x, y, x^2, xy, y^2]
    expected = np.column_stack([np.ones(2), x, y, x**2, x * y, y**2])
    np.testing.assert_allclose(X, expected)


def test_univariate_design_matrix_no_intercept() -> None:
    x = np.array([2.0, 3.0])
    X = univariate_polynomial_design_matrix(x, degree=2, intercept=False)
    expected = np.array([[2.0, 4.0], [3.0, 9.0]])
    np.testing.assert_allclose(X, expected)
