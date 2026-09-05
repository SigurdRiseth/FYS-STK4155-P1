import numpy as np
from numpy.typing import NDArray


def ols(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    rcond: float | None = None,
) -> NDArray[np.float64]:
    """Ordinary least squares via singular value decomposition.

    Solves min_theta ||X @ theta - y||_2^2 using the SVD-based pseudoinverse,
    which is stable for rank-deficient or ill-conditioned X. Does not add an
    intercept column; prepend one to X if you need a bias term.

    Args:
        X: Design matrix, shape (n_samples, n_features).
        y: Targets, shape (n_samples,).
        rcond: Singular values below rcond * max(s) are treated as zero.
            None uses a machine-precision default scaled by max(X.shape).

    Returns:
        beta: Coefficients, shape (n_features,).

    LLM-assisted
    ------------
    Tool: Claude Sonnet 4.8 (September 2026)
    Role: Suggested the rcond parameter and singular-value cutoff logic for the SVD-based
        pseudoinverse.
    Modifications: Integrated the suggestion into the existing OLS implementation and verified the
        resulting behavior.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y disagree on n_samples: {X.shape[0]} vs {y.shape[0]}.")

    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    if rcond is None:
        rcond = np.finfo(np.float64).eps * max(X.shape)

    # LLM-assisted: Claude Sonnet 4.8 (September 2026) suggested the
    # singular-value cutoff and pseudoinverse handling below.
    cutoff = rcond * s[0] if s.size else 0.0
    s_inv = np.where(s > cutoff, 1.0 / s, 0.0)

    return Vt.T @ (s_inv * (U.T @ y))
