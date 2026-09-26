"""Data and design-matrix helpers shared by the experiment and figure scripts.

LLM-assisted: Claude (claude-opus-5-5, Claude Cowork desktop app, September 2026)
wrote/rewrote this file (level 4) to use the shared settings in utils/config.py.
TODO(author): describe your review/changes.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fys_stk4155_p1.data.design_matrix import univariate_polynomial_design_matrix
from fys_stk4155_p1.data.runge import generate_runge_data, runge_function

from . import config as C

RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "results"
TABLE_DIR = Path(__file__).resolve().parents[2] / "docs" / "figures" / "tables"


def data(
    seed: int = C.SEED, n: int = C.N, noise: float = C.NOISE
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """(x, y, x_train, x_test, y_train, y_test) for one data set; the split uses
    the same permutation as `bootstrap_bias_variance_sweep` (random_state=seed)."""
    x, y = generate_runge_data(n=n, noise_std=noise, seed=seed)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=C.TEST_SIZE, random_state=seed)
    return x, y, x_tr, x_te, y_tr, y_te


def design(x_fit: NDArray, degree: int, *x_eval: NDArray) -> list[NDArray]:
    """Polynomial design matrices (x^1..x^d, no intercept column -- the model
    fits its intercept by centering y instead, see regression.base.LinearModel),
    standardized with the statistics of `x_fit` (the training rows), as in the
    CV pipelines."""
    X_fit = univariate_polynomial_design_matrix(x_fit, degree)
    Xs = [univariate_polynomial_design_matrix(xe, degree) for xe in x_eval]
    if degree > 0:
        scaler = StandardScaler().fit(X_fit)
        X_fit = scaler.transform(X_fit)
        Xs = [scaler.transform(X) for X in Xs]
    return [X_fit, *Xs]


def oracle_mse(pred_grid: NDArray, noise: float = C.NOISE) -> float:
    """True expected squared error at a new x ~ U(-1, 1): E_x[(f_hat - f)^2] + sigma^2,
    evaluated on `config.X_GRID` (uses the known f)."""
    return float(np.mean((pred_grid - runge_function(C.X_GRID)) ** 2) + noise**2)


def load_result(name: str) -> dict[str, Any]:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"{path} missing: run `make experiments` first.")
    return json.loads(path.read_text())


def write_table(name: str, tex: str) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / f"{name}.tex"
    path.write_text(tex)
    return path
