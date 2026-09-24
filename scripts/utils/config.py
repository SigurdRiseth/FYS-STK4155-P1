"""Single source of truth for every experiment and report figure.

All `scripts/generate_*.py` and `scripts/run_experiments.py` take their
defaults from here, so every number and figure in the report comes from the
same data-generating process, split and grids. Change a value here and
`make figures` regenerates everything consistently.

LLM-assisted: Claude (claude-opus-5-5, Claude Cowork desktop app, September 2026)
wrote/rewrote this file (level 4) to use the shared settings in utils/config.py.
TODO(author): describe your review/changes.
"""

import numpy as np

# Data: Runge's function on [-1, 1], x ~ U(-1, 1), additive N(0, NOISE^2) noise.
N = 100
NOISE = 0.1
SEED = 42  # reference data set: every single-data-set figure uses this seed
TEST_SIZE = 0.2  # 80/20 train-test split, seeded with SEED

# Model complexity and resampling.
DEGREE_MAX = 20
DEGREES = list(range(0, DEGREE_MAX + 1))
N_BOOTSTRAPS = 100
K_VALUES = (5, 10)  # folds compared in part d)
K_SELECT = 5  # folds used for model selection (part i)

# Penalty grids (lambda in our convention: MSE + lam * ||theta||, see cost.py).
RIDGE_LAMBDAS = np.logspace(-12, 1, 27)
LASSO_LAMBDAS = np.logspace(-8, 0, 17)

# Repetition over data sets for the model-selection statistics (seeds 0..N_SEEDS-1 + SEED).
N_SEEDS = 30

# Sensitivity sweeps (part a/c).
N_VALUES = (50, 100, 200, 400)
NOISE_VALUES = (0.05, 0.1, 0.2, 0.4)

# Fixed "over-parameterized" degree used to illustrate Ridge/Lasso paths.
SHOWCASE_DEGREE = 15

# Gradient descent (parts e-h): problems on which the optimizers are compared.
GD_DEGREE = 5
GD_LAMBDA = 1e-3
GD_BENCH_PROBLEMS = [(d, lam) for d in (5, 10, 15) for lam in (0.0, GD_LAMBDA)]
GD_LEARNING_RATES = np.logspace(-5, 1, 25)
GD_MAX_ITER = 100_000
GD_PARAM_TOL = 1e-3  # ||theta - theta_closed|| / ||theta_closed||
SGD_BATCH_SIZES = (4, 16, 32, 80)
SGD_EPOCHS = 2000

# Own Lasso solver used everywhere: subgradient descent with Adam (regression/lasso.py).
LASSO_GD = {"learning_rate": 0.05, "optimizer": "adam", "max_iter": 5000}

# Dense grid on [-1, 1] for the true (known-f) test error.
X_GRID = np.linspace(-1, 1, 2001)


def seeds() -> list[int]:
    """Data seeds for repeated experiments: 0..N_SEEDS-1 plus the reference SEED."""
    return sorted(set(range(N_SEEDS)) | {SEED})
