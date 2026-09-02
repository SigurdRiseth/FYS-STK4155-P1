# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FYS-STK4155 Project 1: regression on the Franke function using OLS, Ridge,
and LASSO with polynomial design matrices. Built from a template repo for
academic ML coursework — reproducibility (seeded RNG, locked deps, no
committed data) is a first-class requirement, not incidental.

## Commands

Package manager is `uv`; always run Python through `uv run` rather than an
ad hoc environment.

```bash
uv sync                      # install deps + dev tools into .venv
uv run pre-commit install    # enable git hooks (lint, format, nbstripout, mypy, gitleaks)

make test        # uv run pytest (runs with coverage, see pyproject.toml)
make lint         # uv run ruff check .
make format       # uv run ruff format .
make typecheck    # uv run mypy
make check        # lint + typecheck + test
make clean        # remove caches (__pycache__, .pytest_cache, .mypy_cache, .ruff_cache, htmlcov)

uv run pytest tests/test_franke.py::test_polynomial_design_matrix_shape  # single test
uv run pre-commit run --all-files                                       # all hooks manually
uv run python scripts/generate_data.py [--n 1000] [--noise 0.1] [--seed 42]  # regenerate dataset
```

CI (`.github/workflows/ci.yml`) runs `uv sync --locked` (fails if `uv.lock`
is stale relative to `pyproject.toml`), all pre-commit hooks, then the test
suite, on every push to `main` and every PR.

## Architecture

- `src/fys_stk4155_p1/` — the only place reusable/tested logic lives.
  Currently: `franke.py` (the Franke test function and noisy-sample
  generation) and `design_matrix.py` (bivariate polynomial design matrix
  shared by OLS/Ridge/LASSO — monomials `x^i * y^j` for `i + j <= degree`,
  ordered by increasing total degree, column 0 is the intercept).
- `scripts/` — thin CLI entry points that wire together `src/` logic
  (argparse + I/O only). `generate_data.py` samples the Franke dataset and
  writes it to `data/raw/franke.npz`; regression/experiment scripts should
  follow the same pattern: import from the package, keep the script itself
  free of algorithmic logic.
- `tests/` mirrors `src/` structure and asserts determinism (same seed →
  identical output) alongside correctness of shapes/values — follow this
  pattern for new modules (e.g. upcoming OLS/Ridge/LASSO regressors).
- `data/` is gitignored except `README.md`; never commit raw or generated
  data — regenerate via `scripts/`, or `git add -f` with documented
  provenance if a dataset truly must be versioned.
- `notebooks/` is exploration-only (outputs stripped by nbstripout on
  commit); any code producing a reported result belongs in `src/`, invoked
  from a script, so results are regenerated deterministically from one
  command.
- `docs/` holds the report/write-up and generated figures.

## Conventions

- Every function touching randomness takes an explicit `seed` (see
  `generate_franke_data`) — never rely on global RNG state.
- Type hints use `numpy.typing.NDArray[np.float64]`; mypy runs over
  `src` and `tests` with `warn_redundant_casts`/`warn_unused_ignores` on.
- Ruff rule set: pycodestyle (E/W), pyflakes (F), isort (I), pyupgrade (UP),
  bugbear (B), numpy-specific (NPY), pandas-vet (PD); line length 100.
- Docstrings are Google-style with `Args:`/`Returns:`.
