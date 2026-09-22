# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FYS-STK4155 Project 1: regression on the Runge function using OLS, Ridge,
and LASSO with polynomial design matrices. Built from a template repo for
academic ML coursework — reproducibility (seeded RNG, locked deps, no
committed data) is a first-class requirement, not incidental.

## Commands

Package manager is `uv`; always run Python through `uv run` rather than an
ad hoc environment.

```bash
uv sync                      # install deps + dev tools into .venv
uv run pre-commit install    # enable git hooks (lint, format, nbstripout, mypy, gitleaks)

make test         # uv run pytest (runs with coverage, see pyproject.toml)
make lint         # uv run ruff check .
make format       # uv run ruff format .
make typecheck    # uv run mypy
make check        # lint + typecheck + test
make clean        # remove caches (__pycache__, .pytest_cache, .mypy_cache, .ruff_cache, htmlcov)

uv run pytest tests/data/test_runge.py::test_runge_function_at_zero_is_one  # single test
uv run pre-commit run --all-files                                       # all hooks manually
uv run python scripts/generate_data.py [--n 1000] [--noise 0.1] [--seed 42]  # regenerate dataset
```

CI (`.github/workflows/ci.yml`) runs `uv sync --locked` (fails if `uv.lock`
is stale relative to `pyproject.toml`), all pre-commit hooks, then the test
suite, on every push to `main` and every PR.

## Architecture

- `src/fys_stk4155_p1/` — the only place reusable/tested logic lives, split
  by concern: `data/` (Runge sampling, polynomial design matrix), `regression/`
  (shared `LinearModel` base, OLS, Ridge, Lasso (via `GradientDescent`,
  `penalty="l1"` — no closed form), degree sweeps),
  `resampling/` (bootstrap; cross-validation in progress), and `metrics.py`.
- `scripts/` — thin CLI entry points that wire together `src/` logic
  (argparse + I/O only), e.g. `generate_data.py` (writes
  `data/raw/runge.npz`) and the `generate_*_figures.py` scripts (write to
  `docs/figures/`); keep new scripts free of algorithmic logic.
- `tests/` mirrors `src/` structure and asserts determinism (same seed →
  identical output) alongside correctness of shapes/values — follow this
  pattern for new modules.
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
  `generate_runge_data`) — never rely on global RNG state.
- Type hints use `numpy.typing.NDArray[np.float64]`; mypy runs over
  `src` and `tests` with `warn_redundant_casts`/`warn_unused_ignores` on.
- Ruff rule set: pycodestyle (E/W), pyflakes (F), isort (I), pyupgrade (UP),
  bugbear (B), numpy-specific (NPY), pandas-vet (PD); line length 100.
- Docstrings are Google-style with `Args:`/`Returns:`.
- Always use double precision for jax (`jax.config.update("jax_enable_x64", True)`).
