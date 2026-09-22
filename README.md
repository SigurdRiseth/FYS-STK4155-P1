# Project 1 -- FYS-STK4155

Regression on the Runge function using OLS, Ridge, and LASSO with bivariate
polynomial design matrices, fit via both closed-form solutions and gradient-based
optimization (GD, SGD, momentum, AdaGrad, RMSProp, Adam), and evaluated with
bootstrap and cross-validation resampling. Coursework for FYS-STK4155 at UiO.

> **Status:** work in progress — report and results sections below are not
> written up yet.

## Project structure

```
.
├── src/fys_stk4155_p1/   # importable package — all reusable logic lives here
│   ├── data/             # Runge sampling + polynomial design matrices
│   ├── regression/       # OLS, Ridge, LASSO (closed-form + gradient descent)
│   ├── optimization/     # optimizers (GD, SGD, Momentum, AdaGrad, RMSProp, Adam)
│   └── resampling/       # bootstrap, cross-validation
├── tests/                # pytest test suite, mirrors src/ structure
├── scripts/              # thin runnable entry points (data download, experiments)
├── notebooks/            # exploratory Jupyter notebooks (outputs stripped on commit)
├── data/                 # raw/processed data — gitignored, see data/README.md
├── docs/                 # report, write-up, figures
└── .github/workflows/    # CI: pre-commit hooks + tests on every push/PR
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (pinned in
`.python-version`).

```bash
uv sync                  # installs dependencies + dev tools into .venv
uv run pre-commit install  # enable git hooks (lint, format, nbstripout, mypy, gitleaks)
```

## Development

Run tests with `make test` / `uv run pytest`.

## Reproducibility

- Dependencies are pinned via `uv.lock`; always use `uv run`/`uv sync`
  rather than an ad hoc environment.
- Set and log random seeds explicitly wherever randomness is involved
  (train/test splits, weight init, bootstrapping) — don't rely on global
  state.
- Notebooks are for exploration only; code meant to produce a reported
  result belongs in `src/` (tested) and is invoked from a script, so the
  result can be regenerated deterministically from a single command.
- Raw data is never committed — `data/README.md` documents how to (re)obtain
  it.

## License

[GPL-3.0-or-later](LICENSE)
