# scripts/

Standalone, runnable entry points that orchestrate the package in
`src/fys_stk4155_p1/`. Scripts stay thin: reusable logic lives in `src/`
(importable and tested); scripts only parse arguments, wire that logic
together, and write output files.

| script | writes | notes |
|---|---|---|
| `generate_data.py` | `data/raw/runge.npz` (+ a PDF preview with `--pdf`) | the reference data set used everywhere else |
| `run_experiments.py` | `data/results/e{1..5}_*.json` | numerical experiments behind the report's numbers; `{e1,...,e5}` subcommand, see its docstring |
| `generate_ols_figures.py`, `generate_ridge_figures.py`, `generate_lasso_figures.py`, `generate_bias_variance_figures.py`, `generate_cross_validation_figures.py`, `generate_gradient_descent_figures.py`, `generate_sgd_figures.py` | PDFs/tables under `docs/figures/` | one script per report section; read the JSON in `data/results/`, not raw data, wherever an experiment already produced the numbers |
| `utils/config.py` | — | single source of truth for every setting (sample size, noise, degrees, CV folds, penalty grids, seeds) used by the scripts above |
| `utils/common.py`, `utils/plotting.py` | — | shared helpers (data/design-matrix loading, figure style) to keep the scripts themselves thin |

Run any script with `uv run python scripts/<name>.py [args]`, or use the
`make experiments` / `make figures` / `make report` targets (see the root
`README.md`) to regenerate everything in dependency order.
