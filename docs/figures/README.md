# docs/figures/

Figures here are **generated, not committed** — this README is the only
tracked file in this directory (see `.gitignore`). Regenerate them by
running the report-figure scripts in `scripts/`; each one calls
`save_figure()` (`scripts/utils/plotting.py`) and writes PDFs into a
subdirectory named after the experiment.

## Reproducing

```bash
uv sync
uv run python scripts/generate_data.py --pdf    # -> docs/figures/data/
uv run python scripts/generate_ridge_figures.py # -> docs/figures/ridge/
```

Pass `--help` to any script for the options (seed, noise, polynomial
degree, ...) used to produce its figures — every experiment script takes
an explicit `--seed`, so output is reproducible bit-for-bit.
