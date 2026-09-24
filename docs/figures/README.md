# docs/figures/

Figures (PDF) and tables (`tables/*.tex`) here are generated, not committed.
Regenerate all of them, from the settings in `scripts/utils/config.py`, with

```bash
make figures        # runs `make experiments` first if results are missing or stale
```

Subdirectories follow the report: `data/`, `ols/`, `ridge/`, `bias_variance/`,
`cross_validation/` (resampling and model selection), `gradient_descent/`,
`sgd/`, `lasso/`, `tables/`. Style (sizes for the IEEE two-column layout,
fonts, colors) is defined once in `scripts/utils/plotting.py`.
