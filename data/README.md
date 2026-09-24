# data/

```
data/
├── raw/       # generated data set(s) — gitignored, never edited by hand
└── results/   # experiment outputs (JSON) — versioned, see below
```

## raw/

Not committed (see `.gitignore`). Regenerate the reference data set with:

```bash
uv run python scripts/generate_data.py [--n 100] [--noise 0.1] [--seed 42]
```

which writes `raw/runge.npz` (samples of the Runge function,
$f(x) = 1/(1+25x^2)$, plus additive Gaussian noise). Defaults come from
`scripts/utils/config.py`; pass no arguments to reproduce exactly what the
report uses.

## results/

The one exception to "never commit data": `results/*.json` are the numerical
outputs behind every number and figure in the report, versioned so they're
visible without rerunning anything (a course reproducibility requirement).
Regenerate them with `make experiments` (`scripts/run_experiments.py`);
`results/SUMMARY.md` documents what each file contains and the shared
experimental setup (n, noise, split, seeds).
