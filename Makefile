.PHONY: install lint format typecheck test check clean figures experiments report clean-report

install:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test

# --- Experiments and figures --------------------------------------------------
# Every setting lives in scripts/utils/config.py. Experiment results (JSON in
# data/results/) are rebuilt only when the code or the config changes; e2 is
# the slow one (about 1 CPU-hour, parallelized over data seeds).
PY := uv run python
RES := data/results
EXP_DEPS := scripts/run_experiments.py scripts/utils/config.py scripts/utils/common.py \
	$(shell find src -name '*.py')
RESULTS := $(RES)/e1_ols_degree.json $(RES)/e2_model_selection.json $(RES)/e3_lasso.json \
	$(RES)/e4_optimizers.json $(RES)/e5_autodiff.json

$(RES)/e1_ols_degree.json: $(EXP_DEPS)
	$(PY) scripts/run_experiments.py e1
$(RES)/e2_model_selection.json: $(EXP_DEPS)
	$(PY) scripts/run_experiments.py e2
$(RES)/e3_lasso.json: $(EXP_DEPS) $(RES)/e2_model_selection.json
	$(PY) scripts/run_experiments.py e3
$(RES)/e4_optimizers.json: $(EXP_DEPS)
	$(PY) scripts/run_experiments.py e4
$(RES)/e5_autodiff.json: $(EXP_DEPS)
	$(PY) scripts/run_experiments.py e5

experiments: $(RESULTS)

figures: $(RESULTS)
	$(PY) scripts/generate_data.py --pdf
	$(PY) scripts/generate_ols_figures.py
	$(PY) scripts/generate_ridge_figures.py
	$(PY) scripts/generate_bias_variance_figures.py
	$(PY) scripts/generate_cross_validation_figures.py
	$(PY) scripts/generate_gradient_descent_figures.py
	$(PY) scripts/generate_sgd_figures.py
	$(PY) scripts/generate_lasso_figures.py

report: figures
	cd docs && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean-report:
	cd docs && latexmk -c

clean: clean-report
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
