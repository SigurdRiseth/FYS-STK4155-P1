.PHONY: install lint format typecheck test check clean figures report clean-report

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

figures:
	uv run python scripts/generate_data.py --pdf
	uv run python scripts/generate_ols_figures.py
	uv run python scripts/generate_ridge_figures.py
	uv run python scripts/generate_lasso_figures.py
	uv run python scripts/generate_bias_variance_figures.py
	uv run python scripts/generate_cross_validation_figures.py
	uv run python scripts/generate_gradient_descent_figures.py
	uv run python scripts/generate_sgd_figures.py

report: figures
	cd docs && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean-report:
	cd docs && latexmk -c

clean: clean-report
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
