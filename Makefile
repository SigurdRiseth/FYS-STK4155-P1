.PHONY: install lint format typecheck test check clean

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

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
