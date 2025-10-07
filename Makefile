.PHONY: help install install-dev install-all test lint format clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install base dependencies
	uv venv
	uv pip install -e .

install-dev:  ## Install with dev dependencies
	uv venv
	uv pip install -e ".[dev]"
	pre-commit install

install-all:  ## Install all dependencies (dev, graph, viz)
	uv venv
	uv pip install -e ".[dev,graph,viz]"
	pre-commit install

test:  ## Run tests with coverage
	pytest --cov=src --cov-report=term-missing --cov-report=html

lint:  ## Check code quality
	ruff check .

format:  ## Format code with ruff
	ruff format .
	ruff check . --fix

clean:  ## Clean cache and build files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ dist/ build/

jupyter:  ## Start JupyterLab
	jupyter lab

sync:  ## Sync dependencies after git pull
	uv pip install -e ".[dev]"
