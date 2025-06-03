.PHONY: help clean install install-dev test coverage lint format type-check pre-commit run-streamlit docs

# Default target
.DEFAULT_GOAL := help

# Python interpreter
PYTHON := python3
PIP := pip3

help:  ## Show this help message
	@echo "BuffetBot Development Commands"
	@echo "=============================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

clean:  ## Clean up build artifacts and cache files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	@echo "✨ Cleaned up build artifacts and cache files"

install:  ## Install project dependencies
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed"

install-dev:  ## Install development dependencies
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"
	pre-commit install
	@echo "✅ Development dependencies installed"

test:  ## Run tests
	$(PYTHON) -m pytest tests/ -v

test-cov:  ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term

coverage:  ## Generate coverage report
	$(PYTHON) -m pytest tests/ --cov=. --cov-report=html
	@echo "📊 Coverage report generated in htmlcov/"
	@echo "📊 Open htmlcov/index.html in your browser to view"

lint:  ## Run linting checks
	@echo "🔍 Running flake8..."
	$(PYTHON) -m flake8 . --count --statistics
	@echo "🔍 Running pylint..."
	$(PYTHON) -m pylint analysis utils buffetbot --fail-under=8.0
	@echo "✅ Linting complete"

format:  ## Format code with black and isort
	@echo "🎨 Running isort..."
	$(PYTHON) -m isort .
	@echo "🎨 Running black..."
	$(PYTHON) -m black .
	@echo "✅ Code formatting complete"

format-check:  ## Check code formatting without changes
	@echo "🔍 Checking isort..."
	$(PYTHON) -m isort . --check-only --diff
	@echo "🔍 Checking black..."
	$(PYTHON) -m black . --check --diff
	@echo "✅ Format check complete"

type-check:  ## Run type checking with mypy
	@echo "🔍 Running mypy..."
	$(PYTHON) -m mypy . --ignore-missing-imports
	@echo "✅ Type checking complete"

security:  ## Run security checks with bandit
	@echo "🔐 Running bandit security scan..."
	$(PYTHON) -m bandit -r . -ll -x tests/
	@echo "✅ Security scan complete"

pre-commit:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

pre-commit-update:  ## Update pre-commit hooks
	pre-commit autoupdate

run-streamlit:  ## Run the Streamlit glossary app
	streamlit run ui/streamlit/glossary_app.py

run-example:  ## Run the integration example
	$(PYTHON) examples/example_integration.py

run-tests:  ## Run tests using the test runner script
	$(PYTHON) scripts/run_tests.py

docs:  ## Build documentation
	cd docs && make html
	@echo "📚 Documentation built in docs/_build/html/"

requirements:  ## Update requirements.txt from pyproject.toml
	pip-compile pyproject.toml -o requirements.txt --resolver=backtracking

check-all: format-check lint type-check security test  ## Run all checks

dev-setup: clean install-dev  ## Complete development environment setup

# Development workflow commands
commit-ready: format lint type-check test  ## Prepare code for commit

release:  ## Create a new release
	@echo "📦 Building distribution packages..."
	$(PYTHON) -m build
	@echo "✅ Release packages built in dist/"

# Docker commands (if needed in future)
docker-build:  ## Build Docker image
	docker build -t buffetbot:latest .

docker-run:  ## Run Docker container
	docker run -it --rm -p 8501:8501 buffetbot:latest
