# Configuration Files Overview

This document describes all the configuration files added to the BuffetBot project and their purposes.

## Version Control & Ignore Files

### `.gitignore`
- Comprehensive Python gitignore file
- Excludes common Python artifacts: `__pycache__`, `*.pyc`, `.venv`, etc.
- Includes IDE-specific ignores for PyCharm, VS Code
- OS-specific ignores for macOS, Windows, Linux
- Project-specific ignores for logs, cache, and data files

## Code Style & Formatting

### `.editorconfig`
- Ensures consistent coding styles between different editors and IDEs
- Defines indentation, line endings, and charset settings
- Language-specific rules for Python, YAML, JSON, JavaScript, CSS, HTML

### `pyproject.toml`
- Modern Python project configuration (PEP 518)
- Configures:
  - Project metadata (name, version, dependencies)
  - Black formatter settings
  - isort import sorting settings
  - mypy type checking settings
  - pytest configuration
  - Coverage settings

### `setup.cfg`
- Configuration for tools that don't yet support `pyproject.toml`
- Includes settings for:
  - flake8 linting
  - pydocstyle documentation linting
  - bandit security scanning
  - pylint code analysis

## Development Tools

### `.pre-commit-config.yaml`
- Automated code quality checks before commits
- Runs multiple tools:
  - Code formatting (Black, isort)
  - Linting (flake8, mypy)
  - Security checks (bandit)
  - File checks (trailing whitespace, merge conflicts)
- Auto-fixes issues when possible

### `Makefile`
- Convenient commands for development tasks
- Key commands:
  - `make help` - Show all available commands
  - `make install-dev` - Set up development environment
  - `make test` - Run tests
  - `make format` - Format code
  - `make lint` - Run linters
  - `make commit-ready` - Run all checks before committing

### `.python-version`
- Specifies Python version (3.10.13) for pyenv users
- Ensures consistent Python version across development environments

## Project Configuration

### `setup.py`
- Minimal setup.py for backward compatibility
- Actual configuration is in `pyproject.toml`

### `MANIFEST.in`
- Specifies which files to include in Python distributions
- Includes documentation, examples, tests, and UI files
- Excludes unnecessary files like caches and virtual environments

## Documentation

### `CONTRIBUTING.md`
- Guidelines for contributing to the project
- Includes:
  - Development setup instructions
  - Coding standards
  - Pull request process
  - Commit message conventions

### `CHANGELOG.md`
- Tracks all notable changes to the project
- Follows [Keep a Changelog](https://keepachangelog.com/) format
- Uses [Semantic Versioning](https://semver.org/)

## Environment Variables

### `.env` (create from `.env.example`)
- Store sensitive configuration like API keys
- Not tracked in version control (listed in `.gitignore`)
- Example template would include:
  ```
  API_KEY=your_api_key_here
  DATABASE_URL=sqlite:///buffetbot.db
  LOG_LEVEL=INFO
  ```

## Usage Tips

1. **Initial Setup**:
   ```bash
   make install-dev  # Install all development dependencies
   ```

2. **Before Committing**:
   ```bash
   make commit-ready  # Run all checks
   ```

3. **Code Formatting**:
   ```bash
   make format  # Auto-format code
   ```

4. **Running Tests**:
   ```bash
   make test  # Run test suite
   make coverage  # Generate coverage report
   ```

These configuration files follow Python community best practices and make the project more maintainable, consistent, and professional. 