# Contributing to BuffetBot

Thank you for your interest in contributing to BuffetBot! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## How to Contribute

### Reporting Issues

1. Check if the issue already exists in the [issue tracker](https://github.com/your-username/buffetbot/issues)
2. If not, create a new issue with:
   - Clear, descriptive title
   - Detailed description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (Python version, OS, etc.)

### Suggesting Features

1. Check existing issues and discussions for similar suggestions
2. Open a new issue with the "enhancement" label
3. Describe the feature and its use case
4. Explain why this would be valuable to the project

### Contributing Code

#### Setup Development Environment

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/buffetbot.git
   cd buffetbot
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   make install-dev
   # or manually:
   pip install -e ".[dev]"
   pre-commit install
   ```

#### Development Workflow

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the coding standards

3. Run tests and checks:
   ```bash
   make check-all
   # or individually:
   make format
   make lint
   make type-check
   make test
   ```

4. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: add new feature"  # Use conventional commits
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a Pull Request

### Coding Standards

#### Python Style Guide

- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for formatting (88 character line limit)
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Write [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

#### Type Hints

- Add type hints to all function signatures
- Use `from typing import` for type annotations
- Run `mypy` to check types

#### Testing

- Write tests for all new functionality
- Maintain or improve code coverage (minimum 80%)
- Use pytest for testing
- Place tests in the `tests/` directory

#### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Test additions or changes
- `chore:` Maintenance tasks

### Documentation

- Update docstrings for any modified functions/classes
- Update README.md if adding new features
- Add/update examples in the `examples/` directory
- Update glossary if adding new financial metrics

## Pull Request Process

1. Ensure all tests pass and code meets quality standards
2. Update documentation as needed
3. Add entry to CHANGELOG.md
4. Request review from maintainers
5. Address review feedback
6. Squash commits if requested

## Development Commands

```bash
# Run all checks before submitting PR
make commit-ready

# Individual commands
make format      # Format code
make lint        # Run linters
make type-check  # Check types
make test        # Run tests
make coverage    # Generate coverage report

# Run specific examples
make run-example
make run-streamlit
```

## Questions?

Feel free to open an issue for any questions about contributing! 