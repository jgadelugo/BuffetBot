# BuffetBot Project Structure

This document outlines the organization of the BuffetBot project following Python best practices.

```
BuffetBot/
│
├── buffetbot/                      # Main package directory
│   ├── __init__.py                 # Package initialization
│   └── glossary.py                 # Financial metrics glossary module
│
├── analysis/                       # Analysis modules
│   ├── __init__.py
│   ├── growth_analysis.py          # Growth analysis module
│   ├── value_analysis.py           # Value analysis module
│   ├── health_analysis.py          # Health analysis module
│   └── risk_analysis.py            # Risk analysis module
│
├── utils/                          # Utility modules
│   ├── __init__.py
│   ├── data_fetcher.py            # Data fetching utilities
│   ├── cache_manager.py           # Cache management
│   ├── validators.py              # Data validation
│   ├── formatting.py              # Data formatting
│   └── api_config.py              # API configuration
│
├── ui/                            # User interface implementations
│   ├── streamlit/                 # Streamlit web app
│   │   └── glossary_app.py
│   ├── web/                       # Standalone web interface
│   │   └── glossary.html
│   └── react/                     # React components
│       ├── GlossaryComponent.jsx
│       └── GlossaryComponent.css
│
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── conftest.py               # pytest configuration and fixtures
│   ├── test_glossary.py          # Glossary module tests
│   └── README.md                 # Testing documentation
│
├── examples/                      # Example scripts
│   ├── __init__.py
│   ├── example_integration.py    # Integration example
│   └── README.md                 # Examples documentation
│
├── docs/                         # Documentation
│   ├── glossary/                 # Glossary-specific docs
│   │   ├── README.md            # Glossary documentation
│   │   └── ui_implementations.md # UI implementations guide
│   └── ...                      # Other documentation
│
├── scripts/                      # Utility scripts
│   └── run_tests.py             # Test runner script
│
├── data/                        # Data directory (gitignored)
├── logs/                        # Log files (gitignored)
├── cache/                       # Cache directory (gitignored)
├── dashboard/                   # Dashboard components
└── recommend/                   # Recommendation system

## Configuration Files

### Root Directory Files

- `.gitignore` - Version control ignore patterns
- `.editorconfig` - Editor configuration for consistent coding style
- `.pre-commit-config.yaml` - Pre-commit hooks configuration
- `.python-version` - Python version specification for pyenv
- `pyproject.toml` - Modern Python project configuration
- `setup.cfg` - Additional tool configurations
- `setup.py` - Minimal setup script for compatibility
- `Makefile` - Development automation commands
- `requirements.txt` - Project dependencies
- `env.example` - Environment variables template
- `MANIFEST.in` - Package distribution configuration

### Documentation Files

- `README.md` - Main project documentation
- `LICENSE` - Project license
- `CONTRIBUTING.md` - Contribution guidelines
- `CHANGELOG.md` - Version history
- `PROJECT_STRUCTURE.md` - This file
- `CONFIGURATION_FILES.md` - Configuration files overview

## Key Features

1. **Package Structure**: Main code organized under `buffetbot/` package
2. **Separation of Concerns**: UI, tests, examples, and docs in separate directories
3. **Configuration Management**: All config files in root with clear documentation
4. **Development Tools**: Pre-configured linting, formatting, and testing
5. **Multiple UI Options**: Streamlit, standalone HTML, and React components

## Development Workflow

1. Install development environment: `make install-dev`
2. Run tests: `make test`
3. Format code: `make format`
4. Run linting: `make lint`
5. Run Streamlit app: `make run-streamlit`
6. Run examples: `make run-example`

## Import Examples

```python
# Import from main package
from glossary import GLOSSARY, search_metrics

# Import from analysis modules (when properly set up)
from analysis.growth_analysis import analyze_growth
from utils.data_fetcher import fetch_stock_data
```

This structure follows Python packaging best practices and makes the project maintainable, scalable, and easy to distribute. 