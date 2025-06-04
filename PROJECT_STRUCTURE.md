# BuffetBot Project Structure

This document outlines the organization of the BuffetBot project following Python best practices.

```
BuffetBot/
│
├── buffetbot/                      # Main package directory
│   ├── __init__.py                 # Package initialization with core imports
│   ├── glossary.py                 # Financial metrics glossary module
│   ├── cli.py                      # Command-line interface module
│   │
│   ├── analysis/                   # Analysis modules
│   │   ├── __init__.py
│   │   ├── growth_analysis.py      # Growth analysis module
│   │   ├── value_analysis.py       # Value analysis module
│   │   ├── health_analysis.py      # Health analysis module
│   │   ├── risk_analysis.py        # Risk analysis module
│   │   ├── ecosystem.py            # Ecosystem analysis
│   │   ├── options_advisor.py      # Options analysis module
│   │   └── options/                # Options analysis sub-modules
│   │       ├── config/             # Configuration modules
│   │       ├── core/               # Core domain models and logic
│   │       ├── data/               # Data access and services
│   │       └── scoring/            # Scoring algorithms
│   │
│   ├── dashboard/                  # Dashboard components
│   │   ├── __init__.py
│   │   ├── app.py                  # Main dashboard application
│   │   ├── streamlit_app.py        # Streamlit app entry point
│   │   ├── components/             # Reusable UI components
│   │   │   ├── analytics.py
│   │   │   ├── charts.py
│   │   │   ├── disclaimers.py
│   │   │   ├── forecast_panel.py
│   │   │   ├── glossary_utils.py
│   │   │   ├── metrics.py
│   │   │   ├── metrics_display.py
│   │   │   ├── options_settings.py
│   │   │   ├── options_utils.py
│   │   │   ├── price_valuation.py
│   │   │   └── sidebar.py
│   │   ├── config/                 # Dashboard configuration
│   │   │   ├── analytics.py
│   │   │   └── settings.py
│   │   ├── dashboard_utils/        # Dashboard utilities
│   │   │   ├── data_processing.py
│   │   │   ├── data_utils.py
│   │   │   └── formatters.py
│   │   ├── utils/                  # Additional utilities
│   │   │   └── enhanced_options_analysis.py
│   │   └── views/                  # View modules for different tabs
│   │       ├── analyst_forecast.py
│   │       ├── base.py
│   │       ├── financial_health.py
│   │       ├── glossary.py
│   │       ├── growth_metrics.py
│   │       ├── options_advisor.py
│   │       ├── overview.py
│   │       ├── price_analysis.py
│   │       └── risk_analysis.py
│   │
│   ├── data/                       # Data access modules
│   │   ├── __init__.py
│   │   ├── cleaner.py              # Data cleaning utilities
│   │   ├── fetcher.py              # Legacy data fetcher
│   │   ├── forecast_fetcher.py     # Forecast data fetching
│   │   ├── options_fetcher.py      # Options data fetching
│   │   ├── peer_fetcher.py         # Peer comparison data
│   │   ├── source_status.py        # Data source status checking
│   │   └── fetcher/                # New modular fetcher
│   │       ├── fetcher.py
│   │       └── utils/
│   │
│   ├── recommend/                  # Recommendation system
│   │   ├── __init__.py
│   │   └── recommender.py
│   │
│   └── utils/                      # Utility modules
│       ├── __init__.py
│       ├── cache.py                # Cache management
│       ├── config.py               # Configuration utilities
│       ├── correlation_math.py     # Mathematical correlations
│       ├── data_fetcher.py         # Data fetching utilities
│       ├── data_report.py          # Data reporting
│       ├── errors.py               # Custom exceptions
│       ├── logger.py               # Logging utilities
│       ├── options_math.py         # Options mathematical calculations
│       └── validators.py           # Data validation

├── requirements/                   # Dependency specifications
│   ├── README.md                   # Requirements directory documentation
│   ├── base.txt                    # Core dependencies for all environments
│   ├── dev.txt                     # Development dependencies (testing, linting)
│   ├── test.txt                    # Testing-specific dependencies
│   └── prod.txt                    # Production dependencies for Streamlit

├── config/                         # Configuration files
│   ├── README.md                   # Configuration directory documentation
│   ├── .editorconfig               # Editor configuration for consistent style
│   └── .flake8                     # Python linting configuration

├── scripts/                        # Utility scripts
│   ├── run_app.py                  # Application runner
│   ├── run_dashboard.py            # Dashboard runner
│   ├── run_dashboard.sh            # Shell script for dashboard
│   ├── stop_dashboard.sh           # Shell script to stop dashboard
│   ├── run_tests.py                # Simple test runner
│   ├── run_tests_main.py           # Comprehensive test runner
│   └── remove_path_setup.py        # Path cleanup utility

├── tests/                          # Test suite
│   ├── __init__.py
│   ├── conftest.py                 # pytest configuration and fixtures
│   ├── test_glossary.py            # Glossary module tests
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   ├── fixtures/                   # Test fixtures
│   └── README.md                   # Testing documentation

├── examples/                       # Example scripts
│   ├── __init__.py
│   ├── example_integration.py      # Integration example
│   ├── ecosystem_demo.py           # Ecosystem analysis demo
│   ├── data_status_demo.py         # Data status demo
│   └── README.md                   # Examples documentation

├── ui/                             # User interface implementations
│   ├── streamlit/                  # Streamlit web app
│   │   └── glossary_app.py
│   ├── web/                        # Standalone web interface
│   │   └── glossary.html
│   └── react/                      # React components
│       ├── GlossaryComponent.jsx
│       └── GlossaryComponent.css

├── docs/                           # Documentation
│   ├── deployment/                 # Deployment documentation
│   │   ├── DEPLOYMENT.md           # Deployment instructions
│   │   └── RUNNING_THE_APP.md      # Application running guide
│   ├── development/                # Development documentation
│   │   └── CONFIGURATION_FILES.md  # Configuration files overview
│   ├── prompts/                    # AI/LLM prompts and instructions
│   │   ├── MODULARIZATION_PROMPT.md
│   │   ├── ORGANIZE_DOCS_PROMPT.md
│   │   └── REFACTORING_PROMPT.txt
│   └── glossary/                   # Glossary-specific docs
│       ├── README.md               # Glossary documentation
│       └── ui_implementations.md   # UI implementations guide

├── data/                           # Data directory (gitignored)
├── logs/                           # Log files (gitignored)
├── cache/                          # Cache directory (gitignored)
├── archive/                        # Archived files
│
├── requirements.txt                # Symlink to requirements/base.txt (backward compatibility)
├── requirements-streamlit.txt      # Symlink to requirements/prod.txt (backward compatibility)
├── .editorconfig                   # Symlink to config/.editorconfig (tool compatibility)
├── .flake8                         # Symlink to config/.flake8 (tool compatibility)
└── run_dashboard.sh                # Symlink to scripts/run_dashboard.sh (backward compatibility)

## Configuration Files

### Root Directory Files (Essential)

- `.gitignore` - Version control ignore patterns
- `.pre-commit-config.yaml` - Pre-commit hooks configuration
- `.python-version` - Python version specification for pyenv
- `pyproject.toml` - Modern Python project configuration
- `setup.cfg` - Additional tool configurations
- `setup.py` - Minimal setup script for compatibility
- `Makefile` - Development automation commands
- `env.example` - Environment variables template
- `MANIFEST.in` - Package distribution configuration
- `main.py` - Main entry point for Streamlit deployment

### Organized Dependencies & Configuration

- `requirements/` - All dependency specifications organized by environment
- `config/` - Tool-specific configuration files
- Symlinks maintained for backward compatibility

### Documentation Files

- `README.md` - Main project documentation
- `LICENSE` - Project license
- `CONTRIBUTING.md` - Contribution guidelines
- `CHANGELOG.md` - Version history
- `PROJECT_STRUCTURE.md` - This file

## Key Features

1. **Package Structure**: Main code organized under `buffetbot/` package
2. **Organized Dependencies**: Requirements split by environment in `requirements/` directory
3. **Centralized Configuration**: Tool configs organized in `config/` directory
4. **Separation of Concerns**: UI, tests, examples, and docs in separate directories
5. **Development Tools**: Pre-configured linting, formatting, and testing
6. **Multiple UI Options**: Streamlit, standalone HTML, and React components
7. **Utility Scripts**: Consolidated in `scripts/` directory
8. **Modular Architecture**: Clear separation of analysis, data, dashboard, and utilities
9. **Organized Documentation**: Structured docs with deployment, development, and prompt sections
10. **Backward Compatibility**: Symlinks maintain compatibility with existing tools and workflows

## Development Workflow

1. Install development environment: `make install-dev`
2. Run tests: `make test` or `python scripts/run_tests_main.py`
3. Format code: `make format`
4. Run linting: `make lint`
5. Run Streamlit app: `python scripts/run_dashboard.py` or `./run_dashboard.sh`
6. Run examples: `python examples/example_integration.py`

## Dependency Management

### Installing Dependencies

```bash
# Development environment
make install-dev

# Production environment
make install-prod

# Test environment only
make install-test

# Base dependencies only
make install
```

### Managing Requirements

```bash
# Update requirements from pyproject.toml
make requirements

# Install specific environment
pip install -r requirements/dev.txt
```

## Import Examples

```python
# Import from main package
from buffetbot.glossary import GLOSSARY, search_metrics
from buffetbot import fetch_stock_data, get_logger

# Import from analysis modules
from buffetbot.analysis.growth_analysis import analyze_growth
from buffetbot.utils.data_fetcher import fetch_stock_data

# Import from dashboard components
from buffetbot.dashboard.components.charts import create_price_chart
```

This structure follows Python packaging best practices and makes the project maintainable, scalable, and easy to distribute.

## Recent Reorganization Changes

✅ **Completed Reorganization Tasks:**

1. **Moved `glossary.py`** from root → `buffetbot/glossary.py`
2. **Updated all imports** to use `from buffetbot.glossary import ...`
3. **Moved utility scripts** to `scripts/` directory:
   - `run_app.py` → `scripts/run_app.py`
   - `run_dashboard.py` → `scripts/run_dashboard.py`
   - `run_tests.py` → `scripts/run_tests_main.py`
   - `remove_path_setup.py` → `scripts/remove_path_setup.py`
   - `run_dashboard.sh` → `scripts/run_dashboard.sh`
   - `stop_dashboard.sh` → `scripts/stop_dashboard.sh`
4. **Organized documentation** into structured `docs/` directory:
   - `deployment/` - Deployment and running instructions
   - `development/` - Development configuration docs
   - `prompts/` - AI/LLM prompts and refactoring instructions
   - `glossary/` - Glossary-specific documentation
5. **Organized requirements** into `requirements/` directory:
   - `base.txt` - Core dependencies
   - `dev.txt` - Development dependencies
   - `test.txt` - Testing dependencies
   - `prod.txt` - Production/Streamlit dependencies
6. **Organized configuration** into `config/` directory:
   - `.editorconfig` - Editor configuration
   - `.flake8` - Linting configuration
7. **Updated `buffetbot/__init__.py`** to include glossary exports
8. **Cleaned up root directory** by removing misplaced files
9. **Updated all import statements** across the codebase
10. **Created backward compatibility symlinks** for existing workflows
11. **Updated Makefile** with new dependency management commands
12. **Verified tests still pass** after reorganization

The project now follows exemplary Python packaging conventions with:
- Clean root directory with only essential configuration files
- Organized dependencies by environment in `requirements/` directory
- Centralized tool configuration in `config/` directory
- All business logic contained within the `buffetbot` package
- Utility scripts properly organized in `scripts/` directory
- Comprehensive documentation structure in `docs/` directory
- Backward compatibility maintained through symlinks
