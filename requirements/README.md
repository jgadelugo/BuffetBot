# Requirements Directory

This directory contains all dependency specifications for different environments and use cases.

## Files

- **`base.txt`** - Core dependencies required for all environments
- **`dev.txt`** - Development dependencies (testing, linting, formatting tools)
- **`test.txt`** - Testing-specific dependencies (pytest, coverage tools)
- **`prod.txt`** - Production dependencies for Streamlit deployment

## Usage

### Development Environment
```bash
pip install -r requirements/base.txt
pip install -r requirements/dev.txt
```

### Testing Environment
```bash
pip install -r requirements/base.txt
pip install -r requirements/test.txt
```

### Production/Streamlit Environment
```bash
pip install -r requirements/base.txt
pip install -r requirements/prod.txt
```

### Using Makefile (Recommended)
```bash
make install-dev    # Install development dependencies
make install-test   # Install test dependencies
make install-prod   # Install production dependencies
make install        # Install only base dependencies
```

## Backward Compatibility

Root-level symlinks are maintained for compatibility:
- `requirements.txt` → `requirements/base.txt`
- `requirements-streamlit.txt` → `requirements/prod.txt`

## Updating Requirements

Use the Makefile command to update from pyproject.toml:
```bash
make requirements
```
