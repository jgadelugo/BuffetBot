# BuffetBot Tests

This directory contains all test files for the BuffetBot financial analysis toolkit.

## Structure

- `conftest.py` - Pytest configuration and shared fixtures
- `test_glossary.py` - Tests for the glossary_data module

## Running Tests

### Run all tests:
```bash
# From the BuffetBot root directory
python -m pytest tests/

# With verbose output
python -m pytest -v tests/

# With coverage report
python -m pytest --cov=. tests/
```

### Run specific test file:
```bash
python -m pytest tests/test_glossary.py
```

### Run specific test function:
```bash
python -m pytest tests/test_glossary.py::test_glossary
```

## Test Conventions

1. All test files should be named `test_*.py` or `*_test.py`
2. Test functions should start with `test_`
3. Use descriptive names that explain what is being tested
4. Group related tests in test classes when appropriate
5. Use fixtures from `conftest.py` for common test data

## Writing Tests

Example test structure:
```python
import pytest
from glossary_data import get_metric_info

def test_get_metric_info_valid():
    """Test that get_metric_info returns correct data for valid metric."""
    info = get_metric_info("pe_ratio")
    assert info["name"] == "Price-to-Earnings (P/E) Ratio"
    assert info["category"] == "value"

def test_get_metric_info_invalid():
    """Test that get_metric_info raises KeyError for invalid metric."""
    with pytest.raises(KeyError):
        get_metric_info("invalid_metric")
```

## Fixtures

Available fixtures from `conftest.py`:
- `sample_metrics` - Sample calculated metrics dictionary
- `sample_analysis_results` - Sample analysis results with all metric categories
