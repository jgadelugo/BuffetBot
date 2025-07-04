# BuffetBot Tests

This directory contains all test files for the BuffetBot financial analysis toolkit.

## Structure

- `conftest.py` - Pytest configuration and shared fixtures
- `test_glossary.py` - Tests for the glossary_data module
- `test_fallback_logic.py` - Integration test for multi-source fallback logic in data fetchers
- `test_dashboard_error_handling.py` - Tests for dashboard error handling
- `test_options_advisor.py` - Tests for options advisor functionality
- `test_options_fetcher.py` - Tests for options data fetching
- `test_options_math.py` - Tests for options mathematics utilities
- `test_robust_fetchers.py` - Tests for robust data fetching mechanisms
- `test_ticker_change_detection.py` - Tests for ticker change detection
- `test_ui_error_handling.py` - Tests for UI error handling
- `test_ui_scoring_transparency.py` - Tests for UI scoring transparency features
- `test_weight_normalization.py` - Tests for dynamic weight normalization

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

### Run integration tests:
```bash
# Run the fallback logic integration test
python tests/test_fallback_logic.py

# Run UI scoring transparency tests
python tests/test_ui_scoring_transparency.py

# Run weight normalization tests
python tests/test_weight_normalization.py

# Or through pytest
python -m pytest tests/test_fallback_logic.py -v
```

## Test Conventions

1. All test files should be named `test_*.py` or `*_test.py`
2. Test functions should start with `test_`
3. Use descriptive names that explain what is being tested
4. Group related tests in test classes when appropriate
5. Use fixtures from `conftest.py` for common test data
6. Include both unit tests (class-based) and integration tests (standalone functions)

## Writing Tests

Example test structure:
```python
import pytest
from glossary_data import get_metric_info

class TestGlossaryFunctions:
    """Test class for glossary functionality."""

    def test_get_metric_info_valid(self):
        """Test that get_metric_info returns correct data for valid metric."""
        info = get_metric_info("pe_ratio")
        assert info["name"] == "Price-to-Earnings (P/E) Ratio"
        assert info["category"] == "value"

    def test_get_metric_info_invalid(self):
        """Test that get_metric_info raises KeyError for invalid metric."""
        with pytest.raises(KeyError):
            get_metric_info("invalid_metric")

def test_integration_workflow():
    """Integration test for complete workflow."""
    # Test the full workflow end-to-end
    pass
```

## Fixtures

Available fixtures from `conftest.py`:
- `sample_metrics` - Sample calculated metrics dictionary
- `sample_analysis_results` - Sample analysis results with all metric categories

## UI Testing

UI tests in `test_ui_scoring_transparency.py` test the scoring transparency features:
- Data score badge generation with correct color coding
- Partial data detection in recommendation DataFrames
- Score details breakdown display
- Graceful handling of edge cases (empty data, None values, etc.)

These tests can be run both with pytest for detailed unit testing and directly as integration tests.
