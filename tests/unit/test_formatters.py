"""Unit tests for dashboard.utils.formatters module."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from buffetbot.dashboard.dashboard_utils.formatters import (
    safe_format_currency,
    safe_format_number,
    safe_format_percentage,
)


class TestSafeFormatCurrency:
    """Test cases for safe_format_currency function."""

    def test_valid_currency_formatting(self):
        """Test formatting valid currency values."""
        assert safe_format_currency(1234.56) == "$1,234.56"
        assert safe_format_currency(1000000) == "$1,000,000.00"
        assert safe_format_currency(0) == "$0.00"
        assert safe_format_currency(0.99) == "$0.99"

    def test_currency_decimal_places(self):
        """Test custom decimal places for currency."""
        assert safe_format_currency(1234.5678, decimal_places=0) == "$1,235"
        assert safe_format_currency(1234.5678, decimal_places=3) == "$1,234.568"
        assert safe_format_currency(1234.5678, decimal_places=4) == "$1,234.5678"

    def test_none_values(self):
        """Test handling of None values."""
        assert safe_format_currency(None) == "N/A"
        assert safe_format_currency(None, decimal_places=3) == "N/A"

    def test_nan_values(self):
        """Test handling of NaN values."""
        assert safe_format_currency(float("nan")) == "N/A"
        assert safe_format_currency(np.nan) == "N/A"
        assert safe_format_currency(pd.NA) == "N/A"

    def test_invalid_types(self):
        """Test handling of invalid types."""
        assert safe_format_currency("invalid") == "N/A"
        assert safe_format_currency([1, 2, 3]) == "N/A"
        assert safe_format_currency({"value": 100}) == "N/A"

    def test_large_numbers(self):
        """Test formatting very large numbers."""
        assert safe_format_currency(1e12) == "$1,000,000,000,000.00"
        assert safe_format_currency(1.23e15) == "$1,230,000,000,000,000.00"

    def test_negative_numbers(self):
        """Test formatting negative numbers."""
        assert safe_format_currency(-1234.56) == "$-1,234.56"
        assert safe_format_currency(-1000000) == "$-1,000,000.00"


class TestSafeFormatPercentage:
    """Test cases for safe_format_percentage function."""

    def test_valid_percentage_formatting(self):
        """Test formatting valid percentage values."""
        assert safe_format_percentage(0.1234) == "12.3%"
        assert safe_format_percentage(0.5) == "50.0%"
        assert safe_format_percentage(1.0) == "100.0%"
        assert safe_format_percentage(0.0) == "0.0%"

    def test_percentage_decimal_places(self):
        """Test custom decimal places for percentages."""
        assert safe_format_percentage(0.12345, decimal_places=0) == "12%"
        assert safe_format_percentage(0.12345, decimal_places=2) == "12.35%"
        assert safe_format_percentage(0.12345, decimal_places=3) == "12.345%"

    def test_none_values(self):
        """Test handling of None values."""
        assert safe_format_percentage(None) == "N/A"
        assert safe_format_percentage(None, decimal_places=3) == "N/A"

    def test_nan_values(self):
        """Test handling of NaN values."""
        assert safe_format_percentage(float("nan")) == "N/A"
        assert safe_format_percentage(np.nan) == "N/A"
        assert safe_format_percentage(pd.NA) == "N/A"

    def test_invalid_types(self):
        """Test handling of invalid types."""
        assert safe_format_percentage("invalid") == "N/A"
        assert safe_format_percentage([0.1, 0.2]) == "N/A"
        assert safe_format_percentage({"value": 0.5}) == "N/A"

    def test_large_percentages(self):
        """Test formatting very large percentages."""
        assert safe_format_percentage(10.0) == "1000.0%"
        assert safe_format_percentage(100.0) == "10000.0%"

    def test_negative_percentages(self):
        """Test formatting negative percentages."""
        assert safe_format_percentage(-0.1234) == "-12.3%"
        assert safe_format_percentage(-1.0) == "-100.0%"


class TestSafeFormatNumber:
    """Test cases for safe_format_number function."""

    def test_valid_number_formatting(self):
        """Test formatting valid numbers."""
        assert safe_format_number(1234.56) == "1234.56"
        assert safe_format_number(1000000) == "1000000.00"
        assert safe_format_number(0) == "0.00"
        assert safe_format_number(0.99) == "0.99"

    def test_number_decimal_places(self):
        """Test custom decimal places for numbers."""
        assert safe_format_number(1234.5678, decimal_places=0) == "1235"
        assert safe_format_number(1234.5678, decimal_places=1) == "1234.6"
        assert safe_format_number(1234.5678, decimal_places=3) == "1234.568"

    def test_none_values(self):
        """Test handling of None values."""
        assert safe_format_number(None) == "N/A"
        assert safe_format_number(None, decimal_places=3) == "N/A"

    def test_nan_values(self):
        """Test handling of NaN values."""
        assert safe_format_number(float("nan")) == "N/A"
        assert safe_format_number(np.nan) == "N/A"
        assert safe_format_number(pd.NA) == "N/A"

    def test_invalid_types(self):
        """Test handling of invalid types."""
        assert safe_format_number("invalid") == "N/A"
        assert safe_format_number([1, 2, 3]) == "N/A"
        assert safe_format_number({"value": 100}) == "N/A"

    def test_scientific_notation(self):
        """Test handling of scientific notation."""
        assert safe_format_number(1.23e-5) == "0.00"  # Very small numbers
        assert safe_format_number(1.23e10) == "12300000000.00"  # Very large numbers

    def test_negative_numbers(self):
        """Test formatting negative numbers."""
        assert safe_format_number(-1234.56) == "-1234.56"
        assert safe_format_number(-0.001) == "-0.00"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_infinity_values(self):
        """Test handling of infinity values."""
        assert safe_format_currency(float("inf")) == "N/A"
        assert safe_format_percentage(float("-inf")) == "N/A"
        assert safe_format_number(float("inf")) == "N/A"

    def test_zero_decimal_places(self):
        """Test formatting with zero decimal places."""
        assert safe_format_currency(1234.99, decimal_places=0) == "$1,235"
        assert safe_format_percentage(0.1234, decimal_places=0) == "12%"
        assert safe_format_number(1234.99, decimal_places=0) == "1235"

    def test_very_small_numbers(self):
        """Test formatting very small numbers."""
        assert safe_format_currency(0.001) == "$0.00"
        assert safe_format_percentage(0.00001) == "0.0%"
        assert safe_format_number(0.00001, decimal_places=5) == "0.00001"


if __name__ == "__main__":
    pytest.main([__file__])
