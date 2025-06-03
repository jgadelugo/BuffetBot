"""Unit tests for dashboard.utils.data_utils module."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from buffetbot.dashboard.utils.data_utils import (
    safe_get_last_price,
    safe_get_nested_value,
)


class TestSafeGetNestedValue:
    """Test cases for safe_get_nested_value function."""

    def test_valid_nested_access(self):
        """Test accessing valid nested dictionary values."""
        data = {
            "level1": {"level2": {"level3": "target_value"}},
            "simple": "simple_value",
        }

        assert safe_get_nested_value(data, "simple") == "simple_value"
        assert (
            safe_get_nested_value(data, "level1", "level2", "level3") == "target_value"
        )
        assert safe_get_nested_value(data, "level1", "level2") == {
            "level3": "target_value"
        }

    def test_missing_keys(self):
        """Test handling of missing keys."""
        data = {"level1": {"level2": {"level3": "value"}}}

        assert safe_get_nested_value(data, "nonexistent") is None
        assert safe_get_nested_value(data, "level1", "nonexistent") is None
        assert safe_get_nested_value(data, "level1", "level2", "nonexistent") is None
        assert safe_get_nested_value(data, "nonexistent", "level2", "level3") is None

    def test_none_data(self):
        """Test handling of None data."""
        assert safe_get_nested_value(None, "key") is None
        assert safe_get_nested_value(None, "key1", "key2") is None

    def test_non_dict_intermediate_values(self):
        """Test handling of non-dict intermediate values."""
        data = {"level1": {"level2": "string_value"}}  # Not a dict

        assert safe_get_nested_value(data, "level1", "level2", "level3") is None

    def test_empty_dict(self):
        """Test handling of empty dictionary."""
        data = {}
        assert safe_get_nested_value(data, "key") is None

    def test_invalid_data_types(self):
        """Test handling of invalid data types."""
        assert safe_get_nested_value("not_a_dict", "key") is None
        assert safe_get_nested_value(123, "key") is None
        assert safe_get_nested_value([], "key") is None

    def test_numeric_values(self):
        """Test retrieving numeric values."""
        data = {"metrics": {"price": 123.45, "volume": 1000000, "change": -2.5}}

        assert safe_get_nested_value(data, "metrics", "price") == 123.45
        assert safe_get_nested_value(data, "metrics", "volume") == 1000000
        assert safe_get_nested_value(data, "metrics", "change") == -2.5

    def test_complex_data_structures(self):
        """Test handling of complex nested structures."""
        data = {
            "company": {
                "financials": {
                    "income_statement": {
                        "revenue": [100, 110, 120],
                        "expenses": {"operating": 80, "interest": 5},
                    }
                }
            }
        }

        assert safe_get_nested_value(
            data, "company", "financials", "income_statement", "revenue"
        ) == [100, 110, 120]
        assert (
            safe_get_nested_value(
                data,
                "company",
                "financials",
                "income_statement",
                "expenses",
                "operating",
            )
            == 80
        )


class TestSafeGetLastPrice:
    """Test cases for safe_get_last_price function."""

    def test_valid_price_data(self):
        """Test extracting last price from valid price data."""
        price_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [105, 106, 107],
                "Low": [95, 96, 97],
                "Close": [103, 104, 105.5],
                "Volume": [1000, 1100, 1200],
            }
        )

        assert safe_get_last_price(price_data) == 105.5

    def test_single_row_dataframe(self):
        """Test with single row DataFrame."""
        price_data = pd.DataFrame({"Close": [123.45]})

        assert safe_get_last_price(price_data) == 123.45

    def test_none_price_data(self):
        """Test handling of None price data."""
        assert safe_get_last_price(None) is None

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        price_data = pd.DataFrame()
        assert safe_get_last_price(price_data) is None

    def test_missing_close_column(self):
        """Test handling of DataFrame without Close column."""
        price_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [105, 106, 107],
                "Low": [95, 96, 97],
                "Volume": [1000, 1100, 1200],
            }
        )

        assert safe_get_last_price(price_data) is None

    def test_invalid_data_types(self):
        """Test handling of invalid data types."""
        assert safe_get_last_price("not_a_dataframe") is None
        assert safe_get_last_price(123) is None
        assert safe_get_last_price([1, 2, 3]) is None
        assert safe_get_last_price({"Close": [100, 101]}) is None

    def test_close_column_with_nans(self):
        """Test handling of Close column with NaN values."""
        price_data = pd.DataFrame({"Close": [100, 101, np.nan, 104, 105]})

        # Should return 105, the last non-NaN value
        assert safe_get_last_price(price_data) == 105

    def test_all_nan_close_values(self):
        """Test handling of Close column with all NaN values."""
        price_data = pd.DataFrame({"Close": [np.nan, np.nan, np.nan]})

        assert safe_get_last_price(price_data) is None

    def test_close_column_with_mixed_types(self):
        """Test handling of Close column with mixed data types."""
        price_data = pd.DataFrame({"Close": [100, 101, "invalid", 104, 105]})

        # Should handle conversion gracefully
        result = safe_get_last_price(price_data)
        assert result == 105 or result is None  # Depends on pandas handling

    def test_negative_prices(self):
        """Test handling of negative prices (edge case)."""
        price_data = pd.DataFrame({"Close": [100, 101, -1, 104, 105]})

        assert safe_get_last_price(price_data) == 105

    def test_zero_prices(self):
        """Test handling of zero prices."""
        price_data = pd.DataFrame({"Close": [100, 101, 0, 104, 0]})

        assert safe_get_last_price(price_data) == 0.0

    def test_very_large_prices(self):
        """Test handling of very large prices."""
        price_data = pd.DataFrame({"Close": [100, 101, 1e10, 104, 1e15]})

        assert safe_get_last_price(price_data) == 1e15

    def test_scientific_notation_prices(self):
        """Test handling of prices in scientific notation."""
        price_data = pd.DataFrame({"Close": [100, 101, 1.23e-5, 104, 1.56e3]})

        assert safe_get_last_price(price_data) == 1560.0


class TestErrorHandling:
    """Test error handling across all functions."""

    def test_safe_get_nested_value_exceptions(self):
        """Test that safe_get_nested_value handles exceptions gracefully."""
        # Test with data that might cause KeyError, TypeError, or AttributeError
        problematic_data = {"key": object()}  # Object that doesn't support item access

        assert safe_get_nested_value(problematic_data, "key", "subkey") is None

    def test_safe_get_last_price_exceptions(self):
        """Test that safe_get_last_price handles exceptions gracefully."""

        # Create a DataFrame-like object that might cause issues
        class FakeDataFrame:
            def __init__(self):
                self.empty = False
                self.columns = ["Close"]

            def __getitem__(self, key):
                if key == "Close":
                    raise ValueError("Test exception")

        fake_df = FakeDataFrame()
        assert safe_get_last_price(fake_df) is None


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_stock_data_extraction(self):
        """Test extracting data from realistic stock data structure."""
        stock_data = {
            "fundamentals": {"market_cap": 1.5e12, "pe_ratio": 25.6, "beta": 1.2},
            "metrics": {"price_change": 0.025, "volatility": 0.35, "rsi": 65.4},
            "price_data": pd.DataFrame({"Close": [150.1, 151.2, 149.8, 152.3, 153.7]}),
        }

        # Test nested value extraction
        assert safe_get_nested_value(stock_data, "fundamentals", "market_cap") == 1.5e12
        assert safe_get_nested_value(stock_data, "metrics", "rsi") == 65.4
        assert safe_get_nested_value(stock_data, "nonexistent", "key") is None

        # Test price extraction
        price_data = safe_get_nested_value(stock_data, "price_data")
        assert safe_get_last_price(price_data) == 153.7

    def test_missing_data_scenarios(self):
        """Test scenarios with missing or incomplete data."""
        incomplete_data = {
            "fundamentals": {"market_cap": None, "pe_ratio": 25.6},
            "metrics": {},
            "price_data": pd.DataFrame(),  # Empty DataFrame
        }

        assert (
            safe_get_nested_value(incomplete_data, "fundamentals", "market_cap") is None
        )
        assert safe_get_nested_value(incomplete_data, "metrics", "rsi") is None
        assert safe_get_last_price(incomplete_data["price_data"]) is None


if __name__ == "__main__":
    pytest.main([__file__])
