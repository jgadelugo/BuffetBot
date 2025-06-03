#!/usr/bin/env python3
"""
Test script for UI error handling with None values and missing data.

This script tests that the UI helper functions properly handle None values
and missing data structures without raising TypeErrors.
"""

import sys
from pathlib import Path

import pandas as pd

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from buffetbot.dashboard.app import (
    safe_format_currency,
    safe_format_number,
    safe_format_percentage,
    safe_get_last_price,
    safe_get_nested_value,
)


def test_safe_formatting_functions():
    """Test all the safe formatting functions with various inputs."""
    print("Testing safe formatting functions...")

    # Test safe_format_currency
    print("\n1. Testing safe_format_currency:")
    test_cases = [
        (None, "N/A"),
        (float("nan"), "N/A"),
        (123.456, "$123.46"),
        (1234567.89, "$1,234,567.89"),
        (0, "$0.00"),
        (-123.45, "$-123.45"),
        ("invalid", "N/A"),
    ]

    for value, expected in test_cases:
        result = safe_format_currency(value)
        print(f"  {value} -> '{result}' (expected: '{expected}')")
        assert result == expected or (expected == "N/A" and result == "N/A")

    # Test safe_format_percentage
    print("\n2. Testing safe_format_percentage:")
    test_cases = [
        (None, "N/A"),
        (float("nan"), "N/A"),
        (0.1234, "12.3%"),
        (0.0, "0.0%"),
        (-0.05, "-5.0%"),
        (1.5, "150.0%"),
        ("invalid", "N/A"),
    ]

    for value, expected in test_cases:
        result = safe_format_percentage(value)
        print(f"  {value} -> '{result}' (expected: '{expected}')")
        assert result == expected or (expected == "N/A" and result == "N/A")

    # Test safe_format_number
    print("\n3. Testing safe_format_number:")
    test_cases = [
        (None, "N/A"),
        (float("nan"), "N/A"),
        (123.456, "123.46"),
        (0, "0.00"),
        (-123.45, "-123.45"),
        ("invalid", "N/A"),
    ]

    for value, expected in test_cases:
        result = safe_format_number(value)
        print(f"  {value} -> '{result}' (expected: '{expected}')")
        assert result == expected or (expected == "N/A" and result == "N/A")


def test_safe_get_nested_value():
    """Test safe nested value extraction."""
    print("\n4. Testing safe_get_nested_value:")

    test_data = {
        "level1": {"level2": {"value": 42}, "empty": None},
        "numbers": [1, 2, 3],
    }

    test_cases = [
        (test_data, ["level1", "level2", "value"], 42),
        (test_data, ["level1", "empty"], None),
        (test_data, ["level1", "nonexistent"], None),
        (test_data, ["nonexistent"], None),
        (None, ["level1"], None),
        ({}, ["level1"], None),
    ]

    for data, keys, expected in test_cases:
        result = safe_get_nested_value(data, *keys)
        print(f"  {keys} from {type(data).__name__} -> {result} (expected: {expected})")
        assert result == expected


def test_safe_get_last_price():
    """Test safe price extraction from DataFrame."""
    print("\n5. Testing safe_get_last_price:")

    # Valid DataFrame
    valid_df = pd.DataFrame({"Close": [100.0, 101.0, 102.0]})

    # Empty DataFrame
    empty_df = pd.DataFrame()

    # DataFrame without Close column
    no_close_df = pd.DataFrame({"Open": [100.0, 101.0, 102.0]})

    test_cases = [
        (valid_df, 102.0),
        (empty_df, None),
        (no_close_df, None),
        (None, None),
        ("invalid", None),
    ]

    for data, expected in test_cases:
        result = safe_get_last_price(data)
        print(f"  {type(data).__name__} -> {result} (expected: {expected})")
        assert result == expected


def test_edge_cases():
    """Test edge cases and error conditions."""
    print("\n6. Testing edge cases:")

    # Test with extremely large numbers
    large_number = 1e15
    result = safe_format_currency(large_number)
    print(f"  Large number formatting: {result}")

    # Test with very small numbers
    small_number = 1e-10
    result = safe_format_percentage(small_number)
    print(f"  Small percentage formatting: {result}")

    # Test with complex nested structure
    complex_data = {
        "data": {
            "metrics": {
                "price_change": None,
                "volatility": float("nan"),
                "rsi": "invalid",
            }
        }
    }

    print("  Complex nested data extraction:")
    print(
        f"    price_change: {safe_get_nested_value(complex_data, 'data', 'metrics', 'price_change')}"
    )
    print(
        f"    volatility: {safe_get_nested_value(complex_data, 'data', 'metrics', 'volatility')}"
    )
    print(f"    rsi: {safe_get_nested_value(complex_data, 'data', 'metrics', 'rsi')}")


def main():
    """Run all UI error handling tests."""
    print("TESTING UI ERROR HANDLING WITH NONE VALUES AND MISSING DATA")
    print("=" * 65)

    try:
        test_safe_formatting_functions()
        test_safe_get_nested_value()
        test_safe_get_last_price()
        test_edge_cases()

        print("\n" + "=" * 65)
        print("✅ ALL UI ERROR HANDLING TESTS PASSED")
        print("✅ Safe formatting functions handle None values correctly")
        print("✅ No TypeErrors when formatting None or invalid values")
        print("✅ UI will display 'N/A' instead of crashing")

        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
