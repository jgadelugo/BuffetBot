#!/usr/bin/env python3
"""
Test script for the options_fetcher module.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from buffetbot.data.options_fetcher import (
    OptionsDataError,
    fetch_long_dated_calls,
    get_options_summary,
)


def test_fetch_options():
    """Test the options fetcher with a well-known ticker."""
    print("Testing options fetcher...")

    # Test with Apple (AAPL) - usually has good options data
    ticker = "AAPL"
    min_days = 90  # Using shorter period for testing

    try:
        print(
            f"\nFetching call options for {ticker} with min {min_days} days to expiry..."
        )
        options_df = fetch_long_dated_calls(ticker, min_days_to_expiry=min_days)

        if options_df.empty:
            print(f"No options found for {ticker}")
            return

        print(f"Found {len(options_df)} call options")
        print(f"\nColumns: {list(options_df.columns)}")

        # Show first few rows
        print(f"\nFirst 5 options:")
        print(options_df.head())

        # Get summary
        summary = get_options_summary(options_df)
        print(f"\nOptions Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        print("\n✅ Test completed successfully!")

    except OptionsDataError as e:
        print(f"❌ Options data error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def test_error_handling():
    """Test error handling with invalid inputs."""
    print("\nTesting error handling...")

    # Test invalid ticker
    try:
        fetch_long_dated_calls("")
        print("❌ Should have raised error for empty ticker")
    except OptionsDataError:
        print("✅ Correctly handled empty ticker")

    # Test negative days
    try:
        fetch_long_dated_calls("AAPL", min_days_to_expiry=-1)
        print("❌ Should have raised error for negative days")
    except ValueError:
        print("✅ Correctly handled negative days")

    # Test invalid ticker (should handle gracefully)
    try:
        result = fetch_long_dated_calls("INVALIDTICKER123", min_days_to_expiry=30)
        if not result["data_available"] or result["data"].empty:
            print("✅ Correctly handled invalid ticker (returned empty DataFrame)")
        else:
            print("❌ Unexpected: got data for invalid ticker")
    except OptionsDataError:
        print("✅ Correctly raised error for invalid ticker")


if __name__ == "__main__":
    test_fetch_options()
    test_error_handling()
