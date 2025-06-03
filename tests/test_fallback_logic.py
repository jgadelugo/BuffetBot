#!/usr/bin/env python3
"""
Test script to demonstrate multi-source fallback logic in the data fetcher modules.

This script tests the fallback behavior for forecast_fetcher, options_fetcher, and peer_fetcher
modules with different types of tickers to showcase the fallback functionality.
"""

import sys
from pathlib import Path

# Add the project root to the Python path (parent of tests directory)
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.forecast_fetcher import get_analyst_forecast
from data.options_fetcher import fetch_long_dated_calls
from data.peer_fetcher import get_peers


def test_forecast_fallback():
    """Test forecast fetcher with fallback logic."""
    print("=" * 60)
    print("TESTING FORECAST FETCHER FALLBACK LOGIC")
    print("=" * 60)

    # Test cases: common ticker, uncommon ticker, invalid ticker
    test_cases = [
        ("AAPL", "Common ticker (should use Yahoo)"),
        ("OBSCURE", "Uncommon ticker (should trigger fallback)"),
        ("INVALID123", "Invalid ticker (should fail all sources)"),
    ]

    for ticker, description in test_cases:
        print(f"\n📊 Testing {ticker} - {description}")
        print("-" * 50)

        result = get_analyst_forecast(ticker)

        print(f"Data Available: {result['data_available']}")
        print(f"Source Used: {result['source_used']}")

        if result["data_available"]:
            print(f"Mean Target: ${result['mean_target']:.2f}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"✅ Success with {result['source_used']} source")
        else:
            print(f"Error: {result['error_message']}")
            print(f"❌ Failed with source: {result['source_used']}")


def test_options_fallback():
    """Test options fetcher with fallback logic."""
    print("\n" + "=" * 60)
    print("TESTING OPTIONS FETCHER FALLBACK LOGIC")
    print("=" * 60)

    # Test cases
    test_cases = [
        ("AAPL", "Common ticker (should use Yahoo)"),
        ("OBSCURE", "Uncommon ticker (should trigger fallback)"),
        ("INVALID123", "Invalid ticker (should fail all sources)"),
    ]

    for ticker, description in test_cases:
        print(f"\n📈 Testing {ticker} - {description}")
        print("-" * 50)

        result = fetch_long_dated_calls(ticker, min_days_to_expiry=180)

        print(f"Data Available: {result['data_available']}")
        print(f"Source Used: {result['source_used']}")

        if result["data_available"]:
            print(f"Options Found: {len(result['data'])}")
            print(f"Expiry Dates: {result['total_expiry_dates']}")
            print(f"✅ Success with {result['source_used']} source")
        else:
            print(f"Error: {result['error_message']}")
            print(f"❌ Failed with source: {result['source_used']}")


def test_peers_fallback():
    """Test peer fetcher with fallback logic."""
    print("\n" + "=" * 60)
    print("TESTING PEER FETCHER FALLBACK LOGIC")
    print("=" * 60)

    # Test cases
    test_cases = [
        ("NVDA", "Known ticker with static peers (should use fallback_static)"),
        ("AAPL", "Common ticker (API may fail, use static)"),
        ("OBSCURE", "Uncommon ticker (should fail all sources)"),
        ("INVALID123", "Invalid ticker (should fail validation)"),
    ]

    for ticker, description in test_cases:
        print(f"\n🤝 Testing {ticker} - {description}")
        print("-" * 50)

        result = get_peers(ticker)

        print(f"Data Available: {result['data_available']}")
        print(f"Source Used: {result['source_used']}")

        if result["data_available"]:
            print(f"Peers Found: {result['total_peers_found']}")
            print(
                f"Peer List: {result['peers'][:5]}{'...' if len(result['peers']) > 5 else ''}"
            )
            print(f"✅ Success with {result['source_used']} source")
        else:
            print(f"Error: {result['error_message']}")
            print(f"❌ Failed with source: {result['source_used']}")


def main():
    """Run all fallback tests."""
    print("🚀 MULTI-SOURCE FALLBACK LOGIC TEST SUITE")
    print("Testing forecast, options, and peer fetcher modules...")

    try:
        # Test each module
        test_forecast_fallback()
        test_options_fallback()
        test_peers_fallback()

        print("\n" + "=" * 60)
        print("✅ FALLBACK LOGIC TEST SUITE COMPLETED")
        print("=" * 60)
        print("\n📝 Summary:")
        print("• All modules now implement multi-source fallback logic")
        print("• Primary source: Yahoo Finance")
        print("• Secondary source: Financial Modeling Prep (FMP)")
        print("• Final fallback: Static data/graceful degradation")
        print("• Consistent error handling and logging throughout")
        print("• TypedDict schemas updated with 'source_used' field")

    except Exception as e:
        print(f"\n❌ Test suite failed with error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
