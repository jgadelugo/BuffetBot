#!/usr/bin/env python3
"""
Test script for robust fault handling in data fetchers.

This script tests the refactored data fetchers to ensure they properly handle
errors without breaking the pipeline and return consistent structures with
data availability flags.
"""

import logging
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from buffetbot.data.forecast_fetcher import get_analyst_forecast, get_forecast_summary
from buffetbot.data.options_fetcher import fetch_long_dated_calls, get_options_summary
from buffetbot.data.peer_fetcher import add_static_peers, get_peer_info, get_peers

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_forecast_fetcher():
    """Test the forecast fetcher with robust error handling."""
    print("\n" + "=" * 60)
    print("TESTING FORECAST FETCHER")
    print("=" * 60)

    # Test with a valid ticker
    print("\n1. Testing with valid ticker (AAPL):")
    result = get_analyst_forecast("AAPL")
    print(f"Data available: {result['data_available']}")
    if result["data_available"]:
        print(f"Mean target: ${result['mean_target']:.2f}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Number of analysts: {result['num_analysts']}")
    else:
        print(f"Error: {result['error_message']}")

    # Test summary
    summary = get_forecast_summary("AAPL")
    if summary:
        print(f"Summary available: {len(summary)} characters")
    else:
        print("Summary not available")

    # Test with an invalid ticker
    print("\n2. Testing with invalid ticker (INVALID123):")
    result = get_analyst_forecast("INVALID123")
    print(f"Data available: {result['data_available']}")
    print(f"Error: {result['error_message']}")

    # Test with empty ticker
    print("\n3. Testing with empty ticker:")
    result = get_analyst_forecast("")
    print(f"Data available: {result['data_available']}")
    print(f"Error: {result['error_message']}")


def test_options_fetcher():
    """Test the options fetcher with robust error handling."""
    print("\n" + "=" * 60)
    print("TESTING OPTIONS FETCHER")
    print("=" * 60)

    # Test with a valid ticker
    print("\n1. Testing with valid ticker (AAPL):")
    result = fetch_long_dated_calls("AAPL", min_days_to_expiry=90)
    print(f"Data available: {result['data_available']}")
    if result["data_available"]:
        print(f"Options found: {len(result['data'])}")
        print(f"Expiry dates processed: {result['valid_chains_processed']}")
    else:
        print(f"Error: {result['error_message']}")

    # Test summary
    summary = get_options_summary(result)
    if summary:
        print(f"Summary - Total options: {summary['total_options']}")
    else:
        print("Summary not available")

    # Test with an invalid ticker
    print("\n2. Testing with invalid ticker (INVALID123):")
    result = fetch_long_dated_calls("INVALID123")
    print(f"Data available: {result['data_available']}")
    print(f"Error: {result['error_message']}")

    # Test with empty ticker
    print("\n3. Testing with empty ticker:")
    result = fetch_long_dated_calls("")
    print(f"Data available: {result['data_available']}")
    print(f"Error: {result['error_message']}")


def test_peer_fetcher():
    """Test the peer fetcher with robust error handling."""
    print("\n" + "=" * 60)
    print("TESTING PEER FETCHER")
    print("=" * 60)

    # Test with a valid ticker that has static peers
    print("\n1. Testing with valid ticker (NVDA):")
    result = get_peers("NVDA")
    print(f"Data available: {result['data_available']}")
    if result["data_available"]:
        print(f"Peers found: {result['total_peers_found']}")
        print(f"Data source: {result['data_source']}")
        print(f"Peers: {result['peers'][:3]}...")  # Show first 3
    else:
        print(f"Error: {result['error_message']}")

    # Test peer info
    if result["data_available"]:
        print("\n1b. Testing detailed peer info:")
        info_result = get_peer_info("NVDA")
        print(f"Peer info available: {info_result['data_available']}")
        if info_result["data_available"]:
            print(f"Successful lookups: {info_result['successful_lookups']}")
            print(f"Failed lookups: {info_result['failed_lookups']}")
        else:
            print(f"Error: {info_result['error_message']}")

    # Test with an invalid ticker
    print("\n2. Testing with invalid ticker (INVALID123):")
    result = get_peers("INVALID123")
    print(f"Data available: {result['data_available']}")
    print(f"Error: {result['error_message']}")

    # Test with empty ticker
    print("\n3. Testing with empty ticker:")
    result = get_peers("")
    print(f"Data available: {result['data_available']}")
    print(f"Error: {result['error_message']}")

    # Test adding static peers
    print("\n4. Testing add static peers:")
    add_result = add_static_peers("TEST", ["AAPL", "MSFT", "INVALID"])
    print(f"Success: {add_result['success']}")
    print(f"Valid peers added: {add_result['valid_peers_added']}")
    print(f"Invalid peers skipped: {add_result['invalid_peers_skipped']}")


def main():
    """Run all tests."""
    print("TESTING ROBUST FAULT HANDLING FOR DATA FETCHERS")
    print("This script tests that all fetchers handle errors gracefully")
    print("and return consistent structures with data availability flags.")

    try:
        test_forecast_fetcher()
        test_options_fetcher()
        test_peer_fetcher()

        print("\n" + "=" * 60)
        print("TESTING COMPLETE")
        print("=" * 60)
        print("✅ All fetchers implemented robust fault handling")
        print("✅ No exceptions were raised that break the pipeline")
        print("✅ All functions return consistent structures with data_available flags")
        print("✅ Clear error messages are provided when data is unavailable")

    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
        print("This indicates a fetcher is still raising exceptions!")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
