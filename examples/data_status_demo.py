"""
Data Source Status Demo

This script demonstrates the functionality of the data/source_status.py module,
showing how to check data availability status for individual tickers and
get health summaries across multiple tickers.
"""

from data.source_status import (
    get_data_availability_status,
    get_source_health_summary,
    print_data_status,
)


def demo_single_ticker_status():
    """Demonstrate single ticker status checking."""
    print("=" * 60)
    print("DEMO: Single Ticker Data Availability Status")
    print("=" * 60)

    # Test with a popular ticker
    print("\n1. Testing with AAPL (should have good data availability):")
    aapl_status = get_data_availability_status("AAPL")
    print_data_status(aapl_status)

    # Test with a less common ticker
    print("\n2. Testing with NVDA (should have good data availability):")
    nvda_status = get_data_availability_status("NVDA")
    print_data_status(nvda_status)

    # Test with invalid ticker
    print("\n3. Testing with invalid ticker (should fail gracefully):")
    invalid_status = get_data_availability_status("INVALID_TICKER_123")
    print_data_status(invalid_status)


def demo_multi_ticker_health_summary():
    """Demonstrate multi-ticker health summary."""
    print("\n" + "=" * 60)
    print("DEMO: Multi-Ticker Health Summary")
    print("=" * 60)

    # Mixed list of tickers - some valid, some invalid
    test_tickers = [
        "AAPL",  # Should work well
        "MSFT",  # Should work well
        "GOOGL",  # Should work well
        "INVALID",  # Should fail
        "XYZ123",  # Should fail
    ]

    print(f"\nChecking health across {len(test_tickers)} tickers: {test_tickers}")
    print("This may take a moment as we check each data source...\n")

    summary = get_source_health_summary(test_tickers)

    print("📋 HEALTH SUMMARY RESULTS:")
    print("-" * 40)
    print(f"Total tickers checked: {summary['total_tickers_checked']}")
    print(f"Healthy tickers: {summary['healthy_tickers']}")
    print(f"Partially healthy: {summary['partial_tickers']}")
    print(f"Unhealthy tickers: {summary['unhealthy_tickers']}")
    print()

    print("📊 SOURCE SUCCESS RATES:")
    print("-" * 40)
    for source, rate in summary["source_success_rates"].items():
        print(f"{source.capitalize()}: {rate:.1%}")
    print()

    print("🏆 SOURCE RELIABILITY RANKING:")
    print("-" * 40)
    print(f"Most reliable: {summary['most_reliable_source']}")
    print(f"Least reliable: {summary['least_reliable_source']}")


def demo_direct_api_usage():
    """Demonstrate direct API usage with return value inspection."""
    print("\n" + "=" * 60)
    print("DEMO: Direct API Usage & Data Inspection")
    print("=" * 60)

    print("\nChecking status for AAPL and inspecting the returned data structure:")
    status = get_data_availability_status("AAPL")

    print("\n📋 Raw Status Dictionary:")
    print("-" * 30)
    for key, value in status.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")

    print("\n💡 Programmatic Usage Examples:")
    print("-" * 30)
    print(f"• Ticker: {status['ticker']}")
    print(f"• Overall health: {status['overall_health']}")
    print(
        f"• Available sources: {status['available_sources']}/{status['total_sources']}"
    )

    if status["forecast"]["available"]:
        print(f"• Forecast available from: {status['forecast']['source']}")
    else:
        print("• Forecast not available")

    if status["options"]["available"]:
        print(f"• Options available from: {status['options']['source']}")
    else:
        print("• Options not available")

    if status["peers"]["available"]:
        print(f"• Peers available from: {status['peers']['source']}")
    else:
        print("• Peers not available")


def main():
    """Run all demos."""
    print("🚀 BuffetBot Data Source Status Module Demo")
    print("This demo showcases the centralized data availability reporting system.")

    # Run all demo functions
    demo_single_ticker_status()
    demo_multi_ticker_health_summary()
    demo_direct_api_usage()

    print("\n" + "=" * 60)
    print("✅ Demo Complete!")
    print("=" * 60)
    print("\nThe data/source_status.py module provides:")
    print("• get_data_availability_status(ticker) - Check individual ticker status")
    print("• print_data_status(status_dict) - Pretty print status information")
    print("• get_source_health_summary(tickers) - Aggregate health across tickers")
    print("\nAll functions handle errors gracefully and never crash the application.")


if __name__ == "__main__":
    main()
