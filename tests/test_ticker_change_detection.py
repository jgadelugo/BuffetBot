#!/usr/bin/env python3
"""
Test script to verify ticker change detection and data quality score recalculation.
This script simulates the core logic without Streamlit to ensure it works properly.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging

from data.cleaner import clean_financial_data
from data.fetcher import fetch_stock_data
from utils.data_report import DataCollectionReport
from utils.logger import setup_logger

# Set up logging
logger = setup_logger(__name__, "logs/test_ticker_change.log")


def test_ticker_change_simulation():
    """Test the ticker change and data quality score calculation."""

    # Test with different tickers
    test_tickers = ["AAPL", "MSFT", "GOOGL"]

    print("Testing ticker change detection and data quality score calculation...\n")

    for ticker in test_tickers:
        print(f"Processing ticker: {ticker}")
        print("-" * 40)

        try:
            # Simulate the data fetching process
            print(f"Fetching data for {ticker}...")
            raw_data = fetch_stock_data(ticker, years=5)

            # Clean and process data (similar to get_stock_info)
            cleaned_data = clean_financial_data(
                {
                    "income_stmt": raw_data["income_stmt"],
                    "balance_sheet": raw_data["balance_sheet"],
                    "cash_flow": raw_data["cash_flow"],
                }
            )

            # Add other data
            cleaned_data["price_data"] = raw_data["price_data"]
            cleaned_data["fundamentals"] = raw_data["fundamentals"]
            cleaned_data["metrics"] = raw_data["metrics"]

            # Create DataCollectionReport and calculate quality score
            print(f"Calculating data quality score for {ticker}...")
            report = DataCollectionReport(cleaned_data)
            report_data = report.get_report()
            quality_score = report_data.get("data_quality_score", 0)

            print(f"✅ Data quality score for {ticker}: {quality_score:.1f}%")

            # Show some additional info
            availability = report_data.get("data_availability", {})
            available_statements = sum(
                1 for status in availability.values() if status.get("available", False)
            )
            total_statements = len(availability)

            print(f"📊 Available statements: {available_statements}/{total_statements}")

            # Show fundamentals availability
            fundamentals = cleaned_data.get("fundamentals", {})
            fundamental_metrics = len(
                [k for k, v in fundamentals.items() if v is not None and v != 0]
            )
            print(f"💰 Fundamental metrics available: {fundamental_metrics}")

            # Show data quality issues if any
            recommendations = report_data.get("recommendations", [])
            if recommendations:
                print(
                    f"⚠️  Data quality issues found ({len(recommendations)} recommendations):"
                )
                for i, rec in enumerate(recommendations[:3], 1):  # Show first 3
                    print(f"   {i}. {rec}")
                if len(recommendations) > 3:
                    print(f"   ... and {len(recommendations) - 3} more")
            else:
                print("✅ No data quality issues detected")

            # Show data recency
            for statement in ["income_stmt", "balance_sheet", "cash_flow"]:
                status = availability.get(statement, {})
                if status.get("available"):
                    last_date = status.get("last_available_date", "N/A")
                    print(f"📅 {statement}: {last_date}")

            print(f"✅ Successfully processed {ticker}\n")

        except Exception as e:
            print(f"❌ Error processing {ticker}: {str(e)}")
            logger.error(f"Error processing {ticker}: {str(e)}", exc_info=True)
            print()


if __name__ == "__main__":
    test_ticker_change_simulation()
