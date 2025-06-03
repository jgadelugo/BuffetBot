"""Test script for the glossary_data module.

This script demonstrates how to use the glossary module in both backend
and frontend contexts.
"""

import os
import sys

# Add parent directory to path to import buffetbot module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glossary import (
    GLOSSARY,
    get_metric_info,
    get_metric_names,
    get_metrics_by_category,
    search_metrics,
)


def test_glossary():
    """Test various functions of the glossary module."""

    print("=== Financial Metrics Glossary Test ===\n")

    # Test 1: Show total metrics count
    print(f"Total metrics in glossary: {len(GLOSSARY)}")
    print()

    # Test 2: Get metrics by category
    for category in ["growth", "value", "health", "risk"]:
        metrics = get_metrics_by_category(category)
        print(f"{category.title()} metrics: {len(metrics)}")
        print(f"  Examples: {', '.join(list(metrics.keys())[:3])}")
    print()

    # Test 3: Search functionality
    print("Searching for 'debt' related metrics:")
    debt_metrics = search_metrics("debt")
    for key, metric in debt_metrics.items():
        print(f"  - {metric['name']}: {metric['description'][:80]}...")
    print()

    # Test 4: Get specific metric info
    print("Detailed info for 'piotroski_score':")
    try:
        piotroski = get_metric_info("piotroski_score")
        print(f"  Name: {piotroski['name']}")
        print(f"  Category: {piotroski['category']}")
        print(f"  Description: {piotroski['description'][:100]}...")
        print(f"  Formula: {piotroski['formula'][:100]}...")
    except KeyError as e:
        print(f"  Error: {e}")
    print()

    # Test 5: Backend usage example
    print("Backend usage example - Calculate metrics to display:")
    calculated_metrics = {
        "current_ratio": 1.5,
        "debt_to_equity": 0.8,
        "pe_ratio": 15.2,
        "revenue_growth": 0.12,
    }

    for metric_key, value in calculated_metrics.items():
        info = get_metric_info(metric_key)
        print(f"  {info['name']}: {value:.2f}")
        print(f"    Category: {info['category']}")
        print(f"    Interpretation: ", end="")

        # Add interpretation based on metric
        if metric_key == "current_ratio":
            if value > 1.0:
                print("Good liquidity")
            else:
                print("Potential liquidity issues")
        elif metric_key == "debt_to_equity":
            if value < 1.0:
                print("Conservative leverage")
            else:
                print("High leverage")
        elif metric_key == "pe_ratio":
            if value < 15:
                print("Potentially undervalued")
            elif value > 25:
                print("Potentially overvalued")
            else:
                print("Fair valuation")
        elif metric_key == "revenue_growth":
            print(f"{value*100:.1f}% year-over-year growth")
    print()

    # Test 6: Frontend usage example
    print("Frontend usage example - Display metrics by category:")
    health_metrics = get_metrics_by_category("health")
    print("Financial Health Dashboard:")
    for i, (key, metric) in enumerate(list(health_metrics.items())[:5]):
        print(f"  {i+1}. {metric['name']}")
        print(f"     Formula: {metric['formula']}")
        print()


if __name__ == "__main__":
    test_glossary()
