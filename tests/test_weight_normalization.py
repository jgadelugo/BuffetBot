#!/usr/bin/env python3
"""
Test script for Dynamic Weight Normalization

This module tests the normalize_scoring_weights function that dynamically
redistributes scoring weights when data sources are missing, ensuring
fair comparison across tickers regardless of data availability.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.options_advisor import normalize_scoring_weights


def test_normalize_scoring_weights():
    """Test the normalize_scoring_weights function with various scenarios."""

    print("🧪 Testing Dynamic Weight Normalization")
    print("=" * 60)

    # Original weights (standard configuration)
    original_weights = {
        "rsi": 0.20,
        "beta": 0.20,
        "momentum": 0.20,
        "iv": 0.20,
        "forecast": 0.20,
    }

    # Test Case 1: All sources available (baseline)
    print("\n📊 Test Case 1: All sources available")
    available = ["rsi", "beta", "momentum", "iv", "forecast"]
    normalized = normalize_scoring_weights(original_weights, available)
    print(f"Available sources: {available}")
    print(f"Normalized weights: {normalized}")
    print(f"Total weight: {sum(normalized.values()):.6f}")
    assert abs(sum(normalized.values()) - 1.0) < 0.001, "Weights should sum to 1.0"
    assert len(normalized) == 5, "Should have all 5 weights"
    print("✅ PASS: All weights preserved and sum to 1.0")

    # Test Case 2: Missing forecast data (common scenario)
    print("\n📊 Test Case 2: Missing forecast data (typical scenario)")
    available = ["rsi", "beta", "momentum", "iv"]
    normalized = normalize_scoring_weights(original_weights, available)
    print(f"Available sources: {available}")
    print(f"Normalized weights: {normalized}")
    print(f"Total weight: {sum(normalized.values()):.6f}")
    expected_weight = 0.25  # 0.2 / 0.8 = 0.25 for each remaining
    assert abs(sum(normalized.values()) - 1.0) < 0.001, "Weights should sum to 1.0"
    assert (
        abs(normalized["rsi"] - expected_weight) < 0.001
    ), f"Expected {expected_weight}, got {normalized['rsi']}"
    assert "forecast" not in normalized, "Missing source should not be in result"
    print("✅ PASS: Weights properly redistributed proportionally")

    # Test Case 3: Only one source available (extreme scenario)
    print("\n📊 Test Case 3: Only one source available (extreme scenario)")
    available = ["rsi"]
    normalized = normalize_scoring_weights(original_weights, available)
    print(f"Available sources: {available}")
    print(f"Normalized weights: {normalized}")
    print(f"Total weight: {sum(normalized.values()):.6f}")
    assert abs(normalized["rsi"] - 1.0) < 0.001, "Single source should get 100% weight"
    assert len(normalized) == 1, "Should have only one weight"
    print("✅ PASS: Single source gets 100% weight")

    # Test Case 4: Multiple missing sources
    print("\n📊 Test Case 4: Multiple missing sources")
    available = ["rsi", "momentum"]
    normalized = normalize_scoring_weights(original_weights, available)
    print(f"Available sources: {available}")
    print(f"Normalized weights: {normalized}")
    print(f"Total weight: {sum(normalized.values()):.6f}")
    expected_weight = 0.5  # 0.2 / 0.4 = 0.5 for each remaining
    assert abs(sum(normalized.values()) - 1.0) < 0.001, "Weights should sum to 1.0"
    assert (
        abs(normalized["rsi"] - expected_weight) < 0.001
    ), f"Expected {expected_weight}, got {normalized['rsi']}"
    assert (
        abs(normalized["momentum"] - expected_weight) < 0.001
    ), f"Expected {expected_weight}, got {normalized['momentum']}"
    assert len(normalized) == 2, "Should have exactly 2 weights"
    print("✅ PASS: Weights properly redistributed for partial data")

    # Test Case 5: No sources available (edge case)
    print("\n📊 Test Case 5: No sources available (edge case)")
    available = []
    normalized = normalize_scoring_weights(original_weights, available)
    print(f"Available sources: {available}")
    print(f"Normalized weights: {normalized}")
    assert normalized == {}, "Empty dict expected for no sources"
    print("✅ PASS: Empty dict returned for no sources")

    # Test Case 6: Custom weights with missing sources
    print("\n📊 Test Case 6: Custom weights with missing sources")
    custom_weights = {
        "rsi": 0.30,
        "beta": 0.15,
        "momentum": 0.25,
        "iv": 0.20,
        "forecast": 0.10,
    }
    available = [
        "rsi",
        "momentum",
        "iv",
    ]  # Missing beta and forecast (0.25 total weight)
    normalized = normalize_scoring_weights(custom_weights, available)
    print(f"Custom weights: {custom_weights}")
    print(f"Available sources: {available}")
    print(f"Normalized weights: {normalized}")
    print(f"Total weight: {sum(normalized.values()):.6f}")
    # Total available weight: 0.30 + 0.25 + 0.20 = 0.75
    # Expected: rsi = 0.30/0.75 = 0.4, momentum = 0.25/0.75 = 0.333, iv = 0.20/0.75 = 0.267
    assert abs(sum(normalized.values()) - 1.0) < 0.001, "Weights should sum to 1.0"
    assert (
        abs(normalized["rsi"] - 0.4) < 0.001
    ), f"Expected 0.4, got {normalized['rsi']}"
    print("✅ PASS: Custom weights properly normalized")

    print("\n🎉 All tests passed! Dynamic weight normalization is working correctly.")
    print("\n💡 Key Features Validated:")
    print("   ✓ Weights always sum to 1.0")
    print("   ✓ Proportional redistribution when data is missing")
    print("   ✓ Single source gets 100% weight")
    print("   ✓ Graceful handling of edge cases")
    print("   ✓ Maintains original proportions where possible")
    print("   ✓ Works with custom weight configurations")


def main():
    """Run weight normalization tests."""
    print("🚀 Starting Weight Normalization Tests...")
    try:
        test_normalize_scoring_weights()
        print("\n✅ All weight normalization tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
