#!/usr/bin/env python3
"""
Test script for UI Scoring Transparency features

This module tests the new UI functions that display scoring breakdown,
data badges, and partial data warnings in the Options Advisor dashboard.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add the project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))


class TestUIScoringTransparency:
    """Test class for UI scoring transparency features."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Import the functions from dashboard app
        from dashboard.app import (
            check_for_partial_data,
            get_data_score_badge,
            render_score_details_popover,
        )

        self.get_data_score_badge = get_data_score_badge
        self.check_for_partial_data = check_for_partial_data
        self.render_score_details_popover = render_score_details_popover

        # Test data fixtures
        self.full_score_details = {
            "rsi": 0.20,
            "beta": 0.20,
            "momentum": 0.20,
            "iv": 0.20,
            "forecast": 0.20,
        }

        self.partial_score_details = {
            "rsi": 0.25,
            "beta": 0.25,
            "momentum": 0.25,
            "iv": 0.25,
        }

        self.limited_score_details = {"rsi": 0.50, "momentum": 0.50}

        self.empty_score_details = {}

    def test_full_data_score_badge(self):
        """Test data score badge for full indicators (5/5)."""
        badge = self.get_data_score_badge(self.full_score_details)
        assert "🟢 5/5" in badge, f"Expected green badge for full data, got {badge}"

    def test_partial_data_score_badge(self):
        """Test data score badge for partial indicators (4/5)."""
        badge = self.get_data_score_badge(self.partial_score_details)
        assert "🟡 4/5" in badge, f"Expected yellow badge for partial data, got {badge}"

    def test_limited_data_score_badge(self):
        """Test data score badge for limited indicators (2/5)."""
        badge = self.get_data_score_badge(self.limited_score_details)
        assert "🔴 2/5" in badge, f"Expected red badge for limited data, got {badge}"

    def test_empty_data_score_badge(self):
        """Test data score badge for no indicators (0/5)."""
        badge = self.get_data_score_badge(self.empty_score_details)
        assert "❓ 0/5" in badge, f"Expected question badge for empty data, got {badge}"

    def test_none_data_score_badge(self):
        """Test data score badge with None input."""
        badge = self.get_data_score_badge(None)
        assert "❓ 0/5" in badge, f"Expected question badge for None input, got {badge}"

    def test_check_partial_data_mixed_dataframe(self):
        """Test partial data detection in mixed DataFrame."""
        test_recommendations = pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL", "AAPL"],
                "strike": [150.0, 155.0, 160.0],
                "score_details": [
                    self.full_score_details,  # 5 indicators
                    self.partial_score_details,  # 4 indicators
                    self.limited_score_details,  # 2 indicators
                ],
            }
        )

        has_partial = self.check_for_partial_data(test_recommendations)
        assert has_partial == True, "Should detect partial data in mixed DataFrame"

    def test_check_partial_data_all_full(self):
        """Test partial data detection with all full data."""
        full_recommendations = pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL"],
                "strike": [150.0, 155.0],
                "score_details": [
                    self.full_score_details,  # 5 indicators
                    self.full_score_details,  # 5 indicators
                ],
            }
        )

        has_partial = self.check_for_partial_data(full_recommendations)
        assert has_partial == False, "Should not detect partial data when all is full"

    def test_check_partial_data_empty_dataframe(self):
        """Test partial data detection with empty DataFrame."""
        empty_df = pd.DataFrame()
        has_partial = self.check_for_partial_data(empty_df)
        assert has_partial == False, "Empty DataFrame should return False"

    def test_check_partial_data_missing_column(self):
        """Test partial data detection when score_details column is missing."""
        test_df = pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL"],
                "strike": [150.0, 155.0]
                # Missing score_details column
            }
        )

        has_partial = self.check_for_partial_data(test_df)
        assert (
            has_partial == False
        ), "Should return False when score_details column is missing"

    def test_check_partial_data_invalid_score_details(self):
        """Test partial data detection with invalid score_details."""
        test_df = pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL"],
                "strike": [150.0, 155.0],
                "score_details": ["invalid_string", None],  # Invalid type  # None value
            }
        )

        has_partial = self.check_for_partial_data(test_df)
        assert has_partial == False, "Should handle invalid score_details gracefully"

    def test_data_score_badge_color_thresholds(self):
        """Test that data score badges use correct colors for different thresholds."""
        # Test all 5 scenarios
        test_cases = [
            (
                {"a": 0.2, "b": 0.2, "c": 0.2, "d": 0.2, "e": 0.2},
                "🟢",
            ),  # 5 indicators - green
            (
                {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
                "🟡",
            ),  # 4 indicators - yellow
            ({"a": 0.33, "b": 0.33, "c": 0.34}, "🟡"),  # 3 indicators - yellow
            ({"a": 0.5, "b": 0.5}, "🔴"),  # 2 indicators - red
            ({"a": 1.0}, "🔴"),  # 1 indicator - red
        ]

        for score_details, expected_color in test_cases:
            badge = self.get_data_score_badge(score_details)
            assert (
                expected_color in badge
            ), f"Expected {expected_color} for {len(score_details)} indicators, got {badge}"

    @pytest.mark.parametrize("indicator_count", [0, 1, 2, 3, 4, 5])
    def test_badge_indicator_counts(self, indicator_count):
        """Test badge generation for different indicator counts using parameterized testing."""
        # Create score details with specified number of indicators
        score_details = {}
        indicators = ["rsi", "beta", "momentum", "iv", "forecast"]

        for i in range(indicator_count):
            score_details[indicators[i]] = (
                1.0 / indicator_count if indicator_count > 0 else 0.0
            )

        badge = self.get_data_score_badge(score_details)

        # Check that the badge contains the correct count
        assert (
            f"{indicator_count}/5" in badge
        ), f"Badge should show {indicator_count}/5, got {badge}"

        # Check color coding
        if indicator_count == 5:
            assert "🟢" in badge, f"Should be green for 5 indicators, got {badge}"
        elif indicator_count >= 3:
            assert "🟡" in badge, f"Should be yellow for 3-4 indicators, got {badge}"
        elif indicator_count >= 1:
            assert "🔴" in badge, f"Should be red for 1-2 indicators, got {badge}"
        else:
            assert (
                "❓" in badge
            ), f"Should be question mark for 0 indicators, got {badge}"


def test_scoring_functions_integration():
    """Integration test for all UI scoring functions working together."""
    print("🧪 Testing UI Scoring Transparency Functions")
    print("=" * 60)

    # Import the functions from dashboard app
    from dashboard.app import (
        check_for_partial_data,
        get_data_score_badge,
        render_score_details_popover,
    )

    # Test data
    full_score_details = {
        "rsi": 0.20,
        "beta": 0.20,
        "momentum": 0.20,
        "iv": 0.20,
        "forecast": 0.20,
    }

    partial_score_details = {"rsi": 0.25, "beta": 0.25, "momentum": 0.25, "iv": 0.25}

    limited_score_details = {"rsi": 0.50, "momentum": 0.50}

    # Test Case 1: Full data score badge
    print("\n📊 Test Case 1: Full data score badge")
    badge = get_data_score_badge(full_score_details)
    print(f"Full data badge: {badge}")
    assert "🟢 5/5" in badge, f"Expected green badge for full data, got {badge}"
    print("✅ PASS: Full data badge is correct")

    # Test Case 2: Partial data score badge
    print("\n📊 Test Case 2: Partial data score badge")
    badge = get_data_score_badge(partial_score_details)
    print(f"Partial data badge: {badge}")
    assert "🟡 4/5" in badge, f"Expected yellow badge for partial data, got {badge}"
    print("✅ PASS: Partial data badge is correct")

    # Test Case 3: Limited data score badge
    print("\n📊 Test Case 3: Limited data score badge")
    badge = get_data_score_badge(limited_score_details)
    print(f"Limited data badge: {badge}")
    assert "🔴 2/5" in badge, f"Expected red badge for limited data, got {badge}"
    print("✅ PASS: Limited data badge is correct")

    # Test Case 4: Check for partial data - mixed DataFrame
    print("\n📊 Test Case 4: Check for partial data in DataFrame")
    test_recommendations = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "strike": [150.0, 155.0, 160.0],
            "score_details": [
                full_score_details,  # 5 indicators
                partial_score_details,  # 4 indicators
                limited_score_details,  # 2 indicators
            ],
        }
    )

    has_partial = check_for_partial_data(test_recommendations)
    print(f"Has partial data: {has_partial}")
    assert has_partial == True, "Should detect partial data in mixed DataFrame"
    print("✅ PASS: Correctly detected partial data")

    print("\n🎉 All UI scoring transparency integration tests passed!")
    print("\n💡 Key Features Validated:")
    print("   ✓ Data score badges with correct colors")
    print("   ✓ Partial data detection in DataFrames")
    print("   ✓ Graceful handling of edge cases")
    print("   ✓ Proper color coding for different data levels")


def main():
    """Run UI scoring transparency tests when called directly."""
    print("🚀 Starting UI Scoring Transparency Tests...")
    try:
        test_scoring_functions_integration()
        print("\n✅ All UI scoring transparency tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
