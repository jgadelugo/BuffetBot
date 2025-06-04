"""Test suite for TotalScore column access fix."""

from unittest.mock import patch

import pandas as pd
import pytest

from buffetbot.dashboard.views.options_advisor import safe_get_score_column


class TestTotalScoreColumnSafety:
    """Test cases to verify the TotalScore column access fix."""

    def test_safe_get_score_column_with_totalscore(self):
        """Test safe_get_score_column with TotalScore column."""
        df = pd.DataFrame(
            {
                "Strike": [100, 105],
                "TotalScore": [0.85, 0.78],
                "Expiry": ["2024-01-15", "2024-02-15"],
            }
        )

        col_name, values = safe_get_score_column(df)

        assert col_name == "TotalScore"
        assert list(values) == [0.85, 0.78]

    def test_safe_get_score_column_with_compositescore(self):
        """Test safe_get_score_column with CompositeScore column."""
        df = pd.DataFrame(
            {
                "Strike": [100, 105],
                "CompositeScore": [0.92, 0.87],
                "Expiry": ["2024-01-15", "2024-02-15"],
            }
        )

        col_name, values = safe_get_score_column(df)

        assert col_name == "CompositeScore"
        assert list(values) == [0.92, 0.87]

    def test_safe_get_score_column_priority_order(self):
        """Test that TotalScore takes priority over CompositeScore."""
        df = pd.DataFrame(
            {
                "Strike": [100, 105],
                "TotalScore": [0.85, 0.78],
                "CompositeScore": [0.92, 0.87],
                "Expiry": ["2024-01-15", "2024-02-15"],
            }
        )

        col_name, values = safe_get_score_column(df)

        # TotalScore should have priority
        assert col_name == "TotalScore"
        assert list(values) == [0.85, 0.78]

    def test_safe_get_score_column_with_finalscore(self):
        """Test safe_get_score_column with FinalScore column."""
        df = pd.DataFrame(
            {
                "Strike": [100, 105],
                "FinalScore": [0.95, 0.89],
                "Expiry": ["2024-01-15", "2024-02-15"],
            }
        )

        col_name, values = safe_get_score_column(df)

        assert col_name == "FinalScore"
        assert list(values) == [0.95, 0.89]

    def test_safe_get_score_column_with_generic_score(self):
        """Test safe_get_score_column with generic Score column."""
        df = pd.DataFrame(
            {
                "Strike": [100, 105],
                "Score": [0.88, 0.82],
                "Expiry": ["2024-01-15", "2024-02-15"],
            }
        )

        col_name, values = safe_get_score_column(df)

        assert col_name == "Score"
        assert list(values) == [0.88, 0.82]

    def test_safe_get_score_column_no_score_columns(self):
        """Test safe_get_score_column with no score columns."""
        df = pd.DataFrame(
            {
                "Strike": [100, 105],
                "Premium": [5.50, 6.25],
                "Expiry": ["2024-01-15", "2024-02-15"],
            }
        )

        col_name, values = safe_get_score_column(df)

        assert col_name is None
        assert values == 0.0

    def test_safe_get_score_column_empty_dataframe(self):
        """Test safe_get_score_column with empty DataFrame."""
        df = pd.DataFrame()

        col_name, values = safe_get_score_column(df)

        assert col_name is None
        assert values == 0.0

    def test_safe_get_score_column_none_input(self):
        """Test safe_get_score_column with None input."""
        col_name, values = safe_get_score_column(None)

        assert col_name is None
        assert values == 0.0

    def test_safe_get_score_column_custom_fallback(self):
        """Test safe_get_score_column with custom fallback value."""
        df = pd.DataFrame({"Strike": [100, 105], "Premium": [5.50, 6.25]})

        col_name, values = safe_get_score_column(df, fallback_score=0.5)

        assert col_name is None
        assert values == 0.5

    def test_safe_get_score_column_logging(self):
        """Test that appropriate warnings are logged."""
        df = pd.DataFrame({"Strike": [100, 105], "Premium": [5.50, 6.25]})

        with patch("buffetbot.dashboard.views.options_advisor.logger") as mock_logger:
            safe_get_score_column(df)

            # Should log warning about missing score columns
            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "No score columns found" in warning_call
            assert "Available columns:" in warning_call

    def test_safe_get_score_column_debug_logging(self):
        """Test that debug logging works when score column is found."""
        df = pd.DataFrame({"Strike": [100, 105], "CompositeScore": [0.85, 0.78]})

        with patch("buffetbot.dashboard.views.options_advisor.logger") as mock_logger:
            safe_get_score_column(df)

            # Should log debug message about using score column
            mock_logger.debug.assert_called_once_with(
                "Using score column: CompositeScore"
            )

    def test_multiple_score_columns_priority(self):
        """Test priority order with multiple score columns present."""
        df = pd.DataFrame(
            {
                "Strike": [100, 105],
                "Score": [0.70, 0.65],  # Priority 4
                "FinalScore": [0.80, 0.75],  # Priority 3
                "CompositeScore": [0.85, 0.78],  # Priority 2
                "TotalScore": [0.90, 0.83],  # Priority 1 (highest)
            }
        )

        col_name, values = safe_get_score_column(df)

        # Should use TotalScore (highest priority)
        assert col_name == "TotalScore"
        assert list(values) == [0.90, 0.83]

    def test_real_world_scenario_compositescore_only(self):
        """Test real-world scenario where only CompositeScore exists."""
        # This simulates the actual DataFrame structure from options analysis
        df = pd.DataFrame(
            {
                "strike": [100, 105, 110],
                "CompositeScore": [0.85, 0.78, 0.72],
                "expiry": ["2024-01-15", "2024-02-15", "2024-03-15"],
                "RSI": [45.2, 52.1, 38.9],
                "Beta": [1.2, 1.1, 1.3],
                "IV": ["25%", "28%", "30%"],
            }
        )

        col_name, values = safe_get_score_column(df)

        assert col_name == "CompositeScore"
        assert list(values) == [0.85, 0.78, 0.72]
        assert len(values) == 3

    def test_backwards_compatibility(self):
        """Test that the fix maintains backwards compatibility."""
        # Test old-style DataFrame with TotalScore
        old_df = pd.DataFrame({"Strike": [100, 105], "TotalScore": [0.85, 0.78]})

        # Test new-style DataFrame with CompositeScore
        new_df = pd.DataFrame({"Strike": [100, 105], "CompositeScore": [0.85, 0.78]})

        old_col, old_values = safe_get_score_column(old_df)
        new_col, new_values = safe_get_score_column(new_df)

        # Both should work and return the same values
        assert list(old_values) == list(new_values)
        assert old_col == "TotalScore"
        assert new_col == "CompositeScore"


if __name__ == "__main__":
    # Simple test runner
    test_suite = TestTotalScoreColumnSafety()

    print("🔍 Testing TotalScore column access fix...")

    try:
        test_suite.test_safe_get_score_column_with_totalscore()
        print("✅ TotalScore column test passed")

        test_suite.test_safe_get_score_column_with_compositescore()
        print("✅ CompositeScore column test passed")

        test_suite.test_safe_get_score_column_priority_order()
        print("✅ Priority order test passed")

        test_suite.test_safe_get_score_column_no_score_columns()
        print("✅ No score columns test passed")

        test_suite.test_safe_get_score_column_empty_dataframe()
        print("✅ Empty DataFrame test passed")

        test_suite.test_safe_get_score_column_none_input()
        print("✅ None input test passed")

        test_suite.test_multiple_score_columns_priority()
        print("✅ Multiple columns priority test passed")

        test_suite.test_real_world_scenario_compositescore_only()
        print("✅ Real-world scenario test passed")

        test_suite.test_backwards_compatibility()
        print("✅ Backwards compatibility test passed")

        print("🎉 All TotalScore column fix tests passed!")
        print("🔧 TotalScore column error should be resolved in production")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
