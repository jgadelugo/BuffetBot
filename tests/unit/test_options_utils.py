"""Unit tests for dashboard.components.options_utils module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from buffetbot.dashboard.components.options_utils import (
    check_for_partial_data,
    create_styling_functions,
    get_data_score_badge,
)


class TestGetDataScoreBadge:
    """Test cases for get_data_score_badge function."""

    def test_full_score_badge(self):
        """Test badge for complete scoring data."""
        score_details = {
            "rsi": 0.2,
            "beta": 0.2,
            "momentum": 0.2,
            "iv": 0.2,
            "forecast": 0.2,
        }
        # Now dynamically gets total from SCORING_WEIGHTS (5 indicators)
        assert get_data_score_badge(score_details) == "🟢 5/5"

    def test_good_score_badge(self):
        """Test badge for good scoring data (3-4 indicators)."""
        score_details_4 = {"rsi": 0.25, "beta": 0.25, "momentum": 0.25, "iv": 0.25}
        assert get_data_score_badge(score_details_4) == "🟡 4/5"

        score_details_3 = {"rsi": 0.33, "beta": 0.33, "momentum": 0.34}
        assert get_data_score_badge(score_details_3) == "🟡 3/5"

    def test_poor_score_badge(self):
        """Test badge for poor scoring data (1-2 indicators)."""
        score_details_2 = {"rsi": 0.5, "beta": 0.5}
        assert get_data_score_badge(score_details_2) == "🔴 2/5"

        score_details_1 = {"rsi": 1.0}
        assert get_data_score_badge(score_details_1) == "🔴 1/5"

    def test_empty_score_details(self):
        """Test badge for empty score details."""
        assert get_data_score_badge({}) == "❓ 0/5"
        assert get_data_score_badge(None) == "❓ 0/5"

    def test_invalid_score_details(self):
        """Test badge for invalid score details."""
        assert get_data_score_badge("invalid") == "❓ 0/5"
        assert get_data_score_badge([1, 2, 3]) == "❓ 0/5"
        assert get_data_score_badge(123) == "❓ 0/5"

    def test_score_details_with_metadata(self):
        """Test badge when score_details contains metadata fields like risk_tolerance."""
        # This scenario matches the user's issue where score_details has 6 items but only 5 are actual indicators
        score_details_with_metadata = {
            "rsi": 0.2,
            "beta": 0.2,
            "momentum": 0.2,
            "iv": 0.2,
            "forecast": 0.2,
            "risk_tolerance": "Conservative",  # This should be excluded from count
        }
        # Should still show 5/5 because risk_tolerance is metadata, not a scoring indicator
        assert get_data_score_badge(score_details_with_metadata) == "🟢 5/5"

    def test_partial_score_with_metadata(self):
        """Test badge with partial indicators plus metadata."""
        score_details_partial_with_metadata = {
            "rsi": 0.25,
            "beta": 0.25,
            "momentum": 0.25,
            "iv": 0.25,
            "risk_tolerance": "Aggressive",  # Metadata field
            "analysis_date": "2024-01-01",  # Another metadata field
        }
        # Should show 4/5 (only counting actual scoring indicators)
        assert get_data_score_badge(score_details_partial_with_metadata) == "🟡 4/5"


class TestCheckForPartialData:
    """Test cases for check_for_partial_data function."""

    def test_no_partial_data(self):
        """Test when all recommendations have complete data."""
        recommendations = pd.DataFrame(
            {
                "strike": [100, 110, 120],
                "score_details": [
                    {
                        "rsi": 0.2,
                        "beta": 0.2,
                        "momentum": 0.2,
                        "iv": 0.2,
                        "forecast": 0.2,
                    },
                    {
                        "rsi": 0.2,
                        "beta": 0.2,
                        "momentum": 0.2,
                        "iv": 0.2,
                        "forecast": 0.2,
                    },
                    {
                        "rsi": 0.2,
                        "beta": 0.2,
                        "momentum": 0.2,
                        "iv": 0.2,
                        "forecast": 0.2,
                    },
                ],
            }
        )
        assert check_for_partial_data(recommendations) == False

    def test_partial_data_present(self):
        """Test when some recommendations have partial data."""
        recommendations = pd.DataFrame(
            {
                "strike": [100, 110, 120],
                "score_details": [
                    {
                        "rsi": 0.2,
                        "beta": 0.2,
                        "momentum": 0.2,
                        "iv": 0.2,
                        "forecast": 0.2,
                    },  # Complete
                    {"rsi": 0.33, "beta": 0.33, "momentum": 0.34},  # Partial (3/5)
                    {
                        "rsi": 0.2,
                        "beta": 0.2,
                        "momentum": 0.2,
                        "iv": 0.2,
                        "forecast": 0.2,
                    },  # Complete
                ],
            }
        )
        assert check_for_partial_data(recommendations) == True

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame()
        assert check_for_partial_data(empty_df) == False

    def test_missing_score_details_column(self):
        """Test DataFrame without score_details column."""
        recommendations = pd.DataFrame(
            {"strike": [100, 110, 120], "price": [5.0, 6.0, 7.0]}
        )
        assert check_for_partial_data(recommendations) == False

    def test_invalid_score_details(self):
        """Test with invalid score_details values."""
        recommendations = pd.DataFrame(
            {
                "strike": [100, 110, 120],
                "score_details": [
                    {
                        "rsi": 0.2,
                        "beta": 0.2,
                        "momentum": 0.2,
                        "iv": 0.2,
                        "forecast": 0.2,
                    },
                    "invalid_dict",  # Invalid type
                    None,  # None value
                ],
            }
        )
        # Should not crash and should handle gracefully
        result = check_for_partial_data(recommendations)
        assert isinstance(result, bool)

    def test_mixed_completeness(self):
        """Test mixed completeness levels."""
        recommendations = pd.DataFrame(
            {
                "strike": [100, 110, 120, 130],
                "score_details": [
                    {
                        "rsi": 0.2,
                        "beta": 0.2,
                        "momentum": 0.2,
                        "iv": 0.2,
                        "forecast": 0.2,
                    },  # 5/5
                    {
                        "rsi": 0.25,
                        "beta": 0.25,
                        "momentum": 0.25,
                        "iv": 0.25,
                    },  # 4/5 - partial
                    {"rsi": 0.5, "beta": 0.5},  # 2/5 - partial
                    {},  # 0/5 - partial
                ],
            }
        )
        assert check_for_partial_data(recommendations) == True


class TestCreateStylingFunctions:
    """Test cases for create_styling_functions function."""

    def setUp(self):
        """Set up styling functions for testing."""
        (
            self.highlight_rsi,
            self.highlight_score,
            self.highlight_iv,
            self.highlight_forecast,
        ) = create_styling_functions()

    def test_rsi_styling(self):
        """Test RSI value styling."""
        self.setUp()

        # Test overbought (>70)
        result = self.highlight_rsi(75)
        assert "background-color: #ffcdd2" in result
        assert "color: #d32f2f" in result

        # Test oversold (<30)
        result = self.highlight_rsi(25)
        assert "background-color: #c8e6c9" in result
        assert "color: #2e7d32" in result

        # Test neutral (30-70)
        result = self.highlight_rsi(50)
        assert "background-color: #fff3e0" in result
        assert "color: #f57c00" in result

        # Test invalid value
        result = self.highlight_rsi("invalid")
        assert result == ""

    def test_score_styling(self):
        """Test composite score styling."""
        self.setUp()

        # Test high score (>=0.7)
        result = self.highlight_score(0.8)
        assert "background-color: #c8e6c9" in result
        assert "color: #2e7d32" in result
        assert "font-weight: bold" in result

        # Test medium score (0.5-0.7)
        result = self.highlight_score(0.6)
        assert "background-color: #fff3e0" in result
        assert "color: #f57c00" in result
        assert "font-weight: bold" in result

        # Test low score (<0.5)
        result = self.highlight_score(0.3)
        assert "background-color: #ffcdd2" in result
        assert "color: #d32f2f" in result

        # Test invalid value
        result = self.highlight_score("invalid")
        assert result == ""

    def test_iv_styling(self):
        """Test implied volatility styling."""
        self.setUp()

        # Test low IV (<=30%)
        result = self.highlight_iv("25%")
        assert "background-color: #c8e6c9" in result
        assert "color: #2e7d32" in result

        # Test medium IV (30-50%)
        result = self.highlight_iv("40%")
        assert "background-color: #fff3e0" in result
        assert "color: #f57c00" in result

        # Test high IV (>50%)
        result = self.highlight_iv("65%")
        assert "background-color: #ffcdd2" in result
        assert "color: #d32f2f" in result

        # Test invalid value
        result = self.highlight_iv("invalid")
        assert result == ""

    def test_forecast_styling(self):
        """Test forecast confidence styling."""
        self.setUp()

        # Test high confidence (>=70%)
        result = self.highlight_forecast("80%")
        assert "background-color: #c8e6c9" in result
        assert "color: #2e7d32" in result

        # Test medium confidence (40-70%)
        result = self.highlight_forecast("55%")
        assert "background-color: #fff3e0" in result
        assert "color: #f57c00" in result

        # Test low confidence (<40%)
        result = self.highlight_forecast("25%")
        assert "background-color: #ffcdd2" in result
        assert "color: #d32f2f" in result

        # Test invalid value
        result = self.highlight_forecast("invalid")
        assert result == ""

    def test_edge_case_values(self):
        """Test edge case values for all styling functions."""
        self.setUp()

        # Test boundary values for RSI
        assert self.highlight_rsi(30) != ""  # Boundary between oversold and neutral
        assert self.highlight_rsi(70) != ""  # Boundary between neutral and overbought

        # Test boundary values for score
        assert self.highlight_score(0.5) != ""  # Boundary between low and medium
        assert self.highlight_score(0.7) != ""  # Boundary between medium and high

        # Test zero values
        assert self.highlight_rsi(0) != ""
        assert self.highlight_score(0) != ""
        assert self.highlight_iv("0%") != ""
        assert self.highlight_forecast("0%") != ""


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_full_options_analysis_workflow(self):
        """Test a complete options analysis workflow with the utilities."""
        # Create realistic options recommendations data
        recommendations = pd.DataFrame(
            {
                "strike": [100, 105, 110, 115, 120],
                "expiry": [
                    "2024-12-20",
                    "2024-12-20",
                    "2024-12-20",
                    "2024-12-20",
                    "2024-12-20",
                ],
                "lastPrice": [8.5, 6.2, 4.1, 2.8, 1.9],
                "RSI": [45, 52, 38, 67, 73],
                "IV": [0.28, 0.32, 0.25, 0.45, 0.52],
                "Momentum": [0.65, 0.72, 0.58, 0.81, 0.69],
                "ForecastConfidence": [0.75, 0.68, 0.82, 0.55, 0.48],
                "CompositeScore": [0.82, 0.76, 0.71, 0.63, 0.48],
                "score_details": [
                    {
                        "rsi": 0.2,
                        "beta": 0.2,
                        "momentum": 0.2,
                        "iv": 0.2,
                        "forecast": 0.2,
                    },
                    {
                        "rsi": 0.2,
                        "beta": 0.2,
                        "momentum": 0.2,
                        "iv": 0.2,
                        "forecast": 0.2,
                    },
                    {
                        "rsi": 0.25,
                        "beta": 0.25,
                        "momentum": 0.25,
                        "iv": 0.25,
                    },  # Missing forecast
                    {
                        "rsi": 0.33,
                        "beta": 0.33,
                        "momentum": 0.34,
                    },  # Missing IV and forecast
                    {"rsi": 0.5, "beta": 0.5},  # Only RSI and beta
                ],
            }
        )

        # Test partial data detection
        has_partial = check_for_partial_data(recommendations)
        assert has_partial == True

        # Test data score badges for each row
        badges = [
            get_data_score_badge(details)
            for details in recommendations["score_details"]
        ]
        expected_badges = ["🟢 5/5", "🟢 5/5", "🟡 4/5", "🟡 3/5", "🔴 2/5"]
        assert badges == expected_badges

        # Test styling functions work with the data
        (
            highlight_rsi,
            highlight_score,
            highlight_iv,
            highlight_forecast,
        ) = create_styling_functions()

        # Test RSI styling on actual data
        rsi_styles = [highlight_rsi(rsi) for rsi in recommendations["RSI"]]
        assert all(style != "" for style in rsi_styles)

        # Test score styling on actual data
        score_styles = [
            highlight_score(score) for score in recommendations["CompositeScore"]
        ]
        assert all(style != "" for style in score_styles)

        # Test IV styling (convert to percentage format first)
        iv_percentage_strings = [f"{iv:.0%}" for iv in recommendations["IV"]]
        iv_styles = [highlight_iv(iv_str) for iv_str in iv_percentage_strings]
        assert all(style != "" for style in iv_styles)

        # Test forecast styling (convert to percentage format first)
        forecast_percentage_strings = [
            f"{fc:.0%}" for fc in recommendations["ForecastConfidence"]
        ]
        forecast_styles = [
            highlight_forecast(fc_str) for fc_str in forecast_percentage_strings
        ]
        assert all(style != "" for style in forecast_styles)

    def test_empty_recommendations_handling(self):
        """Test handling of empty recommendations."""
        empty_df = pd.DataFrame()

        # Should not cause errors
        assert check_for_partial_data(empty_df) == False

        # Styling functions should handle empty data gracefully
        (
            highlight_rsi,
            highlight_score,
            highlight_iv,
            highlight_forecast,
        ) = create_styling_functions()

        # These should not crash
        assert highlight_rsi("") == ""
        assert highlight_score("") == ""
        assert highlight_iv("") == ""
        assert highlight_forecast("") == ""

    def test_malformed_data_handling(self):
        """Test handling of malformed or unexpected data."""
        malformed_recommendations = pd.DataFrame(
            {
                "strike": [100, 105],
                "score_details": [
                    {"invalid_key": 0.5},  # Unexpected keys
                    ["not", "a", "dict"],  # Wrong type
                ],
            }
        )

        # Should handle gracefully without crashing
        result = check_for_partial_data(malformed_recommendations)
        assert isinstance(result, bool)

        # Badge generation should handle malformed data
        badges = [
            get_data_score_badge(details)
            for details in malformed_recommendations["score_details"]
        ]
        assert all("❓" in badge or "🔴" in badge for badge in badges)


if __name__ == "__main__":
    pytest.main([__file__])
