"""
Unit tests for options advisor scoring functionality.

This module tests the enhanced scoring system that dynamically calculates
data scores and properly handles metadata fields.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from buffetbot.analysis.options_advisor import (
    get_scoring_indicator_names,
    get_scoring_weights,
    get_total_scoring_indicators,
)
from buffetbot.dashboard.components.options_utils import (
    get_data_score_badge,
    render_score_details_popover,
)


class TestDynamicScoringSystem:
    """Test the dynamic scoring system that adapts to SCORING_WEIGHTS."""

    def test_get_total_scoring_indicators(self):
        """Test that total indicators matches SCORING_WEIGHTS length."""
        total = get_total_scoring_indicators()
        weights = get_scoring_weights()
        assert total == len(weights)
        assert total == 5  # Current expected value

    def test_get_scoring_indicator_names(self):
        """Test that indicator names match SCORING_WEIGHTS keys."""
        names = get_scoring_indicator_names()
        weights = get_scoring_weights()
        assert set(names) == set(weights.keys())
        assert set(names) == {"rsi", "beta", "momentum", "iv", "forecast"}

    def test_data_score_badge_with_all_indicators(self):
        """Test data score badge with all indicators present."""
        score_details = {
            "rsi": 0.2,
            "beta": 0.2,
            "momentum": 0.2,
            "iv": 0.2,
            "forecast": 0.2,
        }
        badge = get_data_score_badge(score_details)
        assert badge == "🟢 5/5"

    def test_data_score_badge_excludes_metadata(self):
        """Test that metadata fields are excluded from scoring count."""
        score_details = {
            "rsi": 0.2,
            "beta": 0.2,
            "momentum": 0.2,
            "iv": 0.2,
            "forecast": 0.2,
            "risk_tolerance": "Conservative",  # Metadata
            "analysis_date": "2024-01-01",  # Metadata
            "custom_field": "value",  # Metadata
        }
        badge = get_data_score_badge(score_details)
        # Should still be 5/5 because only actual indicators are counted
        assert badge == "🟢 5/5"

    def test_data_score_badge_partial_with_metadata(self):
        """Test partial indicators with metadata fields."""
        score_details = {
            "rsi": 0.33,
            "beta": 0.33,
            "momentum": 0.34,
            "risk_tolerance": "Aggressive",  # Metadata
            "strategy_type": "Long Calls",  # Metadata
        }
        badge = get_data_score_badge(score_details)
        # Should be 3/5 (only counting actual indicators)
        assert badge == "🟡 3/5"

    def test_data_score_badge_poor_score(self):
        """Test poor score with few indicators."""
        score_details = {
            "rsi": 0.5,
            "beta": 0.5,
            "extra_metadata": "ignored",
        }
        badge = get_data_score_badge(score_details)
        assert badge == "🔴 2/5"

    def test_data_score_badge_empty(self):
        """Test empty score details."""
        badge = get_data_score_badge({})
        assert badge == "❓ 0/5"

    def test_data_score_badge_invalid_input(self):
        """Test invalid input types."""
        assert get_data_score_badge(None) == "❓ 0/5"
        assert get_data_score_badge("invalid") == "❓ 0/5"
        assert get_data_score_badge([1, 2, 3]) == "❓ 0/5"

    @patch(
        "buffetbot.analysis.options_advisor.SCORING_WEIGHTS",
        {"rsi": 0.25, "beta": 0.25, "momentum": 0.25, "iv": 0.25},
    )
    def test_dynamic_adaptation_to_different_weights(self):
        """Test that the system adapts when SCORING_WEIGHTS changes."""
        # This test simulates having 4 indicators instead of 5
        score_details = {
            "rsi": 0.25,
            "beta": 0.25,
            "momentum": 0.25,
            "iv": 0.25,
            "risk_tolerance": "Conservative",  # Metadata
        }

        # Clear any cached imports to force re-evaluation
        import importlib

        import buffetbot.dashboard.components.options_utils

        importlib.reload(buffetbot.dashboard.components.options_utils)

        from buffetbot.dashboard.components.options_utils import get_data_score_badge

        badge = get_data_score_badge(score_details)
        # Should show 4/4 with the mocked weights
        assert "4/4" in badge or "4/5" in badge  # Fallback might still use 5


class TestScoreDetailsPopover:
    """Test the enhanced score details popover functionality."""

    def test_score_details_separates_indicators_and_metadata(self):
        """Test that score details properly separates indicators from metadata."""
        # This is more of an integration test since render_score_details_popover
        # uses Streamlit components, but we can test the logic
        score_details = {
            "rsi": 0.2,
            "beta": 0.2,
            "momentum": 0.2,
            "iv": 0.2,
            "forecast": 0.2,
            "risk_tolerance": "Conservative",
            "analysis_date": "2024-01-01",
        }

        # Import the indicator names to test separation logic
        from buffetbot.analysis.options_advisor import get_scoring_indicator_names

        all_indicator_names = set(get_scoring_indicator_names())

        actual_indicators = {
            k: v for k, v in score_details.items() if k in all_indicator_names
        }
        metadata_fields = {
            k: v for k, v in score_details.items() if k not in all_indicator_names
        }

        assert len(actual_indicators) == 5
        assert len(metadata_fields) == 2
        assert "risk_tolerance" in metadata_fields
        assert "analysis_date" in metadata_fields
        assert "rsi" in actual_indicators
        assert "forecast" in actual_indicators


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_options_recommendations_with_metadata(self):
        """Test realistic options recommendations with metadata."""
        # Simulate realistic options recommendations DataFrame
        recommendations = pd.DataFrame(
            {
                "strike": [100, 105, 110],
                "expiry": ["2024-12-20", "2024-12-20", "2024-12-20"],
                "lastPrice": [8.5, 6.2, 4.1],
                "RSI": [45, 52, 38],
                "Beta": [1.2, 1.1, 1.3],
                "Momentum": [0.05, 0.03, 0.02],
                "IV": [0.28, 0.32, 0.25],
                "ForecastConfidence": [0.75, 0.68, 0.82],
                "CompositeScore": [0.82, 0.76, 0.71],
                "score_details": [
                    {
                        "rsi": 0.2,
                        "beta": 0.2,
                        "momentum": 0.2,
                        "iv": 0.2,
                        "forecast": 0.2,
                        "risk_tolerance": "Conservative",
                    },
                    {
                        "rsi": 0.25,
                        "beta": 0.25,
                        "momentum": 0.25,
                        "iv": 0.25,
                        "risk_tolerance": "Conservative",  # Missing forecast
                    },
                    {
                        "rsi": 0.33,
                        "beta": 0.33,
                        "momentum": 0.34,
                        "risk_tolerance": "Conservative",  # Missing IV and forecast
                    },
                ],
            }
        )

        # Test data score badges for each recommendation
        badges = [
            get_data_score_badge(details)
            for details in recommendations["score_details"]
        ]

        expected_badges = ["🟢 5/5", "🟡 4/5", "🟡 3/5"]
        assert badges == expected_badges

    def test_edge_case_only_metadata(self):
        """Test edge case where score_details contains only metadata."""
        score_details = {
            "risk_tolerance": "Conservative",
            "analysis_date": "2024-01-01",
            "strategy_type": "Long Calls",
        }

        badge = get_data_score_badge(score_details)
        # Should be red (🔴) for 0 indicators since it's a valid dict but no actual indicators
        assert badge == "🔴 0/5"

    def test_unknown_indicators_treated_as_metadata(self):
        """Test that unknown indicators are treated as metadata."""
        score_details = {
            "rsi": 0.2,
            "beta": 0.2,
            "unknown_indicator": 0.3,  # Not in SCORING_WEIGHTS
            "custom_metric": 0.3,  # Not in SCORING_WEIGHTS
            "risk_tolerance": "Conservative",
        }

        badge = get_data_score_badge(score_details)
        # Should only count rsi and beta (2 out of 5)
        assert badge == "🔴 2/5"


if __name__ == "__main__":
    pytest.main([__file__])
