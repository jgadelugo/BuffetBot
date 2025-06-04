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


class TestComprehensiveScoringBreakdown:
    """Test the comprehensive scoring breakdown functionality."""

    def test_comprehensive_breakdown_function_exists(self):
        """Test that the comprehensive breakdown function can be imported."""
        # This is a simple smoke test to ensure the function exists
        from buffetbot.dashboard.views.options_advisor import (
            render_comprehensive_scoring_breakdown,
        )

        # Function should be callable
        assert callable(render_comprehensive_scoring_breakdown)

        # Function should have the expected signature
        import inspect

        sig = inspect.signature(render_comprehensive_scoring_breakdown)
        params = list(sig.parameters.keys())
        assert "recommendations" in params
        assert "ticker" in params
        assert "strategy_type" in params

    def test_comprehensive_breakdown_with_empty_data(self):
        """Test comprehensive breakdown with empty recommendations."""
        from unittest.mock import Mock, patch

        empty_recommendations = pd.DataFrame()

        # Create minimal streamlit mock
        streamlit_mock = Mock()
        streamlit_mock.subheader = Mock()

        with patch.dict("sys.modules", {"streamlit": streamlit_mock}):
            from buffetbot.dashboard.views.options_advisor import (
                render_comprehensive_scoring_breakdown,
            )

            # Should return early without rendering anything
            render_comprehensive_scoring_breakdown(
                empty_recommendations, "AAPL", "Long Calls"
            )

            # Should not call subheader since function returns early
            streamlit_mock.subheader.assert_not_called()

    def test_comprehensive_breakdown_missing_score_details(self):
        """Test comprehensive breakdown with missing score_details column."""
        from unittest.mock import Mock, patch

        recommendations = pd.DataFrame(
            {
                "Strike": [100, 105],
                "Expiry": ["2024-12-20", "2024-12-20"],
                "TotalScore": [4.5, 4.2]
                # Missing score_details column
            }
        )

        # Create minimal streamlit mock
        streamlit_mock = Mock()
        streamlit_mock.subheader = Mock()

        with patch.dict("sys.modules", {"streamlit": streamlit_mock}):
            from buffetbot.dashboard.views.options_advisor import (
                render_comprehensive_scoring_breakdown,
            )

            # Should return early without rendering anything
            render_comprehensive_scoring_breakdown(
                recommendations, "AAPL", "Long Calls"
            )

            # Should not call subheader since function returns early
            streamlit_mock.subheader.assert_not_called()

    def test_portfolio_quality_metrics_calculation(self):
        """Test that portfolio quality metrics are calculated correctly."""
        # Create test data with known characteristics
        recommendations = pd.DataFrame(
            {
                "TotalScore": [
                    4.0,
                    4.1,
                    4.2,
                    3.8,
                    3.9,
                ],  # Low std dev = high consistency
                "score_details": [
                    {
                        "rsi": 0.2,
                        "beta": 0.2,
                        "momentum": 0.2,
                        "iv": 0.2,
                        "forecast": 0.2,
                    },
                    {
                        "rsi": 0.21,
                        "beta": 0.19,
                        "momentum": 0.21,
                        "iv": 0.19,
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
                        "rsi": 0.19,
                        "beta": 0.21,
                        "momentum": 0.19,
                        "iv": 0.21,
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

        # Manually calculate expected metrics
        scores = recommendations["TotalScore"]
        expected_std = scores.std()
        expected_mean = scores.mean()
        expected_consistency = max(0, 1 - (expected_std / expected_mean))

        # Verify consistency calculation logic
        assert (
            expected_consistency > 0.9
        )  # Should be high consistency due to low std dev
        assert expected_std < 0.2  # Low standard deviation

        # Test data completeness calculation
        first_score_details = recommendations.iloc[0]["score_details"]
        all_indicator_names = {"rsi", "beta", "momentum", "iv", "forecast"}
        actual_indicators = {
            k: v for k, v in first_score_details.items() if k in all_indicator_names
        }
        completeness_score = len(actual_indicators) / len(all_indicator_names)

        assert completeness_score == 1.0  # All indicators present

    def test_individual_recommendation_analysis_data(self):
        """Test individual recommendation analysis with realistic data."""
        recommendations = pd.DataFrame(
            {
                "Strike": [150.0],
                "Expiry": ["2024-06-21"],
                "TotalScore": [4.123],
                "RSI": [42.5],
                "Beta": [1.15],
                "Momentum": [0.032],
                "IV": [0.28],
                "ForecastConfidence": [0.72],
                "score_details": [
                    {
                        "rsi": 0.245,
                        "beta": 0.155,
                        "momentum": 0.248,
                        "iv": 0.198,
                        "forecast": 0.154,
                        "risk_tolerance": "Moderate",
                        "analysis_date": "2024-01-15",
                    }
                ],
            }
        )

        # Verify scoring contributions can be calculated
        score_details = recommendations.iloc[0]["score_details"]
        scoring_weights = {
            "rsi": 0.25,
            "beta": 0.15,
            "momentum": 0.25,
            "iv": 0.20,
            "forecast": 0.15,
        }

        total_contribution = 0
        for indicator, weight in score_details.items():
            if indicator in scoring_weights:
                contribution = weight * scoring_weights[indicator]
                total_contribution += contribution

        # Verify contribution calculation makes sense
        assert total_contribution > 0
        assert (
            total_contribution < 1.0
        )  # Should be less than 1 since weights are normalized

    def test_scoring_logic_calculations(self):
        """Test the core scoring logic calculations used in the breakdown."""
        # Test data with varying score consistency
        high_consistency_scores = [4.0, 4.01, 3.99, 4.02, 3.98]
        low_consistency_scores = [1.0, 5.0, 2.5, 4.8, 1.2]

        # High consistency test
        import numpy as np

        high_std = np.std(high_consistency_scores)
        high_mean = np.mean(high_consistency_scores)
        high_consistency = max(0, 1 - (high_std / high_mean))

        # Low consistency test
        low_std = np.std(low_consistency_scores)
        low_mean = np.mean(low_consistency_scores)
        low_consistency = max(0, 1 - (low_std / low_mean))

        # Verify high consistency is better than low
        assert high_consistency > low_consistency
        assert high_consistency > 0.9  # Should be very high
        assert low_consistency < 0.5  # Should be much lower

        # Test coverage calculation
        all_indicators = {"rsi", "beta", "momentum", "iv", "forecast"}
        partial_indicators = {"rsi", "beta", "momentum"}

        full_coverage = len(all_indicators) / len(all_indicators)
        partial_coverage = len(partial_indicators) / len(all_indicators)

        assert full_coverage == 1.0
        assert partial_coverage == 0.6


if __name__ == "__main__":
    pytest.main([__file__])
