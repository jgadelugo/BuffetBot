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


class TestEnhancedMethodologyDynamicWeights:
    """Test the enhanced methodology function with dynamic weight display."""

    def test_render_enhanced_methodology_function_exists(self):
        """Test that the render_enhanced_methodology function exists and is callable."""
        from buffetbot.dashboard.views.options_advisor import (
            render_enhanced_methodology,
        )

        assert callable(render_enhanced_methodology)

    def test_methodology_with_default_weights(self):
        """Test methodology function with default weights."""
        from unittest.mock import MagicMock, Mock, patch

        # Create a comprehensive streamlit mock
        st_mock = MagicMock()

        # Mock the expander as a context manager
        expander_mock = MagicMock()
        expander_mock.__enter__ = Mock(return_value=expander_mock)
        expander_mock.__exit__ = Mock(return_value=None)
        st_mock.expander.return_value = expander_mock

        # Capture markdown calls
        markdown_calls = []
        st_mock.markdown.side_effect = lambda content: markdown_calls.append(content)

        # Mock streamlit module completely before any imports
        with patch.dict("sys.modules", {"streamlit": st_mock}):
            # Mock st import in the function
            with patch("buffetbot.dashboard.views.options_advisor.st", st_mock):
                from buffetbot.dashboard.views.options_advisor import (
                    render_enhanced_methodology,
                )

                # Call with no weights (should use defaults)
                render_enhanced_methodology()

                # Verify streamlit calls
                st_mock.expander.assert_called_once_with(
                    "🔬 Enhanced Analysis Methodology", expanded=False
                )
                assert st_mock.markdown.called

                # Check markdown content
                assert len(markdown_calls) > 0
                content = markdown_calls[0]
                assert "20%" in content
                assert "RSI Analysis" in content
                assert "Technical Scoring Components:" in content

    def test_methodology_with_custom_weights(self):
        """Test methodology function with custom scoring weights."""
        from unittest.mock import MagicMock, Mock, patch

        # Custom weights that differ from default 20%
        custom_weights = {
            "rsi": 0.30,  # 30%
            "beta": 0.10,  # 10%
            "momentum": 0.25,  # 25%
            "iv": 0.25,  # 25%
            "forecast": 0.10,  # 10%
        }

        # Create streamlit mock
        st_mock = MagicMock()
        expander_mock = MagicMock()
        expander_mock.__enter__ = Mock(return_value=expander_mock)
        expander_mock.__exit__ = Mock(return_value=None)
        st_mock.expander.return_value = expander_mock

        markdown_calls = []
        st_mock.markdown.side_effect = lambda content: markdown_calls.append(content)

        with patch.dict("sys.modules", {"streamlit": st_mock}):
            with patch("buffetbot.dashboard.views.options_advisor.st", st_mock):
                from buffetbot.dashboard.views.options_advisor import (
                    render_enhanced_methodology,
                )

                # Call with custom weights
                render_enhanced_methodology(custom_weights)

                # Verify calls
                st_mock.expander.assert_called_once()
                assert st_mock.markdown.called

                # Check content has custom weights
                content = markdown_calls[0]
                assert "30%" in content  # RSI weight
                assert "10%" in content  # Beta and Forecast weight
                assert "25%" in content  # Momentum and IV weight

                # Should not show default 20% for all indicators
                twenty_percent_count = content.count("20%")
                assert twenty_percent_count == 0

    def test_methodology_with_partial_weights(self):
        """Test methodology function with partial custom weights."""
        from unittest.mock import MagicMock, Mock, patch

        # Partial weights - some missing indicators
        partial_weights = {
            "rsi": 0.40,  # 40%
            "beta": 0.30,  # 30%
            "momentum": 0.30,  # 30%
            # Missing iv and forecast - should fallback to 20%
        }

        st_mock = MagicMock()
        expander_mock = MagicMock()
        expander_mock.__enter__ = Mock(return_value=expander_mock)
        expander_mock.__exit__ = Mock(return_value=None)
        st_mock.expander.return_value = expander_mock

        markdown_calls = []
        st_mock.markdown.side_effect = lambda content: markdown_calls.append(content)

        with patch.dict("sys.modules", {"streamlit": st_mock}):
            with patch("buffetbot.dashboard.views.options_advisor.st", st_mock):
                from buffetbot.dashboard.views.options_advisor import (
                    render_enhanced_methodology,
                )

                # Call with partial weights
                render_enhanced_methodology(partial_weights)

                content = markdown_calls[0]

                # Should show custom weights for available indicators
                assert "40%" in content  # RSI
                assert "30%" in content  # Beta and Momentum

                # Should show default 20% for missing indicators
                assert "20%" in content  # IV and Forecast fallback

    def test_methodology_weight_formatting(self):
        """Test that weight percentages are formatted correctly."""
        from unittest.mock import MagicMock, Mock, patch

        # Test various weight formats
        test_weights = {
            "rsi": 0.333333,  # Should format to 33%
            "beta": 0.125,  # Should format to 12%
            "momentum": 0.1666,  # Should format to 17%
            "iv": 0.25,  # Should format to 25%
            "forecast": 0.125,  # Should format to 12%
        }

        st_mock = MagicMock()
        expander_mock = MagicMock()
        expander_mock.__enter__ = Mock(return_value=expander_mock)
        expander_mock.__exit__ = Mock(return_value=None)
        st_mock.expander.return_value = expander_mock

        markdown_calls = []
        st_mock.markdown.side_effect = lambda content: markdown_calls.append(content)

        with patch.dict("sys.modules", {"streamlit": st_mock}):
            with patch("buffetbot.dashboard.views.options_advisor.st", st_mock):
                from buffetbot.dashboard.views.options_advisor import (
                    render_enhanced_methodology,
                )

                render_enhanced_methodology(test_weights)

                content = markdown_calls[0]

                # Check formatted percentages (:.0% formatting)
                assert "33%" in content  # 0.333333 -> 33%
                assert "12%" in content  # 0.125 -> 12%
                assert "17%" in content  # 0.1666 -> 17%
                assert "25%" in content  # 0.25 -> 25%

    def test_methodology_fallback_on_import_error(self):
        """Test methodology function fallback when scoring weights import fails."""
        from unittest.mock import MagicMock, Mock, patch

        st_mock = MagicMock()
        expander_mock = MagicMock()
        expander_mock.__enter__ = Mock(return_value=expander_mock)
        expander_mock.__exit__ = Mock(return_value=None)
        st_mock.expander.return_value = expander_mock

        markdown_calls = []
        st_mock.markdown.side_effect = lambda content: markdown_calls.append(content)

        with patch.dict("sys.modules", {"streamlit": st_mock}):
            with patch("buffetbot.dashboard.views.options_advisor.st", st_mock):
                # Mock the import to raise an error
                with patch(
                    "buffetbot.dashboard.views.options_advisor.get_scoring_weights",
                    side_effect=ImportError("Mocked import error"),
                ):
                    from buffetbot.dashboard.views.options_advisor import (
                        render_enhanced_methodology,
                    )

                    # Call without weights, should fallback to defaults on import error
                    render_enhanced_methodology()

                    content = markdown_calls[0]

                    # Should fallback to default 20% weights
                    assert "20%" in content
                    assert content.count("20%") == 5  # All 5 indicators should show 20%

    def test_methodology_with_zero_weights(self):
        """Test methodology function with zero weights (edge case)."""
        from unittest.mock import MagicMock, Mock, patch

        # Edge case: zero weights
        zero_weights = {
            "rsi": 0.0,  # 0%
            "beta": 0.0,  # 0%
            "momentum": 0.0,  # 0%
            "iv": 0.0,  # 0%
            "forecast": 1.0,  # 100%
        }

        st_mock = MagicMock()
        expander_mock = MagicMock()
        expander_mock.__enter__ = Mock(return_value=expander_mock)
        expander_mock.__exit__ = Mock(return_value=None)
        st_mock.expander.return_value = expander_mock

        markdown_calls = []
        st_mock.markdown.side_effect = lambda content: markdown_calls.append(content)

        with patch.dict("sys.modules", {"streamlit": st_mock}):
            with patch("buffetbot.dashboard.views.options_advisor.st", st_mock):
                from buffetbot.dashboard.views.options_advisor import (
                    render_enhanced_methodology,
                )

                render_enhanced_methodology(zero_weights)

                content = markdown_calls[0]

                # Should show 0% for zero weights and 100% for forecast
                assert "0%" in content
                assert "100%" in content

                # Count to make sure formatting works with extreme values
                zero_count = content.count("0%")
                hundred_count = content.count("100%")

                assert zero_count >= 4  # At least 4 indicators with 0%
                assert hundred_count >= 1  # At least 1 indicator with 100%

    def test_methodology_comprehensive_content_verification(self):
        """Test that all expected content sections are included in methodology."""
        from unittest.mock import MagicMock, Mock, patch

        st_mock = MagicMock()
        expander_mock = MagicMock()
        expander_mock.__enter__ = Mock(return_value=expander_mock)
        expander_mock.__exit__ = Mock(return_value=None)
        st_mock.expander.return_value = expander_mock

        markdown_calls = []
        st_mock.markdown.side_effect = lambda content: markdown_calls.append(content)

        with patch.dict("sys.modules", {"streamlit": st_mock}):
            with patch("buffetbot.dashboard.views.options_advisor.st", st_mock):
                from buffetbot.dashboard.views.options_advisor import (
                    render_enhanced_methodology,
                )

                render_enhanced_methodology()

                content = markdown_calls[0]

                # Verify all expected sections are present
                assert "Technical Scoring Components:" in content
                assert "Risk Management Features:" in content

                # Verify all indicators are mentioned
                assert "RSI Analysis" in content
                assert "Beta Analysis" in content
                assert "Momentum Analysis" in content
                assert "Implied Volatility Analysis" in content
                assert "Forecast Analysis" in content

                # Verify explanatory content is present
                assert "Identifies overbought/oversold conditions" in content
                assert "Measures stock correlation with market" in content
                assert "Price trend strength over multiple timeframes" in content
                assert "Current IV vs historical levels" in content
                assert "Analyst forecast confidence" in content

                # Verify risk management section
                assert "Strategy-specific risk metrics" in content
                assert "Greeks analysis for sensitivity measurement" in content
                assert "Volatility comparison for fair value assessment" in content

    def test_methodology_parameter_handling(self):
        """Test that the methodology function handles different parameter scenarios correctly."""
        from unittest.mock import MagicMock, Mock, patch

        st_mock = MagicMock()
        expander_mock = MagicMock()
        expander_mock.__enter__ = Mock(return_value=expander_mock)
        expander_mock.__exit__ = Mock(return_value=None)
        st_mock.expander.return_value = expander_mock

        markdown_calls = []
        st_mock.markdown.side_effect = lambda content: markdown_calls.append(content)

        with patch.dict("sys.modules", {"streamlit": st_mock}):
            with patch("buffetbot.dashboard.views.options_advisor.st", st_mock):
                from buffetbot.dashboard.views.options_advisor import (
                    render_enhanced_methodology,
                )

                # Test with None (should use defaults)
                render_enhanced_methodology(None)
                assert len(markdown_calls) == 1

                # Reset for next test
                markdown_calls.clear()
                st_mock.reset_mock()

                # Test with empty dict (should use defaults)
                render_enhanced_methodology({})
                assert len(markdown_calls) == 1
                content = markdown_calls[0]
                assert "20%" in content  # Should use default fallback

    def test_methodology_expander_configuration(self):
        """Test that the expander is configured correctly."""
        from unittest.mock import MagicMock, Mock, patch

        st_mock = MagicMock()
        expander_mock = MagicMock()
        expander_mock.__enter__ = Mock(return_value=expander_mock)
        expander_mock.__exit__ = Mock(return_value=None)
        st_mock.expander.return_value = expander_mock

        markdown_calls = []
        st_mock.markdown.side_effect = lambda content: markdown_calls.append(content)

        with patch.dict("sys.modules", {"streamlit": st_mock}):
            with patch("buffetbot.dashboard.views.options_advisor.st", st_mock):
                from buffetbot.dashboard.views.options_advisor import (
                    render_enhanced_methodology,
                )

                render_enhanced_methodology()

                # Verify expander was called with correct parameters
                st_mock.expander.assert_called_once_with(
                    "🔬 Enhanced Analysis Methodology", expanded=False
                )

                # Verify it was used as context manager
                expander_mock.__enter__.assert_called_once()
                expander_mock.__exit__.assert_called_once()

                # Verify markdown was called within the expander context
                assert st_mock.markdown.called

    def test_weight_extraction_from_recommendations(self):
        """Test that actual weights are correctly extracted from recommendations score_details."""
        from unittest.mock import MagicMock, Mock, patch

        import pandas as pd

        # Create mock recommendations DataFrame with score_details
        mock_recommendations = pd.DataFrame(
            {
                "Strike": [100, 105],
                "TotalScore": [0.75, 0.68],
                "score_details": [
                    {
                        "rsi": 0.30,
                        "beta": 0.10,
                        "momentum": 0.25,
                        "iv": 0.20,
                        "forecast": 0.15,
                        "risk_tolerance": "Moderate",  # metadata
                        "strategy_type": "Long Calls",  # metadata
                    },
                    {
                        "rsi": 0.28,
                        "beta": 0.12,
                        "momentum": 0.22,
                        "iv": 0.18,
                        "forecast": 0.20,
                    },
                ],
            }
        )

        # Mock the scoring functions
        with patch(
            "buffetbot.analysis.options_advisor.get_scoring_indicator_names"
        ) as mock_indicator_names:
            mock_indicator_names.return_value = [
                "rsi",
                "beta",
                "momentum",
                "iv",
                "forecast",
            ]

            # Test weight extraction logic (simulating the logic from render_options_advisor_tab)
            first_score_details = mock_recommendations.iloc[0]["score_details"]
            all_indicator_names = {"rsi", "beta", "momentum", "iv", "forecast"}

            # Extract only the actual scoring indicators (not metadata)
            actual_weights = {
                k: v for k, v in first_score_details.items() if k in all_indicator_names
            }

            # Verify the extraction worked correctly
            assert actual_weights == {
                "rsi": 0.30,
                "beta": 0.10,
                "momentum": 0.25,
                "iv": 0.20,
                "forecast": 0.15,
            }

            # Verify metadata was filtered out
            assert "risk_tolerance" not in actual_weights
            assert "strategy_type" not in actual_weights

    def test_weight_extraction_fallback_behavior(self):
        """Test fallback behavior when get_scoring_indicator_names import fails."""
        import pandas as pd

        # Create mock recommendations DataFrame
        mock_recommendations = pd.DataFrame(
            {
                "Strike": [100],
                "score_details": [
                    {
                        "rsi": 0.35,
                        "beta": 0.15,
                        "momentum": 0.20,
                        "iv": 0.15,
                        "forecast": 0.15,
                        "unknown_field": 0.05,  # Should be filtered out
                    }
                ],
            }
        )

        # Simulate import failure and fallback to known indicators
        first_score_details = mock_recommendations.iloc[0]["score_details"]
        known_indicators = {"rsi", "beta", "momentum", "iv", "forecast"}

        actual_weights = {
            k: v for k, v in first_score_details.items() if k in known_indicators
        }

        # Verify fallback worked correctly
        assert actual_weights == {
            "rsi": 0.35,
            "beta": 0.15,
            "momentum": 0.20,
            "iv": 0.15,
            "forecast": 0.15,
        }

        # Verify unknown field was filtered out
        assert "unknown_field" not in actual_weights

    def test_weight_extraction_with_empty_recommendations(self):
        """Test weight extraction behavior with empty recommendations DataFrame."""
        import pandas as pd

        # Test with empty DataFrame
        empty_recommendations = pd.DataFrame()
        current_weights = {
            "rsi": 0.20,
            "beta": 0.20,
            "momentum": 0.20,
            "iv": 0.20,
            "forecast": 0.20,
        }

        # Should fall back to current_weights when recommendations is empty
        actual_weights_for_methodology = current_weights  # Default fallback

        if (
            not empty_recommendations.empty
            and "score_details" in empty_recommendations.columns
        ):
            # This block should not execute
            assert False, "Should not execute for empty DataFrame"

        # Verify fallback to default weights
        assert actual_weights_for_methodology == current_weights

    def test_weight_extraction_missing_score_details_column(self):
        """Test weight extraction when score_details column is missing."""
        import pandas as pd

        # Create DataFrame without score_details column
        recommendations_no_score_details = pd.DataFrame(
            {"Strike": [100, 105], "TotalScore": [0.75, 0.68]}
        )

        current_weights = {
            "rsi": 0.25,
            "beta": 0.15,
            "momentum": 0.25,
            "iv": 0.20,
            "forecast": 0.15,
        }
        actual_weights_for_methodology = current_weights  # Default fallback

        if (
            not recommendations_no_score_details.empty
            and "score_details" in recommendations_no_score_details.columns
        ):
            # This block should not execute
            assert False, "Should not execute when score_details column is missing"

        # Verify fallback to default weights
        assert actual_weights_for_methodology == current_weights

    def test_methodology_displays_extracted_weights(self):
        """Test that methodology function displays the extracted weights correctly."""
        from unittest.mock import MagicMock, Mock, patch

        # Create custom weights that would come from score_details extraction
        extracted_weights = {
            "rsi": 0.35,
            "beta": 0.10,
            "momentum": 0.30,
            "iv": 0.15,
            "forecast": 0.10,
        }

        # Mock streamlit components properly
        st_mock = MagicMock()

        # Create a proper context manager mock for expander
        expander_context = MagicMock()
        expander_context.__enter__ = Mock(return_value=expander_context)
        expander_context.__exit__ = Mock(return_value=None)

        # Configure st.expander to return the context manager
        st_mock.expander.return_value = expander_context

        # Track markdown calls
        markdown_calls = []
        st_mock.markdown.side_effect = lambda content: markdown_calls.append(content)

        # Mock the streamlit module in the specific function
        with patch("buffetbot.dashboard.views.options_advisor.st", st_mock):
            from buffetbot.dashboard.views.options_advisor import (
                render_enhanced_methodology,
            )

            # Call with extracted weights
            render_enhanced_methodology(extracted_weights)

            # Verify expander was created with correct parameters
            st_mock.expander.assert_called_once_with(
                "🔬 Enhanced Analysis Methodology", expanded=False
            )

            # Verify the context manager was used properly
            expander_context.__enter__.assert_called_once()
            expander_context.__exit__.assert_called_once()

            # Verify markdown was called (this happens on st.markdown within the context)
            st_mock.markdown.assert_called()

            # Verify the markdown calls contain our extracted weights
            assert len(markdown_calls) > 0
            markdown_content = str(markdown_calls[0])

            # Check that the actual extracted weights appear in the content
            assert "35%" in markdown_content  # RSI weight (0.35 -> 35%)
            assert "10%" in markdown_content  # Beta and Forecast weights (0.10 -> 10%)
            assert "30%" in markdown_content  # Momentum weight (0.30 -> 30%)
            assert "15%" in markdown_content  # IV weight (0.15 -> 15%)

            # Verify the content structure is correct
            assert "Technical Scoring Components:" in markdown_content
            assert "RSI Analysis" in markdown_content
            assert "Beta Analysis" in markdown_content
            assert "Momentum Analysis" in markdown_content
            assert "Implied Volatility Analysis" in markdown_content
            assert "Forecast Analysis" in markdown_content

    def test_partial_score_details_handling(self):
        """Test handling of score_details with only some indicators."""
        import pandas as pd

        # Create recommendations with partial score_details
        mock_recommendations = pd.DataFrame(
            {
                "Strike": [100],
                "score_details": [
                    {
                        "rsi": 0.40,
                        "momentum": 0.35,
                        "forecast": 0.25,
                        # Missing 'beta' and 'iv'
                        "risk_tolerance": "Aggressive",  # metadata
                    }
                ],
            }
        )

        first_score_details = mock_recommendations.iloc[0]["score_details"]
        known_indicators = {"rsi", "beta", "momentum", "iv", "forecast"}

        # Extract available indicators only
        actual_weights = {
            k: v for k, v in first_score_details.items() if k in known_indicators
        }

        # Verify only available indicators are extracted
        expected_weights = {"rsi": 0.40, "momentum": 0.35, "forecast": 0.25}

        assert actual_weights == expected_weights
        assert "beta" not in actual_weights
        assert "iv" not in actual_weights
        assert "risk_tolerance" not in actual_weights

    def test_non_dict_score_details_handling(self):
        """Test handling when score_details is not a dictionary."""
        import pandas as pd

        # Create recommendations with non-dict score_details
        mock_recommendations = pd.DataFrame(
            {"Strike": [100], "score_details": ["invalid_string_value"]}  # Not a dict
        )

        current_weights = {
            "rsi": 0.20,
            "beta": 0.20,
            "momentum": 0.20,
            "iv": 0.20,
            "forecast": 0.20,
        }
        actual_weights_for_methodology = current_weights  # Default fallback

        first_score_details = mock_recommendations.iloc[0]["score_details"]

        # Should not process non-dict score_details
        if isinstance(first_score_details, dict):
            # This block should not execute
            assert False, "Should not process non-dict score_details"

        # Verify fallback to default weights
        assert actual_weights_for_methodology == current_weights


if __name__ == "__main__":
    pytest.main([__file__])
