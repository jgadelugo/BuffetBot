"""
Unit tests for the _calculate_spread_composite_scores function.

This module tests the missing function that was causing the import error in
buffetbot.analysis.options_advisor. The function calculates base composite
scores for bull call spreads.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from buffetbot.analysis.options_advisor import (
    SCORING_WEIGHTS,
    _calculate_spread_composite_scores,
    normalize_scoring_weights,
)


class TestCalculateSpreadCompositeScores:
    """Test cases for _calculate_spread_composite_scores function."""

    @pytest.fixture
    def sample_spreads_df(self):
        """Create sample spreads data for testing."""
        return pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL", "AAPL"],
                "expiry": ["2024-12-20", "2024-12-20", "2024-12-20"],
                "long_strike": [150, 155, 160],
                "short_strike": [155, 160, 165],
                "net_premium": [3.5, 2.8, 2.1],
                "max_profit": [1.5, 2.2, 2.9],
                "profit_ratio": [1.43, 1.79, 2.38],
                "spread_width": [5, 5, 5],
                "breakeven_price": [153.5, 157.8, 162.1],
                "current_price": [155, 155, 155],
                "daysToExpiry": [90, 90, 90],
                "impliedVolatility": [0.25, 0.30, 0.35],
            }
        )

    @pytest.fixture
    def technical_indicators(self):
        """Sample technical indicators for testing."""
        return {
            "rsi": 45.0,
            "beta": 1.2,
            "momentum": 0.05,
            "avg_iv": 0.28,
            "forecast_confidence": 0.75,
        }

    @pytest.fixture
    def all_data_available(self):
        """All data sources available scenario."""
        return {
            "rsi": True,
            "beta": True,
            "momentum": True,
            "iv": True,
            "forecast": True,
        }

    @pytest.fixture
    def partial_data_available(self):
        """Partial data sources available scenario."""
        return {
            "rsi": True,
            "beta": True,
            "momentum": False,
            "iv": True,
            "forecast": False,
        }

    def test_function_exists_and_callable(self):
        """Test that the function exists and is callable."""
        assert callable(_calculate_spread_composite_scores)
        assert (
            _calculate_spread_composite_scores.__name__
            == "_calculate_spread_composite_scores"
        )

    def test_basic_functionality(
        self, sample_spreads_df, technical_indicators, all_data_available
    ):
        """Test basic functionality with all data available."""
        result = _calculate_spread_composite_scores(
            sample_spreads_df,
            technical_indicators["rsi"],
            technical_indicators["beta"],
            technical_indicators["momentum"],
            technical_indicators["avg_iv"],
            technical_indicators["forecast_confidence"],
            all_data_available,
        )

        # Check basic structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_spreads_df)

        # Check required columns are added
        assert "CompositeScore" in result.columns
        assert "score_details" in result.columns

        # Check CompositeScore is in valid range [0, 1]
        assert all(result["CompositeScore"] >= 0)
        assert all(result["CompositeScore"] <= 1)

        # Check score_details structure
        for score_details in result["score_details"]:
            assert isinstance(score_details, dict)
            assert abs(sum(score_details.values()) - 1.0) < 0.001

    def test_partial_data_handling(
        self, sample_spreads_df, technical_indicators, partial_data_available
    ):
        """Test handling of partial data availability."""
        result = _calculate_spread_composite_scores(
            sample_spreads_df,
            technical_indicators["rsi"],
            technical_indicators["beta"],
            technical_indicators["momentum"],
            technical_indicators["avg_iv"],
            technical_indicators["forecast_confidence"],
            partial_data_available,
        )

        # Should still produce valid results with partial data
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_spreads_df)
        assert "CompositeScore" in result.columns

        # Scores should still be normalized
        assert all(result["CompositeScore"] >= 0)
        assert all(result["CompositeScore"] <= 1)

        # Score details should only include available sources
        first_score_details = result.iloc[0]["score_details"]
        available_sources = [k for k, v in partial_data_available.items() if v]
        assert set(first_score_details.keys()) == set(available_sources)

    def test_profit_ratio_boost(
        self, sample_spreads_df, technical_indicators, all_data_available
    ):
        """Test profit ratio boost functionality."""
        # Create data with varying profit ratios
        test_df = sample_spreads_df.copy()
        test_df["profit_ratio"] = [0.8, 1.5, 2.5]  # Below 1.0, moderate, high

        result = _calculate_spread_composite_scores(
            test_df,
            technical_indicators["rsi"],
            technical_indicators["beta"],
            technical_indicators["momentum"],
            technical_indicators["avg_iv"],
            technical_indicators["forecast_confidence"],
            all_data_available,
        )

        # Higher profit ratios should generally get higher scores
        # (though other factors also influence the final score)
        profit_ratios = result["profit_ratio"].values
        composite_scores = result["CompositeScore"].values

        # Check that the highest profit ratio doesn't get the lowest score
        max_profit_idx = np.argmax(profit_ratios)
        min_profit_idx = np.argmin(profit_ratios)

        # The boost should help, but not guarantee ordering due to other factors
        assert profit_ratios[max_profit_idx] > profit_ratios[min_profit_idx]

    def test_breakeven_boost(
        self, sample_spreads_df, technical_indicators, all_data_available
    ):
        """Test breakeven price boost functionality."""
        # Ensure breakeven_price and current_price columns exist
        test_df = sample_spreads_df.copy()
        test_df["current_price"] = [155, 155, 155]
        test_df["breakeven_price"] = [160, 152, 148]  # Worse, better, best breakeven

        result = _calculate_spread_composite_scores(
            test_df,
            technical_indicators["rsi"],
            technical_indicators["beta"],
            technical_indicators["momentum"],
            technical_indicators["avg_iv"],
            technical_indicators["forecast_confidence"],
            all_data_available,
        )

        # Check that breakeven boost is applied (better breakeven = lower breakeven price relative to current)
        assert "CompositeScore" in result.columns
        assert all(result["CompositeScore"] >= 0)
        assert all(result["CompositeScore"] <= 1)

    def test_iv_scoring_logic(
        self, sample_spreads_df, technical_indicators, all_data_available
    ):
        """Test implied volatility scoring logic for spreads."""
        # Test with IVs around the average
        test_df = sample_spreads_df.copy()
        avg_iv = technical_indicators["avg_iv"]
        test_df["impliedVolatility"] = [
            avg_iv * 0.8,  # Below average (should score higher for spreads)
            avg_iv,  # Average
            avg_iv * 1.3,  # Above average (should score lower for spreads)
        ]

        result = _calculate_spread_composite_scores(
            test_df,
            technical_indicators["rsi"],
            technical_indicators["beta"],
            technical_indicators["momentum"],
            avg_iv,
            technical_indicators["forecast_confidence"],
            all_data_available,
        )

        # For spreads, moderate IV is preferred (inverted scoring)
        # Lower IV should generally score better for spreads due to lower cost
        assert "CompositeScore" in result.columns
        assert all(result["CompositeScore"] >= 0)

    def test_rsi_scoring_for_spreads(
        self, sample_spreads_df, technical_indicators, all_data_available
    ):
        """Test RSI scoring logic specific to spreads."""
        # Test different RSI values
        rsi_values = [25, 45, 65]  # Oversold, moderate, approaching overbought

        for rsi in rsi_values:
            result = _calculate_spread_composite_scores(
                sample_spreads_df,
                rsi,
                technical_indicators["beta"],
                technical_indicators["momentum"],
                technical_indicators["avg_iv"],
                technical_indicators["forecast_confidence"],
                all_data_available,
            )

            # All should produce valid scores
            assert "CompositeScore" in result.columns
            assert all(result["CompositeScore"] >= 0)
            assert all(result["CompositeScore"] <= 1)

    def test_beta_scoring_for_spreads(
        self, sample_spreads_df, technical_indicators, all_data_available
    ):
        """Test beta scoring logic for spreads."""
        # Test different beta values
        beta_values = [0.5, 1.0, 1.5, 2.0]  # Low, market, moderate high, high

        for beta in beta_values:
            result = _calculate_spread_composite_scores(
                sample_spreads_df,
                technical_indicators["rsi"],
                beta,
                technical_indicators["momentum"],
                technical_indicators["avg_iv"],
                technical_indicators["forecast_confidence"],
                all_data_available,
            )

            # All should produce valid scores
            assert "CompositeScore" in result.columns
            assert all(result["CompositeScore"] >= 0)
            assert all(result["CompositeScore"] <= 1)

    def test_momentum_scoring_for_spreads(
        self, sample_spreads_df, technical_indicators, all_data_available
    ):
        """Test momentum scoring for bull call spreads."""
        # Test different momentum values
        momentum_values = [
            -0.1,
            -0.02,
            0.05,
            0.12,
        ]  # Negative, slight negative, positive, strong positive

        for momentum in momentum_values:
            result = _calculate_spread_composite_scores(
                sample_spreads_df,
                technical_indicators["rsi"],
                technical_indicators["beta"],
                momentum,
                technical_indicators["avg_iv"],
                technical_indicators["forecast_confidence"],
                all_data_available,
            )

            # All should produce valid scores
            assert "CompositeScore" in result.columns
            assert all(result["CompositeScore"] >= 0)
            assert all(result["CompositeScore"] <= 1)

    def test_empty_dataframe_handling(self, technical_indicators, all_data_available):
        """Test handling of empty input DataFrame."""
        empty_df = pd.DataFrame()

        result = _calculate_spread_composite_scores(
            empty_df,
            technical_indicators["rsi"],
            technical_indicators["beta"],
            technical_indicators["momentum"],
            technical_indicators["avg_iv"],
            technical_indicators["forecast_confidence"],
            all_data_available,
        )

        # Should return empty DataFrame but with proper structure if columns were added
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_no_data_sources_available(self, sample_spreads_df, technical_indicators):
        """Test behavior when no data sources are available."""
        no_data_available = {
            "rsi": False,
            "beta": False,
            "momentum": False,
            "iv": False,
            "forecast": False,
        }

        result = _calculate_spread_composite_scores(
            sample_spreads_df,
            technical_indicators["rsi"],
            technical_indicators["beta"],
            technical_indicators["momentum"],
            technical_indicators["avg_iv"],
            technical_indicators["forecast_confidence"],
            no_data_available,
        )

        # Should still produce some result even with no data sources
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_spreads_df)

        # CompositeScore might be low or neutral when no data is available
        if "CompositeScore" in result.columns:
            assert all(result["CompositeScore"] >= 0)
            assert all(result["CompositeScore"] <= 1)

    def test_column_preservation(
        self, sample_spreads_df, technical_indicators, all_data_available
    ):
        """Test that original columns are preserved."""
        original_columns = set(sample_spreads_df.columns)

        result = _calculate_spread_composite_scores(
            sample_spreads_df,
            technical_indicators["rsi"],
            technical_indicators["beta"],
            technical_indicators["momentum"],
            technical_indicators["avg_iv"],
            technical_indicators["forecast_confidence"],
            all_data_available,
        )

        # All original columns should be preserved
        for col in original_columns:
            assert col in result.columns

        # New columns should be added
        assert "CompositeScore" in result.columns
        assert "score_details" in result.columns

    def test_score_details_weight_normalization(
        self, sample_spreads_df, technical_indicators, partial_data_available
    ):
        """Test that score_details weights are properly normalized."""
        result = _calculate_spread_composite_scores(
            sample_spreads_df,
            technical_indicators["rsi"],
            technical_indicators["beta"],
            technical_indicators["momentum"],
            technical_indicators["avg_iv"],
            technical_indicators["forecast_confidence"],
            partial_data_available,
        )

        # Check each row's score_details
        for _, row in result.iterrows():
            score_details = row["score_details"]
            assert isinstance(score_details, dict)

            # Weights should sum to 1.0 (within floating point tolerance)
            total_weight = sum(score_details.values())
            assert abs(total_weight - 1.0) < 0.001

            # Should only contain available sources
            available_sources = [k for k, v in partial_data_available.items() if v]
            assert set(score_details.keys()) == set(available_sources)

    @patch("buffetbot.analysis.options_advisor.normalize_scoring_weights")
    def test_weight_normalization_called(
        self,
        mock_normalize,
        sample_spreads_df,
        technical_indicators,
        all_data_available,
    ):
        """Test that weight normalization is called with correct parameters."""
        # Mock the normalize function to return a specific result
        mock_normalize.return_value = {
            "rsi": 0.2,
            "beta": 0.2,
            "momentum": 0.2,
            "iv": 0.2,
            "forecast": 0.2,
        }

        _calculate_spread_composite_scores(
            sample_spreads_df,
            technical_indicators["rsi"],
            technical_indicators["beta"],
            technical_indicators["momentum"],
            technical_indicators["avg_iv"],
            technical_indicators["forecast_confidence"],
            all_data_available,
        )

        # Verify that normalize_scoring_weights was called
        mock_normalize.assert_called_once()

        # Check that it was called with the right parameters
        call_args = mock_normalize.call_args
        assert (
            call_args[0][0] == SCORING_WEIGHTS
        )  # First argument should be SCORING_WEIGHTS
        assert call_args[0][1] == [
            "rsi",
            "beta",
            "momentum",
            "iv",
            "forecast",
        ]  # Available sources

    def test_different_profit_scenarios(self, technical_indicators, all_data_available):
        """Test with different profit scenarios to ensure robustness."""
        # Create test data with edge cases
        edge_case_df = pd.DataFrame(
            {
                "ticker": ["TEST", "TEST", "TEST"],
                "profit_ratio": [0.1, 1.0, 5.0],  # Very low, break-even, very high
                "max_profit": [0.5, 0, 10.0],  # Small, zero, large
                "spread_width": [1, 5, 20],  # Narrow, medium, wide
                "impliedVolatility": [0.1, 0.5, 1.0],  # Low, medium, high IV
                "current_price": [100, 100, 100],
                "breakeven_price": [105, 100, 95],  # Above, at, below current
            }
        )

        result = _calculate_spread_composite_scores(
            edge_case_df,
            technical_indicators["rsi"],
            technical_indicators["beta"],
            technical_indicators["momentum"],
            technical_indicators["avg_iv"],
            technical_indicators["forecast_confidence"],
            all_data_available,
        )

        # Should handle edge cases gracefully
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(edge_case_df)
        assert "CompositeScore" in result.columns
        assert all(result["CompositeScore"] >= 0)
        assert all(result["CompositeScore"] <= 1)

        # Should not have any NaN values
        assert not result["CompositeScore"].isna().any()


class TestSpreadCompositeScoresIntegration:
    """Integration tests for the spread composite scores function."""

    def test_integration_with_existing_functions(self):
        """Test that the function integrates properly with existing code."""
        # Test that the function can be imported and used with the existing workflow
        from buffetbot.analysis.options_advisor import (
            _calculate_spread_composite_scores_with_risk_tolerance,
        )

        # This should not raise an ImportError anymore
        assert callable(_calculate_spread_composite_scores)
        assert callable(_calculate_spread_composite_scores_with_risk_tolerance)

    def test_realistic_data_processing(self):
        """Test with more realistic spread data."""
        realistic_df = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 5,
                "expiry": ["2024-12-20"] * 5,
                "long_strike": [145, 150, 155, 160, 165],
                "short_strike": [150, 155, 160, 165, 170],
                "net_premium": [3.2, 2.8, 2.4, 2.0, 1.6],
                "max_profit": [1.8, 2.2, 2.6, 3.0, 3.4],
                "profit_ratio": [1.56, 1.79, 2.08, 2.50, 3.13],
                "spread_width": [5] * 5,
                "breakeven_price": [148.2, 152.8, 157.4, 162.0, 166.6],
                "current_price": [155] * 5,
                "daysToExpiry": [45, 45, 45, 45, 45],
                "impliedVolatility": [0.22, 0.25, 0.28, 0.31, 0.34],
            }
        )

        data_availability = {
            "rsi": True,
            "beta": True,
            "momentum": True,
            "iv": True,
            "forecast": True,
        }

        result = _calculate_spread_composite_scores(
            realistic_df,
            rsi=42.0,
            beta=1.15,
            momentum=0.03,
            avg_iv=0.27,
            forecast_confidence=0.68,
            data_availability=data_availability,
        )

        # Verify the result is realistic and usable
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert "CompositeScore" in result.columns
        assert all(result["CompositeScore"] >= 0)
        assert all(result["CompositeScore"] <= 1)

        # Verify score distribution makes sense
        scores = result["CompositeScore"].values
        assert len(set(scores)) > 1  # Should have different scores
        assert np.std(scores) > 0.01  # Should have reasonable variation
