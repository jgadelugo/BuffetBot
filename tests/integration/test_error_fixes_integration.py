"""
Integration tests for error fixes.

This module tests that all error fixes work together in integrated scenarios
that simulate real-world usage of the options analysis system.
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from buffetbot.analysis.options.core.domain_models import (
    AnalysisRequest,
    RiskTolerance,
    StrategyType,
    TimeHorizon,
)
from buffetbot.analysis.options_advisor import (
    analyze_options_strategy,
    normalize_scoring_weights,
)


class TestErrorFixesIntegration:
    """Integration tests for all error fixes working together."""

    def test_end_to_end_with_new_time_horizons(self):
        """Test end-to-end workflow with new TimeHorizon values."""
        # Test all new TimeHorizon values
        new_horizons = [TimeHorizon.ONE_YEAR, TimeHorizon.EIGHTEEN_MONTHS]

        for horizon in new_horizons:
            # Create request with new TimeHorizon - should not raise errors
            request = AnalysisRequest(
                ticker="AAPL",
                strategy_type=StrategyType.LONG_CALLS,
                time_horizon=horizon,
                risk_tolerance=RiskTolerance.MODERATE,
                min_days=180,
                top_n=5,
            )

            # Verify all attributes are set correctly
            assert request.time_horizon == horizon
            assert request.time_horizon.value in [
                "One Year (12 months)",
                "18 Months (1.5 years)",
            ]

    def test_weight_normalization_edge_cases_integration(self):
        """Test weight normalization with various data availability scenarios."""
        # Scenario 1: All data sources available
        weights = {"rsi": 0.2, "beta": 0.2, "momentum": 0.2, "iv": 0.2, "forecast": 0.2}
        all_sources = ["rsi", "beta", "momentum", "iv", "forecast"]
        result1 = normalize_scoring_weights(weights, all_sources)
        assert abs(sum(result1.values()) - 1.0) < 0.001
        assert len(result1) == 5

        # Scenario 2: Partial data sources (common in real usage)
        partial_sources = ["rsi", "beta", "momentum"]  # Missing iv and forecast
        result2 = normalize_scoring_weights(weights, partial_sources)
        assert abs(sum(result2.values()) - 1.0) < 0.001
        assert len(result2) == 3

        # Scenario 3: Single data source (edge case)
        single_source = ["rsi"]
        result3 = normalize_scoring_weights(weights, single_source)
        assert result3 == {"rsi": 1.0}

        # Scenario 4: No data sources (fallback case)
        no_sources = []
        result4 = normalize_scoring_weights(weights, no_sources)
        assert result4 == {"neutral": 1.0}

    @patch("buffetbot.analysis.options_advisor.fetch_long_dated_calls")
    @patch("buffetbot.analysis.options_advisor.fetch_price_history")
    @patch("buffetbot.analysis.options_advisor.get_analyst_forecast")
    @patch("buffetbot.analysis.options_advisor.get_peers")
    def test_full_analysis_workflow_with_fixes(
        self, mock_peers, mock_forecast, mock_price, mock_options
    ):
        """Test full analysis workflow incorporating all fixes."""
        # Mock data to simulate successful analysis
        mock_options_df = pd.DataFrame(
            {
                "strike": [100, 105, 110],
                "lastPrice": [5.0, 3.0, 1.5],
                "impliedVolatility": [0.25, 0.28, 0.30],
                "volume": [100, 200, 150],
                "daysToExpiry": [200, 200, 200],
                "expiry": ["2025-12-19"] * 3,
            }
        )

        mock_options.return_value = {
            "data": mock_options_df,
            "data_available": True,
            "error_message": None,
        }

        # Mock price history
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        mock_price.return_value = pd.Series(
            [100 + i * 0.1 for i in range(100)], index=dates
        )

        # Mock forecast
        mock_forecast.return_value = {"confidence": 0.7, "mean_target": 120.0}

        # Mock peers
        mock_peers.return_value = {"data_available": True, "peers": ["MSFT", "GOOGL"]}

        # Test analysis with new TimeHorizon values
        try:
            result = analyze_options_strategy(
                strategy_type="Long Calls",
                ticker="AAPL",
                min_days=180,
                top_n=5,
                risk_tolerance="Moderate",
                time_horizon="One Year (12 months)",  # Using new enum value
            )

            # Should return a DataFrame without errors
            assert isinstance(result, pd.DataFrame)

        except Exception as e:
            # If there are still issues, they should be documented
            pytest.fail(f"Full analysis failed with error: {str(e)}")

    def test_risk_tolerance_and_time_horizon_combinations(self):
        """Test all combinations of risk tolerance and time horizon."""
        risk_tolerances = [
            RiskTolerance.CONSERVATIVE,
            RiskTolerance.MODERATE,
            RiskTolerance.AGGRESSIVE,
        ]

        time_horizons = [
            TimeHorizon.SHORT_TERM,
            TimeHorizon.MEDIUM_TERM,
            TimeHorizon.LONG_TERM,
            TimeHorizon.ONE_YEAR,  # New
            TimeHorizon.EIGHTEEN_MONTHS,  # New
        ]

        # Test all combinations
        for risk in risk_tolerances:
            for horizon in time_horizons:
                request = AnalysisRequest(
                    ticker="TEST",
                    strategy_type=StrategyType.LONG_CALLS,
                    risk_tolerance=risk,
                    time_horizon=horizon,
                    min_days=60,
                    top_n=3,
                )

                # Should create successfully without errors
                assert request.risk_tolerance == risk
                assert request.time_horizon == horizon

    def test_data_resilience_scenarios(self):
        """Test system resilience with various data scenarios."""
        # Test weight normalization with different failure scenarios
        base_weights = {
            "rsi": 0.2,
            "beta": 0.2,
            "momentum": 0.2,
            "iv": 0.2,
            "forecast": 0.2,
        }

        data_scenarios = [
            # Scenario: RSI calculation fails
            (["beta", "momentum", "iv", "forecast"], 4),
            # Scenario: Network issues, only RSI available
            (["rsi"], 1),
            # Scenario: All technical indicators fail
            ([], 1),  # Should use neutral fallback
            # Scenario: Only forecast available
            (["forecast"], 1),
            # Scenario: Mixed success
            (["rsi", "iv"], 2),
        ]

        for available_sources, expected_length in data_scenarios:
            result = normalize_scoring_weights(base_weights, available_sources)

            # Should always return valid weights that sum to 1.0
            assert len(result) == expected_length
            assert abs(sum(result.values()) - 1.0) < 0.001

            # Special case for no sources
            if not available_sources:
                assert "neutral" in result
                assert result["neutral"] == 1.0

    def test_enum_string_conversion_consistency(self):
        """Test that enum string conversions work consistently across the system."""
        # Test TimeHorizon string conversions that were problematic before
        problematic_strings = ["One Year (12 months)", "18 Months (1.5 years)"]

        for time_str in problematic_strings:
            # Should convert to enum successfully
            horizon = TimeHorizon(time_str)
            assert horizon.value == time_str

            # Should work in AnalysisRequest
            request = AnalysisRequest(
                ticker="TEST",
                strategy_type=StrategyType.LONG_CALLS,
                time_horizon=time_str,  # String conversion
            )
            assert request.time_horizon.value == time_str

    def test_backwards_compatibility_preserved(self):
        """Test that all fixes preserve backwards compatibility."""
        # Original TimeHorizon values should still work
        original_horizons = [
            "Short-term (1-3 months)",
            "Medium-term (3-6 months)",
            "Long-term (6+ months)",
        ]

        for horizon_str in original_horizons:
            # Should work exactly as before
            horizon = TimeHorizon(horizon_str)
            assert horizon.value == horizon_str

            # Should work in analysis request
            request = AnalysisRequest(
                ticker="AAPL",
                strategy_type=StrategyType.LONG_CALLS,
                time_horizon=horizon_str,
            )
            assert request.time_horizon.value == horizon_str

        # Original API calls should still work
        original_weights = {"rsi": 0.5, "beta": 0.5}
        result = normalize_scoring_weights(original_weights, ["rsi", "beta"])
        assert isinstance(result, dict)
        assert "rsi" in result
        assert "beta" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
