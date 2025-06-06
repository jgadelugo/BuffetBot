"""
Tests for VaR (Value at Risk) Metrics functionality.

Professional test suite ensuring comprehensive coverage of VaR calculations,
breach analysis, and tail risk features with proper error handling
and performance validation.
"""

import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from buffetbot.features.risk.var_metrics import VaRMetrics


class TestVaRMetrics:
    """Professional test suite for VaR metrics functionality."""

    def test_historical_var_basic(
        self, risk_returns_data, expected_risk_ranges, risk_test_utils
    ):
        """Test basic historical VaR calculation."""
        returns = risk_returns_data["stock_returns"]

        result = VaRMetrics.historical_var(returns)

        # Structure validation
        assert isinstance(result, dict)
        assert "var_95" in result
        assert "var_99" in result
        assert isinstance(result["var_95"], pd.Series)
        assert isinstance(result["var_99"], pd.Series)

        # Value validation using utility function
        if not result["var_95"].empty:
            is_valid = risk_test_utils.validate_var_values(
                result["var_95"], expected_risk_ranges["var_95"]
            )
            assert is_valid, "VaR 95% values outside expected range"

        # VaR 99% should be more negative than VaR 95%
        valid_mask = result["var_95"].notna() & result["var_99"].notna()
        if valid_mask.any():
            assert (result["var_99"][valid_mask] <= result["var_95"][valid_mask]).all()

    def test_historical_var_custom_confidence(self, risk_returns_data):
        """Test historical VaR with custom confidence levels."""
        returns = risk_returns_data["stock_returns"]
        custom_levels = [0.90, 0.95, 0.99, 0.995]

        result = VaRMetrics.historical_var(returns, confidence_levels=custom_levels)

        # Check all requested confidence levels are present
        expected_keys = [f"var_{int(level*100)}" for level in custom_levels]
        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], pd.Series)

        # VaR should be monotonically decreasing with confidence level
        valid_data = pd.DataFrame({key: result[key] for key in expected_keys}).dropna()
        if not valid_data.empty:
            for i in range(len(expected_keys) - 1):
                current_key = expected_keys[i]
                next_key = expected_keys[i + 1]
                assert (valid_data[next_key] <= valid_data[current_key]).all()

    def test_historical_var_insufficient_data(self, insufficient_risk_data):
        """Test VaR calculation with insufficient data."""
        result = VaRMetrics.historical_var(insufficient_risk_data)

        # Should return empty series for insufficient data
        assert result["var_95"].empty or result["var_95"].isna().all()
        assert result["var_99"].empty or result["var_99"].isna().all()

    def test_parametric_var_normal_distribution(
        self, risk_returns_data, expected_risk_ranges
    ):
        """Test parametric VaR using normal distribution."""
        returns = risk_returns_data["stock_returns"]

        result = VaRMetrics.parametric_var(returns, distribution="normal")

        assert "var_95" in result
        assert "var_99" in result
        assert isinstance(result["var_95"], pd.Series)
        assert isinstance(result["var_99"], pd.Series)

        # Check values are in expected ranges
        var_95_valid = (result["var_95"] >= expected_risk_ranges["var_95"][0]) & (
            result["var_95"] <= expected_risk_ranges["var_95"][1]
        )
        assert var_95_valid.all() or result["var_95"].isna().all()

    def test_parametric_var_t_distribution(self, risk_returns_data):
        """Test parametric VaR using t-distribution."""
        returns = risk_returns_data["stock_returns"]

        result = VaRMetrics.parametric_var(returns, distribution="t")

        assert "var_95" in result
        assert "var_99" in result

        # t-distribution VaR should generally be more conservative (more negative)
        # than normal distribution VaR for fat-tailed data
        normal_result = VaRMetrics.parametric_var(returns, distribution="normal")

        # Compare where both have valid values
        valid_mask = result["var_95"].notna() & normal_result["var_95"].notna()
        if valid_mask.any():
            # Both should be negative for loss estimates
            assert (result["var_95"][valid_mask] < 0).all()
            assert (normal_result["var_95"][valid_mask] < 0).all()

    def test_parametric_var_invalid_distribution(self, risk_returns_data):
        """Test parametric VaR with invalid distribution."""
        returns = risk_returns_data["stock_returns"]

        # Should handle invalid distribution gracefully by defaulting to normal
        result = VaRMetrics.parametric_var(returns, distribution="invalid")

        # Should still return valid structure
        assert "var_95" in result
        assert "var_99" in result
        assert isinstance(result["var_95"], pd.Series)
        assert isinstance(result["var_99"], pd.Series)

    def test_expected_shortfall_calculation(
        self, risk_returns_data, expected_risk_ranges
    ):
        """Test Expected Shortfall (Conditional VaR) calculation."""
        returns = risk_returns_data["stock_returns"]

        result = VaRMetrics.expected_shortfall(returns)

        assert "es_95" in result
        assert "es_99" in result
        assert isinstance(result["es_95"], pd.Series)
        assert isinstance(result["es_99"], pd.Series)

        # ES should be more negative than VaR (worse than VaR)
        var_result = VaRMetrics.historical_var(returns)

        valid_mask = result["es_95"].notna() & var_result["var_95"].notna()
        if valid_mask.any():
            assert (
                result["es_95"][valid_mask] <= var_result["var_95"][valid_mask]
            ).all()

    def test_var_breach_analysis(self, risk_returns_data):
        """Test VaR breach analysis functionality."""
        returns = risk_returns_data["stock_returns"]
        var_result = VaRMetrics.historical_var(returns)

        if not var_result["var_95"].empty:
            breach_analysis = VaRMetrics.var_breach_analysis(
                returns, var_result["var_95"], confidence_level=0.95
            )

            assert "breach_rate" in breach_analysis
            assert "expected_breach_rate" in breach_analysis
            assert "total_breaches" in breach_analysis
            assert "breaches" in breach_analysis

            # Breach rate should be between 0 and 1
            assert 0 <= breach_analysis["breach_rate"] <= 1

            # Expected breach rate should match confidence level (with floating point tolerance)
            assert math.isclose(
                breach_analysis["expected_breach_rate"], 0.05, rel_tol=1e-10
            )

            # Total breaches should be non-negative integer
            assert isinstance(breach_analysis["total_breaches"], (int, np.integer))
            assert breach_analysis["total_breaches"] >= 0

    def test_tail_risk_features(self, risk_returns_data):
        """Test tail risk feature calculations."""
        returns = risk_returns_data["stock_returns"]

        result = VaRMetrics.tail_risk_features(returns)

        expected_features = [
            "skewness",
            "kurtosis",
            "tail_ratio",
            "extreme_loss_freq",
            "tail_expectation",
        ]
        for feature in expected_features:
            assert feature in result
            assert isinstance(result[feature], pd.Series)

        # Validate feature ranges
        if not result["skewness"].empty:
            # Skewness can be any real number, but should be reasonable
            assert result["skewness"].abs().max() < 10

        if not result["kurtosis"].empty:
            # Kurtosis should be reasonable for financial data
            valid_kurtosis = result["kurtosis"].dropna()
            if len(valid_kurtosis) > 0:
                assert (
                    valid_kurtosis.max() < 100
                )  # Very high but reasonable upper bound

        if not result["extreme_loss_freq"].empty:
            # Extreme loss frequency should be between 0 and 1
            valid_extreme_loss = result["extreme_loss_freq"].dropna()
            if len(valid_extreme_loss) > 0:
                assert ((valid_extreme_loss >= 0) & (valid_extreme_loss <= 1)).all()

    def test_extreme_returns_handling(self, extreme_returns):
        """Test VaR calculations with extreme return values."""
        result = VaRMetrics.historical_var(extreme_returns)

        # Should handle extreme values without crashing
        assert isinstance(result, dict)
        assert "var_95" in result
        assert "var_99" in result

        # VaR values should capture extreme losses
        if not result["var_95"].empty:
            var_95_values = result["var_95"].dropna()
            if len(var_95_values) > 0:
                # With extreme negative returns, VaR should be quite negative
                assert var_95_values.min() < -0.05  # At least -5%

    def test_returns_with_nans(self, returns_with_nans):
        """Test VaR calculations with NaN values in returns."""
        result = VaRMetrics.historical_var(returns_with_nans)

        # Should handle NaN values gracefully
        assert isinstance(result, dict)
        assert "var_95" in result
        assert "var_99" in result

        # Results may have NaN values, but should not crash
        assert isinstance(result["var_95"], pd.Series)
        assert isinstance(result["var_99"], pd.Series)

    def test_window_size_parameter(self, risk_returns_data):
        """Test VaR calculation with different window sizes."""
        returns = risk_returns_data["stock_returns"]

        # Test with different window sizes
        result_small = VaRMetrics.historical_var(returns, window=30)
        result_large = VaRMetrics.historical_var(returns, window=120)

        # Both should return valid results
        assert isinstance(result_small["var_95"], pd.Series)
        assert isinstance(result_large["var_95"], pd.Series)

        # Larger window should have fewer NaN values at the beginning
        assert result_large["var_95"].count() <= result_small["var_95"].count()

    @patch("buffetbot.features.risk.var_metrics.logger")
    def test_error_handling_and_logging(self, mock_logger, risk_returns_data):
        """Test error handling and logging functionality."""
        # Test with invalid input
        invalid_input = "not_a_series"

        result = VaRMetrics.historical_var(invalid_input)

        # Should return empty result and log error
        assert isinstance(result, dict)
        mock_logger.error.assert_called()

    def test_performance_requirements(self, performance_data):
        """Test that VaR calculations meet performance requirements."""
        # Use the performance data fixture (1000 points)
        returns = pd.Series(np.random.normal(0, 0.02, 1000))

        import time

        start_time = time.time()

        # Run multiple VaR calculations
        VaRMetrics.historical_var(returns)
        VaRMetrics.parametric_var(returns)
        VaRMetrics.expected_shortfall(returns)
        VaRMetrics.tail_risk_features(returns)

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete within reasonable time (3 seconds for 1000 points)
        assert execution_time < 3.0, f"VaR calculations too slow: {execution_time:.3f}s"
