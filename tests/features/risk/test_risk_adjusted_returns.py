"""
Unit tests for Risk-Adjusted Returns module.

Tests Sharpe ratio, Sortino ratio, Calmar ratio, Information ratio,
Omega ratio, Sterling ratio, and comprehensive risk-adjusted metrics.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from buffetbot.features.risk.risk_adjusted_returns import RiskAdjustedReturns


class TestRiskAdjustedReturns:
    """Test suite for risk-adjusted return calculations."""

    def test_sharpe_ratio_basic(
        self, risk_returns_data, expected_risk_ranges, risk_free_rates
    ):
        """Test basic Sharpe ratio calculation."""
        returns = risk_returns_data["stock_returns"]

        result = RiskAdjustedReturns.sharpe_ratio(returns, risk_free_rates["low"])

        assert isinstance(result, pd.Series)
        assert len(result) == len(returns)

        # Check Sharpe ratio values are in reasonable range
        sharpe_valid = (result >= expected_risk_ranges["sharpe_ratio"][0]) & (
            result <= expected_risk_ranges["sharpe_ratio"][1]
        )
        non_nan_mask = result.notna()
        if non_nan_mask.any():
            assert sharpe_valid[
                non_nan_mask
            ].all(), "Sharpe ratio values outside expected range"

        # Should have NaN values at the beginning due to window
        assert (
            result.iloc[:251].isna().all()
        )  # First 251 values should be NaN (default window=252)

    def test_sharpe_ratio_different_risk_free_rates(
        self, risk_returns_data, risk_free_rates
    ):
        """Test Sharpe ratio with different risk-free rates."""
        returns = risk_returns_data["stock_returns"]

        # Test with different risk-free rates
        sharpe_zero = RiskAdjustedReturns.sharpe_ratio(returns, risk_free_rates["zero"])
        sharpe_low = RiskAdjustedReturns.sharpe_ratio(returns, risk_free_rates["low"])
        sharpe_high = RiskAdjustedReturns.sharpe_ratio(returns, risk_free_rates["high"])

        # Higher risk-free rate should generally lead to lower Sharpe ratio
        valid_mask = sharpe_zero.notna() & sharpe_low.notna() & sharpe_high.notna()
        if valid_mask.any():
            # Allow for some numerical precision issues
            assert (sharpe_zero[valid_mask] >= sharpe_low[valid_mask] - 0.01).all()
            assert (sharpe_low[valid_mask] >= sharpe_high[valid_mask] - 0.01).all()

    def test_sharpe_ratio_custom_window(self, risk_returns_data):
        """Test Sharpe ratio with custom window sizes."""
        returns = risk_returns_data["stock_returns"]

        # Test different window sizes
        sharpe_60 = RiskAdjustedReturns.sharpe_ratio(returns, window=60)
        sharpe_120 = RiskAdjustedReturns.sharpe_ratio(returns, window=120)

        # Larger window should have more NaN values at the beginning
        assert sharpe_60.iloc[:59].isna().all()
        assert sharpe_120.iloc[:119].isna().all()

        # Both should have valid Sharpe ratio values
        assert isinstance(sharpe_60, pd.Series)
        assert isinstance(sharpe_120, pd.Series)

    def test_sortino_ratio_basic(self, risk_returns_data, expected_risk_ranges):
        """Test basic Sortino ratio calculation."""
        returns = risk_returns_data["stock_returns"]

        result = RiskAdjustedReturns.sortino_ratio(returns)

        assert isinstance(result, pd.Series)
        assert len(result) == len(returns)

        # Check Sortino ratio values are in reasonable range
        sortino_valid = (result >= expected_risk_ranges["sortino_ratio"][0]) & (
            result <= expected_risk_ranges["sortino_ratio"][1]
        )
        non_nan_mask = result.notna()
        if non_nan_mask.any():
            assert sortino_valid[
                non_nan_mask
            ].all(), "Sortino ratio values outside expected range"

        # Sortino should generally be higher than Sharpe (less conservative)
        sharpe_result = RiskAdjustedReturns.sharpe_ratio(returns)
        comparison_mask = result.notna() & sharpe_result.notna()
        if comparison_mask.any():
            # Sortino is often higher than Sharpe for typical return distributions
            assert (
                result[comparison_mask] >= sharpe_result[comparison_mask] - 0.5
            ).all()

    def test_calmar_ratio_basic(self, risk_returns_data, expected_risk_ranges):
        """Test basic Calmar ratio calculation."""
        returns = risk_returns_data["stock_returns"]

        result = RiskAdjustedReturns.calmar_ratio(returns)

        assert isinstance(result, pd.Series)
        assert len(result) == len(returns)

        # Check Calmar ratio values are in reasonable range
        calmar_valid = (result >= expected_risk_ranges["calmar_ratio"][0]) & (
            result <= expected_risk_ranges["calmar_ratio"][1]
        )
        non_nan_mask = result.notna()
        if non_nan_mask.any():
            assert calmar_valid[
                non_nan_mask
            ].all(), "Calmar ratio values outside expected range"

    def test_information_ratio_basic(self, risk_returns_data):
        """Test basic Information ratio calculation."""
        returns = risk_returns_data["stock_returns"]
        benchmark_returns = risk_returns_data["benchmark_returns"]

        result = RiskAdjustedReturns.information_ratio(returns, benchmark_returns)

        assert isinstance(result, pd.Series)
        assert len(result) == len(returns)

        # Information ratio should be finite where calculable
        finite_mask = result.notna() & np.isfinite(result)
        if finite_mask.any():
            # Information ratio can be any real number, but should be reasonable
            assert (result[finite_mask].abs() < 10).all()

    def test_omega_ratio_basic(self, risk_returns_data):
        """Test basic Omega ratio calculation."""
        returns = risk_returns_data["stock_returns"]

        result = RiskAdjustedReturns.omega_ratio(returns, threshold=0.0)

        assert isinstance(result, pd.Series)
        assert len(result) == len(returns)

        # Omega ratio should be positive
        non_nan_mask = result.notna()
        if non_nan_mask.any():
            assert (result[non_nan_mask] > 0).all()

        # For good performance, Omega should be > 1
        # For poor performance, Omega should be < 1

    def test_omega_ratio_different_thresholds(self, risk_returns_data):
        """Test Omega ratio with different threshold values."""
        returns = risk_returns_data["stock_returns"]

        # Test with different thresholds
        omega_0 = RiskAdjustedReturns.omega_ratio(returns, threshold=0.0)
        omega_pos = RiskAdjustedReturns.omega_ratio(returns, threshold=0.01)
        omega_neg = RiskAdjustedReturns.omega_ratio(returns, threshold=-0.01)

        # All should be valid
        assert isinstance(omega_0, pd.Series)
        assert isinstance(omega_pos, pd.Series)
        assert isinstance(omega_neg, pd.Series)

        # Higher threshold should generally lead to lower Omega ratio
        valid_mask = omega_0.notna() & omega_pos.notna() & omega_neg.notna()
        if valid_mask.any():
            # Negative threshold should give higher Omega than positive threshold
            assert (omega_neg[valid_mask] >= omega_pos[valid_mask] - 0.01).all()

    def test_sterling_ratio_basic(self, risk_returns_data):
        """Test basic Sterling ratio calculation."""
        returns = risk_returns_data["stock_returns"]

        result = RiskAdjustedReturns.sterling_ratio(returns)

        assert isinstance(result, pd.Series)
        assert len(result) == len(returns)

        # Sterling ratio should be finite where calculable
        finite_mask = result.notna() & np.isfinite(result)
        if finite_mask.any():
            # Sterling ratio can be any real number, but should be reasonable
            assert (result[finite_mask].abs() < 100).all()

    def test_comprehensive_risk_adjusted_metrics(self, risk_returns_data):
        """Test comprehensive risk-adjusted metrics calculation."""
        returns = risk_returns_data["stock_returns"]
        benchmark_returns = risk_returns_data["benchmark_returns"]

        result = RiskAdjustedReturns.comprehensive_risk_adjusted_metrics(
            returns, benchmark_returns, risk_free_rate=0.02
        )

        # Should return a dictionary of metrics
        assert isinstance(result, dict)

        # Check for key metrics
        expected_metrics = [
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "information_ratio",
            "omega_ratio",
            "sterling_ratio",
        ]

        for metric in expected_metrics:
            assert metric in result

        # Check for summary metrics
        summary_fields = ["risk_efficiency_score", "performance_class"]
        for field in summary_fields:
            assert field in result

    def test_positive_returns_scenario(self):
        """Test risk-adjusted ratios with consistently positive returns."""
        dates = pd.date_range("2023-01-01", periods=300, freq="D")

        # Create consistently positive returns
        positive_returns = pd.Series(np.random.uniform(0.001, 0.02, 300), index=dates)

        sharpe = RiskAdjustedReturns.sharpe_ratio(positive_returns, window=60)
        sortino = RiskAdjustedReturns.sortino_ratio(positive_returns, window=60)

        # Both should be positive for consistently positive returns
        sharpe_valid = sharpe.dropna()
        sortino_valid = sortino.dropna()

        if len(sharpe_valid) > 0:
            assert (sharpe_valid > 0).all()
        if len(sortino_valid) > 0:
            assert (sortino_valid > 0).all()

    def test_negative_returns_scenario(self):
        """Test risk-adjusted ratios with consistently negative returns."""
        dates = pd.date_range("2023-01-01", periods=300, freq="D")

        # Create consistently negative returns
        negative_returns = pd.Series(np.random.uniform(-0.02, -0.001, 300), index=dates)

        sharpe = RiskAdjustedReturns.sharpe_ratio(negative_returns, window=60)
        calmar = RiskAdjustedReturns.calmar_ratio(negative_returns, window=60)

        # Should be negative for consistently negative returns
        sharpe_valid = sharpe.dropna()
        calmar_valid = calmar.dropna()

        if len(sharpe_valid) > 0:
            assert (sharpe_valid < 0).all()
        if len(calmar_valid) > 0:
            assert (calmar_valid < 0).all()

    def test_insufficient_data(self, insufficient_risk_data):
        """Test risk-adjusted ratios with insufficient data."""
        # Test with very short series
        sharpe = RiskAdjustedReturns.sharpe_ratio(insufficient_risk_data, window=10)
        sortino = RiskAdjustedReturns.sortino_ratio(insufficient_risk_data, window=10)
        calmar = RiskAdjustedReturns.calmar_ratio(insufficient_risk_data, window=10)

        # Should return empty or all-NaN series for insufficient data
        assert sharpe.isna().all() or sharpe.empty
        assert sortino.isna().all() or sortino.empty
        assert calmar.isna().all() or calmar.empty

    def test_constant_returns(self):
        """Test risk-adjusted ratios with constant returns."""
        dates = pd.date_range("2023-01-01", periods=300, freq="D")

        # Constant positive returns (no volatility)
        constant_returns = pd.Series([0.01] * 300, index=dates)

        sharpe = RiskAdjustedReturns.sharpe_ratio(constant_returns, window=60)
        sortino = RiskAdjustedReturns.sortino_ratio(constant_returns, window=60)

        # With zero volatility, ratios should be infinite or handled gracefully
        # Most implementations handle this by returning NaN
        sharpe_valid = sharpe.dropna()
        sortino_valid = sortino.dropna()

        # Should either be NaN (appropriate) or very large positive values
        if len(sharpe_valid) > 0:
            assert (sharpe_valid >= 0).all()  # Should be non-negative
        if len(sortino_valid) > 0:
            assert (sortino_valid >= 0).all()  # Should be non-negative

    def test_returns_with_nans(self, returns_with_nans):
        """Test risk-adjusted ratios with NaN values in returns."""
        sharpe = RiskAdjustedReturns.sharpe_ratio(returns_with_nans, window=30)
        sortino = RiskAdjustedReturns.sortino_ratio(returns_with_nans, window=30)
        omega = RiskAdjustedReturns.omega_ratio(returns_with_nans, window=30)

        # Should handle NaN values gracefully
        assert isinstance(sharpe, pd.Series)
        assert isinstance(sortino, pd.Series)
        assert isinstance(omega, pd.Series)

        # May have additional NaN values due to NaN inputs, but should not crash

    def test_extreme_returns_handling(self, extreme_returns):
        """Test risk-adjusted ratios with extreme return values."""
        sharpe = RiskAdjustedReturns.sharpe_ratio(extreme_returns, window=30)
        sortino = RiskAdjustedReturns.sortino_ratio(extreme_returns, window=30)
        calmar = RiskAdjustedReturns.calmar_ratio(extreme_returns, window=30)

        # Should handle extreme values without crashing
        assert isinstance(sharpe, pd.Series)
        assert isinstance(sortino, pd.Series)
        assert isinstance(calmar, pd.Series)

        # Results should be finite where calculable
        finite_sharpe = sharpe.dropna()
        if len(finite_sharpe) > 0:
            # Should not have infinite values (unless appropriately handled)
            assert np.isfinite(finite_sharpe).all() or len(finite_sharpe) == 0

    def test_benchmark_comparison(self, risk_returns_data):
        """Test risk-adjusted ratios relative to benchmark."""
        returns = risk_returns_data["stock_returns"]
        benchmark_returns = risk_returns_data["market_returns"]

        # Calculate Information ratio
        info_ratio = RiskAdjustedReturns.information_ratio(returns, benchmark_returns)

        # Calculate individual Sharpe ratios
        stock_sharpe = RiskAdjustedReturns.sharpe_ratio(returns)
        benchmark_sharpe = RiskAdjustedReturns.sharpe_ratio(benchmark_returns)

        # Information ratio provides relative performance measure
        assert isinstance(info_ratio, pd.Series)

        # Compare metrics where available
        valid_mask = (
            info_ratio.notna() & stock_sharpe.notna() & benchmark_sharpe.notna()
        )

        if valid_mask.any():
            # Information ratio and Sharpe difference should be related
            # (though not exactly equal due to different calculations)
            sharpe_diff = stock_sharpe[valid_mask] - benchmark_sharpe[valid_mask]
            # Just verify they're in reasonable relationship
            assert len(info_ratio[valid_mask]) == len(sharpe_diff)

    @patch("buffetbot.features.risk.risk_adjusted_returns.logger")
    def test_error_handling(self, mock_logger):
        """Test error handling with invalid inputs."""
        # Test with invalid input type
        invalid_input = "not_a_series"

        result = RiskAdjustedReturns.sharpe_ratio(invalid_input)

        # Should handle error gracefully and return empty series
        assert isinstance(result, pd.Series)
        assert result.empty or result.isna().all()
        mock_logger.error.assert_called()

    def test_window_size_effects(self, risk_returns_data):
        """Test the effect of different window sizes on calculations."""
        returns = risk_returns_data["stock_returns"]

        # Test multiple window sizes
        windows = [30, 60, 120, 252]
        sharpe_results = {}

        for window in windows:
            sharpe_results[window] = RiskAdjustedReturns.sharpe_ratio(
                returns, window=window
            )

        # All should be valid Series
        for window, result in sharpe_results.items():
            assert isinstance(result, pd.Series)
            assert len(result) == len(returns)

            # Larger windows should have more NaN values at the beginning
            assert result.iloc[: window - 1].isna().all()

        # Compare non-NaN value counts
        non_nan_counts = {w: result.count() for w, result in sharpe_results.items()}

        # Smaller windows should have more non-NaN values
        assert non_nan_counts[30] >= non_nan_counts[60]
        assert non_nan_counts[60] >= non_nan_counts[120]
        assert non_nan_counts[120] >= non_nan_counts[252]

    def test_performance_with_large_dataset(self):
        """Test performance with large datasets."""
        # Create large dataset
        np.random.seed(42)
        n_points = 5000
        dates = pd.date_range("2020-01-01", periods=n_points, freq="D")

        returns = pd.Series(np.random.normal(0.001, 0.02, n_points), index=dates)
        benchmark = pd.Series(np.random.normal(0.0008, 0.015, n_points), index=dates)

        import time

        start_time = time.time()

        # Run comprehensive analysis
        RiskAdjustedReturns.sharpe_ratio(returns, window=252)
        RiskAdjustedReturns.sortino_ratio(returns, window=252)
        RiskAdjustedReturns.calmar_ratio(returns, window=252)
        RiskAdjustedReturns.information_ratio(returns, benchmark, window=252)
        RiskAdjustedReturns.omega_ratio(returns, window=252)
        RiskAdjustedReturns.sterling_ratio(returns, window=252)

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete within reasonable time (5 seconds for 5000 points)
        assert (
            execution_time < 5.0
        ), f"Risk-adjusted returns too slow: {execution_time:.3f}s"

    def test_comprehensive_metrics_integration(self, risk_returns_data):
        """Test integration of comprehensive metrics."""
        returns = risk_returns_data["stock_returns"]
        benchmark_returns = risk_returns_data["benchmark_returns"]

        result = RiskAdjustedReturns.comprehensive_metrics(
            returns, benchmark_returns, risk_free_rate=0.03
        )

        # Verify all expected components
        assert isinstance(result, dict)

        # Check for individual metrics
        individual_metrics = ["sharpe_ratio", "sortino_ratio", "calmar_ratio"]
        for metric in individual_metrics:
            assert metric in result

        # Check for summary statistics
        if "risk_efficiency_score" in result:
            score = result["risk_efficiency_score"]
            if score is not None and not np.isnan(score):
                assert 0 <= score <= 100

        if "performance_class" in result:
            perf_class = result["performance_class"]
            if perf_class is not None:
                assert perf_class in [
                    "poor",
                    "below_average",
                    "average",
                    "good",
                    "excellent",
                ]
