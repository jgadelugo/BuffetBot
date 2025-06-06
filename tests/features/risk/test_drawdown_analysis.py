"""
Tests for Drawdown Analysis functionality.

Professional test suite ensuring comprehensive coverage of drawdown calculations,
maximum drawdown analysis, clustering, and recovery metrics with proper
error handling and performance validation.
"""

import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from buffetbot.features.risk.drawdown_analysis import DrawdownAnalysis


class TestDrawdownAnalysis:
    """Professional test suite for drawdown analysis functionality."""

    def test_basic_drawdown_calculation(
        self, price_data_with_drawdowns, risk_test_utils
    ):
        """Test basic drawdown calculation functionality."""
        prices = price_data_with_drawdowns

        result = DrawdownAnalysis.calculate_drawdowns(prices)

        # Structure validation using updated validator
        assert risk_test_utils.validate_drawdown_output(result)

        # Validate data types and ranges
        assert isinstance(result["drawdown"], pd.Series)
        assert isinstance(result["underwater"], pd.Series)
        assert isinstance(result["drawdown_duration"], pd.Series)
        assert isinstance(result["cumulative_max"], pd.Series)

        # Drawdown should be non-positive (0 or negative)
        valid_drawdown = result["drawdown"].dropna()
        if len(valid_drawdown) > 0:
            assert (valid_drawdown <= 0).all(), "Drawdown values should be non-positive"

        # Cumulative max should be non-decreasing
        valid_cummax = result["cumulative_max"].dropna()
        if len(valid_cummax) > 0:
            assert (
                valid_cummax.diff().dropna() >= 0
            ).all(), "Cumulative max should be non-decreasing"

        # Underwater should be boolean
        assert result["underwater"].dtype == bool or result["underwater"].empty

        # Drawdown duration should be non-negative integers
        valid_duration = result["drawdown_duration"].dropna()
        if len(valid_duration) > 0:
            assert (
                valid_duration >= 0
            ).all(), "Drawdown duration should be non-negative"

    def test_maximum_drawdown_analysis(self, price_data_with_drawdowns):
        """Test maximum drawdown analysis."""
        prices = price_data_with_drawdowns

        result = DrawdownAnalysis.maximum_drawdown_analysis(prices)

        # Structure validation - check for actual field names
        required_fields = [
            "max_drawdown",
            "peak_date",
            "trough_date",
            "recovery_date",
            "drawdown_duration",
            "recovery_duration",
            "total_duration",
        ]
        for field in required_fields:
            assert field in result

        # Max drawdown should be non-positive
        assert result["max_drawdown"] <= 0, "Max drawdown should be non-positive"

        # Duration fields should be non-negative
        assert result["drawdown_duration"] >= 0
        assert result["recovery_duration"] >= 0
        assert result["total_duration"] >= 0

        # Date logic validation
        if result["peak_date"] is not None and result["trough_date"] is not None:
            assert (
                result["peak_date"] <= result["trough_date"]
            ), "Peak should occur before trough"

        if result["trough_date"] is not None and result["recovery_date"] is not None:
            assert (
                result["trough_date"] <= result["recovery_date"]
            ), "Trough should occur before recovery"

    def test_rolling_max_drawdown(self, price_data_with_drawdowns):
        """Test rolling maximum drawdown calculation."""
        prices = price_data_with_drawdowns

        result = DrawdownAnalysis.rolling_max_drawdown(prices, window=60)

        assert isinstance(result, pd.Series)

        # Rolling max drawdown should be non-positive
        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert (
                valid_values <= 0
            ).all(), "Rolling max drawdown should be non-positive"

        # Should have fewer non-NaN values than the original series
        assert result.count() <= len(prices)

    def test_drawdown_clusters(self, price_data_with_drawdowns):
        """Test drawdown clustering analysis."""
        prices = price_data_with_drawdowns

        result = DrawdownAnalysis.drawdown_clusters(prices, min_recovery=0.1)

        # Structure validation - check for actual field names
        assert "clusters" in result  # Changed from 'cluster_details'
        assert "cluster_count" in result  # Changed from 'num_clusters'

        assert isinstance(result["clusters"], pd.DataFrame)
        assert isinstance(result["num_clusters"], int)
        assert result["num_clusters"] >= 0

        # Validate cluster DataFrame structure if not empty
        if not result["clusters"].empty:
            expected_columns = [
                "peak_date",
                "trough_date",
                "recovery_date",
                "max_drawdown",
                "duration",
                "severity_score",
            ]
            for col in expected_columns:
                assert col in result["clusters"].columns

    def test_recovery_analysis(self, price_data_with_drawdowns):
        """Test recovery analysis functionality."""
        prices = price_data_with_drawdowns

        result = DrawdownAnalysis.recovery_analysis(prices)

        # Structure validation - check for actual field names
        expected_fields = [
            "time_to_recovery",
            "avg_recovery_time",
            "recovery_rate",
        ]  # Changed recovery_times to time_to_recovery
        for field in expected_fields:
            assert field in result

        assert isinstance(result["recovery_times"], pd.Series)
        assert isinstance(result["avg_recovery_time"], (int, float, type(None)))
        assert isinstance(result["recovery_rate"], (int, float))

        # Recovery rate should be between 0 and 1
        assert 0 <= result["recovery_rate"] <= 1

        # Average recovery time should be non-negative if not None
        if result["avg_recovery_time"] is not None:
            assert result["avg_recovery_time"] >= 0

    def test_drawdown_risk_features(self, price_data_with_drawdowns):
        """Test drawdown risk feature extraction."""
        prices = price_data_with_drawdowns

        result = DrawdownAnalysis.drawdown_risk_features(prices)

        assert isinstance(result, dict)

        # Expected features (updated to match actual implementation)
        expected_features = [
            "max_drawdown",
            "avg_cluster_depth",
            "avg_drawdown_duration",
            "avg_recovery_time",
            "current_drawdown",
        ]

        for feature in expected_features:
            assert feature in result
            assert isinstance(result[feature], (int, float))

        # Validate feature ranges
        assert result["max_drawdown"] <= 0
        assert result["avg_drawdown"] <= 0
        assert result["drawdown_frequency"] >= 0
        assert result["pain_index"] >= 0  # Pain index should be non-negative
        assert result["ulcer_index"] >= 0  # Ulcer index should be non-negative

    def test_empty_price_series(self):
        """Test handling of empty price series."""
        empty_prices = pd.Series(dtype=float)

        result = DrawdownAnalysis.calculate_drawdowns(empty_prices)

        # Should return empty series for all fields
        assert result["drawdown"].empty
        assert result["underwater"].empty
        assert result["drawdown_duration"].empty
        assert result["cumulative_max"].empty

    def test_single_price_point(self):
        """Test handling of single price point."""
        single_price = pd.Series([100.0])

        result = DrawdownAnalysis.calculate_drawdowns(single_price)

        # Should handle single point gracefully
        assert len(result["drawdown"]) == 1
        assert result["drawdown"].iloc[0] == 0.0  # No drawdown with single point
        assert not result["underwater"].iloc[0]  # Not underwater
        assert result["drawdown_duration"].iloc[0] == 0  # Zero duration

    def test_prices_with_nans(self, price_data_with_drawdowns):
        """Test handling of NaN values in price series."""
        prices = price_data_with_drawdowns.copy()
        # Introduce some NaN values
        prices.iloc[10:15] = np.nan

        result = DrawdownAnalysis.calculate_drawdowns(prices)

        # Should handle NaN values gracefully
        assert isinstance(result["drawdown"], pd.Series)
        assert isinstance(result["underwater"], pd.Series)
        assert isinstance(result["drawdown_duration"], pd.Series)
        assert isinstance(result["cumulative_max"], pd.Series)

    def test_strictly_decreasing_prices(self):
        """Test drawdown calculation with strictly decreasing prices."""
        decreasing_prices = pd.Series(np.arange(100, 50, -1), dtype=float)

        result = DrawdownAnalysis.calculate_drawdowns(decreasing_prices)

        # All points except first should be in drawdown
        underwater_count = result["underwater"].sum()
        assert (
            underwater_count > 0
        ), "Should have underwater periods with decreasing prices"

        # Max drawdown should be significant
        max_dd_analysis = DrawdownAnalysis.maximum_drawdown_analysis(decreasing_prices)
        assert max_dd_analysis["max_drawdown"] < -30, "Should have significant drawdown"

    def test_v_shaped_recovery(self):
        """Test drawdown analysis with V-shaped price recovery."""
        # Create V-shaped price series: decline then recovery
        decline = np.linspace(100, 50, 25)
        recovery = np.linspace(50, 120, 25)
        v_shaped_prices = pd.Series(np.concatenate([decline, recovery[1:]]))

        result = DrawdownAnalysis.maximum_drawdown_analysis(v_shaped_prices)

        # Should identify peak, trough, and recovery
        assert result["peak_date"] is not None
        assert result["trough_date"] is not None
        assert result["recovery_date"] is not None

        # Recovery should occur after trough
        peak_idx = v_shaped_prices.index.get_loc(result["peak_date"])
        trough_idx = v_shaped_prices.index.get_loc(result["trough_date"])
        recovery_idx = v_shaped_prices.index.get_loc(result["recovery_date"])

        assert peak_idx < trough_idx < recovery_idx

    @patch("buffetbot.features.risk.drawdown_analysis.logger")
    def test_error_handling_and_logging(self, mock_logger):
        """Test error handling and logging functionality."""
        # Test with invalid input
        invalid_input = "not_a_series"

        result = DrawdownAnalysis.calculate_drawdowns(invalid_input)

        # Should return empty result and log error
        assert isinstance(result, dict)
        mock_logger.error.assert_called()

    def test_performance_requirements(self):
        """Test that drawdown calculations meet performance requirements."""
        # Create large price series (1000 points)
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)
        prices = pd.Series((1 + returns).cumprod() * 100)

        import time

        start_time = time.time()

        # Run multiple drawdown calculations
        DrawdownAnalysis.calculate_drawdowns(prices)
        DrawdownAnalysis.maximum_drawdown_analysis(prices)
        DrawdownAnalysis.rolling_max_drawdown(prices)
        DrawdownAnalysis.drawdown_clusters(prices)
        DrawdownAnalysis.recovery_analysis(prices)
        DrawdownAnalysis.drawdown_risk_features(prices)

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete within reasonable time (3 seconds for 1000 points)
        assert (
            execution_time < 3.0
        ), f"Drawdown calculations too slow: {execution_time:.3f}s"

    def test_window_size_parameter(self, price_data_with_drawdowns):
        """Test rolling calculations with different window sizes."""
        prices = price_data_with_drawdowns

        # Test different window sizes
        result_small = DrawdownAnalysis.rolling_max_drawdown(prices, window=30)
        result_large = DrawdownAnalysis.rolling_max_drawdown(prices, window=120)

        # Both should return valid results
        assert isinstance(result_small, pd.Series)
        assert isinstance(result_large, pd.Series)

        # Larger window should have fewer non-NaN values
        assert result_large.count() <= result_small.count()

    def test_mathematical_consistency(self, price_data_with_drawdowns):
        """Test mathematical consistency of drawdown calculations."""
        prices = price_data_with_drawdowns

        basic_result = DrawdownAnalysis.calculate_drawdowns(prices)
        max_dd_result = DrawdownAnalysis.maximum_drawdown_analysis(prices)

        # Max drawdown from basic calculation should match max_dd_analysis
        basic_max_dd = basic_result["drawdown"].min()
        max_dd_analysis = max_dd_result["max_drawdown"]

        # Allow for small floating point differences
        assert math.isclose(
            basic_max_dd, max_dd_analysis, rel_tol=1e-10
        ), f"Max drawdown mismatch: {basic_max_dd} vs {max_dd_analysis}"
